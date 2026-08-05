"""NPKGRIDS v1.08 crop-specific fertilizer application rates.

NPKGRIDS (Nguyen, Tang, Conchedda et al. 2024) publishes one netCDF per crop on
a global 0.05 degree grid, each holding an N, a P2O5 and a K2O application rate
in kg/ha together with a data-quality score and a provenance code:

======================  ==========================================
``Nrate``               N application rate [kg-N/ha]
``P2O5rate``            P2O5 application rate [kg-P2O5/ha]
``K2Orate``             K2O application rate [kg-K2O/ha]
``<nutrient>qual``      data quality, 1 = highest .. 0 = lowest
``<nutrient>set``       source dataset id (see the file attributes)
======================  ==========================================

Two sentinels matter. ``-1`` marks ocean and is never a rate. A ``0`` marks a
land pixel the crop is not grown on (or whose rate is undefined) — averaging
those into a target cell would dilute the rate of the pixels that *do* grow the
crop, so by default only positive pixels contribute and a cell with none is left
NaN (and thus dropped from the export). Set ``npk.include_zero_rate`` to treat
the zeros as genuine zero-application land instead.
"""

from __future__ import annotations

import logging
from pathlib import Path

import numpy as np
import xarray as xr

from data4simplace.config import PipelineConfig
from data4simplace.grid import TargetGrid

logger = logging.getLogger(__name__)

#: NPKGRIDS variable -> the canonical name carried on the target grid. The
#: canonical names keep the oxide form the source reports; the exporter converts
#: to elemental P/K when it inverts a rate into a product amount.
RATE_VARIABLES: dict[str, str] = {
    "Nrate": "N",
    "P2O5rate": "P2O5",
    "K2Orate": "K2O",
}
#: Companion quality fields, carried through for provenance/filtering.
QUALITY_VARIABLES: dict[str, str] = {
    "Nqual": "N_quality",
    "P2O5qual": "P2O5_quality",
    "K2Oqual": "K2O_quality",
}
#: Ocean / no-data sentinel declared in the per-variable ``Ocean`` attribute.
OCEAN_SENTINEL = -1.0


class NPKGridsHandler:
    """Load one NPKGRIDS crop file and align it to the 10 km target grid.

    Parameters
    ----------
    config:
        The validated pipeline configuration. ``paths.npk_root`` must point at
        the directory of NPKGRIDS netCDF files and ``npk.crop`` names the crop.
    """

    def __init__(self, config: PipelineConfig) -> None:
        self._config = config
        self._npk = config.npk
        self._grid = TargetGrid.from_config(config.grid)

    # ------------------------------------------------------------------ #
    # Discovery
    # ------------------------------------------------------------------ #
    def resolve_file(self) -> Path:
        """Locate the netCDF for the configured crop.

        Matching is on the filename stem's crop suffix rather than a fixed
        version string, so a future NPKGRIDS release drops in without a config
        change.

        Raises
        ------
        ValueError
            If ``paths.npk_root`` is unset or is not a directory.
        FileNotFoundError
            If no file for the crop exists under the root.
        """
        root = self._config.paths.npk_root
        if root is None:
            raise ValueError("npk.source 'npkgrids' requires paths.npk_root to be set")
        root = Path(root)
        if not root.is_dir():
            raise ValueError(f"paths.npk_root is not a directory: {root}")

        crop = self._npk.crop.lower()
        matches = sorted(p for p in root.glob("*.nc") if p.stem.lower().endswith(f"_{crop}"))
        if not matches:
            available = sorted(p.stem.rsplit("_", 1)[-1] for p in root.glob("*.nc"))
            raise FileNotFoundError(
                f"No NPKGRIDS file for crop {self._npk.crop!r} under {root}. "
                f"Available crops: {', '.join(available[:20])}"
                + (" ..." if len(available) > 20 else "")
            )
        if len(matches) > 1:
            logger.warning(
                "Multiple NPKGRIDS files match crop %r (%s); using the last",
                self._npk.crop,
                ", ".join(p.name for p in matches),
            )
        return matches[-1]

    # ------------------------------------------------------------------ #
    # Loading
    # ------------------------------------------------------------------ #
    def _subset(self, path: Path) -> xr.Dataset:
        """Open the crop file and cut it to the target grid's bounding box.

        The file is global at 0.05 degrees (7200 x 3600); subsetting first keeps
        the whole stage well inside a single tile's memory budget.
        """
        ds = xr.open_dataset(path)
        grid = self._config.grid
        # NPKGRIDS stores latitude south-to-north, so the slice runs min -> max.
        pad = grid.resolution_deg
        return ds.sel(
            lon=slice(grid.min_lon - pad, grid.max_lon + pad),
            lat=slice(grid.min_lat - pad, grid.max_lat + pad),
        )

    def _clean(self, rate: xr.DataArray) -> xr.DataArray:
        """Mask the ocean sentinel and (by default) the zero-rate land pixels."""
        cleaned = rate.where(rate > OCEAN_SENTINEL / 2.0)
        if not self._npk.include_zero_rate:
            cleaned = cleaned.where(cleaned > 0.0)
        return cleaned

    def load(self) -> xr.Dataset:
        """Load, clean and regrid the crop's N/P2O5/K2O rates.

        Returns
        -------
        xarray.Dataset
            ``N``, ``P2O5`` and ``K2O`` in kg/ha on the target grid, plus the
            ``*_quality`` scores when ``npk.keep_quality`` is set. Cells with no
            qualifying source pixel are NaN.
        """
        path = self.resolve_file()
        source = self._subset(path)
        logger.info(
            "NPKGRIDS %s: %s, %d x %d source pixels",
            self._npk.crop,
            path.name,
            source.sizes.get("lat", 0),
            source.sizes.get("lon", 0),
        )

        # The quality fields are also needed as an input to min_quality, even
        # when they are not themselves exported.
        wanted = dict(RATE_VARIABLES)
        if self._npk.keep_quality or self._npk.min_quality is not None:
            wanted.update(QUALITY_VARIABLES)

        cleaned: dict[str, xr.DataArray] = {}
        for src_name, name in wanted.items():
            if src_name not in source.data_vars:
                logger.warning("NPKGRIDS file %s has no variable %s", path.name, src_name)
                continue
            layer = source[src_name].load()
            if src_name in RATE_VARIABLES:
                layer = self._clean(layer)
            else:
                # Quality shares the ocean sentinel but not the zero convention.
                layer = layer.where(layer > OCEAN_SENTINEL / 2.0)
            cleaned[name] = layer

        if not cleaned:
            logger.warning("NPKGRIDS file %s carried none of the expected rates", path.name)
            return xr.Dataset()

        stacked = xr.Dataset(cleaned)
        if self._npk.min_quality is not None:
            stacked = self._apply_quality_filter(stacked)
        if not self._npk.keep_quality:
            stacked = stacked.drop_vars(QUALITY_VARIABLES.values(), errors="ignore")

        regridded = self._grid.regrid(stacked, method="mean")
        assert isinstance(regridded, xr.Dataset)  # a Dataset in, a Dataset out
        for name, source_name in ((v, k) for k, v in RATE_VARIABLES.items()):
            if name in regridded:
                regridded[name].attrs = {
                    "long_name": f"{name} application rate ({self._npk.crop})",
                    "units": f"kg-{name}/ha",
                    "source": path.name,
                    "source_variable": source_name,
                }
        regridded.attrs["npkgrids_file"] = str(path)
        regridded.attrs["npkgrids_crop"] = self._npk.crop

        valid = int(np.isfinite(regridded["N"].values).sum()) if "N" in regridded else 0
        logger.info("NPKGRIDS aligned to %d target cells with an N rate", valid)
        return regridded

    def _apply_quality_filter(self, data: xr.Dataset) -> xr.Dataset:
        """Drop source pixels whose nutrient quality score is below the floor."""
        threshold = float(self._npk.min_quality or 0.0)
        out = data.copy()
        for rate_name, quality_name in (
            ("N", "N_quality"),
            ("P2O5", "P2O5_quality"),
            ("K2O", "K2O_quality"),
        ):
            if rate_name in out and quality_name in out:
                out[rate_name] = out[rate_name].where(out[quality_name] >= threshold)
        return out
