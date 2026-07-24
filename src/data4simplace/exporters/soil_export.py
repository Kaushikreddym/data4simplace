"""SIMPLACE soil profile file generator.

The SIMPLACE soil reference (``soil.csv``) is a **wide** table: one row per
``location``, with per-layer soil properties encoded as ``<stem>_<N>`` columns
(e.g. ``clay_1`` … ``clay_6``) plus single-value profile attributes. This
exporter reproduces that structure from the regridded SoilGrids dataset:

* **Spatial properties** (``clay``, ``sand``, ``bulkdensity``, ``carbon``,
  ``PH``) are taken from SoilGrids, remapped from SoilGrids' standard depth
  intervals onto the SIMPLACE layer boundaries by overlap-weighted averaging.
* **Hydraulic properties** (``soilwater_fc``/``_wp``/``_sat``/``_init``) are
  filled from the Saxton–Rawls pedotransfer output when available.
* **Non-derivable columns** (soiltype, mineral N, CaCO3, P pools, initialisation
  constants …) are carried over from the reference template row so the file
  stays runnable; without a reference they fall back to the missing sentinel.

Keying: outputs use our own 10 km grid cell as ``location`` (each carries a full
lat/lon). The reference ``location`` IDs are not reused because the project's
``location.csv`` provides latitude only — sampling SoilGrids needs longitude too.
"""

from __future__ import annotations

import logging
import re
from pathlib import Path

import numpy as np
import pandas as pd
import xarray as xr

from data4simplace.config import PipelineConfig
from data4simplace.exporters.base_exporter import BaseExporter, ReferenceSpec

logger = logging.getLogger(__name__)

# Reference column stem  ->  source SoilGrids variable (units already unscaled:
# clay/sand in %, bulkdensity in kg/dm3, carbon in g/kg, PH on the pH scale).
_STEM_TO_SOILGRIDS = {
    "clay": "clay",
    "sand": "sand",
    "bulkdensity": "bdod",
    "carbon": "soc",
    "PH": "phh2o",
}
# Reference column stem  ->  Saxton-Rawls PTF variable (volumetric, m3/m3).
_STEM_TO_PTF = {
    "soilwater_fc": "theta_fc",
    "soilwater_wp": "theta_wp",
    "soilwater_sat": "theta_sat",
    "soilwater_init": "theta_fc",  # initial water content initialised at field capacity
}
# SIMPLACE layer bottom depths (m) used when the reference does not specify them.
_DEFAULT_LAYER_BOTTOMS_M = (0.1, 0.3, 0.5, 0.7, 1.0, 2.0)
# Matches a per-layer column ``<stem>_<N>``.
_LAYER_COL_RE = re.compile(r"^(?P<stem>.+)_(?P<n>\d+)$")


def _parse_interval_cm(label: str) -> tuple[float, float]:
    """Parse a SoilGrids depth label such as ``"5-15cm"`` -> ``(5.0, 15.0)``."""
    token = label.lower().replace("cm", "").strip()
    top, bottom = token.split("-")
    return float(top), float(bottom)


def _bottoms_to_intervals_cm(bottoms_cm: list[float]) -> list[tuple[float, float]]:
    """Turn cumulative bottom depths into ``(top, bottom)`` intervals."""
    intervals: list[tuple[float, float]] = []
    top = 0.0
    for bottom in bottoms_cm:
        intervals.append((top, bottom))
        top = bottom
    return intervals


def remap_depth_weighted(
    values: np.ndarray,
    src_intervals: list[tuple[float, float]],
    dst_intervals: list[tuple[float, float]],
) -> np.ndarray:
    """Overlap-weighted remap of a depth-resolved cube onto new layers.

    Parameters
    ----------
    values:
        Array shaped ``(n_src_layers, ...)``.
    src_intervals, dst_intervals:
        Lists of ``(top, bottom)`` depth intervals (same unit) for the source
        and destination layering.

    Returns
    -------
    numpy.ndarray
        Array shaped ``(n_dst_layers, ...)``; destination layers with no source
        overlap are ``NaN``.
    """
    out = np.full((len(dst_intervals),) + values.shape[1:], np.nan, dtype="float64")
    for j, (d_top, d_bot) in enumerate(dst_intervals):
        acc = np.zeros(values.shape[1:], dtype="float64")
        wsum = 0.0
        for i, (s_top, s_bot) in enumerate(src_intervals):
            overlap = max(0.0, min(d_bot, s_bot) - max(d_top, s_top))
            if overlap > 0:
                acc = acc + np.nan_to_num(values[i]) * overlap
                wsum += overlap
        if wsum > 0:
            out[j] = acc / wsum
    return out


class SoilExporter(BaseExporter):
    """Export a SIMPLACE wide-format soil profile file from soil data."""

    kind = "soil"

    def __init__(self, config: PipelineConfig, reference_path: str | Path | None) -> None:
        super().__init__(config, reference_path)
        self._template: dict[str, object] | None = None

    def fallback_spec(self) -> ReferenceSpec:
        """Compact wide schema used when no reference file is available."""
        stems = ["clay", "sand", "bulkdensity", "carbon", "PH",
                 "soilwater_fc", "soilwater_wp", "soilwater_sat", "soilwater_init"]
        columns = ["location"]
        columns += [f"{stem}_{n}" for n in range(1, 7) for stem in stems]
        columns += [f"SoilLayerDepth_{n}" for n in range(1, 7)]
        return ReferenceSpec(
            delimiter=",",
            columns=columns,
            missing_value=str(self._config.missing_value),
        )

    # ------------------------------------------------------------------ #
    # Reference template (constant / non-derivable columns)
    # ------------------------------------------------------------------ #
    def _reference_template(self) -> dict[str, object]:
        """First reference row as ``{column: value}`` for carry-over defaults."""
        if self._template is not None:
            return self._template
        ref_file = self._resolve_reference_file()
        if ref_file is None:
            self._template = {}
            return self._template
        row = pd.read_csv(ref_file, sep=self.spec.delimiter, nrows=1).iloc[0]
        self._template = {str(k): v for k, v in row.to_dict().items()}
        return self._template

    def _layer_bottoms_cm(self, n_layers: int) -> list[float]:
        """SIMPLACE layer bottom depths (cm), read from the reference if present."""
        template = self._reference_template()
        bottoms: list[float] = []
        for n in range(1, n_layers + 1):
            val = template.get(f"SoilLayerDepth_{n}", template.get(f"depth_{n}"))
            if val is None:
                bottoms = []
                break
            bottoms.append(float(val) * 100.0)  # metres -> cm
        if not bottoms:
            bottoms = [b * 100.0 for b in _DEFAULT_LAYER_BOTTOMS_M][:n_layers]
        return bottoms

    @staticmethod
    def _n_layers(columns: list[str]) -> int:
        """Highest per-layer suffix present in the reference columns."""
        suffixes = [int(m.group("n")) for c in columns if (m := _LAYER_COL_RE.match(c))]
        return max(suffixes) if suffixes else 6

    # ------------------------------------------------------------------ #
    # Frame construction
    # ------------------------------------------------------------------ #
    def build_frame(
        self,
        soil: xr.Dataset,
        cell_table: pd.DataFrame,
        hydraulic: xr.Dataset | None = None,
    ) -> pd.DataFrame:
        """Build the wide-format soil table (one row per grid cell / location).

        Parameters
        ----------
        soil:
            Regridded soil dataset; variables carry a ``depth`` dimension.
        cell_table:
            Grid cell table (``SimplaceID``, row, col, lat, lon).
        hydraulic:
            Optional Saxton-Rawls dataset aligned to ``soil`` (with ``depth``).
        """
        if "depth" not in soil.dims:
            raise ValueError("Soil dataset must carry a 'depth' dimension")

        columns = self.spec.columns or self.fallback_spec().columns
        n_layers = self._n_layers(columns)
        src_intervals = [_parse_interval_cm(str(d)) for d in soil["depth"].values]
        dst_intervals = _bottoms_to_intervals_cm(self._layer_bottoms_cm(n_layers))
        template = self._reference_template()

        # Materialise once, then remap every derivable variable onto SIMPLACE
        # layers: {stem: array shaped (n_layers, n_lat, n_lon)}.
        soil = soil.load()
        remapped: dict[str, np.ndarray] = {}
        for stem, var in _STEM_TO_SOILGRIDS.items():
            if var in soil.data_vars:
                remapped[stem] = remap_depth_weighted(
                    np.asarray(soil[var].values), src_intervals, dst_intervals
                )
        if hydraulic is not None and "depth" in hydraulic.dims:
            hydraulic = hydraulic.load()
            for stem, var in _STEM_TO_PTF.items():
                if var in hydraulic.data_vars:
                    remapped[stem] = remap_depth_weighted(
                        np.asarray(hydraulic[var].values), src_intervals, dst_intervals
                    )

        rows = np.asarray(cell_table["row"], dtype=int)
        cols = np.asarray(cell_table["col"], dtype=int)
        sentinel = self.spec.missing_value

        data: dict[str, object] = {}
        for col in columns:
            if col == "location":
                data[col] = np.asarray(cell_table["SimplaceID"], dtype=np.int64)
                continue
            match = _LAYER_COL_RE.match(col)
            if match and match.group("stem") in remapped:
                layer = int(match.group("n")) - 1
                cube = remapped[match.group("stem")]
                if layer < cube.shape[0]:
                    values = cube[layer, rows, cols]
                    data[col] = np.round(values, 5)
                    continue
            # Non-derivable: carry the reference constant, else the sentinel.
            data[col] = template.get(col, sentinel)

        frame = pd.DataFrame(data, columns=columns)

        # Drop cells with no valid derived soil data (e.g. masked-out cells).
        derived_cols = [
            c for c in columns
            if (m := _LAYER_COL_RE.match(c)) and m.group("stem") in remapped
        ]
        if derived_cols:
            valid = ~frame[derived_cols].apply(pd.to_numeric, errors="coerce").isna().all(axis=1)
            frame = frame[valid].reset_index(drop=True)
        return frame

    def export(
        self,
        soil: xr.Dataset,
        cell_table: pd.DataFrame,
        output_dir: str | Path,
        hydraulic: xr.Dataset | None = None,
    ) -> Path:
        """Write the wide-format soil file; return its path."""
        frame = self.build_frame(soil, cell_table, hydraulic)
        out_path = Path(output_dir) / "soil" / "soil.csv"
        return self.write_csv(frame, out_path)
