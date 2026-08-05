"""Irrigated / rainfed classification of the target grid from harvested area.

A SIMPLACE run needs to know whether a cell's crop is irrigated, and neither
gridded product states that directly — both publish *areas*. The class here turns
the irrigated and rainfed harvested-area grids of a crop into one binary label
per target cell:

.. math::

    f = \\frac{A_\\text{irrigated}}{A_\\text{irrigated} + A_\\text{rainfed}}
    \\qquad
    v_\\text{IRR} = \\begin{cases} 1 & f > \\text{threshold} \\\\
                                   0 & \\text{otherwise} \\end{cases}

``irrigation.threshold`` sets the cut (default 0.5). A cell holding less than
``irrigation.min_crop_area_ha`` of the crop is **unclassified**: its fraction is
NaN and, because SIMPLACE wants a plain flag, it is written as ``0`` — the same
value a rainfed cell gets. The distinction survives in
:attr:`IrrigationClassification.fraction` and
:attr:`IrrigationClassification.source_id` for anyone who needs it.

Sources
-------
``irrigation.source`` picks between two independent products:

``mirca``
    MIRCA-OS v0.1 Monthly Growing Area Grids, 5 arcmin, global. One netCDF per
    crop, sub-crop and system (``_ir`` / ``_rf``) holding a monthly growing area
    in ha. The annual harvested area is MIRCA-OS's own definition — the sum over
    sub-crops of each sub-crop's peak month — computed for both systems.

``ecira``
    ECIRA v2.0, 1 km, EU/EEA only. ``Crop_IR`` is the irrigated area and
    ``Crop_A`` the growing area (its README guarantees ``Crop_A = Crop_IR +
    Crop_RF``), so the fraction is a plain ratio of two rasters. ``Crop_RF`` is
    used instead when ``Crop_A`` is not unpacked.

``merged`` (default)
    ECIRA wherever it classifies the cell, MIRCA-OS elsewhere. ECIRA takes
    precedence because MIRCA-OS inherits its irrigated/rainfed split from
    national statistics that report **zero** irrigated cereals for whole
    countries (Germany, Romania, the UK, Sweden, Hungary, Poland) and zero
    irrigated maize for Portugal, and separately loses Italy's irrigated wheat
    between its own crop calendar and its grids. Outside ECIRA's EU/EEA
    footprint MIRCA-OS is the only option, so the merged layer covers the whole
    configured domain. See ``notebooks/irrigation_mirca_ecira_comparison.ipynb``.

Crop groups
-----------
ECIRA has no wheat class — its ``CERE`` is *cereals excluding maize and rice* —
so a wheat run is classified against the cereal aggregate on both sides
(:data:`SIMPLACE_CROP_GROUPS` maps ``winter_wheat`` to ``cereals``). Set
``irrigation.crop_group`` to override the mapping.

Both products carry hectares per cell, an **extensive** quantity, so regridding
conserves the sum rather than averaging: MIRCA-OS by exact 1-D overlap weights
(:func:`conservative_regrid`), ECIRA by binning its 1 km pixel centres, which
never splits a pixel's hectares across cells.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Literal, Optional

import numpy as np
import pandas as pd
import rasterio
import xarray as xr
from pyproj import Transformer

from data4simplace.config import PipelineConfig
from data4simplace.grid import TargetGrid

logger = logging.getLogger(__name__)

#: Crop groups both products can be aggregated to.
CropGroup = Literal["maize", "wheat", "cereals"]

#: Crop group -> the MIRCA-OS file stems summed to build it. Wheat is two
#: sub-crops (seasons); ``cereals`` is ECIRA's ``CERE`` class, i.e. cereals
#: excluding maize and rice.
MIRCA_SUBCROPS: dict[str, tuple[str, ...]] = {
    "maize": ("Maize",),
    "wheat": ("Wheat_1", "Wheat_2"),
    "cereals": ("Wheat_1", "Wheat_2", "Barley", "Rye", "Millet", "Sorghum"),
}

#: Crop group -> the ECIRA crop code. ``wheat`` resolves to the cereal class
#: because ECIRA publishes no wheat layer.
ECIRA_CODES: dict[str, str] = {
    "maize": "LMAIZ",
    "wheat": "CERE",
    "cereals": "CERE",
}

#: SIMPLACE crop name (``npk.simplace_crop``) -> crop group. Wheat maps to
#: ``cereals`` so both products contribute the same class.
SIMPLACE_CROP_GROUPS: dict[str, str] = {
    "winter_wheat": "cereals",
    "spring_wheat": "cereals",
    "wheat": "cereals",
    "winter_barley": "cereals",
    "spring_barley": "cereals",
    "barley": "cereals",
    "rye": "cereals",
    "maize": "maize",
    "grain_maize": "maize",
    "silage_maize": "maize",
}

#: ``source_id`` values on :class:`IrrigationClassification`.
SOURCE_NONE, SOURCE_ECIRA, SOURCE_MIRCA = 0, 1, 2
_SOURCE_NAMES = {SOURCE_NONE: "unclassified", SOURCE_ECIRA: "ECIRA", SOURCE_MIRCA: "MIRCA-OS"}


# --------------------------------------------------------------------------- #
# Regridding
# --------------------------------------------------------------------------- #
def _overlap_weights(src_edges: np.ndarray, tgt_edges: np.ndarray) -> np.ndarray:
    """Fraction of each source cell's 1-D extent falling in each target cell.

    Both edge vectors must be strictly increasing. Returns ``(n_src, n_tgt)``.
    """
    lo = np.maximum(src_edges[:-1, None], tgt_edges[None, :-1])
    hi = np.minimum(src_edges[1:, None], tgt_edges[None, 1:])
    return np.clip(hi - lo, 0.0, None) / np.diff(src_edges)[:, None]


def conservative_regrid(
    values: np.ndarray,
    src_lat: np.ndarray,
    src_lon: np.ndarray,
    grid: TargetGrid,
) -> np.ndarray:
    """Area-conservatively remap an extensive lon/lat field onto ``grid``.

    Separable 1-D overlap weights redistribute each source cell's total into the
    target cells it overlaps, so the sum is preserved (up to what falls outside
    the target box). Area is assumed uniform within a source cell, which is the
    only assumption available at 5 arcmin.

    Parameters
    ----------
    values:
        ``(n_src_lat, n_src_lon)`` extensive field (hectares per cell).
    src_lat, src_lon:
        Source cell *centres* of a regular grid, either direction.
    grid:
        The target grid. The result follows its north-to-south row order.

    Returns
    -------
    numpy.ndarray
        ``grid.shape`` array in the same units as ``values``.
    """
    lat = np.asarray(src_lat, dtype="float64")
    lon = np.asarray(src_lon, dtype="float64")
    arr = np.nan_to_num(np.asarray(values, dtype="float64"))

    if lat.size < 2 or lon.size < 2:
        raise ValueError("conservative_regrid needs at least 2 source cells per axis")
    if lat[0] > lat[-1]:  # work in ascending order on both axes
        lat, arr = lat[::-1], arr[::-1, :]
    if lon[0] > lon[-1]:
        lon, arr = lon[::-1], arr[:, ::-1]

    dlat = float(np.diff(lat).mean())
    dlon = float(np.diff(lon).mean())
    src_lat_edges = np.append(lat - dlat / 2.0, lat[-1] + dlat / 2.0)
    src_lon_edges = np.append(lon - dlon / 2.0, lon[-1] + dlon / 2.0)

    res = grid.resolution_deg
    tgt_lat_centers = np.sort(grid.lat_centers)  # ascending for the weights
    tgt_lat_edges = np.append(tgt_lat_centers - res / 2.0, tgt_lat_centers[-1] + res / 2.0)
    tgt_lon_edges = np.append(grid.lon_centers - res / 2.0, grid.lon_centers[-1] + res / 2.0)

    w_lat = _overlap_weights(src_lat_edges, tgt_lat_edges)
    w_lon = _overlap_weights(src_lon_edges, tgt_lon_edges)
    out = w_lat.T @ arr @ w_lon
    # The grid's own rows run north -> south.
    return out[::-1, :] if grid.lat_centers[0] > grid.lat_centers[-1] else out


# --------------------------------------------------------------------------- #
# The rule
# --------------------------------------------------------------------------- #
def irrigated_fraction(
    irrigated: np.ndarray, total: np.ndarray, min_crop_area_ha: float
) -> np.ndarray:
    """Share of a cell's crop area that is irrigated.

    NaN where the cell holds less than ``min_crop_area_ha`` of the crop: without
    a floor, a rounding difference on a fraction of a hectare would produce a
    confident label.
    """
    total = np.asarray(total, dtype="float64")
    irrigated = np.asarray(irrigated, dtype="float64")
    fraction = np.divide(
        irrigated, total, out=np.full(total.shape, np.nan), where=total > 0.0
    )
    return np.where(total >= min_crop_area_ha, fraction, np.nan)


def classify(fraction: np.ndarray, threshold: float) -> np.ndarray:
    """``1`` where the irrigated share exceeds ``threshold``, else ``0``.

    Unclassified cells (NaN fraction) become ``0``, i.e. they are exported as
    rainfed — SIMPLACE takes a flag, not a three-state code.
    """
    fraction = np.asarray(fraction, dtype="float64")
    return np.where(np.isfinite(fraction) & (fraction > threshold), 1, 0).astype("int8")


def resolve_crop_group(config: PipelineConfig) -> str:
    """The crop group to classify: the config override, else the SIMPLACE crop.

    Raises
    ------
    ValueError
        If ``npk.simplace_crop`` has no mapping and ``irrigation.crop_group`` is
        unset, since guessing the wrong crop would silently label the run.
    """
    configured = config.irrigation.crop_group
    if configured is not None:
        return str(configured)

    crop = config.npk.simplace_crop.strip().lower()
    group = SIMPLACE_CROP_GROUPS.get(crop)
    if group is None:
        raise ValueError(
            f"No irrigation crop group known for npk.simplace_crop {crop!r}; "
            f"set irrigation.crop_group to one of {sorted(MIRCA_SUBCROPS)} "
            f"(known crops: {sorted(SIMPLACE_CROP_GROUPS)})"
        )
    logger.info("Irrigation crop group for %r resolved to %r", crop, group)
    return group


# --------------------------------------------------------------------------- #
# Result
# --------------------------------------------------------------------------- #
@dataclass(frozen=True)
class IrrigationClassification:
    """Per-cell irrigated / rainfed labels on the target grid.

    Attributes
    ----------
    virr:
        ``(n_lat, n_lon)`` ``int8`` of 1 (irrigated) / 0 (rainfed or
        unclassified) — the value written to the management file.
    fraction:
        The irrigated share the label came from; NaN where unclassified. This is
        what distinguishes a genuine rainfed cell from one with too little crop.
    source_id:
        Which product decided each cell: :data:`SOURCE_ECIRA`,
        :data:`SOURCE_MIRCA` or :data:`SOURCE_NONE`.
    crop_group:
        The crop group classified (``maize`` / ``wheat`` / ``cereals``).
    source, threshold, min_crop_area_ha, year:
        The settings the classification was produced with.
    """

    virr: np.ndarray
    fraction: np.ndarray
    source_id: np.ndarray
    crop_group: str
    source: str
    threshold: float
    min_crop_area_ha: float
    year: int

    @property
    def n_classified(self) -> int:
        """Cells with enough crop area to carry a meaningful label."""
        return int(np.isfinite(self.fraction).sum())

    @property
    def n_irrigated(self) -> int:
        """Cells labelled irrigated."""
        return int((self.virr == 1).sum())

    def column(self, cell_table: pd.DataFrame) -> np.ndarray:
        """The ``vIRR`` value for every row of a cell table.

        The classification is already on the target grid, so the table's
        ``row``/``col`` index it directly — one vectorised gather rather than a
        per-cell lookup, which matters at Europe's ~10^5 cells.
        """
        if cell_table.empty:
            return np.empty(0, dtype="int8")
        rows = cell_table["row"].to_numpy()
        cols = cell_table["col"].to_numpy()
        return self.virr[rows, cols].astype("int8")

    def to_dataset(self, grid: TargetGrid) -> xr.Dataset:
        """The classification as a NetCDF-ready dataset."""
        coords = {"lat": grid.lat_centers, "lon": grid.lon_centers}
        data = xr.Dataset(
            {
                "vIRR": (("lat", "lon"), self.virr),
                "irrigated_fraction": (("lat", "lon"), self.fraction.astype("float32")),
                "source_id": (("lat", "lon"), self.source_id),
            },
            coords=coords,
        )
        data["vIRR"].attrs = {
            "long_name": f"irrigated (1) / rainfed (0) class for {self.crop_group}",
            "rule": f"irrigated share of harvested area > {self.threshold}",
            "note": "unclassified cells are written as 0; see irrigated_fraction",
        }
        data["irrigated_fraction"].attrs = {
            "long_name": "irrigated share of the crop's harvested area",
            "units": "1",
            "min_crop_area_ha": self.min_crop_area_ha,
        }
        data["source_id"].attrs = {
            "flag_values": f"{SOURCE_NONE} {SOURCE_ECIRA} {SOURCE_MIRCA}",
            "flag_meanings": "unclassified ECIRA MIRCA-OS",
        }
        data.attrs = {
            "title": f"Irrigation classification ({self.crop_group}, {self.year})",
            "source": self.source,
            "threshold": self.threshold,
            "crop_group": self.crop_group,
        }
        return data

    def summary(self) -> str:
        """One-line description for the run log."""
        counts = {
            name: int((self.source_id == code).sum())
            for code, name in _SOURCE_NAMES.items()
        }
        return (
            f"{self.crop_group} ({self.source}, f > {self.threshold}): "
            f"{self.n_irrigated:,} irrigated of {self.n_classified:,} classified "
            f"cells; from " + ", ".join(f"{v:,} {k}" for k, v in counts.items())
        )


# --------------------------------------------------------------------------- #
# Classifier
# --------------------------------------------------------------------------- #
class IrrigationClassifier:
    """Build the per-cell irrigated / rainfed classification for a crop.

    Parameters
    ----------
    config:
        The validated pipeline configuration. Reads the ``irrigation`` block,
        ``paths.mirca_root`` / ``paths.ecira_root`` and — when
        ``irrigation.crop_group`` is unset — ``npk.simplace_crop``.

    Examples
    --------
    >>> classification = IrrigationClassifier(config).classify()  # doctest: +SKIP
    >>> frame["vIRR"] = classification.column(cell_table)         # doctest: +SKIP
    """

    def __init__(self, config: PipelineConfig) -> None:
        self._config = config
        self._settings = config.irrigation
        self._grid = TargetGrid.from_config(config.grid)
        self._crop_group = resolve_crop_group(config)
        # ECIRA's pixel -> target-cell index is expensive to build and identical
        # for every raster, so it is cached across the two reads.
        self._ecira_index: Optional[tuple[np.ndarray, np.ndarray, tuple[int, int]]] = None

    @property
    def crop_group(self) -> str:
        """The crop group being classified."""
        return self._crop_group

    # ------------------------------------------------------------------ #
    # Entry point
    # ------------------------------------------------------------------ #
    def classify(self) -> IrrigationClassification:
        """Classify every target cell.

        Returns
        -------
        IrrigationClassification
            Labels, the fractions they came from and the deciding product.

        Raises
        ------
        FileNotFoundError
            If a required source file is missing.
        ValueError
            If the configured source has no root path.
        """
        source = self._settings.source
        shape = self._grid.shape
        nan = np.full(shape, np.nan)

        mirca = self._mirca_fraction() if source in ("mirca", "merged") else nan
        ecira = self._ecira_fraction() if source in ("ecira", "merged") else nan

        if source == "mirca":
            fraction, source_id = mirca, np.where(np.isfinite(mirca), SOURCE_MIRCA, SOURCE_NONE)
        elif source == "ecira":
            fraction, source_id = ecira, np.where(np.isfinite(ecira), SOURCE_ECIRA, SOURCE_NONE)
        else:
            # ECIRA first: MIRCA-OS reports whole countries as fully rainfed.
            use_ecira = np.isfinite(ecira)
            fraction = np.where(use_ecira, ecira, mirca)
            source_id = np.where(
                use_ecira,
                SOURCE_ECIRA,
                np.where(np.isfinite(mirca), SOURCE_MIRCA, SOURCE_NONE),
            )

        result = IrrigationClassification(
            virr=classify(fraction, self._settings.threshold),
            fraction=fraction.astype("float32"),
            source_id=source_id.astype("int8"),
            crop_group=self._crop_group,
            source=source,
            threshold=self._settings.threshold,
            min_crop_area_ha=self._settings.min_crop_area_ha,
            year=self._settings.year,
        )
        logger.info("Irrigation classification: %s", result.summary())
        return result

    # ------------------------------------------------------------------ #
    # MIRCA-OS
    # ------------------------------------------------------------------ #
    def _mirca_dir(self) -> Path:
        """The MIRCA-OS directory holding the year's netCDFs.

        Accepts either the year folder itself or a parent containing one.
        """
        root = self._config.paths.mirca_root
        if root is None:
            raise ValueError(
                f"irrigation.source {self._settings.source!r} needs paths.mirca_root"
            )
        root = Path(root)
        year_dir = root / str(self._settings.year)
        if year_dir.is_dir():
            return year_dir
        if not root.is_dir():
            raise FileNotFoundError(f"paths.mirca_root is not a directory: {root}")
        return root

    def _open_mirca(self, path: Path) -> xr.DataArray:
        """Open one MIRCA-OS monthly file, cropped to the grid bbox.

        The stored coordinates are cell **corners**, not centres: longitude runs
        to exactly 180.0 and latitude to -90.0 over 4320 x 2160 cells, so each
        value is its cell's east/south edge. Left uncorrected the grid sits half
        a cell (~4.6 km) out of place — half a target cell.
        """
        data = xr.open_dataset(path)["harvested_area"].sortby("month")
        dlon = float(np.diff(data["longitude"].values).mean())
        dlat = float(np.diff(data["latitude"].values).mean())  # negative: descending
        data = data.assign_coords(
            longitude=data["longitude"] - dlon / 2.0,
            latitude=data["latitude"] - dlat / 2.0,
        )

        g = self._config.grid
        pad = 5.0 * g.resolution_deg
        lats = data["latitude"].values
        lat_slice = (
            slice(g.max_lat + pad, g.min_lat - pad)
            if lats[0] > lats[-1]
            else slice(g.min_lat - pad, g.max_lat + pad)
        )
        return data.sel(
            longitude=slice(g.min_lon - pad, g.max_lon + pad), latitude=lat_slice
        ).load()

    def _mirca_harvested_area(self, system: str) -> np.ndarray:
        """Annual harvested area of the crop group on the target grid, ha.

        MIRCA-OS's own annual definition: the sum over sub-crops of each
        sub-crop's peak month. The sub-crop maxima are summed rather than the
        monthly sums maximised because the sub-crops are separate seasons.
        """
        folder = self._mirca_dir()
        year = self._settings.year
        total: Optional[np.ndarray] = None
        coords: Optional[tuple[np.ndarray, np.ndarray]] = None

        for stem in MIRCA_SUBCROPS[self._crop_group]:
            path = folder / f"MIRCA-OS_{stem}_{year}_{system}.nc"
            if not path.is_file():
                raise FileNotFoundError(f"MIRCA-OS file not found: {path}")
            monthly = self._open_mirca(path)
            coords = (monthly["latitude"].values, monthly["longitude"].values)
            peak = np.nan_to_num(monthly.values).max(axis=0)
            total = peak if total is None else total + peak

        assert total is not None and coords is not None  # the group is never empty
        return conservative_regrid(total, coords[0], coords[1], self._grid)

    def _mirca_fraction(self) -> np.ndarray:
        """MIRCA-OS irrigated share of the crop group's harvested area."""
        irrigated = self._mirca_harvested_area("ir")
        rainfed = self._mirca_harvested_area("rf")
        logger.info(
            "MIRCA-OS %s %d: %.0f kha irrigated, %.0f kha rainfed in the domain",
            self._crop_group, self._settings.year, irrigated.sum() / 1e3, rainfed.sum() / 1e3,
        )
        return irrigated_fraction(
            irrigated, irrigated + rainfed, self._settings.min_crop_area_ha
        )

    # ------------------------------------------------------------------ #
    # ECIRA
    # ------------------------------------------------------------------ #
    def _ecira_path(self, product: str) -> Optional[Path]:
        """Path of one ECIRA product for the crop group, or ``None``.

        ``product`` is ``IR``, ``A`` or ``RF``; the file naming is
        ``<product_dir>/<year>/<CODE>_<product>_A_<year>.tif`` with ``Crop_A``
        dropping the redundant product token.
        """
        root = self._config.paths.ecira_root
        if root is None:
            raise ValueError(
                f"irrigation.source {self._settings.source!r} needs paths.ecira_root"
            )
        year = self._settings.year
        code = ECIRA_CODES[self._crop_group]
        stem = f"{code}_A_{year}" if product == "A" else f"{code}_{product}_A_{year}"
        path = Path(root) / f"Crop_{product}" / str(year) / f"{stem}.tif"
        return path if path.is_file() else None

    def _ecira_pixel_index(self, template: Path) -> tuple[np.ndarray, np.ndarray, tuple[int, int]]:
        """Map every ECIRA pixel centre to a flat target-cell index.

        Built once and reused: the transform is the same for every raster. A
        pixel's hectares are never split across target cells, so the domain
        total is conserved exactly; the only error is the sub-kilometre
        displacement of pixels straddling a cell edge, which is small against a
        ~10 km cell.
        """
        if self._ecira_index is not None:
            return self._ecira_index

        with rasterio.open(template) as src:
            shape = (src.height, src.width)
            transform, crs = src.transform, src.crs

        to_wgs84 = Transformer.from_crs(crs, self._grid.crs, always_xy=True)
        res = self._grid.resolution_deg
        lat_edges = np.sort(np.append(self._grid.lat_centers - res / 2.0,
                                      self._grid.lat_centers[-1:] + res / 2.0))
        lon_edges = np.append(self._grid.lon_centers - res / 2.0,
                              self._grid.lon_centers[-1:] + res / 2.0)
        n_lat, n_lon = self._grid.shape
        north_up = self._grid.lat_centers[0] > self._grid.lat_centers[-1]

        flat = np.full(shape, -1, dtype=np.int32)
        cols = np.arange(shape[1])
        a, b, c, d, e, f = (transform.a, transform.b, transform.c,
                            transform.d, transform.e, transform.f)
        for start in range(0, shape[0], 512):
            stop = min(start + 512, shape[0])
            cc, rr = np.meshgrid(cols + 0.5, np.arange(start, stop) + 0.5)
            lon, lat = to_wgs84.transform(c + a * cc + b * rr, f + d * cc + e * rr)

            ix = np.searchsorted(lon_edges, lon, side="right") - 1
            iy = np.searchsorted(lat_edges, lat, side="right") - 1
            inside = (ix >= 0) & (ix < n_lon) & (iy >= 0) & (iy < n_lat)
            # lat_edges is ascending; the grid's rows may run north -> south.
            row = (n_lat - 1 - iy) if north_up else iy
            flat[start:stop, :] = np.where(inside, row * n_lon + ix, -1)

        index = flat.ravel()
        self._ecira_index = (index, index >= 0, shape)
        logger.info(
            "ECIRA: %d of %d pixels inside the target grid",
            int((index >= 0).sum()), index.size,
        )
        return self._ecira_index

    def _ecira_area(self, path: Path, template: Path) -> np.ndarray:
        """Sum one ECIRA raster's hectares into the target cells."""
        index, valid, _ = self._ecira_pixel_index(template)
        with rasterio.open(path) as src:
            values = np.nan_to_num(src.read(1).astype("float64")).ravel()
        keep = valid & (values != 0.0)
        n_lat, n_lon = self._grid.shape
        return np.bincount(
            index[keep], weights=values[keep], minlength=n_lat * n_lon
        ).reshape(n_lat, n_lon)

    def _ecira_fraction(self) -> np.ndarray:
        """ECIRA irrigated share of the crop group's growing area.

        The denominator is ``Crop_A`` (= ``Crop_IR + Crop_RF`` per the ECIRA
        README); where that product is not unpacked, ``Crop_IR + Crop_RF`` is
        summed instead.
        """
        irrigated_path = self._ecira_path("IR")
        if irrigated_path is None:
            raise FileNotFoundError(
                f"No ECIRA Crop_IR raster for {self._crop_group} "
                f"({ECIRA_CODES[self._crop_group]}) in {self._settings.year} under "
                f"{self._config.paths.ecira_root}"
            )

        irrigated = self._ecira_area(irrigated_path, irrigated_path)
        total_path = self._ecira_path("A")
        if total_path is not None:
            total = self._ecira_area(total_path, irrigated_path)
        else:
            rainfed_path = self._ecira_path("RF")
            if rainfed_path is None:
                raise FileNotFoundError(
                    f"ECIRA has neither a Crop_A nor a Crop_RF raster for "
                    f"{self._crop_group} in {self._settings.year}; the irrigated "
                    f"share needs a denominator"
                )
            logger.info("ECIRA Crop_A not available; using Crop_IR + Crop_RF")
            total = irrigated + self._ecira_area(rainfed_path, irrigated_path)

        logger.info(
            "ECIRA %s %d: %.0f kha irrigated of %.0f kha grown in the domain",
            self._crop_group, self._settings.year, irrigated.sum() / 1e3, total.sum() / 1e3,
        )
        return irrigated_fraction(irrigated, total, self._settings.min_crop_area_ha)
