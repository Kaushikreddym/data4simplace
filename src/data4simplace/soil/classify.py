"""Soil classification and per-target-cell aggregation at 250 m.

One module for the two halves of the CLAUDE.md dominant-soil-type workflow,
which are only useful together: a fine pixel has to be *classified* before it can
be masked, and the mask is only useful because the *aggregation* that follows
reduces the surviving pixels onto the 10 km grid.

The aggregation is **true 2-D binning**: every fine pixel is assigned to exactly
one target cell, then all of a cell's pixels are reduced together. That is
required because the rules (majority vote, geometric mean, pH H+ mean) are not
separable across axes, so the sequential per-axis reduction of
``TargetGrid.regrid`` would give wrong answers. All reducers are
**equal-weight**: the fine field is expected to have been masked to cropland and
to one soil class already, so every surviving pixel counts the same.

Two texture-based classifications are available:

``usda``
    The 12-class USDA texture class of the topsoil layer.
``usda_profile``
    A composite **(topsoil, rooting-zone)** class: the topsoil USDA class paired
    with the USDA class of the thickness-weighted mean texture over the rooting
    zone (5-100 cm by default), giving 12 x 12 = 144 codes. A surface class
    alone cannot express the layered profiles that dominate NE Germany (sandy
    cover over loamy till), where the subsoil sets plant-available water.

Equal-weight was chosen over cropland-fraction weighting for the majority vote:
pixels are first hard-filtered to >= ``soil.cropland_min_fraction`` cropland,
then counted equally.

Contents, in order: the binning primitives, the per-variable reducers, the
texture classifications, the dominant-class selection and multi-class
composition, and finally valid-cell detection and the nearest-neighbour gap fill.
"""

from __future__ import annotations

import logging
import re
from typing import Callable, Mapping

import numpy as np
import pandas as pd
import xarray as xr

from data4simplace.grid import TargetGrid

logger = logging.getLogger(__name__)

# A reducer maps the 1-D array of a cell's (finite) pixel values to one scalar.
Reducer = Callable[[np.ndarray], float]

# Per-cell summary statistics offered by :func:`bin_describe` (pandas names).
CELL_STATISTICS = ("mean", "median", "std", "kurt", "count")


# ---------------------------------------------------------------------------- #
# Binning primitives
# ---------------------------------------------------------------------------- #
def _edges(centers: np.ndarray, res: float) -> np.ndarray:
    """Ascending cell edges for an array of centres (any order)."""
    c = np.sort(centers)
    return np.concatenate([c - res / 2.0, c[-1:] + res / 2.0])


def cell_assignment(
    field: xr.DataArray, grid: TargetGrid, finite_only: bool = True
) -> tuple[np.ndarray, np.ndarray]:
    """Map every fine pixel to its target cell.

    Parameters
    ----------
    field:
        A 2-D ``(lat, lon)`` fine-resolution field.
    grid:
        The target 10 km grid.
    finite_only:
        Drop pixels whose value is not finite (the default). Pass ``False`` to
        keep them, e.g. when the values are integer class codes.

    Returns
    -------
    (cell, values):
        Flat target-cell indices (``row * n_lon + col``) and the matching pixel
        values, for the pixels that fall inside the grid.
    """
    from data4simplace.grid import _standardise_lonlat_names

    field = _standardise_lonlat_names(field)
    lat_centers = grid.lat_centers  # north -> south
    lon_centers = grid.lon_centers  # west -> east
    n_lat, n_lon = lat_centers.size, lon_centers.size

    lat_edges = _edges(lat_centers, grid.resolution_deg)  # ascending
    lon_edges = _edges(lon_centers, grid.resolution_deg)  # ascending

    fine_lat = np.asarray(field["lat"].values, dtype="float64")
    fine_lon = np.asarray(field["lon"].values, dtype="float64")

    # Fine pixel -> target column, and -> ascending-lat index (0 = southernmost).
    col = np.searchsorted(lon_edges, fine_lon, side="right") - 1
    lat_asc = np.searchsorted(lat_edges, fine_lat, side="right") - 1
    valid_lon = (col >= 0) & (col < n_lon)
    valid_lat = (lat_asc >= 0) & (lat_asc < n_lat)
    # Target latitudes run north-to-south, so flip the ascending index.
    row = np.where(valid_lat, (n_lat - 1) - lat_asc, -1)

    # Broadcast per-axis indices to the full 2-D pixel grid.
    col_2d, row_2d = np.meshgrid(col, row)
    vcol_2d, vrow_2d = np.meshgrid(valid_lon, valid_lat)

    values = np.asarray(field.values).ravel()
    cell = (row_2d * n_lon + col_2d).ravel()
    keep = (vrow_2d & vcol_2d).ravel()
    if finite_only:
        keep = keep & np.isfinite(values.astype("float64"))
    return cell[keep], values[keep]


def bin_reduce(
    field: xr.DataArray, grid: TargetGrid, reducer: Reducer, fill: float = np.nan
) -> xr.DataArray:
    """Reduce a fine 2-D field onto ``grid`` cells with ``reducer``.

    Parameters
    ----------
    field:
        A 2-D ``(lat, lon)`` fine-resolution field. Non-finite pixels are
        dropped before reduction.
    grid:
        The target 10 km grid.
    reducer:
        Function applied to each cell's 1-D array of finite pixel values.
    fill:
        Value for target cells that contain no valid pixels.

    Returns
    -------
    xarray.DataArray
        The reduced field on the target grid (``lat`` north-to-south).
    """
    n_lat, n_lon = grid.lat_centers.size, grid.lon_centers.size
    cell, values = cell_assignment(field, grid)

    out = np.full(n_lat * n_lon, fill, dtype="float64")
    if cell.size:
        df = pd.DataFrame({"cell": cell, "v": values.astype("float64")})
        agg = df.groupby("cell")["v"].agg(reducer)
        out[agg.index.to_numpy()] = agg.to_numpy()

    return xr.DataArray(
        out.reshape(n_lat, n_lon),
        dims=("lat", "lon"),
        coords={"lat": grid.lat_centers, "lon": grid.lon_centers},
        name=field.name,
    )


def _grouped_kurtosis(df: pd.DataFrame) -> pd.Series:
    """Bias-corrected excess kurtosis (Fisher's G2) per group of ``df``.

    Matches ``pandas.Series.kurt``: a normal sample scores ~0. Computed from
    central moments in two vectorised passes (group mean, then group sums of the
    squared and 4th-power deviations) rather than per-group Python calls, since
    the grouping here has one group per target cell. NaN for groups with fewer
    than four values or with zero spread, where G2 is undefined.
    """
    grouped = df.groupby("cell")["v"]
    n = grouped.count()
    deviation = df["v"] - df["cell"].map(grouped.mean())
    moments = (
        df.assign(d2=deviation**2, d4=deviation**4)
        .groupby("cell")[["d2", "d4"]]
        .sum()
    )
    variance = moments["d2"] / (n - 1)
    with np.errstate(invalid="ignore", divide="ignore"):
        g2 = (
            n * (n + 1) * moments["d4"] / ((n - 1) * (n - 2) * (n - 3) * variance**2)
            - 3.0 * (n - 1) ** 2 / ((n - 2) * (n - 3))
        )
    return g2.where((n >= 4) & (variance > 0))


def bin_describe(
    field: xr.DataArray, grid: TargetGrid, statistics: tuple[str, ...] = CELL_STATISTICS
) -> xr.Dataset:
    """Summary statistics of a fine field per target cell.

    Computes all requested statistics in a single pass, which is what the
    three-primary-class stage needs: the same masked pixels described by mean,
    median, spread and shape rather than reduced to one number.

    Parameters
    ----------
    field:
        A 2-D ``(lat, lon)`` fine field, already masked to the class of interest.
    grid:
        The target 10 km grid.
    statistics:
        Any of ``mean``, ``median``, ``std``, ``kurt``, ``count`` (pandas
        aggregations; ``std`` is the sample standard deviation and ``kurt`` the
        excess kurtosis, so a normal distribution scores 0).

    Returns
    -------
    xarray.Dataset
        One variable per statistic on the target grid. Cells without pixels are
        NaN (``count`` is 0); ``std`` needs two pixels and ``kurt`` four, and is
        NaN below that.
    """
    unknown = set(statistics) - set(CELL_STATISTICS)
    if unknown:
        raise ValueError(f"Unsupported statistics {sorted(unknown)}; use {CELL_STATISTICS}")

    n_lat, n_lon = grid.lat_centers.size, grid.lon_centers.size
    cell, values = cell_assignment(field, grid)

    out = {stat: np.full(n_lat * n_lon, np.nan, dtype="float64") for stat in statistics}
    if "count" in out:
        out["count"][:] = 0.0
    if cell.size:
        df = pd.DataFrame({"cell": cell, "v": values.astype("float64")})
        grouped = df.groupby("cell")["v"]
        # ``kurt`` is not a groupby aggregation in every pandas version, so it is
        # computed from the group's central moments below.
        direct = [stat for stat in statistics if stat != "kurt"]
        if direct:
            agg = grouped.agg(direct)
            index = agg.index.to_numpy()
            for stat in direct:
                out[stat][index] = agg[stat].to_numpy()
        if "kurt" in statistics:
            kurt = _grouped_kurtosis(df)
            out["kurt"][kurt.index.to_numpy()] = kurt.to_numpy()

    coords = {"lat": grid.lat_centers, "lon": grid.lon_centers}
    return xr.Dataset(
        {
            stat: xr.DataArray(values_.reshape(n_lat, n_lon), dims=("lat", "lon"),
                               coords=coords)
            for stat, values_ in out.items()
        }
    )

# ---------------------------------------------------------------------------- #
# Reducers (1-D, operate on a cell's finite pixel values)
# ---------------------------------------------------------------------------- #
def cell_mean(x: np.ndarray) -> float:
    """Arithmetic mean (linear variables: clay, silt, sand, bdod, cfvo)."""
    return float(np.mean(x))


def cell_geomean(x: np.ndarray) -> float:
    """Geometric mean (log-normal variables: soc, nitrogen).

    Non-positive pixels are dropped (log undefined); an all-non-positive cell
    yields NaN.
    """
    x = x[x > 0]
    if x.size == 0:
        return float("nan")
    return float(np.exp(np.mean(np.log(x))))


def cell_ph(x: np.ndarray) -> float:
    """pH aggregation via hydrogen-ion activity (phh2o).

    Convert to ``[H+] = 10**(-pH)``, average, then back-convert with
    ``-log10(mean)``.
    """
    h = np.power(10.0, -x)
    return float(-np.log10(np.mean(h)))


def cell_median(x: np.ndarray) -> float:
    """Median of a cell's pixel values.

    Used for every variable when ``soil.export_statistic`` is ``median``: the
    median commutes with the monotone transforms behind the mean rules (the
    median of ``ln X`` is the log of the median, likewise for pH and ``[H+]``),
    so no per-variable rule is needed.
    """
    return float(np.median(x))


# Variable -> reducer. Anything not listed uses ``cell_mean``.
RULES: dict[str, Reducer] = {
    "soc": cell_geomean,
    "nitrogen": cell_geomean,
    "phh2o": cell_ph,
}


def reducer_for(layer: str, statistic: str = "mean") -> Reducer:
    """Return the aggregation reducer for a soil ``layer``.

    Parameters
    ----------
    layer:
        SoilGrids layer name.
    statistic:
        ``mean`` applies the variable-specific mean rules (arithmetic, geometric
        for ``soc``/``nitrogen``, H+ mean for ``phh2o``); ``median`` uses
        :func:`cell_median` for every layer.
    """
    if statistic == "median":
        return cell_median
    if statistic != "mean":
        raise ValueError(f"Unsupported export statistic {statistic!r}; use 'mean' or 'median'")
    return RULES.get(layer, cell_mean)


# ---------------------------------------------------------------------------- #
# Texture classification
# ---------------------------------------------------------------------------- #
# USDA soil texture classes (12), 1-based integer codes. 0 is reserved for
# "unclassified" (e.g. masked/no-data pixels).
USDA_CLASSES: dict[int, str] = {
    1: "clay",
    2: "silty_clay",
    3: "sandy_clay",
    4: "clay_loam",
    5: "silty_clay_loam",
    6: "sandy_clay_loam",
    7: "loam",
    8: "silt_loam",
    9: "silt",
    10: "sandy_loam",
    11: "loamy_sand",
    12: "sand",
}

# Number of USDA texture classes; the base of the composite (topsoil, rootzone)
# code used by ``usda_profile`` mode.
N_USDA = len(USDA_CLASSES)

# Default rooting-zone window for ``usda_profile``: below the 0-5 cm topsoil and
# down to 100 cm, the depth range that supplies most of the crop's water.
ROOTZONE_TOP_CM = 5.0
ROOTZONE_BOTTOM_CM = 100.0


def usda_texture_class(
    sand: xr.DataArray, silt: xr.DataArray, clay: xr.DataArray
) -> xr.DataArray:
    """Classify pixels into the 12 USDA texture classes.

    Follows the standard USDA texture triangle boundaries. Inputs are mass
    fractions in **percent** (0-100); they need not sum to exactly 100 (the
    triangle rules use pairwise thresholds). Pixels where any input is NaN are
    assigned class ``0`` (unclassified).

    Parameters
    ----------
    sand, silt, clay:
        Sand/silt/clay mass fractions in percent, on a common grid.

    Returns
    -------
    xarray.DataArray
        Integer USDA class codes (0-12) with the inputs' coordinates.
    """
    s = sand.astype("float64")
    si = silt.astype("float64")
    c = clay.astype("float64")

    # Canonical USDA texture triangle, using the standard boundary formulas
    # (percent). Conditions are evaluated as an ``if/elif`` chain: ``assign``
    # only fills pixels still unclassified, so the listed order is the priority.
    cls = xr.zeros_like(s)

    def assign(code: int, cond: xr.DataArray) -> None:
        nonlocal cls
        cls = xr.where((cls == 0) & cond, code, cls)

    assign(12, (si + 1.5 * c) < 15)                                            # sand
    assign(11, ((si + 1.5 * c) >= 15) & ((si + 2.0 * c) < 30))                 # loamy sand
    assign(                                                                    # sandy loam
        10,
        ((c >= 7) & (c < 20) & (s > 52) & ((si + 2.0 * c) >= 30))
        | ((c < 7) & (si < 50) & (s > 43)),
    )
    assign(7, (c >= 7) & (c < 27) & (si >= 28) & (si < 50) & (s <= 52))        # loam
    assign(                                                                    # silt loam
        8,
        ((si >= 50) & (c >= 12) & (c < 27)) | ((si >= 50) & (si < 80) & (c < 12)),
    )
    assign(9, (si >= 80) & (c < 12))                                           # silt
    assign(6, (c >= 20) & (c < 35) & (si < 28) & (s > 45))                     # sandy clay loam
    assign(4, (c >= 27) & (c < 40) & (s > 20) & (s <= 45))                     # clay loam
    assign(5, (c >= 27) & (c < 40) & (s <= 20))                               # silty clay loam
    assign(3, (c >= 35) & (s > 45))                                            # sandy clay
    assign(2, (c >= 40) & (si >= 40))                                          # silty clay
    assign(1, (c >= 40) & (si < 40) & (s <= 45))                               # clay

    # Restore NaN-driven pixels to unclassified 0.
    valid = sand.notnull() & silt.notnull() & clay.notnull()
    cls = xr.where(valid, cls, 0)
    return cls.astype("int16").rename("usda_class")


def depth_bounds_cm(depth: str) -> tuple[float, float]:
    """Parse a SoilGrids depth label (e.g. ``"5-15cm"``) into ``(top, bottom)`` cm.

    Raises
    ------
    ValueError
        If the label does not follow the ``<top>-<bottom>cm`` convention.
    """
    match = re.fullmatch(r"\s*(\d+(?:\.\d+)?)-(\d+(?:\.\d+)?)\s*cm\s*", depth)
    if match is None:
        raise ValueError(f"Unrecognised SoilGrids depth label {depth!r}")
    return float(match.group(1)), float(match.group(2))


def rootzone_mean(
    by_depth: Mapping[str, xr.DataArray],
    top_cm: float = ROOTZONE_TOP_CM,
    bottom_cm: float = ROOTZONE_BOTTOM_CM,
) -> xr.DataArray:
    """Thickness-weighted mean of a property over a depth window.

    Each depth layer contributes in proportion to how much of it lies inside
    ``[top_cm, bottom_cm]``, so a window need not align with the SoilGrids layer
    boundaries. Layers outside the window are ignored; pixels that are NaN in a
    layer simply do not contribute that layer's weight.

    Parameters
    ----------
    by_depth:
        ``{depth_label: DataArray}`` for one property, on a common grid.
    top_cm, bottom_cm:
        The depth window, in centimetres below the surface.

    Returns
    -------
    xarray.DataArray
        The weighted mean; NaN where no layer contributed.

    Raises
    ------
    ValueError
        If no supplied depth overlaps the window.
    """
    if bottom_cm <= top_cm:
        raise ValueError(f"Empty depth window: {top_cm}-{bottom_cm} cm")

    numerator: xr.DataArray | None = None
    denominator: xr.DataArray | None = None
    used: list[str] = []
    for depth, da in by_depth.items():
        lo, hi = depth_bounds_cm(depth)
        overlap = min(hi, bottom_cm) - max(lo, top_cm)
        if overlap <= 0:
            continue
        valid = da.notnull()
        contribution = da.fillna(0.0) * overlap
        weight = valid * overlap
        numerator = contribution if numerator is None else numerator + contribution
        denominator = weight if denominator is None else denominator + weight
        used.append(depth)

    if numerator is None or denominator is None:
        raise ValueError(
            f"No depth in {list(by_depth)} overlaps the {top_cm}-{bottom_cm} cm window"
        )
    logger.debug("Rooting-zone mean over %s (%g-%g cm)", used, top_cm, bottom_cm)
    return (numerator / denominator.where(denominator > 0)).rename("rootzone_mean")


def usda_profile_class(
    topsoil_class: xr.DataArray, rootzone_class: xr.DataArray
) -> xr.DataArray:
    """Combine a topsoil and a rooting-zone USDA class into one composite code.

    The composite code is ``(topsoil - 1) * 12 + rootzone``, i.e. 1-144 over the
    12 x 12 combinations, and ``0`` where either part is unclassified. Codes stay
    small enough for the majority vote in :func:`dominant_class_per_cell`.

    Parameters
    ----------
    topsoil_class, rootzone_class:
        Integer USDA class codes (0-12) on a common grid.

    Returns
    -------
    xarray.DataArray
        Integer composite class codes (0-144).
    """
    top = topsoil_class.astype("int16")
    root = rootzone_class.astype("int16")
    code = (top - 1) * N_USDA + root
    code = xr.where((top > 0) & (root > 0), code, 0)
    return code.astype("int16").rename("usda_profile_class")


def split_profile_class(code: int) -> tuple[int, int]:
    """Split a composite code back into its ``(topsoil, rootzone)`` USDA classes.

    Returns ``(0, 0)`` for the unclassified code ``0``.
    """
    code = int(code)
    if code <= 0:
        return (0, 0)
    return ((code - 1) // N_USDA + 1, (code - 1) % N_USDA + 1)


def profile_class_name(code: int) -> str:
    """Human-readable name of a composite class, e.g. ``"loamy_sand/clay_loam"``.

    The two parts are ``topsoil/rooting-zone``; ``"unclassified"`` for ``0``.
    """
    top, root = split_profile_class(code)
    if top == 0:
        return "unclassified"
    return f"{USDA_CLASSES[top]}/{USDA_CLASSES[root]}"


def class_name(code: int, mode: str = "usda") -> str:
    """Name a class code for either texture mode (``usda`` or ``usda_profile``)."""
    if mode == "usda_profile":
        return profile_class_name(code)
    return USDA_CLASSES.get(int(code), "unclassified")


# ---------------------------------------------------------------------------- #
# Dominant class selection and multi-class composition
# ---------------------------------------------------------------------------- #
def dominant_class_per_cell(
    class_da: xr.DataArray, grid: TargetGrid
) -> xr.DataArray:
    """Select the majority (mode) class within each 10 km target cell.

    Pixels are counted equally; class ``0`` (unclassified) is ignored. Cells
    with no valid pixels get ``0``.

    Parameters
    ----------
    class_da:
        Integer class codes on the fine (250 m) grid, ``lat``/``lon`` coords.
    grid:
        The target 10 km grid whose cells define the aggregation bins.

    Returns
    -------
    xarray.DataArray
        Integer dominant class per target cell, on the coarse grid.
    """
    coarse = bin_reduce(class_da, grid, _mode_reducer, fill=0.0)
    return coarse.astype("int16").rename("dominant_class")


def class_composition(
    class_da: xr.DataArray,
    grid: TargetGrid,
    n_classes: int = 3,
    pixel_area: xr.DataArray | None = None,
) -> tuple[xr.Dataset, xr.Dataset]:
    """Rank the classes in each target cell and measure the cell's heterogeneity.

    One pass over the fine pixels yields both halves of the multi-class workflow:
    the per-rank composition the top-N export needs, and the cell-level
    uncertainty metrics that say how much of the cell a single exported profile
    actually stands for.

    Parameters
    ----------
    class_da:
        Integer class codes on the fine grid; ``0`` (unclassified) is ignored.
    grid:
        The target 10 km grid.
    n_classes:
        How many ranks to keep. Ties are broken by the lower class code, matching
        the majority vote of :func:`dominant_class_per_cell`, so rank 1 is always
        the dominant class.
    pixel_area:
        Per-pixel surface area (km2) on ``class_da``'s grid, from
        :func:`~data4simplace.spatial.area.pixel_area_km2`. When given, the
        per-rank ``area_km2``/``area_fraction`` and the cell's
        ``total_area_km2`` are reported and the uncertainty metrics are computed
        from area fractions; without it they fall back to pixel fractions.

    Returns
    -------
    (ranked, uncertainty):
        ``ranked`` carries ``class_code`` (0 where a cell has fewer classes than
        ranks), ``pixels`` and ``share_percent`` (percent of the cell's
        classified pixels) on ``(rank, lat, lon)``, plus ``area_km2`` and
        ``area_fraction`` when ``pixel_area`` is given.

        ``uncertainty`` carries, on ``(lat, lon)``:

        ``n_classes``
            How many distinct classes the cell holds (**all** of them, not just
            the ranked ones).
        ``cropland_pixels`` / ``total_area_km2``
            The cell's classified extent.
        ``dominance_ratio``
            ``1 - p1``: the share of the cell the dominant class does *not*
            cover. 0 for a uniform cell.
        ``shannon_entropy``
            ``-sum(p_i * ln p_i) / ln(N)`` over **all** N classes in the cell,
            so it is 0 for a uniform cell and 1 when every class is equally
            represented, comparably across cells with different class counts.
    """
    if n_classes < 1:
        raise ValueError(f"n_classes must be >= 1, got {n_classes}")

    n_lat, n_lon = grid.lat_centers.size, grid.lon_centers.size
    n_cells = n_lat * n_lon
    cell, codes = cell_assignment(class_da, grid, finite_only=False)

    with_area = pixel_area is not None
    if with_area:
        area_cell, area_values = cell_assignment(pixel_area, grid, finite_only=False)
        if area_values.size != codes.size or not np.array_equal(area_cell, cell):
            raise ValueError(
                "pixel_area must be on the same fine grid as the class field"
            )
        areas = area_values.astype("float64")
    else:
        # Without areas every pixel weighs 1, so the fractions below are pixel
        # fractions and the reported "area" columns are simply omitted.
        areas = np.ones(codes.size, dtype="float64")

    code_arr = np.zeros((n_classes, n_cells), dtype="int64")
    pixel_arr = np.zeros((n_classes, n_cells), dtype="int64")
    share_arr = np.full((n_classes, n_cells), np.nan, dtype="float64")
    area_arr = np.full((n_classes, n_cells), np.nan, dtype="float64")
    fraction_arr = np.full((n_classes, n_cells), np.nan, dtype="float64")

    classes_per_cell = np.zeros(n_cells, dtype="int64")
    pixels_per_cell = np.zeros(n_cells, dtype="int64")
    area_per_cell = np.zeros(n_cells, dtype="float64")
    dominance = np.full(n_cells, np.nan, dtype="float64")
    entropy = np.full(n_cells, np.nan, dtype="float64")

    valid = codes > 0
    if valid.any():
        counts = (
            pd.DataFrame(
                {
                    "cell": cell[valid],
                    "code": codes[valid].astype("int64"),
                    "area": areas[valid],
                }
            )
            .groupby(["cell", "code"], sort=False)
            .agg(pixels=("area", "size"), area=("area", "sum"))
            .reset_index()
        )
        per_cell = counts.groupby("cell")
        counts["share"] = 100.0 * counts["pixels"] / per_cell["pixels"].transform("sum")
        counts["fraction"] = counts["area"] / per_cell["area"].transform("sum")

        # --- Cell-level metrics, over every class present --------------------
        totals = counts.groupby("cell").agg(
            n_classes=("code", "size"),
            pixels=("pixels", "sum"),
            area=("area", "sum"),
        )
        cell_idx = totals.index.to_numpy()
        classes_per_cell[cell_idx] = totals["n_classes"].to_numpy()
        pixels_per_cell[cell_idx] = totals["pixels"].to_numpy()
        area_per_cell[cell_idx] = totals["area"].to_numpy()

        p = counts["fraction"].to_numpy()
        with np.errstate(divide="ignore", invalid="ignore"):
            plogp = np.where(p > 0, p * np.log(p), 0.0)
        shannon = (
            pd.Series(-plogp, index=counts["cell"].to_numpy()).groupby(level=0).sum()
        )
        n_present = totals["n_classes"].to_numpy().astype("float64")
        # A single-class cell has zero entropy; ln(1) = 0 would divide by zero.
        normaliser = np.where(n_present > 1, np.log(n_present), 1.0)
        entropy[cell_idx] = np.where(
            n_present > 1, shannon.reindex(cell_idx).to_numpy() / normaliser, 0.0
        )

        # --- Ranked composition ----------------------------------------------
        # Most pixels first; the lower code wins a tie (as the majority vote does).
        counts = counts.sort_values(
            ["cell", "pixels", "code"], ascending=[True, False, True]
        )
        counts["rank"] = counts.groupby("cell").cumcount()

        first = counts[counts["rank"] == 0]
        dominance[first["cell"].to_numpy()] = 1.0 - first["fraction"].to_numpy()

        top = counts[counts["rank"] < n_classes]
        rank_idx = top["rank"].to_numpy()
        top_cells = top["cell"].to_numpy()
        code_arr[rank_idx, top_cells] = top["code"].to_numpy()
        pixel_arr[rank_idx, top_cells] = top["pixels"].to_numpy()
        share_arr[rank_idx, top_cells] = top["share"].to_numpy()
        area_arr[rank_idx, top_cells] = top["area"].to_numpy()
        fraction_arr[rank_idx, top_cells] = top["fraction"].to_numpy()

    dims = ("rank", "lat", "lon")
    coords = {
        "rank": np.arange(1, n_classes + 1),
        "lat": grid.lat_centers,
        "lon": grid.lon_centers,
    }
    shape = (n_classes, n_lat, n_lon)
    ranked_vars: dict[str, tuple] = {
        "class_code": (dims, code_arr.reshape(shape).astype("int16")),
        "pixels": (dims, pixel_arr.reshape(shape).astype("int32")),
        "share_percent": (dims, share_arr.reshape(shape).astype("float32")),
    }
    if with_area:
        ranked_vars["area_km2"] = (dims, area_arr.reshape(shape).astype("float32"))
        ranked_vars["area_fraction"] = (
            dims,
            fraction_arr.reshape(shape).astype("float32"),
        )
    ranked = xr.Dataset(ranked_vars, coords=coords)

    cell_dims = ("lat", "lon")
    cell_coords = {"lat": grid.lat_centers, "lon": grid.lon_centers}
    cell_shape = (n_lat, n_lon)
    uncertainty_vars: dict[str, tuple] = {
        "n_classes": (cell_dims, classes_per_cell.reshape(cell_shape).astype("int16")),
        "cropland_pixels": (
            cell_dims,
            pixels_per_cell.reshape(cell_shape).astype("int32"),
        ),
        "dominance_ratio": (cell_dims, dominance.reshape(cell_shape).astype("float32")),
        "shannon_entropy": (cell_dims, entropy.reshape(cell_shape).astype("float32")),
    }
    if with_area:
        uncertainty_vars["total_area_km2"] = (
            cell_dims,
            np.where(classes_per_cell > 0, area_per_cell, np.nan)
            .reshape(cell_shape)
            .astype("float32"),
        )
    uncertainty = xr.Dataset(uncertainty_vars, coords=cell_coords)

    return ranked, uncertainty


def rank_classes_per_cell(
    class_da: xr.DataArray, grid: TargetGrid, n_classes: int = 3
) -> xr.Dataset:
    """The ``n_classes`` most frequent classes in each target cell.

    Thin wrapper over :func:`class_composition` for callers that only need the
    ranked composition (``class_code``, ``pixels``, ``share_percent`` on
    ``(rank, lat, lon)``) and not the areas or uncertainty metrics.
    """
    ranked, _ = class_composition(class_da, grid, n_classes)
    return ranked


def dominant_pixel_mask(
    class_da: xr.DataArray, dominant: xr.DataArray
) -> xr.DataArray:
    """Boolean fine-grid mask: keep pixels matching their cell's dominant class.

    Parameters
    ----------
    class_da:
        Fine-grid integer class codes.
    dominant:
        Coarse-grid dominant class (from :func:`dominant_class_per_cell`).

    Returns
    -------
    xarray.DataArray
        Boolean mask aligned to ``class_da`` (True = keep).
    """
    # Broadcast the coarse dominant class back onto every fine pixel.
    dom_fine = dominant.reindex(
        lat=class_da["lat"], lon=class_da["lon"], method="nearest"
    )
    mask = (class_da == dom_fine) & (class_da != 0)
    return mask.rename("dominant_keep")


def _mode_reducer(x: np.ndarray) -> float:
    """Majority vote over a cell's class codes (equal-weight, zeros ignored).

    Reducer for :func:`bin_reduce`: returns the most frequent positive integer
    class; ``0`` when no valid votes remain.
    """
    x = x[x > 0].astype("int64")
    if x.size == 0:
        return 0.0
    return float(np.bincount(x).argmax())


# ---------------------------------------------------------------------------- #
# Valid-cell detection
# ---------------------------------------------------------------------------- #
def valid_soil_cells(soil: xr.Dataset) -> xr.DataArray | None:
    """Boolean ``(lat, lon)`` mask: True where the cell carries any soil value.

    A cell counts as valid when at least one property at one depth is finite --
    the same rule the soil exporter uses to drop rows, so the exported cell set
    and the rows of ``soil.csv`` cannot disagree.

    With ``soil.fill_missing`` off (the default) nothing is borrowed from a
    neighbour, so this is exactly the set of cells whose profile was aggregated
    from their **own** 250 m pixels.

    Returns ``None`` when the dataset carries no ``(lat, lon)`` variable, so
    callers can fall back to keeping every cell.
    """
    fields = [da for da in soil.data_vars.values() if {"lat", "lon"} <= set(da.dims)]
    if not fields:
        return None

    valid: xr.DataArray | None = None
    for da in fields:
        finite = da.notnull()
        extra = [d for d in finite.dims if d not in ("lat", "lon")]
        if extra:
            finite = finite.any(dim=extra)
        valid = finite if valid is None else (valid | finite)

    assert valid is not None  # fields is non-empty
    return valid.rename("valid_soil_cells")


# ---------------------------------------------------------------------------- #
# Gap filling
# ---------------------------------------------------------------------------- #
def _fill_nearest_2d(arr: np.ndarray) -> np.ndarray:
    """Fill NaNs in a 2-D array from the nearest finite cell (Euclidean)."""
    from scipy import ndimage

    mask = ~np.isfinite(arr)
    if not mask.any() or mask.all():
        return arr
    idx = ndimage.distance_transform_edt(
        mask, return_distances=False, return_indices=True
    )
    return arr[tuple(idx)]


def fill_missing_cells(
    obj: xr.DataArray | xr.Dataset, within: xr.DataArray | None = None
) -> xr.DataArray | xr.Dataset:
    """Nearest-neighbour fill of empty ``(lat, lon)`` cells (coastal/islands).

    Operates per 2-D lat/lon slice, looping over any extra dims (e.g. ``depth``).
    Returns the input unchanged if SciPy is unavailable (with a warning).

    Parameters
    ----------
    obj:
        The field(s) to fill, carrying ``lat``/``lon`` dims.
    within:
        Optional boolean ``(lat, lon)`` mask bounding the fill. Only cells that
        are True are filled; the rest stay NaN. Without it the search is
        unbounded and would manufacture values far offshore, so callers pass the
        cropland cell mask -- the set of cells that will actually be exported.
    """
    try:
        import scipy  # noqa: F401
    except ImportError:  # pragma: no cover - environment without SciPy
        import logging

        logging.getLogger(__name__).warning(
            "SciPy not available; skipping nearest-neighbour gap fill"
        )
        return obj

    def _fill_da(da: xr.DataArray) -> xr.DataArray:
        if "lat" not in da.dims or "lon" not in da.dims:
            return da
        filled = xr.apply_ufunc(
            _fill_nearest_2d,
            da,
            input_core_dims=[["lat", "lon"]],
            output_core_dims=[["lat", "lon"]],
            vectorize=True,
            keep_attrs=True,
        )
        if within is None:
            return filled
        # ``apply_ufunc`` moves the core dims last; restore the input order.
        filled = filled.transpose(*da.dims)
        # Outside the mask, fall back to the original value: gaps stay gaps, and
        # a cell that already held a value keeps it.
        return filled.where(within, other=da)

    if isinstance(obj, xr.Dataset):
        return obj.map(_fill_da)
    return _fill_da(obj)
