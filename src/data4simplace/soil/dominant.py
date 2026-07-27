"""Dominant soil-type classification and per-target-cell selection.

Implements the dominant-soil-type stage of the CLAUDE.md workflow: derive a
texture class from unscaled sand/silt/clay at the native 250 m resolution,
choose the dominant class per 10 km target cell by pixel majority (equal-weight,
over cropland-filtered pixels), and mask the 250 m stack to keep only pixels of
that dominant class.

Two texture-based classifications are available:

``usda``
    The 12-class USDA texture class of the topsoil layer.
``usda_profile``
    A composite **(topsoil, rooting-zone)** class: the topsoil USDA class paired
    with the USDA class of the thickness-weighted mean texture over the rooting
    zone (5-100 cm by default), giving 12 x 12 = 144 codes. A surface class
    alone cannot express the layered profiles that dominate NE Germany (sandy
    cover over loamy till), where the subsoil sets plant-available water.

Equal-weight was chosen over cropland-fraction weighting: pixels are first hard-
filtered to >= ``soil.cropland_min_fraction`` cropland, then counted equally.
"""

from __future__ import annotations

import logging
import re
from typing import Mapping

import numpy as np
import pandas as pd
import xarray as xr

from data4simplace.grid import TargetGrid

logger = logging.getLogger(__name__)

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
    from data4simplace.soil.aggregate import bin_reduce

    coarse = bin_reduce(class_da, grid, _mode_reducer, fill=0.0)
    return coarse.astype("int16").rename("dominant_class")

def rank_classes_per_cell(
    class_da: xr.DataArray, grid: TargetGrid, n_classes: int = 3
) -> xr.Dataset:
    """The ``n_classes`` most frequent classes in each target cell.

    The single dominant class is rank 1, so this generalises
    :func:`dominant_class_per_cell`: the extra ranks describe how much of a cell
    the dominant class actually represents, which is what an inter-class
    uncertainty estimate needs. Ties are broken by the lower class code, matching
    the majority vote.

    Parameters
    ----------
    class_da:
        Integer class codes on the fine grid; ``0`` (unclassified) is ignored.
    grid:
        The target 10 km grid.
    n_classes:
        How many ranks to keep.

    Returns
    -------
    xarray.Dataset
        ``class_code`` (0 where a cell has fewer classes than ranks), ``pixels``
        and ``share_percent`` (percent of the cell's classified pixels), all with
        dims ``(rank, lat, lon)`` and a 1-based ``rank`` coordinate.
    """
    from data4simplace.soil.aggregate import cell_assignment

    if n_classes < 1:
        raise ValueError(f"n_classes must be >= 1, got {n_classes}")

    n_lat, n_lon = grid.lat_centers.size, grid.lon_centers.size
    cell, codes = cell_assignment(class_da, grid, finite_only=False)

    code_arr = np.full((n_classes, n_lat * n_lon), 0, dtype="int64")
    pixel_arr = np.zeros((n_classes, n_lat * n_lon), dtype="int64")
    share_arr = np.full((n_classes, n_lat * n_lon), np.nan, dtype="float64")

    valid = codes > 0
    if valid.any():
        counts = (
            pd.DataFrame({"cell": cell[valid], "code": codes[valid].astype("int64")})
            .groupby(["cell", "code"], sort=False)
            .size()
            .rename("pixels")
            .reset_index()
        )
        counts["share"] = 100.0 * counts["pixels"] / counts.groupby("cell")[
            "pixels"
        ].transform("sum")
        # Most pixels first; the lower code wins a tie (as the majority vote does).
        counts = counts.sort_values(
            ["cell", "pixels", "code"], ascending=[True, False, True]
        )
        counts["rank"] = counts.groupby("cell").cumcount()
        counts = counts[counts["rank"] < n_classes]
        rank_idx = counts["rank"].to_numpy()
        cell_idx = counts["cell"].to_numpy()
        code_arr[rank_idx, cell_idx] = counts["code"].to_numpy()
        pixel_arr[rank_idx, cell_idx] = counts["pixels"].to_numpy()
        share_arr[rank_idx, cell_idx] = counts["share"].to_numpy()

    dims = ("rank", "lat", "lon")
    coords = {
        "rank": np.arange(1, n_classes + 1),
        "lat": grid.lat_centers,
        "lon": grid.lon_centers,
    }
    shape = (n_classes, n_lat, n_lon)
    return xr.Dataset(
        {
            "class_code": (dims, code_arr.reshape(shape).astype("int16")),
            "pixels": (dims, pixel_arr.reshape(shape).astype("int32")),
            "share_percent": (dims, share_arr.reshape(shape).astype("float32")),
        },
        coords=coords,
    )


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

    Reducer for :func:`~data4simplace.soil.aggregate.bin_reduce`: returns the
    most frequent positive integer class; ``0`` when no valid votes remain.
    """
    x = x[x > 0].astype("int64")
    if x.size == 0:
        return 0.0
    return float(np.bincount(x).argmax())
