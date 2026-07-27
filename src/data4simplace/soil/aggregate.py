"""Per-target-cell aggregation of fine (250 m) soil fields.

Reduces a fine ``lat``/``lon`` field onto the 10 km :class:`TargetGrid` using
**true 2-D binning**: every fine pixel is assigned to exactly one target cell,
then all of a cell's pixels are reduced together. This is required because the
soil aggregation rules (majority vote, geometric mean, pH H+ mean) are not
separable across axes, so the sequential per-axis reduction used by
``TargetGrid.regrid`` would give wrong answers.

All reducers here are **equal-weight** (per the agreed workflow): the fine field
is expected to have already been masked to cropland and to the dominant soil
class, so every surviving pixel counts the same.
"""

from __future__ import annotations

from typing import Callable

import numpy as np
import pandas as pd
import xarray as xr

from data4simplace.grid import TargetGrid

# A reducer maps the 1-D array of a cell's (finite) pixel values to one scalar.
Reducer = Callable[[np.ndarray], float]

# Per-cell summary statistics offered by :func:`bin_describe` (pandas names).
CELL_STATISTICS = ("mean", "median", "std", "kurt", "count")


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


def fill_missing_cells(obj: xr.DataArray | xr.Dataset) -> xr.DataArray | xr.Dataset:
    """Nearest-neighbour fill of empty ``(lat, lon)`` cells (coastal/islands).

    Operates per 2-D lat/lon slice, looping over any extra dims (e.g. ``depth``).
    Returns the input unchanged if SciPy is unavailable (with a warning).
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
        return xr.apply_ufunc(
            _fill_nearest_2d,
            da,
            input_core_dims=[["lat", "lon"]],
            output_core_dims=[["lat", "lon"]],
            vectorize=True,
            keep_attrs=True,
        )

    if isinstance(obj, xr.Dataset):
        return obj.map(_fill_da)
    return _fill_da(obj)
