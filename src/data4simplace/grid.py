"""The unified 10 km target grid and ``SimplaceID`` cell identifiers.

Every handler resamples/aligns onto the :class:`TargetGrid` defined here, and
every exporter uses the same :meth:`TargetGrid.cell_table` so that a given cell
carries one stable ``SimplaceID`` across weather, soil and management outputs.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
import xarray as xr

from data4simplace.config import GridConfig

#: Relative slack on the source-vs-target spacing comparison in
#: :meth:`TargetGrid.regrid`. Source axes stored as float32 (MSWX) measure a
#: fraction of a part per million off their nominal resolution.
_RES_TOL = 1e-3


@dataclass(frozen=True)
class TargetGrid:
    """A regular lon/lat grid of cell centres at a fixed resolution.

    The grid is built from a bounding box and resolution. Cell centres are
    offset by half a cell from the box edges. ``SimplaceID`` is assigned in
    row-major (north-to-south, west-to-east) order and is deterministic for a
    given bounding box + resolution.
    """

    min_lon: float
    max_lon: float
    min_lat: float
    max_lat: float
    resolution_deg: float
    crs: str = "EPSG:4326"

    @classmethod
    def from_config(cls, grid: GridConfig) -> "TargetGrid":
        """Build a :class:`TargetGrid` from a validated :class:`GridConfig`."""
        return cls(
            min_lon=grid.min_lon,
            max_lon=grid.max_lon,
            min_lat=grid.min_lat,
            max_lat=grid.max_lat,
            resolution_deg=grid.resolution_deg,
            crs=grid.crs,
        )

    @property
    def lon_centers(self) -> np.ndarray:
        """1-D array of longitude cell centres (west -> east)."""
        res = self.resolution_deg
        return np.arange(self.min_lon + res / 2.0, self.max_lon, res)

    @property
    def lat_centers(self) -> np.ndarray:
        """1-D array of latitude cell centres (north -> south)."""
        res = self.resolution_deg
        return np.arange(self.max_lat - res / 2.0, self.min_lat, -res)

    @property
    def shape(self) -> tuple[int, int]:
        """``(n_lat, n_lon)`` size of the grid."""
        return (self.lat_centers.size, self.lon_centers.size)

    def cell_table(self) -> pd.DataFrame:
        """Return a table of all cells with a stable ``SimplaceID``.

        Columns: ``SimplaceID`` (1-based int), ``row``, ``col``, ``lat``, ``lon``.
        """
        lats = self.lat_centers
        lons = self.lon_centers
        lon_mesh, lat_mesh = np.meshgrid(lons, lats)
        rows, cols = np.meshgrid(np.arange(lats.size), np.arange(lons.size), indexing="ij")
        n = lats.size * lons.size
        table = pd.DataFrame(
            {
                "SimplaceID": np.arange(1, n + 1, dtype=np.int64),
                "row": rows.ravel(),
                "col": cols.ravel(),
                "lat": lat_mesh.ravel(),
                "lon": lon_mesh.ravel(),
            }
        )
        return table

    def target_dataarray(self) -> xr.DataArray:
        """An empty template :class:`~xarray.DataArray` on the target grid."""
        return xr.DataArray(
            data=np.full(self.shape, np.nan, dtype="float32"),
            dims=("lat", "lon"),
            coords={"lat": self.lat_centers, "lon": self.lon_centers},
            name="template",
        )

    def regrid(self, data: xr.DataArray | xr.Dataset, method: str = "mean") -> xr.DataArray | xr.Dataset:
        """Aggregate a finer field onto this grid by binned reduction.

        Three cases, decided from the source spacing:

        * **Same resolution** (within :data:`_RES_TOL`) - a pure nearest pick via
          ``reindex``. There is nothing to aggregate: each target cell holds
          exactly one source cell. Taking the binned path here would be both a
          no-op numerically and ruinous computationally - ``groupby_bins`` emits
          ``n_lat x n_lon`` tasks *per chunk*, so a 46-year MSWX tile chunked by
          30 days builds ~8.5 million dask tasks per variable and exhausts the
          node in ``dask.optimization.fuse`` before computing anything.
        * **Coarser source** - nearest-neighbour interpolation.
        * **Finer source** - ``groupby_bins`` on both axes, reducing many fine
          cells into each target cell.

        The same-resolution test needs a tolerance rather than ``>=``: MSWX
        stores its axes as float32, so a nominal 0.1 degree grid measures
        0.099999822676 and would otherwise be misjudged as "finer".

        Parameters
        ----------
        data:
            Source object carrying ``lat``/``lon`` (or ``y``/``x``) coordinates.
        method:
            Reduction applied within each target cell (``"mean"``, ``"median"``,
            ``"min"``, ``"max"``, ``"sum"``).
        """
        data = _standardise_lonlat_names(data)
        src_res = float(np.abs(np.diff(data["lon"].values)).mean()) if data["lon"].size > 1 else self.resolution_deg

        if abs(src_res - self.resolution_deg) <= self.resolution_deg * _RES_TOL:
            # Half a cell: the nearest source centre either is this cell's own
            # or belongs to a neighbour, and float32 axis drift is ~1e-4 deg.
            return data.reindex(
                lat=self.lat_centers,
                lon=self.lon_centers,
                method="nearest",
                tolerance=self.resolution_deg / 2.0,
            )

        if src_res >= self.resolution_deg:
            return data.interp(lat=self.lat_centers, lon=self.lon_centers, method="nearest")

        lon_edges = _edges(self.lon_centers, self.resolution_deg)
        lat_edges = _edges(np.sort(self.lat_centers), self.resolution_deg)

        binned = (
            data.groupby_bins("lon", lon_edges, labels=self.lon_centers)
            .reduce(_reducer(method), dim="lon")
            .groupby_bins("lat", lat_edges, labels=np.sort(self.lat_centers))
            .reduce(_reducer(method), dim="lat")
            .rename({"lon_bins": "lon", "lat_bins": "lat"})
        )
        # Restore the north-to-south latitude ordering used throughout.
        return binned.reindex(lat=self.lat_centers, lon=self.lon_centers)


def _standardise_lonlat_names(obj: xr.DataArray | xr.Dataset) -> xr.DataArray | xr.Dataset:
    """Rename ``x``/``y``/``longitude``/``latitude`` coords to ``lon``/``lat``."""
    renames: dict[str, str] = {}
    for cand, target in (("x", "lon"), ("longitude", "lon"), ("y", "lat"), ("latitude", "lat")):
        if cand in obj.coords and target not in obj.coords:
            renames[cand] = target
    return obj.rename(renames) if renames else obj


def _edges(centers: np.ndarray, res: float) -> np.ndarray:
    """Cell edges for an array of monotonically increasing centres."""
    return np.concatenate([centers - res / 2.0, centers[-1:] + res / 2.0])


def _reducer(method: str):
    """Map a reduction name to a NumPy nan-aware callable."""
    reducers = {
        "mean": np.nanmean,
        "median": np.nanmedian,
        "min": np.nanmin,
        "max": np.nanmax,
        "sum": np.nansum,
    }
    if method not in reducers:
        raise ValueError(f"Unsupported regrid method {method!r}; choose from {sorted(reducers)}")
    return reducers[method]
