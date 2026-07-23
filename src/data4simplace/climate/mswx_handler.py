"""MSWX daily climate loader and 10 km spatial extractor.

MSWX is distributed as one netCDF per variable per day, named ``YYYYDDD.nc``
(year + day-of-year) on a global 0.1 degree grid. This handler opens the files
for a requested date window with dask chunking, subsets to the target bounding
box, harmonises variable names, and aggregates onto the unified target grid.
"""

from __future__ import annotations

import logging
from datetime import date, datetime, timedelta
from pathlib import Path

import pandas as pd
import xarray as xr

from data4simplace.config import PipelineConfig
from data4simplace.grid import TargetGrid

logger = logging.getLogger(__name__)


class MSWXHandler:
    """Load and regrid MSWX climate variables onto the target grid.

    Parameters
    ----------
    config:
        The validated pipeline configuration.
    """

    def __init__(self, config: PipelineConfig) -> None:
        self._config = config
        self._root = Path(config.paths.mswx_root)
        self._grid = TargetGrid.from_config(config.grid)
        self._variables = config.climate.variables or {}
        self._chunks = config.climate.chunks

    # ------------------------------------------------------------------ #
    # Date helpers
    # ------------------------------------------------------------------ #
    def _date_range(self) -> list[date]:
        """Inclusive list of dates from the configured time window."""
        start = datetime.fromisoformat(self._config.time.start).date()
        end = datetime.fromisoformat(self._config.time.end).date()
        if end < start:
            raise ValueError("time.end precedes time.start")
        return [start + timedelta(days=i) for i in range((end - start).days + 1)]

    @staticmethod
    def _filename(day: date) -> str:
        """MSWX file stem ``YYYYDDD`` for a calendar date."""
        return f"{day.year}{day.timetuple().tm_yday:03d}.nc"

    # ------------------------------------------------------------------ #
    # Loading
    # ------------------------------------------------------------------ #
    def _open_variable(self, folder: str, canonical: str) -> xr.DataArray | None:
        """Open all daily files for one MSWX variable folder as a time series."""
        var_dir = self._root / folder
        if not var_dir.is_dir():
            logger.warning("MSWX variable folder missing, skipping: %s", var_dir)
            return None

        days = self._date_range()
        paths = [var_dir / self._filename(d) for d in days]
        existing = [(d, p) for d, p in zip(days, paths) if p.is_file()]
        if not existing:
            logger.warning("No MSWX files for %s in requested window", folder)
            return None

        arrays: list[xr.DataArray] = []
        for day, path in existing:
            ds = xr.open_dataset(path, chunks={"lat": self._chunks.get("lat", 512),
                                               "lon": self._chunks.get("lon", 512)})
            # Each MSWX file holds exactly one data variable.
            data_var = next(iter(ds.data_vars))
            da = ds[data_var].squeeze(drop=True)
            da = da.expand_dims(time=[pd.Timestamp(day)])
            arrays.append(da.rename(canonical))

        combined = xr.concat(arrays, dim="time")
        combined = self._subset_bbox(combined)
        return combined.chunk({"time": self._chunks.get("time", 30)})

    def _subset_bbox(self, da: xr.DataArray) -> xr.DataArray:
        """Clip a global array to the target bounding box (with a 1-cell halo)."""
        g = self._config.grid
        halo = g.resolution_deg
        lat_ascending = bool(da["lat"][0] < da["lat"][-1])
        lat_slice = (
            slice(g.min_lat - halo, g.max_lat + halo)
            if lat_ascending
            else slice(g.max_lat + halo, g.min_lat - halo)
        )
        return da.sel(lat=lat_slice, lon=slice(g.min_lon - halo, g.max_lon + halo))

    # ------------------------------------------------------------------ #
    # Public API
    # ------------------------------------------------------------------ #
    def load(self) -> xr.Dataset:
        """Load all configured variables, aligned onto the target grid.

        Returns
        -------
        xarray.Dataset
            Dataset with dims ``(time, lat, lon)`` on the 10 km grid, one data
            variable per configured MSWX field. Precipitation is aggregated with
            a sum-preserving mean; all others use the mean.
        """
        if not self._variables:
            raise ValueError("No climate.variables configured for MSWX processing")

        regridded: dict[str, xr.DataArray] = {}
        for folder, canonical in self._variables.items():
            da = self._open_variable(folder, canonical)
            if da is None:
                continue
            method = "mean"  # 0.1deg source ~= target; nearest handled in regrid
            out = self._grid.regrid(da, method=method)
            regridded[canonical] = out
            logger.info("Loaded MSWX variable %s -> %s", folder, canonical)

        if not regridded:
            raise RuntimeError("No MSWX variables could be loaded for the requested window")

        return xr.Dataset(regridded)
