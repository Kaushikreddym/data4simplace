"""Daily gridded climate loader and 10 km spatial extractor.

Two sources share this reader, because they share a layout:

``mswx``
    The observational MSWX archive: one netCDF per variable per day, named
    ``YYYYDDD.nc`` (year + day-of-year), on a global 0.1 degree grid.
``cmip6_bc``
    CMIP6 output bias-corrected and downscaled onto MSWX by climdata, one
    ``<model>_<experiment>/`` folder per run, in the same ``<VAR>/YYYYDDD.nc``
    layout and units. Its frame is regional (35-70 N, 25 W-45 E) and its axes
    are named ``latitude``/``longitude``; a tile outside the frame loads
    nothing rather than failing.

``climate.layout: yearly`` reads climdata's one-file-per-year cache instead of
the daily files. The values are identical (checked bit for bit on four
variables); only the number of file opens differs.

Reading strategy
----------------
An MSWX file holds a 1800x3600 float32 global frame chunked on disk at
``(1, 32, 32)``, so the loader subsets **per file, at read time**: ``isel`` on a
NumPy-backed (not dask-backed) variable makes HDF5 decompress only the chunks
the window intersects. Measured over 120 files for one 5 degree tile, this reads
7.9 MB instead of 661 MB. The climdata files are chunked one whole frame per
day, so there every day is decompressed in full whichever layout is read; the
cost that remains is the open, which is why ``yearly`` exists.

The reads are spread over a **process** pool. HDF5 serialises reads on a global
lock, so threads do not help here - a 16-thread pool benchmarked slower than
serial (5.2 s vs 4.1 s), while 16 processes finished the same work in 0.66 s.
"""

from __future__ import annotations

import logging
import os
from concurrent.futures import ProcessPoolExecutor
from datetime import date, datetime, timedelta
from pathlib import Path

import numpy as np
import pandas as pd
import xarray as xr

from data4simplace.config import PipelineConfig
from data4simplace.grid import TargetGrid

logger = logging.getLogger(__name__)

#: Past this, the per-file reads stop scaling and only add open() contention
#: on the shared filesystem (16 procs: 0.66 s, 32 procs: 0.83 s for 120 files).
_MAX_READ_WORKERS = 16

_LAT_NAMES = ("lat", "latitude")
_LON_NAMES = ("lon", "longitude")

#: ``(path, lat0, lat1, lon0, lon1, lat_name, lon_name)``
ReadTask = tuple[str, int, int, int, int, str, str]


def _axis_name(ds: xr.Dataset, candidates: tuple[str, ...]) -> str:
    """The dataset's spelling of a horizontal axis."""
    name = next((c for c in candidates if c in ds.coords or c in ds.dims), None)
    if name is None:
        raise KeyError(f"none of {candidates} in {sorted(ds.coords)}")
    return name


def _window(ds: xr.Dataset, task: ReadTask) -> xr.DataArray:
    """The first data variable of ``ds``, clipped to the task's index window."""
    _, lat0, lat1, lon0, lon1, lat_name, lon_name = task
    var = ds[next(iter(ds.data_vars))]
    # The variable is NumPy-backed, so this isel pushes straight down to a
    # partial HDF5 read of the intersecting chunks only.
    return var.isel({lat_name: slice(lat0, lat1), lon_name: slice(lon0, lon1)})


def _read_window(task: ReadTask) -> np.ndarray | None:
    """Read one daily frame, already clipped to the tile window.

    Module-level and tuple-argumented so it is picklable by
    :class:`~concurrent.futures.ProcessPoolExecutor`.

    Returns
    -------
    numpy.ndarray or None
        The ``(nlat, nlon)`` window, or ``None`` if the file could not be read
        (a corrupt or truncated day must not abort a 46-year run).
    """
    try:
        # cache=False keeps h5netcdf from retaining each frame; engine is pinned
        # to h5netcdf because it is measurably faster than netcdf4 here.
        with xr.open_dataset(task[0], engine="h5netcdf", cache=False) as ds:
            return np.asarray(_window(ds, task).squeeze(drop=True).values, dtype="float32")
    except Exception as exc:  # noqa: BLE001 - one bad day must not kill the tile
        logger.warning("Unreadable climate file %s: %s", task[0], exc)
        return None


def _read_year(
    args: tuple[ReadTask, str, str],
) -> tuple[pd.DatetimeIndex, np.ndarray] | None:
    """Read the days of one yearly file that fall inside ``[start, end]``.

    Returns ``(dates, (ndays, nlat, nlon) array)``, or ``None`` if unreadable.
    """
    task, start, end = args
    try:
        with xr.open_dataset(task[0], engine="h5netcdf", cache=False) as ds:
            index = ds.indexes["time"]
            if not isinstance(index, pd.DatetimeIndex):
                index = index.to_datetimeindex()
            days = index.normalize()
            hit = np.flatnonzero((days >= start) & (days <= end))
            if hit.size == 0:
                return pd.DatetimeIndex([]), np.empty((0, 0, 0), "float32")
            frame = _window(ds, task).isel(time=slice(int(hit[0]), int(hit[-1]) + 1))
            return days[hit[0] : hit[-1] + 1], np.asarray(frame.values, dtype="float32")
    except Exception as exc:  # noqa: BLE001
        logger.warning("Unreadable climate file %s: %s", task[0], exc)
        return None


class MSWXHandler:
    """Load and regrid MSWX-layout climate variables onto the target grid.

    Parameters
    ----------
    config:
        The validated pipeline configuration. ``climate.source`` selects the
        archive, ``climate.layout`` the file granularity.
    """

    def __init__(self, config: PipelineConfig) -> None:
        self._config = config
        climate = config.climate
        self._source = climate.source
        self._layout = climate.layout
        self._root = self._resolve_root(config)
        self._grid = TargetGrid.from_config(config.grid)
        self._variables = climate.variables or {}
        self._chunks = climate.chunks
        self._workers = self._resolve_workers(climate.read_workers)
        # Axis names, global axes and the tile's index window, resolved once
        # from the first readable file: every file of a source shares one grid.
        # ``_window`` stays None after resolution when the tile misses the frame.
        self._names: tuple[str, str] = ("lat", "lon")
        self._axes: tuple[np.ndarray, np.ndarray] | None = None
        self._window: tuple[int, int, int, int] | None = None
        self._resolved = False

    # ------------------------------------------------------------------ #
    # Setup
    # ------------------------------------------------------------------ #
    @staticmethod
    def _resolve_root(config: PipelineConfig) -> Path:
        """The directory holding ``<VAR>/`` folders (daily) or the yearly tree."""
        if config.climate.source == "mswx":
            return Path(config.paths.mswx_root)
        assert config.paths.cmip6_bc_root is not None  # enforced by PipelineConfig
        root = Path(config.paths.cmip6_bc_root)
        return root if config.climate.layout == "yearly" else root / str(config.climate.experiment)

    @staticmethod
    def _resolve_workers(configured: int | None) -> int:
        """Worker count for the read pool, defaulting to the SLURM allocation."""
        if configured is not None:
            return max(1, int(configured))
        alloc = os.environ.get("SLURM_CPUS_PER_TASK") or os.environ.get(
            "SLURM_JOB_CPUS_PER_NODE"
        )
        try:
            cpus = int(str(alloc).split("(")[0]) if alloc else (os.cpu_count() or 1)
        except ValueError:
            cpus = os.cpu_count() or 1
        return max(1, min(cpus, _MAX_READ_WORKERS))

    def _resolve_window(self, samples: list[Path]) -> tuple[np.ndarray, np.ndarray] | None:
        """Cache the axes and the tile's index window into them.

        Indices rather than label slices, so the window is valid regardless of
        whether latitude runs north-to-south or south-to-north.

        Tries successive files: the MSWX mirror contains zero-byte days, and the
        first file of a window must not be able to sink the whole variable.

        Returns ``None`` when the tile lies outside the source frame -- a
        regional source simply has no weather there.
        """
        if self._resolved:
            return self._axes

        lat = lon = None
        for sample in samples:
            try:
                with xr.open_dataset(sample, engine="h5netcdf") as ds:
                    self._names = (_axis_name(ds, _LAT_NAMES), _axis_name(ds, _LON_NAMES))
                    lat = np.asarray(ds[self._names[0]].values)
                    lon = np.asarray(ds[self._names[1]].values)
                break
            except Exception as exc:  # noqa: BLE001
                logger.warning("Cannot read grid from %s: %s", sample, exc)
        if lat is None or lon is None:
            raise RuntimeError(
                f"No readable climate file among {len(samples)} candidates; "
                "cannot establish the source grid"
            )

        self._resolved = True
        g = self._config.grid
        halo = g.resolution_deg  # one target cell of slack for the regrid
        lat_hit = np.where((lat >= g.min_lat - halo) & (lat <= g.max_lat + halo))[0]
        lon_hit = np.where((lon >= g.min_lon - halo) & (lon <= g.max_lon + halo))[0]
        if lat_hit.size == 0 or lon_hit.size == 0:
            logger.info(
                "Target grid (%g..%g N, %g..%g E) lies outside the %s frame "
                "(%.2f..%.2f N, %.2f..%.2f E); no weather for this tile",
                g.min_lat, g.max_lat, g.min_lon, g.max_lon, self._source,
                lat.min(), lat.max(), lon.min(), lon.max(),
            )
            return None

        lat0, lat1 = int(lat_hit[0]), int(lat_hit[-1]) + 1
        lon0, lon1 = int(lon_hit[0]), int(lon_hit[-1]) + 1
        self._window = (lat0, lat1, lon0, lon1)
        self._axes = (lat[lat0:lat1], lon[lon0:lon1])
        logger.info(
            "%s window: %d x %d cells of the %d x %d source grid (%d read workers)",
            self._source, lat1 - lat0, lon1 - lon0, lat.size, lon.size, self._workers,
        )
        return self._axes

    def _task(self, path: Path) -> ReadTask:
        assert self._window is not None
        return (str(path), *self._window, *self._names)

    # ------------------------------------------------------------------ #
    # Dates and files
    # ------------------------------------------------------------------ #
    def _bounds(self) -> tuple[date, date]:
        start = datetime.fromisoformat(self._config.time.start).date()
        end = datetime.fromisoformat(self._config.time.end).date()
        if end < start:
            raise ValueError("time.end precedes time.start")
        return start, end

    def _date_range(self) -> list[date]:
        """Inclusive list of dates from the configured time window."""
        start, end = self._bounds()
        return [start + timedelta(days=i) for i in range((end - start).days + 1)]

    @staticmethod
    def _filename(day: date) -> str:
        """Daily file stem ``YYYYDDD`` for a calendar date."""
        return f"{day.year}{day.timetuple().tm_yday:03d}.nc"

    def _yearly_files(self, canonical: str) -> dict[int, Path]:
        """``{year: path}`` of the yearly files covering the window."""
        start, end = self._bounds()
        found: dict[int, Path] = {}
        for year in range(start.year, end.year + 1):
            pattern = self._config.climate.yearly_pattern.format(
                experiment=self._config.climate.experiment, var=canonical, year=year
            )
            if hits := sorted(self._root.glob(pattern)):
                found[year] = hits[0]
        return found

    def absent(self, canonical: str) -> bool:
        """Whether the source holds no files at all for a canonical variable."""
        if self._layout == "yearly":
            return not self._yearly_files(canonical)
        folder = next(f for f, c in self._variables.items() if c == canonical)
        return not (self._root / folder).is_dir()

    def filled(self) -> list[str]:
        """``climate.climatology_fill`` variables the source lacks, so are filled."""
        return [v for v in self._config.climate.climatology_fill if self.absent(v)]

    def missing_required(self) -> list[str]:
        """Required canonical variables the source lacks and nothing fills.

        Cheap -- a directory test or one glob per variable -- so a run can be
        refused at ``--dry-run`` instead of after reading the rest.
        """
        fill = set(self._config.climate.climatology_fill)
        return [v for v in self._config.climate.required if v not in fill and self.absent(v)]

    # ------------------------------------------------------------------ #
    # MSWX climatology fill
    # ------------------------------------------------------------------ #
    def _climatology_path(self, canonical: str) -> Path | None:
        climate, g = self._config.climate, self._config.grid
        if climate.climatology_cache is None:
            return None
        y0, y1 = climate.climatology_years
        return Path(climate.climatology_cache) / (
            f"{canonical}_{y0}-{y1}_w{climate.climatology_window_days}_"
            f"{g.min_lat:g}_{g.max_lat:g}_{g.min_lon:g}_{g.max_lon:g}.nc"
        )

    def _climatology(self, canonical: str) -> xr.DataArray:
        """Smoothed MSWX day-of-year mean of one variable on this grid, ``(dayofyear, lat, lon)``.

        Cached per tile: every experiment of a scenario run asks for the same one.
        """
        path = self._climatology_path(canonical)
        if path is not None and path.is_file():
            with xr.open_dataarray(path) as cached:
                return cached.load()

        climate = self._config.climate
        y0, y1 = climate.climatology_years
        folder = next(f for f, c in self._variables.items() if c == canonical)
        mswx = self._config.model_copy(update={
            "climate": climate.model_copy(update={
                "source": "mswx", "layout": "daily", "experiment": None,
                "required": [], "climatology_fill": [], "variables": {folder: canonical},
            }),
            "time": self._config.time.model_copy(
                update={"start": f"{y0}-01-01", "end": f"{y1}-12-31"}
            ),
        })
        loaded = MSWXHandler(mswx).load()
        if loaded is None or canonical not in loaded:
            raise RuntimeError(f"MSWX holds no {folder} for the {canonical} climatology")
        daily = loaded[canonical].load()
        mean = daily.groupby(daily["time"].dt.dayofyear).mean("time")
        # Circular running mean: 36 years leave visible day-to-day noise in a
        # raw day-of-year mean, which would imprint as a fake seasonal wiggle.
        w = climate.climatology_window_days
        half = w // 2
        values = mean.values
        padded = np.concatenate([values[-half:], values, values[: w - 1 - half]]) if w > 1 else values
        smooth = np.lib.stride_tricks.sliding_window_view(padded, w, axis=0).mean(-1)
        clim = mean.copy(data=smooth.astype("float32")).rename(canonical)
        clim.attrs.update(
            source=f"MSWX {folder} day-of-year mean {y0}-{y1}, {w}-day circular running mean"
        )
        if path is not None:
            path.parent.mkdir(parents=True, exist_ok=True)
            partial = path.with_suffix(f".{os.getpid()}.part")
            clim.to_netcdf(partial)
            partial.replace(path)  # atomic: tiles of parallel experiments share it
        logger.info("MSWX %s climatology %d-%d built for this tile", canonical, y0, y1)
        return clim

    def _fill(self, canonical: str, times: pd.DatetimeIndex) -> xr.DataArray:
        clim = self._climatology(canonical)
        out = clim.sel(dayofyear=xr.DataArray(times.dayofyear, dims="time"))
        out = out.drop_vars("dayofyear").assign_coords(time=times).transpose("time", "lat", "lon")
        return out.chunk({"time": self._chunks.get("time", 30)})

    # ------------------------------------------------------------------ #
    # Loading
    # ------------------------------------------------------------------ #
    def _map(self, fn, tasks: list):
        if self._workers == 1:
            return [fn(t) for t in tasks]
        # chunksize batches the tiny per-file tasks so dispatch overhead does
        # not dominate the read itself.
        with ProcessPoolExecutor(max_workers=self._workers) as pool:
            return list(pool.map(fn, tasks, chunksize=8 if fn is _read_window else 1))

    def _daily_frames(self, folder: str) -> tuple[pd.DatetimeIndex, np.ndarray] | None:
        var_dir = self._root / folder
        if not var_dir.is_dir():
            logger.warning("Climate variable folder missing, skipping: %s", var_dir)
            return None

        days = self._date_range()
        # Zero-byte days exist in the mirror; drop them here so they are counted
        # as gaps rather than surfacing as an open() failure per worker.
        existing: list[tuple[date, Path]] = []
        empty = 0
        for d in days:
            p = var_dir / self._filename(d)
            try:
                if not p.is_file():
                    continue
                if p.stat().st_size == 0:
                    empty += 1
                    continue
            except OSError:
                continue
            existing.append((d, p))

        if missing := len(days) - len(existing):
            logger.warning(
                "%s: %d of %d days unavailable (%.1f%%; %d zero-byte, %d absent)",
                folder, missing, len(days), 100.0 * missing / len(days),
                empty, missing - empty,
            )
        if not existing:
            logger.warning("No climate files for %s in requested window", folder)
            return None

        if self._resolve_window([p for _, p in existing[:20]]) is None:
            return None
        frames = self._map(_read_window, [self._task(p) for _, p in existing])

        kept = [(d, f) for (d, _), f in zip(existing, frames) if f is not None]
        if not kept:
            logger.warning("Every climate file for %s failed to read", folder)
            return None
        if len(kept) < len(existing):
            logger.warning(
                "%s: %d of %d files unreadable, continuing without them",
                folder, len(existing) - len(kept), len(existing),
            )
        return pd.DatetimeIndex([pd.Timestamp(d) for d, _ in kept]), np.stack([f for _, f in kept])

    def _yearly_frames(self, canonical: str) -> tuple[pd.DatetimeIndex, np.ndarray] | None:
        files = self._yearly_files(canonical)
        start, end = self._bounds()
        if missing := sorted(set(range(start.year, end.year + 1)) - set(files)):
            logger.warning(
                "%s: no yearly file for %d of %d years (first: %d)",
                canonical, len(missing), end.year - start.year + 1, missing[0],
            )
        if not files:
            logger.warning("No yearly climate files for %s under %s", canonical, self._root)
            return None

        if self._resolve_window(list(files.values())[:5]) is None:
            return None
        lo, hi = pd.Timestamp(start), pd.Timestamp(end)
        parts = [
            p for p in self._map(_read_year, [(self._task(f), lo, hi) for f in files.values()])
            if p is not None and len(p[0])
        ]
        if not parts:
            logger.warning("Every yearly climate file for %s failed to read", canonical)
            return None
        return (
            pd.DatetimeIndex(np.concatenate([d.values for d, _ in parts])),
            np.concatenate([a for _, a in parts]),
        )

    def _open_variable(self, folder: str, canonical: str) -> xr.DataArray | None:
        """Every day of one variable in the window, clipped to the tile."""
        read = (
            self._yearly_frames(canonical) if self._layout == "yearly"
            else self._daily_frames(folder)
        )
        if read is None:
            return None
        times, stacked = read
        lat_vals, lon_vals = self._axes  # type: ignore[misc]
        da = xr.DataArray(
            stacked,
            dims=("time", "lat", "lon"),
            coords={"time": times, "lat": lat_vals, "lon": lon_vals},
            name=canonical,
        )
        return da.chunk({"time": self._chunks.get("time", 30)})

    # ------------------------------------------------------------------ #
    # Public API
    # ------------------------------------------------------------------ #
    def load(self) -> xr.Dataset | None:
        """Load all configured variables, aligned onto the target grid.

        Returns
        -------
        xarray.Dataset or None
            Dims ``(time, lat, lon)`` on the 10 km grid, one data variable per
            configured field; ``None`` when the target grid lies wholly outside
            the source frame.

        Raises
        ------
        ValueError
            If a ``climate.required`` variable has no files. Checked before any
            read, so an incomplete CMIP6 experiment fails in seconds.
        RuntimeError
            If the grid intersects the source but no variable could be loaded.
        """
        if not self._variables:
            raise ValueError("No climate.variables configured")
        if missing := self.missing_required():
            raise ValueError(
                f"{self._source} source {self._root} lacks required variable(s) "
                f"{', '.join(missing)} (climate.required)"
            )

        filled = self.filled()
        regridded: dict[str, xr.DataArray] = {}
        for folder, canonical in self._variables.items():
            if canonical in filled:
                continue
            da = self._open_variable(folder, canonical)
            if self._resolved and self._window is None:
                return None
            if da is None:
                continue
            regridded[canonical] = self._grid.regrid(da, method="mean")
            logger.info("Loaded %s variable %s -> %s", self._source, folder, canonical)

        if not regridded:
            raise RuntimeError("No climate variables could be loaded for the requested window")
        times = pd.DatetimeIndex(next(iter(regridded.values()))["time"].values)
        for canonical in filled:
            regridded[canonical] = self._fill(canonical, times)
            logger.warning(
                "%s absent from %s: filled with the MSWX day-of-year climatology",
                canonical, self._root,
            )
        if missing := sorted(set(self._config.climate.required) - set(regridded)):
            raise RuntimeError(f"Required climate variable(s) failed to load: {', '.join(missing)}")
        return xr.Dataset(regridded)
