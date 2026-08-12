"""Per-cell crop calendar from a gridded planting/harvest product.

The SIMPLACE export carries no sowing date, so every downstream model has had to
invent one — the torchcrop runner sowed every cell from Crete to Lapland on day
270, which is the single largest error in that run. This module samples a
gridded calendar onto the target grid so the date becomes an input rather than
an assumption.

Two products are supported:

``sage``
    Sacks et al. (2010) SAGE crop calendar, ``<crop>.crop.calendar.fill.nc``: a
    0.5 degree **climatology** — no year dimension — of planting and harvest
    days-of-year, each with a start/end window. Cells no reporting unit covers
    are extrapolated from the nearest one and flagged (the ``index`` variable is
    NaN exactly there), which is a third of the European domain.

``ggcmi``
    GGCMI phase 3 crop calendar, one netCDF per crop with ``planting_day`` and
    ``maturity_day`` at 0.5 degrees. Built for gridded crop models rather than
    assembled from census reports, so it is the escape hatch when the SAGE dates
    are also the reference an evaluation compares against.

**Sampling is nearest-neighbour, never interpolated.** These are dates from
discrete reporting units: the mean of DOY 300 and DOY 40 is not a date, and a
bilinear blend across a reporting boundary invents a season no one grows. The
coarse-source branch of :meth:`~data4simplace.grid.TargetGrid.regrid` already
does exactly this, so the sampling is the same machinery every other stage uses.
"""

from __future__ import annotations

import logging
from pathlib import Path

import numpy as np
import xarray as xr

from data4simplace.config import PipelineConfig
from data4simplace.grid import TargetGrid

logger = logging.getLogger(__name__)

__all__ = [
    "CALENDAR_VARIABLES",
    "GGCMI_CROPS",
    "SAGE_CROPS",
    "calendar_path",
    "load_calendar",
    "resolve_calendar_crop",
]

#: Variables this module puts on the target grid. ``calendar_filled`` and
#: ``calendar_present`` are 0/1 flags carried as floats so they survive the
#: nearest-neighbour sampling and the NaN bookkeeping unchanged.
CALENDAR_VARIABLES: tuple[str, ...] = (
    "sowing_doy",
    "sowing_start_doy",
    "sowing_end_doy",
    "harvest_doy",
    "season_length_days",
    "calendar_filled",
    "calendar_present",
)

#: ``npk.simplace_crop`` -> SAGE crop name as it appears in the filename.
#: Winter and spring cereals are separate products, so the mapping is on the
#: SIMPLACE crop rather than on the coarser NPKGRIDS crop.
SAGE_CROPS: dict[str, str] = {
    "winter_wheat": "Wheat.Winter",
    "spring_wheat": "Wheat",
    "wheat": "Wheat",
    "winter_barley": "Barley.Winter",
    "spring_barley": "Barley",
    "barley": "Barley",
    "winter_rye": "Rye.Winter",
    "rye": "Rye.Winter",
    "winter_rapeseed": "Rapeseed.Winter",
    "rapeseed": "Rapeseed.Winter",
    "oats": "Oats",
    "winter_oats": "Oats.Winter",
    "maize": "Maize",
    "grain_maize": "Maize",
    "potato": "Potatoes",
    "potatoes": "Potatoes",
    "sugarbeet": "Sugarbeets",
    "rice": "Rice",
    "soybean": "Soybeans",
    "sorghum": "Sorghum",
    "millet": "Millet",
    "cotton": "Cotton",
}

#: ``npk.simplace_crop`` -> GGCMI crop code used in its filenames (``wwh`` is
#: winter wheat, ``swh`` spring wheat, ``mai`` maize ...).
GGCMI_CROPS: dict[str, str] = {
    "winter_wheat": "wwh",
    "spring_wheat": "swh",
    "wheat": "wwh",
    "maize": "mai",
    "grain_maize": "mai",
    "rice": "ri1",
    "soybean": "soy",
}

#: SAGE netCDF variable -> canonical name. ``tot.days`` is a timedelta and is
#: handled separately.
_SAGE_COLUMNS: dict[str, str] = {
    "plant": "sowing_doy",
    "plant.start": "sowing_start_doy",
    "plant.end": "sowing_end_doy",
    "harvest": "harvest_doy",
}

#: GGCMI netCDF variable -> canonical name. GGCMI publishes no planting window,
#: so ``sowing_start_doy``/``sowing_end_doy`` stay absent and the exporter writes
#: the sentinel for them.
_GGCMI_COLUMNS: dict[str, str] = {
    "planting_day": "sowing_doy",
    "maturity_day": "harvest_doy",
}


def resolve_calendar_crop(config: PipelineConfig) -> str:
    """The calendar product's crop name for this run.

    ``site.calendar_crop`` wins when set; otherwise ``npk.simplace_crop`` is
    mapped through :data:`SAGE_CROPS` / :data:`GGCMI_CROPS`.

    Raises
    ------
    ValueError
        If the SIMPLACE crop has no mapping and no override is configured.
        Guessing would silently sow the wrong crop's calendar.
    """
    configured = config.site.calendar_crop
    if configured is not None:
        return str(configured)

    source = config.site.calendar_source
    table = SAGE_CROPS if source == "sage" else GGCMI_CROPS
    crop = config.npk.simplace_crop.strip().lower()
    name = table.get(crop)
    if name is None:
        raise ValueError(
            f"No {source} calendar crop known for npk.simplace_crop {crop!r}; "
            f"set site.calendar_crop explicitly (known crops: {sorted(table)})"
        )
    logger.info("Calendar crop for %r resolved to %r (%s)", crop, name, source)
    return name


def calendar_path(root: Path, crop: str, source: str = "sage") -> Path:
    """Locate one crop's calendar file under ``root``.

    Raises
    ------
    FileNotFoundError
        If no file matches. SAGE ships most crops gzipped and only the ones in
        use unpacked, so the message says which file to gunzip.
    """
    root = Path(root)
    if source == "sage":
        path = root / f"{crop}.crop.calendar.fill.nc"
        if path.is_file():
            return path
        gzipped = path.with_suffix(".nc.gz")
        hint = f" ({gzipped.name} exists - gunzip it first)" if gzipped.is_file() else ""
        raise FileNotFoundError(f"SAGE calendar not found: {path}{hint}")

    # GGCMI filenames carry a scenario and an irrigation suffix that vary
    # between releases, so the crop code is matched inside the stem instead.
    matches = sorted(
        p for p in root.glob("*.nc")
        if f"_{crop}_" in p.name or p.stem.endswith(f"_{crop}")
    )
    if not matches:
        raise FileNotFoundError(
            f"No GGCMI calendar for crop {crop!r} under {root}; "
            f"found: {', '.join(p.name for p in sorted(root.glob('*.nc'))[:10]) or 'nothing'}"
        )
    if len(matches) > 1:
        logger.warning(
            "Multiple GGCMI calendars match crop %r (%s); using the last",
            crop, ", ".join(p.name for p in matches),
        )
    return matches[-1]


def _subset(dataset: xr.Dataset, config: PipelineConfig) -> xr.Dataset:
    """Sort the axes and cut the global field to the target bbox with a halo.

    The halo is one source cell, so a target cell on the domain edge still has a
    nearest source centre to pick — without it the outermost row would sample
    outside the subset and come back NaN.
    """
    grid = config.grid
    dataset = dataset.sortby(["lat", "lon"])
    halo = 0.5  # one source cell of either product
    return dataset.sel(
        lon=slice(grid.min_lon - halo, grid.max_lon + halo),
        lat=slice(grid.min_lat - halo, grid.max_lat + halo),
    )


def _rename_coords(dataset: xr.Dataset) -> xr.Dataset:
    """Normalise ``longitude``/``latitude`` to ``lon``/``lat``."""
    renames = {
        name: target
        for name, target in (("longitude", "lon"), ("latitude", "lat"))
        if name in dataset.coords and target not in dataset.coords
    }
    return dataset.rename(renames) if renames else dataset


def _to_days(layer: xr.DataArray) -> xr.DataArray:
    """A season-length field as whole days, whatever it is stored as.

    Both products publish the length as a **timedelta**, which xarray decodes to
    nanoseconds; read as a plain float that is 2.5e16 rather than 295. A field
    already stored as a number is passed through, so the conversion is decided
    by the dtype rather than by which product it came from.
    """
    if np.issubdtype(layer.dtype, np.timedelta64):
        return layer / np.timedelta64(1, "D")
    return layer.astype("float64")


def _read_sage(path: Path) -> xr.Dataset:
    """Canonical calendar fields from a SAGE ``.crop.calendar.fill.nc``."""
    with xr.open_dataset(path) as raw:
        raw = _rename_coords(raw)
        parts = {
            out: raw[name].astype("float64")
            for name, out in _SAGE_COLUMNS.items()
            if name in raw
        }
        if "tot.days" in raw:
            parts["season_length_days"] = _to_days(raw["tot.days"])
        # 'index' is the id of the reporting unit a cell's dates come from, and
        # is NaN exactly where SAGE extrapolated the cell from a neighbour. That
        # is the only quality flag the product carries.
        if "index" in raw:
            parts["calendar_filled"] = raw["index"].isnull().astype("float64")
        return xr.Dataset(parts).load()


def _read_ggcmi(path: Path) -> xr.Dataset:
    """Canonical calendar fields from a GGCMI phase-3 calendar file."""
    with xr.open_dataset(path) as raw:
        raw = _rename_coords(raw)
        parts = {
            out: raw[name].astype("float64")
            for name, out in _GGCMI_COLUMNS.items()
            if name in raw
        }
        if not parts:
            raise ValueError(
                f"{path.name} carries neither planting_day nor maturity_day; "
                f"it has {sorted(raw.data_vars)}"
            )
        dataset = xr.Dataset(parts)
        if "growing_season_length" in raw:
            dataset["season_length_days"] = _to_days(raw["growing_season_length"])
        elif {"sowing_doy", "harvest_doy"} <= set(dataset.data_vars):
            # GGCMI dates a season by two days-of-year, so a winter crop's
            # length wraps the year end.
            span = dataset["harvest_doy"] - dataset["sowing_doy"]
            dataset["season_length_days"] = span.where(span > 0, span + 365.0)
        # Every GGCMI cell is modelled, so none is flagged as extrapolated.
        dataset["calendar_filled"] = xr.zeros_like(dataset["sowing_doy"])
        return dataset.load()


def load_calendar(config: PipelineConfig, grid: TargetGrid) -> xr.Dataset:
    """Sample the configured crop calendar onto the target grid.

    Returns
    -------
    xarray.Dataset
        The :data:`CALENDAR_VARIABLES` on ``(lat, lon)``. ``calendar_present``
        is 1 where the product supplied a sowing date and 0 where the cell fell
        outside the crop's mask; everything else is NaN there, to be gap-filled
        or defaulted downstream rather than silently zeroed.

    Raises
    ------
    ValueError
        If ``paths.calendar_root`` is unset.
    FileNotFoundError
        If the crop's file is absent from that root.
    """
    root = config.paths.calendar_root
    if root is None:
        raise ValueError(
            "flags.run_site_processing needs paths.calendar_root to point at the "
            "crop-calendar directory (site.calendar_source selects the product)"
        )

    source = config.site.calendar_source
    crop = resolve_calendar_crop(config)
    path = calendar_path(Path(root), crop, source)
    dataset = _read_sage(path) if source == "sage" else _read_ggcmi(path)
    dataset = _subset(dataset, config)
    logger.info(
        "%s calendar %s: %d x %d source cells in the domain",
        source, path.name, dataset.sizes.get("lat", 0), dataset.sizes.get("lon", 0),
    )

    # Recorded before the sample so a target cell can be told apart from a
    # source cell: 1 = the product covers this cell, 0 = outside the crop mask.
    dataset["calendar_present"] = xr.where(dataset["sowing_doy"].notnull(), 1.0, 0.0)

    sampled = grid.regrid(dataset, method="mean")
    assert isinstance(sampled, xr.Dataset)  # a Dataset in, a Dataset out

    # regrid() returns NaN outside the source footprint; a cell the product does
    # not cover is "not present", not "unknown whether present".
    sampled["calendar_present"] = sampled["calendar_present"].fillna(0.0)
    for name in CALENDAR_VARIABLES:
        if name not in sampled:
            sampled[name] = xr.full_like(sampled["calendar_present"], np.nan)

    sampled.attrs.update(
        calendar_source=source, calendar_crop=crop, calendar_file=str(path)
    )
    covered = int(sampled["calendar_present"].sum())
    filled = int((sampled["calendar_filled"] == 1.0).sum())
    logger.info(
        "Calendar sampled onto %d target cells (%d covered, of which %d were "
        "extrapolated by the product itself)",
        sampled["calendar_present"].size, covered, filled,
    )
    return sampled[list(CALENDAR_VARIABLES)]
