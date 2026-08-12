"""Loader for the GDHY global gridded yield product.

GDHY v1.2/v1.3 (Iizumi & Sakai 2020, *Scientific Data*) publishes one NetCDF
per crop and year holding a single variable, ``var``, in **t/ha**, on a 0.5°
grid whose longitudes run 0-360. This module reads a range of years, puts the
longitudes on -180..180, crops to the domain and returns a tidy frame.

Two properties of the product bound every result computed from it, and neither
is visible in the file:

* **It is not an observation of the grid cell.** GDHY downscales reported
  national and sub-national yield statistics onto the grid using a
  satellite-derived growing-season vegetation index and a crop mask. Within one
  country the interannual signal is largely the national statistic; what varies
  between cells of that country is the satellite anomaly and the mask. So a
  high spatial correlation partly measures agreement with a crop mask, and the
  interannual signal is not independent between neighbouring cells.
* **It is a market-moisture statistic**, inheriting the moisture basis of the
  census data behind it, whereas the run reports grain dry matter. Nothing here
  converts between them — see :func:`utils.aggregate.to_dry_matter`.

The crop directory matters as much as the crop: ``wheat_winter`` and
``wheat_spring`` are a split of ``wheat``, and pairing a winter-wheat run
against the merged product would compare it with a field that is partly spring
wheat in exactly the northern countries where the two differ most.
"""

from __future__ import annotations

import logging
from pathlib import Path

import numpy as np
import pandas as pd
import xarray as xr

from .config import (
    EUROPE_BBOX,
    GDHY_CROP,
    GDHY_HARVEST_YEAR_OFFSET,
    GDHY_ROOT,
    GDHY_YIELD_TEMPLATE,
    GRID_RES_DEG,
)
from .grid import crop_to_bbox, grid_cell_id, snap_to_grid

logger = logging.getLogger(__name__)

__all__ = ["available_years", "load_gdhy", "load_gdhy_year"]


def _crop_dir(root: Path | None, crop: str) -> Path:
    return (root or GDHY_ROOT) / crop


def available_years(
    root: Path | None = None, crop: str = GDHY_CROP
) -> list[int]:
    """The years GDHY publishes for ``crop``, ascending.

    Args:
        root: GDHY root holding one directory per crop.
        crop: Crop sub-directory, e.g. ``wheat_winter``.

    Returns:
        Sorted file years — the **file** year, before
        :data:`utils.config.GDHY_HARVEST_YEAR_OFFSET` is applied.

    Raises:
        FileNotFoundError: If the crop directory does not exist.
    """
    directory = _crop_dir(root, crop)
    if not directory.is_dir():
        raise FileNotFoundError(
            f"GDHY crop directory not found: {directory}. Set GDHY_ROOT or "
            "pick another crop."
        )
    years = sorted(
        int(path.stem.split("_")[-1]) for path in directory.glob("yield_*.nc4")
    )
    logger.info("GDHY %s: %d years, %d-%d", crop, len(years), years[0], years[-1])
    return years


def load_gdhy_year(
    year: int,
    root: Path | None = None,
    crop: str = GDHY_CROP,
    bbox: tuple[float, float, float, float] | None = EUROPE_BBOX,
) -> pd.DataFrame:
    """One GDHY year as a tidy frame of valid cells.

    Args:
        year: **File** year.
        root: GDHY root.
        crop: Crop sub-directory.
        bbox: Domain to crop to; ``None`` keeps the globe.

    Returns:
        Columns ``lon``, ``lat`` (cell centres on -180..180) and ``obs_yield``
        [t/ha], one row per cell where the crop is grown. Cells outside the
        crop mask are NaN in the file and are dropped here.

    Raises:
        FileNotFoundError: If the year's file is missing.
    """
    path = _crop_dir(root, crop) / GDHY_YIELD_TEMPLATE.format(year=year)
    if not path.is_file():
        raise FileNotFoundError(f"GDHY file not found: {path}")

    with xr.open_dataset(path) as dataset:
        field = dataset["var"]
        # 0-360 to -180..180. Sorting afterwards is not needed for a tidy
        # frame, but it keeps the array monotonic for anyone who slices it.
        field = field.assign_coords(
            lon=((field["lon"] + 180.0) % 360.0) - 180.0
        ).sortby("lon")
        frame = field.to_dataframe(name="obs_yield").reset_index()

    frame = frame.dropna(subset=["obs_yield"])
    if bbox is not None:
        frame = crop_to_bbox(frame, bbox)
    # Snap so the coordinates are bit-identical to the run's binned centres and
    # to the SAGE calendar's, whatever rounding the source file carries.
    frame["lon"], frame["lat"] = snap_to_grid(frame["lon"], frame["lat"])
    return frame[["lon", "lat", "obs_yield"]].reset_index(drop=True)


def load_gdhy(
    years: tuple[int, int] | None = None,
    root: Path | None = None,
    crop: str = GDHY_CROP,
    bbox: tuple[float, float, float, float] | None = EUROPE_BBOX,
    harvest_year_offset: int = GDHY_HARVEST_YEAR_OFFSET,
    resolution: float = GRID_RES_DEG,
) -> pd.DataFrame:
    """Read a range of GDHY years into one frame keyed by grid cell and year.

    Args:
        years: Inclusive ``(first, last)`` **harvest** years to keep, i.e. the
            label the run uses. ``None`` reads everything published.
        root: GDHY root.
        crop: Crop sub-directory.
        bbox: Domain to crop to.
        harvest_year_offset: Added to the file year to get the harvest year.
            See :data:`utils.config.GDHY_HARVEST_YEAR_OFFSET`.
        resolution: Grid spacing, for the ``grid_id``.

    Returns:
        Columns ``grid_id``, ``lon``, ``lat``, ``year`` (harvest year) and
        ``obs_yield`` [t/ha]. Cells the crop is not grown in are absent rather
        than NaN, so the row count is the true sample size.

    Raises:
        ValueError: If no year survives the filter.
    """
    published = available_years(root, crop)
    wanted = [
        file_year for file_year in published
        if years is None
        or years[0] <= file_year + harvest_year_offset <= years[1]
    ]
    if not wanted:
        raise ValueError(
            f"no GDHY year in {years} (published {published[0]}-{published[-1]}, "
            f"offset {harvest_year_offset:+d})"
        )

    frames = []
    for file_year in wanted:
        block = load_gdhy_year(file_year, root, crop, bbox)
        block["year"] = np.int16(file_year + harvest_year_offset)
        frames.append(block)

    out = pd.concat(frames, ignore_index=True)
    out["grid_id"] = grid_cell_id(out["lon"], out["lat"], resolution)
    logger.info(
        "GDHY %s: %d cell-years, %d cells, harvest years %d-%d",
        crop, len(out), out["grid_id"].nunique(),
        int(out["year"].min()), int(out["year"].max()),
    )
    return out[["grid_id", "lon", "lat", "year", "obs_yield"]]
