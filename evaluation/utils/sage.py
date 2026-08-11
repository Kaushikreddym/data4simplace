"""Loader for the SAGE (Sacks et al.) global crop calendar.

``Wheat.Winter.crop.calendar.fill.nc`` holds one climatological field per
quantity on a 0.5° grid: a planting date, a harvest date, the start and end of
each, and the days between them. There is **no year dimension** — the calendar
is a single "typical" season assembled from agricultural-census and
extension-service reports collected around 1990-2000.

Three properties shape what can be asked of it:

* **It is a climatology, so nothing here tests interannual skill.** Every
  metric computed against it is spatial: does the run place the *average*
  season correctly across Europe. A model that tracks a warm year perfectly and
  one that ignores weather entirely score the same.
* **Two thirds of the European cells are interpolated, not reported.** The
  ``fill`` file extrapolates from the nearest reporting unit wherever none
  covers the cell; ``filled`` marks those cells, and every table here can be
  split on it. The fill also flattens real gradients — a filled cell carries
  its donor's dates exactly.
* **A date comes with a window.** ``plant.start``/``plant.end`` span 10 to 100
  days in Europe. Whether the simulated date falls inside that window is the
  more meaningful test; :func:`utils.doy.doy_in_window` computes it.
"""

from __future__ import annotations

import logging
from pathlib import Path

import numpy as np
import pandas as pd
import xarray as xr

from .config import EUROPE_BBOX, GRID_RES_DEG, SAGE_CROP, SAGE_ROOT, SAGE_TEMPLATE
from .grid import crop_to_bbox, grid_cell_id, snap_to_grid

logger = logging.getLogger(__name__)

__all__ = ["SAGE_COLUMNS", "calendar_path", "load_sage_calendar"]

#: NetCDF variable -> output column. The dots in the NetCDF names are awkward
#: in a DataFrame (``frame.plant.start`` is an attribute lookup), so they
#: become underscores.
SAGE_COLUMNS: dict[str, str] = {
    "plant": "plant",
    "plant.start": "plant_start",
    "plant.end": "plant_end",
    "harvest": "harvest",
    "harvest.start": "harvest_start",
    "harvest.end": "harvest_end",
}

#: Timedelta variables, converted to whole days.
_DURATION_COLUMNS: dict[str, str] = {
    "plant.range": "plant_range_days",
    "harvest.range": "harvest_range_days",
    "tot.days": "season_length_days",
}


def calendar_path(
    root: Path | None = None, crop: str = SAGE_CROP
) -> Path:
    """Path of one crop's calendar file.

    Args:
        root: Directory holding the calendars.
        crop: SAGE crop name as it appears in the filename, e.g.
            ``Wheat.Winter``.

    Returns:
        The ``.nc`` path.

    Raises:
        FileNotFoundError: If it is absent. The directory ships most crops
            gzipped and only the ones in use unpacked, so the message says so.
    """
    path = (root or SAGE_ROOT) / SAGE_TEMPLATE.format(crop=crop)
    if not path.is_file():
        gzipped = path.with_suffix(".nc.gz")
        hint = (
            f" ({gzipped.name} exists — gunzip it first)" if gzipped.is_file() else ""
        )
        raise FileNotFoundError(f"SAGE calendar not found: {path}{hint}")
    return path


def load_sage_calendar(
    root: Path | None = None,
    crop: str = SAGE_CROP,
    bbox: tuple[float, float, float, float] | None = EUROPE_BBOX,
    resolution: float = GRID_RES_DEG,
) -> pd.DataFrame:
    """The crop calendar as a tidy frame of valid cells.

    Args:
        root: Directory holding the calendars.
        crop: SAGE crop name, e.g. ``Wheat.Winter``.
        bbox: Domain to crop to; ``None`` keeps the globe.
        resolution: Grid spacing, for the ``grid_id``.

    Returns:
        One row per cell where the crop is grown, with ``grid_id``, ``lon``,
        ``lat``, the six day-of-year columns of :data:`SAGE_COLUMNS`,
        ``plant_range_days``, ``harvest_range_days``, ``season_length_days``
        and ``filled`` — ``True`` where the cell was extrapolated from another
        reporting unit rather than reported.

    Raises:
        FileNotFoundError: If the calendar file is absent.
    """
    path = calendar_path(root, crop)

    with xr.open_dataset(path) as dataset:
        dataset = dataset.rename({"longitude": "lon", "latitude": "lat"})
        parts = {
            out: dataset[name].to_series() for name, out in SAGE_COLUMNS.items()
        }
        for name, out in _DURATION_COLUMNS.items():
            parts[out] = dataset[name].to_series() / np.timedelta64(1, "D")
        # 'index' identifies the reporting unit a cell's dates come from; it is
        # NaN exactly where the cell was filled from a neighbour, which is the
        # only place the quality flag is recorded.
        parts["filled"] = dataset["index"].to_series().isna()
        frame = pd.DataFrame(parts).reset_index()

    frame = frame.dropna(subset=["plant", "harvest"])
    if bbox is not None:
        frame = crop_to_bbox(frame, bbox)

    # The file's coordinates carry a ~3e-6 degree offset (89.75000254 rather
    # than 89.75), which is nothing physically but enough to make the
    # coordinates differ from the run's snapped centres. Snap both sides to the
    # same values so a frame carrying either can be mapped or merged.
    frame["lon"], frame["lat"] = snap_to_grid(frame["lon"], frame["lat"], resolution)
    frame["grid_id"] = grid_cell_id(frame["lon"], frame["lat"], resolution)
    frame["filled"] = frame["filled"].astype(bool)

    logger.info(
        "SAGE %s: %d cells in the domain, %d (%.0f%%) filled from a neighbour",
        crop, len(frame), int(frame["filled"].sum()),
        100.0 * float(frame["filled"].mean()) if len(frame) else 0.0,
    )
    ordered = [
        "grid_id", "lon", "lat", *SAGE_COLUMNS.values(),
        *_DURATION_COLUMNS.values(), "filled",
    ]
    return frame[ordered].reset_index(drop=True)
