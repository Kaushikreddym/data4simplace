"""Write the SIMPLACE project CSV — one line per simulated cell.

The project file *is* the parallelism axis: ``simplace run -l=START-END`` runs a
range of its lines, so the line count sizes the array job and a line range is
one task's work. It is also the only place the cell's identity, its
coordinates and its weather filename are joined.

The Brandenburg project resource declares:

    projectid, simulationid, vColumn, vRow, vLocationID, vlat, vlon,
    vNUTSID, vSTATE_ID, vSTATE_NAME

Every one of those comes from the export except the three administrative
fields, which no export carries. They are written as an explicit ``NA`` rather
than a plausible-looking code: the solution echoes ``vNUTSID`` and
``vSTATE_ID`` straight into its daily output, so a fabricated value would
travel into the results looking like a real region.
"""

from __future__ import annotations

import logging
from pathlib import Path

import numpy as np
import pandas as pd

from cropmodelling4eu.config import GridConfig
from cropmodelling4eu.export.cells import id_to_lonlat, id_to_rowcol
from cropmodelling4eu.simplace.solution import SolutionDocument

logger = logging.getLogger(__name__)

__all__ = [
    "ADMIN_PLACEHOLDER",
    "build_location_frame",
    "build_project_frame",
    "line_ranges",
    "project_columns",
    "write_project_csv",
]

#: Written into administrative columns the export cannot fill.
ADMIN_PLACEHOLDER = "NA"

#: How a project column is filled. Anything not here and not administrative is
#: reported rather than guessed.
_BUILDERS = {
    "projectid": "id",
    "simulationid": "one",
    "vColumn": "col",
    "vRow": "row",
    "vLocationID": "id",
    "vlat": "lat",
    "vlon": "lon",
    "location": "id",
    "vLocationId": "id",
    "Latitude": "lat",
    "Longitude": "lon",
    "vLatitude": "lat",
    "vLongitude": "lon",
    "vWGS84_lat": "lat",
    "vWGS84_lon": "lon",
    "vAltitude": "altitude",
    # A fixed sowing date, for a solution whose phenology latches on one.
    "vSowingDOY": "sowing_doy",
    "vIDPL": "sowing_doy",
    # A *rule-based* solution instead takes a planting window and sows on the
    # first day inside it that its weather rules accept. That is a better use
    # of the SAGE calendar than its midpoint, since SAGE publishes the window.
    "vSowWindowStartDOY": "sowing_window_start",
    "vSowWindowLengthDays": "sowing_window_days",
    "vIRR": "irrigated",
    "vIRRIGATION": "irrigated",
}

#: Columns known to be administrative labels the export cannot supply.
_ADMIN_COLUMNS = {"vNUTSID", "vSTATE_ID", "vSTATE_NAME", "vRegion", "vCountry"}


def build_project_frame(
    ids: np.ndarray,
    grid: GridConfig,
    columns: list[str],
    altitude: np.ndarray | None = None,
    sowing_doy: np.ndarray | None = None,
    irrigated: pd.Series | None = None,
    sowing_window: tuple[np.ndarray, np.ndarray] | None = None,
    constants: dict[str, object] | None = None,
) -> pd.DataFrame:
    """Build the project table for ``ids`` in the order ``columns`` declares.

    Parameters
    ----------
    ids:
        The cells to simulate, one line each.
    grid:
        The export's grid — what turns a ``SimplaceID`` into the row and column
        the weather filename is built from.
    columns:
        The project resource's declared column ids, in order.
    altitude, sowing_doy, irrigated, sowing_window:
        Per-cell site attributes, for solutions whose project file carries
        them. ``None`` leaves the column to the administrative placeholder,
        with a warning naming it.
    constants:
        Column name -> value written verbatim, for the scenario keys a
        solution needs but the export has no opinion about (``vCrop``,
        ``vTrt``, ``vClimPerCO2_ID`` ...). Explicit, because a wrong scenario
        key selects the wrong parameter set rather than failing.

    Returns
    -------
    pandas.DataFrame
        One row per id, columns exactly as declared.
    """
    ids = np.asarray(ids, dtype=np.int64)
    row, col = id_to_rowcol(ids, grid)
    lon, lat = id_to_lonlat(ids, grid)
    window_start, window_days = sowing_window or (None, None)

    sources: dict[str, object] = {
        "id": ids,
        "one": 1,
        "row": row,
        "col": col,
        "lat": np.round(lat, 8),
        "lon": np.round(lon, 8),
        "altitude": altitude,
        "sowing_doy": sowing_doy,
        "sowing_window_start": window_start,
        "sowing_window_days": window_days,
        "irrigated": (
            irrigated.reindex(ids).fillna(0).astype(int).to_numpy()
            if irrigated is not None
            else None
        ),
    }
    constants = constants or {}

    data: dict[str, object] = {}
    unfilled: list[str] = []
    for column in columns:
        if column in constants:
            data[column] = constants[column]
            continue
        key = _BUILDERS.get(column)
        value = sources.get(key) if key else None
        if value is None:
            data[column] = ADMIN_PLACEHOLDER
            if column not in _ADMIN_COLUMNS:
                unfilled.append(column)
        else:
            data[column] = value

    # The reverse of `unfilled`, and the more dangerous half: a site attribute
    # the export computed that no declared column consumes is dropped without
    # a trace, and the solution silently keeps its own hard-coded default --
    # which is how a per-cell SAGE calendar turns into one continental sowing
    # date. Only the solution's project header can consume it, so this is a
    # warning about the template, not about the export.
    # A fixed date and a planting window are two ways to say the same thing, so
    # only a header carrying *neither* leaves the solution sowing on its own
    # default. A rule-based project declares the window and no date, which is
    # correct and must stay quiet.
    consumed = {_BUILDERS.get(column) for column in columns} | set(constants)
    sowing_keys = ("sowing_doy", "sowing_window_start")
    if any(sources.get(key) is not None for key in sowing_keys) and not (
        consumed & set(sowing_keys)
    ):
        logger.warning(
            "The project resource declares no sowing column (vSowingDOY / "
            "vIDPL for a fixed date, vSowWindowStartDOY for a rule-based "
            "window), so the export's per-cell calendar is not passed to "
            "SIMPLACE and the solution's own default sows every cell on the "
            "same day. Declared columns: %s", columns,
        )

    if unfilled:
        logger.warning(
            "Project column(s) %s are not administrative but the export has "
            "nothing to fill them with; written as %r. Check whether the "
            "solution reads them.", unfilled, ADMIN_PLACEHOLDER,
        )
    return pd.DataFrame(data, columns=columns)


def write_project_csv(
    frame: pd.DataFrame, path: str | Path, divider: str = ","
) -> Path:
    """Write the project CSV and log its line count (the array bound)."""
    out = Path(path)
    out.parent.mkdir(parents=True, exist_ok=True)
    frame.to_csv(out, index=False, sep=divider)
    logger.info(
        "Wrote %s: %d lines (simplace run -l=1-%d covers them all)",
        out, len(frame), len(frame),
    )
    return out


def project_columns(document: SolutionDocument, resource_id: str = "proj") -> list[str]:
    """The declared column ids of a project XML's resource.

    Raises
    ------
    KeyError
        If the document declares no resources, which means it is not a project
        file or its header is empty.
    """
    for resource in document.resources:
        if resource.id == resource_id or len(document.resources) == 1:
            return list(resource.columns)
    if not document.resources:
        raise KeyError(f"{document.path.name} declares no <resource> header")
    raise KeyError(
        f"{document.path.name} has no resource {resource_id!r}; it has "
        f"{[r.id for r in document.resources]}"
    )


def build_location_frame(
    ids: np.ndarray,
    grid: GridConfig,
    altitude: np.ndarray | None = None,
    sun_inclination: float = -4.0,
) -> pd.DataFrame:
    """The ``location.csv`` the Brandenburg solution reads daily.

    ``location, Latitude, SunInclination, Altitude`` — per-cell latitude drives
    day length and the radiation geometry, so this **must** be generated rather
    than copied from the template: a copied file would give every European cell
    Brandenburg's 52.6 °N, which changes the season everywhere else.

    ``sun_inclination`` is the solar elevation defining day length; the
    reference project uses -4°, a civil-twilight convention, for every location.
    """
    ids = np.asarray(ids, dtype=np.int64)
    _, lat = id_to_lonlat(ids, grid)
    return pd.DataFrame(
        {
            "location": ids,
            "Latitude": np.round(lat, 8),
            "SunInclination": sun_inclination,
            "Altitude": (
                np.round(altitude, 1) if altitude is not None else 0.0
            ),
        }
    )


def line_ranges(n_lines: int, lines_per_task: int) -> list[tuple[int, int]]:
    """Split ``n_lines`` into inclusive, 1-based ``(start, end)`` task ranges.

    Contiguous, not round-robin — unlike the torchcrop shards. SIMPLACE selects
    lines by range, so a round-robin split is not expressible; the cost is that
    tasks covering Scandinavia and Iberia have different wall times, which the
    array's throttle absorbs.
    """
    if n_lines < 1:
        return []
    if lines_per_task < 1:
        raise ValueError("lines_per_task must be >= 1")
    return [
        (start + 1, min(start + lines_per_task, n_lines))
        for start in range(0, n_lines, lines_per_task)
    ]
