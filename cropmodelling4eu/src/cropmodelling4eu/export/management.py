"""Read the export's fertilizer schedule and the composition it refers to.

The schedule is a long table of applications keyed to a development stage
(``DVS``) rather than a date, because the date is weather-dependent and the
export does not know the weather. Turning a DVS into a day is the runner's job
(see :func:`cropmodelling4eu.torchcrop.run.fertilizer_from_dvs`); turning a row
into nutrients is this module's.

Two layouts are supported, matching what data4simplace writes:

**Wide (Brandenburg).** ``location, FertilizerScenario, crop, Event, vType,
DVS, Amount`` plus an appended ``vIRR``. ``Amount`` is grams of *product*, so
it is multiplied by the carrier's elemental content from
``fertilizer_composition.xml`` — 1 g of KAS is 0.27 g N.

**Long (EU SUSTAg).** ``Location, ENZ, vCrop, Year, vIRRIGATION, Number, DVS,
Amount``. There is no ``vType``: ``Amount`` is already the nutrient, so no
composition file is involved and none is required.

Which one a file is is decided by whether it has a ``vType`` column, so a
composition file is demanded only when one is actually needed.
"""

from __future__ import annotations

import logging
import xml.etree.ElementTree as ET
from pathlib import Path

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

__all__ = [
    "NUTRIENTS",
    "irrigation_flags",
    "load_composition",
    "read_fertilizer_plans",
]

#: Order of the nutrient axis in a plan array.
NUTRIENTS = ("N", "P", "K")

#: Candidate spellings of the columns that vary between layouts.
_KEY_COLUMNS = ("location", "Location")
_IRRIGATION_COLUMNS = ("vIRR", "vIRRIGATION", "Irrigation")
_EVENT_COLUMNS = ("Event", "Number")


def load_composition(path: str | Path) -> pd.DataFrame:
    """Fertilizer type -> elemental N, P, K content [g element / g product].

    Parses ``fertilizer_composition.xml``, the SIMPLACE file declaring what
    each carrier contains. Only the mineral fractions are read: organic N is
    not what a mineral dressing delivers on the day it is applied.

    Raises
    ------
    FileNotFoundError
        If ``path`` does not exist.
    ValueError
        If the file declares no fertilizer.
    """
    path = Path(path)
    if not path.is_file():
        raise FileNotFoundError(f"Fertilizer composition file not found: {path}")

    root = ET.parse(path).getroot()
    rows: dict[str, dict[str, float]] = {}
    for fertilizer in root.findall("fertilizer"):
        params = {
            p.get("id"): (p.text or "").strip() for p in fertilizer.findall("parameter")
        }
        name = params.get("Fertilizertype")
        if not name:
            continue
        rows[name] = {
            "N": float(params.get("NitrateAndAmmonium", 0.0) or 0.0),
            "P": float(params.get("Phosphorus", 0.0) or 0.0),
            "K": float(params.get("Potassium", 0.0) or 0.0),
        }
    if not rows:
        raise ValueError(f"{path} declares no <fertilizer> entries")
    logger.info("Composition %s: %d carriers (%s)", path.name, len(rows), ", ".join(sorted(rows)))
    return pd.DataFrame(rows).T


def read_fertilizer_plans(
    path: str | Path, composition_file: str | Path | None = None
) -> dict[int, np.ndarray]:
    """Per-cell event arrays ``[n_events, 4]`` of ``(DVS, N, P, K)`` in g/m².

    Sorted by DVS within each cell, so a runner can place the events against a
    monotone development-stage trajectory with one ``searchsorted``.

    Parameters
    ----------
    path:
        The schedule CSV, in either layout.
    composition_file:
        ``fertilizer_composition.xml``. Required only for a wide schedule,
        whose amounts are products rather than nutrients.

    Raises
    ------
    ValueError
        If a wide schedule needs a composition file and none was given, or if
        it uses a carrier the composition file does not declare.
    """
    path = Path(path)
    frame = pd.read_csv(path, sep=None, engine="python")
    key = _first_present(frame.columns, _KEY_COLUMNS)
    if key is None:
        raise ValueError(
            f"{path} has no location column (looked for {_KEY_COLUMNS}); "
            f"it has {list(frame.columns)[:8]}"
        )

    if "vType" in frame.columns:
        events = _nutrients_from_products(frame, path, composition_file)
    else:
        # Long layout: the amount is the nutrient already. Which nutrient a row
        # carries is not recorded, so the schedule cannot be split by element;
        # it is reported as N, which is what the LINTUL N balance consumes.
        logger.info(
            "%s has no vType column: reading it as the long layout, where "
            "Amount is the nutrient in g/m2", path.name,
        )
        events = frame.assign(
            N=frame["Amount"].astype(float), P=0.0, K=0.0
        )

    event_column = _first_present(frame.columns, _EVENT_COLUMNS)
    sort_by = [key, "DVS"] if event_column is None else [key, "DVS", event_column]
    events = events.sort_values(sort_by, kind="stable")

    columns = ["DVS", *NUTRIENTS]
    plans = {
        _as_id(location): group[columns].to_numpy(dtype="float64")
        for location, group in events.groupby(key, sort=False)
    }
    logger.info(
        "Fertilizer plans %s: %d cells, %d rows, %.1f events per cell",
        path.name, len(plans), len(events), len(events) / max(len(plans), 1),
    )
    return plans


def _nutrients_from_products(
    frame: pd.DataFrame, path: Path, composition_file: str | Path | None
) -> pd.DataFrame:
    """Multiply each product amount by its carrier's elemental contents."""
    if composition_file is None:
        raise ValueError(
            f"{path.name} carries product amounts (it has a vType column), so "
            f"it needs fertilizer_composition.xml to convert them to "
            f"nutrients. Set paths.composition_file."
        )
    composition = load_composition(composition_file)
    joined = frame.join(composition, on="vType")

    unknown = sorted(set(joined.loc[joined["N"].isna(), "vType"].unique()))
    if unknown:
        raise ValueError(
            f"{path.name} uses carrier(s) {unknown} that "
            f"{Path(composition_file).name} does not declare "
            f"(known: {sorted(composition.index)})"
        )
    for element in NUTRIENTS:
        joined[element] = joined["Amount"].astype(float) * joined[element].astype(float)
    return joined


def irrigation_flags(path: str | Path) -> pd.Series:
    """Per-cell irrigated (1) / rainfed (0) flag from the schedule.

    The flag is a per-location attribute repeated across a cell's event rows,
    so the maximum over a cell recovers it whichever layout wrote it.
    Cells absent from the schedule are absent here too; a caller filling them
    should fill with 0, since an unclassified cell is written as rainfed.
    """
    frame = pd.read_csv(path, sep=None, engine="python")
    key = _first_present(frame.columns, _KEY_COLUMNS)
    column = _first_present(frame.columns, _IRRIGATION_COLUMNS)
    if key is None or column is None:
        logger.info(
            "%s carries no irrigation column (looked for %s); every cell will "
            "run rainfed", Path(path).name, _IRRIGATION_COLUMNS,
        )
        return pd.Series(dtype="int64")
    flags = frame.groupby(key)[column].max().astype("int64")
    flags.index = flags.index.map(_as_id)
    logger.info(
        "Irrigation: %d of %d scheduled cells irrigated", int(flags.sum()), len(flags)
    )
    return flags


def _first_present(columns, candidates: tuple[str, ...]) -> str | None:
    """The first of ``candidates`` present in ``columns``."""
    present = {str(c) for c in columns}
    return next((c for c in candidates if c in present), None)


def _as_id(value) -> int:
    """A location value as an integer ``SimplaceID``."""
    return int(value)
