"""Export layouts: the wide Brandenburg tables and the long row-per-depth ones.

SIMPLACE reads a soil profile in one of two shapes, and which one a solution
expects is declared in its own ``<resource>`` header:

**Wide** — one row per ``location``, per-layer properties encoded as
``<stem>_<N>`` columns (``clay_1`` ... ``clay_6``). This is the Brandenburg
reference the pipeline was built against, and it stays the default.

**Long** — one row per ``(location, depth)``, every property declared
``datatype="DOUBLEARRAY"`` with a ``key`` column::

    <res id="location"     datatype="CHAR"        key="vLocationId"/>
    <res id="depth"        datatype="DOUBLEARRAY"/>
    <res id="soilwater_wp" datatype="DOUBLEARRAY" unit="cm3*cm-3"/>
    <res id="soilwater_fc_global" datatype="DOUBLE" unit="cm3*cm-3"/>

SIMPLACE collapses the rows sharing a key into per-depth arrays itself, so the
long form is a first-class idiom rather than a variant spelling — and because
the depth axis is data rather than column names, a long file can carry
SoilGrids' **native** horizons with no depth remap at all.

Two long dialects exist in the SIMPLACE projects on this system and they
disagree on both names and units, so a layout is *reference-driven*: point
``reference.soil_file_long`` at the file whose structure you want and the
matching dialect is selected from its columns.

Nothing here guesses. A canonical property with no entry in the selected
dialect is dropped with a warning naming it, never written under an approximate
name: a silent mis-map between ``soilwater_fc`` and ``soilwater_red`` produces a
plausible file and a wrong crop.
"""

from __future__ import annotations

import logging
import re
from dataclasses import dataclass, field
from typing import Callable, Literal, Optional

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

__all__ = [
    "DERIVED_COLUMNS",
    "LongColumn",
    "LongDialect",
    "MANAGEMENT_DIALECTS",
    "SOIL_DIALECTS",
    "detect_layout",
    "select_dialect",
]

Layout = Literal["wide", "long"]

#: Matches a wide per-layer column ``<stem>_<N>``.
_LAYER_SUFFIX_RE = re.compile(r"^.+_(\d+)$")

#: Column names that mark a table's depth axis as rows rather than suffixes.
_DEPTH_COLUMN_NAMES = ("depth", "Depth", "DEPTH", "depth_m", "layer_depth")


# --------------------------------------------------------------------------- #
# Derived columns
# --------------------------------------------------------------------------- #
#
# A dialect column is normally one canonical property times a unit factor. These
# few need more than one input, so they are named functions over the tidy
# profile table rather than factors.


def _cn_ratio(frame: pd.DataFrame) -> np.ndarray:
    """Organic carbon to total nitrogen ratio [g/g], both from SoilGrids."""
    carbon = frame.get("carbon")
    nitrogen = frame.get("nitrogen")
    if carbon is None or nitrogen is None:
        return np.full(len(frame), np.nan)
    with np.errstate(divide="ignore", invalid="ignore"):
        return np.where(
            nitrogen.to_numpy() > 0,
            carbon.to_numpy() / nitrogen.to_numpy(),
            np.nan,
        )


def _stock_to_concentration(stock_kg_ha: np.ndarray, frame: pd.DataFrame) -> np.ndarray:
    """kg N/ha in a layer -> mg N/kg soil.

    A hectare of a layer ``t`` cm thick holds ``10^4 * t`` dm3 of soil, so its
    mass is ``10^4 * t * bulkdensity`` kg and::

        mg/kg = kg/ha * 10^6 / (10^4 * t * bd) = kg/ha * 100 / (t * bd)

    This inverts the stock the wide export writes, so a long and a wide file
    from the same run describe the same nitrogen.
    """
    thickness = (frame["depth_bottom_cm"] - frame["depth_top_cm"]).to_numpy()
    bulk = (
        frame["bulkdensity"].to_numpy()
        if "bulkdensity" in frame
        else np.full(len(frame), np.nan)
    )
    with np.errstate(divide="ignore", invalid="ignore"):
        return np.where(
            (thickness > 0) & (bulk > 0), stock_kg_ha * 100.0 / (thickness * bulk), np.nan
        )


def _ammonium_mg_kg(frame: pd.DataFrame) -> np.ndarray:
    """Initial ammonium as a concentration rather than a stock."""
    if "ammonium" not in frame:
        return np.full(len(frame), np.nan)
    return _stock_to_concentration(frame["ammonium"].to_numpy(), frame)


def _nitrate_mg_kg(frame: pd.DataFrame) -> np.ndarray:
    """Initial nitrate as a concentration rather than a stock."""
    if "nitrate" not in frame:
        return np.full(len(frame), np.nan)
    return _stock_to_concentration(frame["nitrate"].to_numpy(), frame)


def _depth_m(frame: pd.DataFrame) -> np.ndarray:
    """Layer **bottom** depth in metres, which is what both dialects mean by depth."""
    return frame["depth_bottom_cm"].to_numpy() / 100.0


def _thickness_m(frame: pd.DataFrame) -> np.ndarray:
    """Layer thickness in metres."""
    return (frame["depth_bottom_cm"] - frame["depth_top_cm"]).to_numpy() / 100.0


#: Name -> function over the tidy profile table. Referenced by
#: :attr:`LongColumn.derive` so a dialect stays a declarative table.
DERIVED_COLUMNS: dict[str, Callable[[pd.DataFrame], np.ndarray]] = {
    "cn_ratio": _cn_ratio,
    "ammonium_mg_kg": _ammonium_mg_kg,
    "nitrate_mg_kg": _nitrate_mg_kg,
    "depth_m": _depth_m,
    "thickness_m": _thickness_m,
}

#: Canonical properties each derived column consumes. Without this a dialect
#: that carries ``NH4_mg_kg`` would still be reported as having "no column for
#: ammonium", which is exactly the false alarm that teaches people to ignore
#: the warning that matters.
DERIVED_INPUTS: dict[str, set[str]] = {
    "cn_ratio": {"carbon", "nitrogen"},
    "ammonium_mg_kg": {"ammonium", "bulkdensity"},
    "nitrate_mg_kg": {"nitrate", "bulkdensity"},
    "depth_m": set(),
    "thickness_m": set(),
}


# --------------------------------------------------------------------------- #
# Dialects
# --------------------------------------------------------------------------- #


@dataclass(frozen=True)
class LongColumn:
    """One column of a long-format file.

    Attributes
    ----------
    name:
        Column name as the dialect spells it.
    source:
        Canonical property it is taken from, in the pipeline's own units. Unset
        for derived and constant columns.
    factor:
        Multiplied onto ``source`` on the way out — the unit conversion. For
        example SUSTAg's ``OC`` is g/100g where the pipeline's ``carbon`` is
        g/kg, so the factor is 0.1.
    derive:
        Key into :data:`DERIVED_COLUMNS` for a column needing more than one
        input (a C:N ratio, a stock-to-concentration conversion).
    constant:
        Written verbatim into every row. Used for the profile-level identifiers
        a solution keys on.
    """

    name: str
    source: Optional[str] = None
    factor: float = 1.0
    derive: Optional[str] = None
    constant: Optional[object] = None

    def values(self, frame: pd.DataFrame) -> np.ndarray | object:
        """This column's values for a tidy profile table."""
        if self.constant is not None:
            return self.constant
        if self.derive is not None:
            return DERIVED_COLUMNS[self.derive](frame)
        if self.source is None or self.source not in frame:
            return np.full(len(frame), np.nan)
        values = frame[self.source].to_numpy()
        # Only numeric columns carry a unit factor; a label column (the crop
        # name, the location id) always has factor 1.0, and multiplying it would
        # be a type error rather than a conversion.
        return values * self.factor if self.factor != 1.0 else values


@dataclass(frozen=True)
class LongDialect:
    """A long-format file's column names, units and key structure.

    Attributes
    ----------
    name:
        Identifier used in logs and config.
    delimiter:
        Field separator, matching the solution's ``<divider>``.
    key_column:
        The column SIMPLACE groups rows by (``key=`` in the resource header).
    depth_column:
        The column carrying the depth axis, or ``None`` for a table whose rows
        are events rather than layers (the fertilizer schedule).
    columns:
        Every column, in write order.
    """

    name: str
    delimiter: str
    key_column: str
    depth_column: Optional[str]
    columns: list[LongColumn] = field(default_factory=list)

    @property
    def column_names(self) -> list[str]:
        return [column.name for column in self.columns]

    def sources(self) -> set[str]:
        """Canonical properties this dialect carries, directly or derived."""
        direct = {c.source for c in self.columns if c.source is not None}
        derived: set[str] = set()
        for column in self.columns:
            if column.derive is not None:
                derived |= DERIVED_INPUTS.get(column.derive, set())
        return direct | derived

    def build(self, frame: pd.DataFrame) -> pd.DataFrame:
        """Render a tidy profile/event table into this dialect's columns."""
        return pd.DataFrame(
            {column.name: column.values(frame) for column in self.columns},
            index=frame.index,
        )[self.column_names]


#: EU SUSTAg (`Rotation_monocrop/data/soil/SLIM_soil_EU_SUSTAg.csv`), read by
#: `EU_SUSTAg_macsurClimateRO_v2.sol.xml`. Comma-separated, keyed on
#: ``location``, one row per depth.
SUSTAG_SOIL = LongDialect(
    name="sustag",
    delimiter=",",
    key_column="location",
    depth_column="Depth",
    columns=[
        LongColumn("location", source="location"),
        LongColumn("Depth", derive="depth_m"),
        # LL/DUL/SAT are the lower limit, drained upper limit and saturation --
        # the same three water contents the wide schema calls
        # soilwater_wp/_fc/_sat.
        LongColumn("LL", source="soilwater_wp"),
        LongColumn("soilwater_red", source="soilwater_red"),
        LongColumn("DUL", source="soilwater_fc"),
        LongColumn("SAT", source="soilwater_sat"),
        LongColumn("BD", source="bulkdensity"),
        LongColumn("OC", source="carbon", factor=0.1),        # g/kg -> g/100g
        LongColumn("CN", derive="cn_ratio"),
        LongColumn("ph", source="PH"),
        LongColumn("sand", source="sand"),
        LongColumn("silt", source="silt"),
        LongColumn("clay", source="clay"),
        LongColumn("soilwater_res", source="soilwater_res"),
        LongColumn("soilwater_init", source="soilwater_init"),
        LongColumn("NH4_mg_kg", derive="ammonium_mg_kg"),
        LongColumn("NO3_mg_kg", derive="nitrate_mg_kg"),
        LongColumn("soilwater_fc_global", source="soilwater_fc_global"),
        LongColumn("soilwater_sat_global", source="soilwater_sat_global"),
    ],
)

#: ERA5 potential-yields project
#: (`ERA5_SIMPLACE/data/soil/20250616_CKASoilDataCefitAP5_Slim.csv`).
#: Semicolon-separated and keyed on ``soiltype`` rather than a cell id.
ERA5_SOIL = LongDialect(
    name="era5",
    delimiter=";",
    key_column="soiltype",
    depth_column="depth",
    columns=[
        LongColumn("soiltype", source="location"),
        LongColumn("depth", derive="depth_m"),
        LongColumn("soilwater_fc", source="soilwater_fc"),
        LongColumn("soilwater_wp", source="soilwater_wp"),
        LongColumn("soilwater_sat", source="soilwater_sat"),
        LongColumn("soilwater_res", source="soilwater_res"),
        LongColumn("soilwater_init", source="soilwater_init"),
        LongColumn("bulkdensity_perc", source="bulkdensity"),
        LongColumn("clay_perc", source="clay"),
        LongColumn("sand_perc", source="sand"),
        LongColumn("carbon_perc", source="carbon", factor=0.1),  # g/kg -> g/100g
        LongColumn("soilwater_fc_global", source="soilwater_fc_global"),
        LongColumn("soilwater_sat_global", source="soilwater_sat_global"),
    ],
)

#: EU SUSTAg **as the v2 solution declares it**
#: (`EU_SUSTAg_macsurClimateRO_initialize_v2.sol.xml`, resource ``soil``).
#:
#: Its ``<res id=...>`` names differ from the SUSTAg CSV that ships beside it —
#: the solution says ``soilwater_wp``/``soilwater_fc``/``bulkdensity``/
#: ``carbon`` where the file says ``LL``/``DUL``/``BD``/``OC``. Writing what the
#: **solution** declares removes the question of how SIMPLACE reconciles the
#: two, which is the one thing about that pair nobody can answer from the
#: files alone.
#:
#: ``ammonium``/``nitrate`` are declared ``mg/kg``, so they are the
#: concentration form rather than the wide layout's kg/ha stock.
SUSTAG_V2_SOIL = LongDialect(
    name="sustag_v2",
    delimiter=",",
    key_column="location",
    depth_column="depth",
    columns=[
        LongColumn("location", source="location"),
        LongColumn("depth", derive="depth_m"),
        LongColumn("soilwater_wp", source="soilwater_wp"),
        LongColumn("soilwater_red", source="soilwater_red"),
        LongColumn("soilwater_fc", source="soilwater_fc"),
        LongColumn("soilwater_sat", source="soilwater_sat"),
        LongColumn("bulkdensity", source="bulkdensity"),
        LongColumn("carbon", source="carbon", factor=0.1),   # g/kg -> g/100g
        LongColumn("CN", derive="cn_ratio"),
        LongColumn("pH", source="PH"),
        LongColumn("sand", source="sand"),
        LongColumn("silt", source="silt"),
        LongColumn("clay", source="clay"),
        LongColumn("soilwater_res", source="soilwater_res"),
        LongColumn("soilwater_init", source="soilwater_init"),
        LongColumn("ammonium", derive="ammonium_mg_kg"),
        LongColumn("nitrate", derive="nitrate_mg_kg"),
        LongColumn("soilwater_fc_global", source="soilwater_fc_global"),
        LongColumn("soilwater_sat_global", source="soilwater_sat_global"),
    ],
)

#: Soil dialects by name.
SOIL_DIALECTS: dict[str, LongDialect] = {
    SUSTAG_SOIL.name: SUSTAG_SOIL,
    SUSTAG_V2_SOIL.name: SUSTAG_V2_SOIL,
    ERA5_SOIL.name: ERA5_SOIL,
}

#: EU SUSTAg fertilizer schedule
#: (`Rotation_monocrop/data/management/fertilizer.csv`). Unlike the Brandenburg
#: schedule it carries no ``vType``: the amount is the **nutrient**, not a
#: product, so no carrier content divides it. Irrigation is a *key* column here
#: rather than an appended one, and the schedule repeats per year.
SUSTAG_MANAGEMENT = LongDialect(
    name="sustag",
    delimiter=",",
    key_column="Location",
    depth_column=None,
    columns=[
        LongColumn("Location", source="location"),
        LongColumn("ENZ", source="ENZ"),
        LongColumn("vCrop", source="crop"),
        LongColumn("Year", source="Year"),
        LongColumn("vIRRIGATION", source="vIRR"),
        LongColumn("Number", source="Event"),
        LongColumn("DVS", source="DVS"),
        LongColumn("Amount", source="Amount"),
    ],
)

#: Management dialects by name.
MANAGEMENT_DIALECTS: dict[str, LongDialect] = {SUSTAG_MANAGEMENT.name: SUSTAG_MANAGEMENT}


# --------------------------------------------------------------------------- #
# Detection
# --------------------------------------------------------------------------- #


def detect_layout(columns: list[str]) -> Layout:
    """Whether a set of column names describes a wide or a long table.

    Wide tables put the depth axis in the column names (``clay_1`` ...
    ``clay_6``); long tables put it in a ``depth`` column and repeat the key.
    The test is therefore mechanical and needs no configuration to *read* a
    reference file.
    """
    layered = sum(1 for c in columns if _LAYER_SUFFIX_RE.match(str(c)))
    has_depth_column = any(str(c) in _DEPTH_COLUMN_NAMES for c in columns)
    # A long table may still carry a couple of suffixed names (SUSTAg has
    # NH4_mg_kg but no numeric suffixes); requiring several is what separates a
    # genuine per-layer block from an incidental underscore.
    return "long" if has_depth_column and layered < 3 else "wide"


def select_dialect(
    columns: list[str], dialects: dict[str, LongDialect], kind: str = "soil"
) -> LongDialect:
    """The dialect whose columns best match a reference file's.

    Matching is by overlap rather than by an exact set, so a project that has
    added a column to its own copy still resolves. A tie or a total miss falls
    back to the first dialect with a warning naming what was compared, because
    silently choosing between two unit conventions is the one failure mode this
    module exists to prevent.
    """
    present = {str(c) for c in columns}
    scored = {
        name: len(present & set(dialect.column_names))
        for name, dialect in dialects.items()
    }
    best = max(scored, key=lambda name: scored[name])
    if scored[best] == 0:
        fallback = next(iter(dialects))
        logger.warning(
            "No known %s long dialect matches the reference columns %s; "
            "falling back to %r. Column names and units may be wrong -- check "
            "the solution's <resource> header.",
            kind, sorted(present)[:8], fallback,
        )
        return dialects[fallback]
    logger.info(
        "%s long dialect %r matched %d/%d reference columns",
        kind, best, scored[best], len(present),
    )
    return dialects[best]
