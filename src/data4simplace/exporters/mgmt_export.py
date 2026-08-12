"""SIMPLACE management / fertilizer schedule exporter.

The reference schedule (``fertilizer_<crop>.csv``) is a long table — one row per
location, event and fertilizer type — carrying *product* amounts in g/m^2 at a
development stage (``DVS``) rather than nutrient rates::

    location,FertilizerScenario,crop,Event,vType,DVS,Amount
    49612,2,winter_wheat,1,PK,0.001,40
    49612,2,winter_wheat,2,KAS,0.25,32

Every location in that file carries the same flat scenario. This exporter keeps
the reference's *structure* — its event count, DVS timing and the relative split
of N across the top dressings — and replaces the flat amounts with each cell's
own NPKGRIDS rates. Two conversions are involved:

1. **Oxide to element.** NPKGRIDS reports P and K as P2O5 / K2O, while
   ``fertilizer_composition.xml`` declares elemental contents.
2. **Nutrient to product.** ``Amount`` is grams of *product*, so a nutrient
   demand is divided by the carrier's content — 1 g N is 3.70 g of KAS (27 %
   mineral N) but 2.17 g of Urea (46 %).

The reference's compound ``PK`` carrier cannot honour two independent rates: its
P:K ratio is fixed at 0.0792:0.083 g/g, so scaling it to a cell's P2O5 rate
fixes K at whatever that ratio delivers. The compound event is therefore split
into the two straight carriers ``npk.p_fertilizer`` / ``npk.k_fertilizer``
(``P`` = P2O5, ``K`` = K2O in the reference composition file), applied at the
same DVS, so both nutrients land exactly on their NPKGRIDS rate.

Cells NPKGRIDS has no rate for are dropped rather than filled with the reference
constants, so no exported location carries a fabricated application.

When ``flags.run_irrigation_classification`` is set the schedule also carries a
``vIRR`` column (name configurable via ``irrigation.column``): 1 where the
irrigated share of the crop's harvested area exceeds ``irrigation.threshold``,
0 for rainfed *and* for cells with too little of the crop to classify. It is a
per-location attribute, so every event row of a cell repeats it. See
:mod:`data4simplace.management.irrigation`.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd
import xarray as xr

from data4simplace.config import PipelineConfig
from data4simplace.exporters.base_exporter import BaseExporter, ReferenceSpec, parse_reference_csv
from data4simplace.exporters.layout import (
    MANAGEMENT_DIALECTS,
    LongDialect,
    select_dialect,
)
from data4simplace.management.irrigation import IrrigationClassification
from data4simplace.npk.composition import (
    K2O_TO_K,
    P2O5_TO_P,
    FertilizerComposition,
    default_composition_path,
    parse_fertilizer_composition,
)
from data4simplace.site.window import SowingWindow

logger = logging.getLogger(__name__)

#: Schedule columns, in the reference's order. Used only as the offline
#: fallback; a real run takes the order from the reference file.
_SCHEDULE_COLUMNS = [
    "location",
    "FertilizerScenario",
    "crop",
    "Event",
    "vType",
    "DVS",
    "Amount",
]
#: Column holding the cell identifier in the reference schedule.
_LOCATION_COLUMN = "location"

#: kg/ha -> g/m^2. 1 kg/ha = 1000 g / 10,000 m^2.
_KG_HA_TO_G_M2 = 0.1

#: Offline fallback: the Brandenburg winter-wheat scenario and the contents of
#: the three carriers it needs, so a run without the SIMPLACE reference files
#: still produces a structurally valid schedule.
_FALLBACK_EVENTS: list[tuple[str, float, float]] = [
    # (vType, DVS, reference Amount)
    ("PK", 0.001, 40.0),
    ("KAS", 0.25, 32.0),
    ("KAS", 0.4, 16.0),
    ("KAS", 0.9, 16.0),
]
_FALLBACK_COMPOSITIONS: dict[str, FertilizerComposition] = {
    "PK": FertilizerComposition(name="PK", phosphorus=0.0792, potassium=0.083),
    "KAS": FertilizerComposition(
        name="KAS", mineral_n=0.27, nitrate=0.135, ammonium=0.135
    ),
    "P": FertilizerComposition(name="P", phosphorus=P2O5_TO_P),
    "K": FertilizerComposition(name="K", potassium=K2O_TO_K),
}


@dataclass(frozen=True)
class ScheduleEvent:
    """One application in the per-cell schedule.

    Attributes
    ----------
    vtype:
        The ``vType`` written to the CSV.
    dvs:
        Development stage the application is triggered at.
    nutrient:
        Which cell rate drives the amount: ``"N"``, ``"P"`` or ``"K"``.
    share:
        Fraction of the cell's rate for that nutrient applied here. The N
        dressings split the rate; the single P and K events always take 1.
    content:
        The carrier's content of the nutrient, g element / g product.
    """

    vtype: str
    dvs: float
    nutrient: str
    share: float
    content: float

    def amounts(self, nutrient_g_m2: np.ndarray) -> np.ndarray:
        """Product amounts (g/m^2) delivering ``share`` of an elemental rate."""
        return nutrient_g_m2 * self.share / self.content


@dataclass(frozen=True)
class ScheduleTemplate:
    """The per-cell schedule shape recovered from the reference file.

    Attributes
    ----------
    events:
        Applications in write order (ascending ``DVS``, reference order within
        a stage).
    scenario:
        The ``FertilizerScenario`` column value.
    """

    events: list[ScheduleEvent]
    scenario: int

    @property
    def nutrients(self) -> set[str]:
        """The nutrient rates this schedule consumes."""
        return {event.nutrient for event in self.events}


class ManagementExporter(BaseExporter):
    """Export a SIMPLACE fertilizer schedule from gridded NPK rates."""

    kind = "management"

    def __init__(self, config: PipelineConfig, reference_path: str | Path | None) -> None:
        super().__init__(config, reference_path)
        self._npk_config = config.npk
        self._template: Optional[ScheduleTemplate] = None

    def fallback_spec(self) -> ReferenceSpec:
        """Documented default matching the SIMPLACE fertilizer schedule layout."""
        return ReferenceSpec(
            delimiter=",",
            columns=list(_SCHEDULE_COLUMNS),
            missing_value=str(self._config.missing_value),
        )

    # ------------------------------------------------------------------ #
    # Fertilizer compositions
    # ------------------------------------------------------------------ #
    def _compositions(self) -> dict[str, FertilizerComposition]:
        """Load ``fertilizer_composition.xml``, or the offline fallback."""
        path = self._npk_config.composition_file or default_composition_path(
            self._reference_path
        )
        if path is None:
            logger.warning(
                "No fertilizer_composition.xml found (npk.composition_file unset "
                "and none next to the management reference); using the built-in "
                "contents for %s", ", ".join(sorted(_FALLBACK_COMPOSITIONS))
            )
            return dict(_FALLBACK_COMPOSITIONS)
        compositions = parse_fertilizer_composition(path)
        # The straight P/K carriers must exist even when the reference schedule
        # itself only uses the compound; fall back to the pure oxides.
        for name in (self._npk_config.p_fertilizer, self._npk_config.k_fertilizer):
            if name not in compositions and name in _FALLBACK_COMPOSITIONS:
                compositions[name] = _FALLBACK_COMPOSITIONS[name]
        return compositions

    # ------------------------------------------------------------------ #
    # Reference schedule -> template
    # ------------------------------------------------------------------ #
    def _reference_events(self) -> list[tuple[str, float, float]]:
        """``(vType, DVS, Amount)`` of one location's reference schedule.

        Every location in the reference carries the same flat scenario, so the
        first location's rows define the pattern for all cells.
        """
        ref_file = self._resolve_reference_file()
        if ref_file is None:
            logger.warning(
                "No management reference; using the built-in winter-wheat "
                "scenario (%d events)", len(_FALLBACK_EVENTS)
            )
            return list(_FALLBACK_EVENTS)

        spec = parse_reference_csv(ref_file, default_missing=str(self._config.missing_value))
        frame = pd.read_csv(ref_file, sep=spec.delimiter)
        for column in ("vType", "DVS", "Amount"):
            if column not in frame.columns:
                raise ValueError(
                    f"Management reference {ref_file} has no {column!r} column; "
                    f"found {list(frame.columns)}"
                )
        if _LOCATION_COLUMN in frame.columns:
            first = frame[_LOCATION_COLUMN].iloc[0]
            frame = frame[frame[_LOCATION_COLUMN] == first]
        if "Event" in frame.columns:
            frame = frame.sort_values("Event", kind="stable")

        return [
            (str(row.vType), float(row.DVS), float(row.Amount))
            for row in frame.itertuples(index=False)
        ]

    def _reference_scenario(self) -> int:
        """The ``FertilizerScenario`` value: config override, else reference."""
        if self._npk_config.fertilizer_scenario is not None:
            return int(self._npk_config.fertilizer_scenario)
        ref_file = self._resolve_reference_file()
        if ref_file is not None:
            spec = parse_reference_csv(
                ref_file, default_missing=str(self._config.missing_value)
            )
            head = pd.read_csv(ref_file, sep=spec.delimiter, nrows=1)
            if "FertilizerScenario" in head.columns:
                return int(head["FertilizerScenario"].iloc[0])
        return 2

    def template(self) -> ScheduleTemplate:
        """Build (and cache) the per-cell schedule template.

        Raises
        ------
        ValueError
            If a reference ``vType`` is absent from the composition file, if the
            configured N/P/K carriers are unknown or carry no such nutrient, or
            if ``npk.n_split`` does not match the reference's N-event count.
        """
        if self._template is not None:
            return self._template

        compositions = self._compositions()
        reference = self._reference_events()
        unknown = {vtype for vtype, _, _ in reference if vtype not in compositions}
        if unknown:
            raise ValueError(
                f"Fertilizer type(s) {sorted(unknown)} used by the management "
                f"reference are not declared in the composition file "
                f"(known: {sorted(compositions)})"
            )

        events: list[ScheduleEvent] = []
        events.extend(self._nutrient_events(reference, compositions))
        events.extend(self._n_events(reference, compositions))
        # Ascending DVS, reference order within a stage (a stable sort).
        events.sort(key=lambda event: event.dvs)

        if not events:
            raise ValueError(
                "The management reference yields no applications; check that "
                "its vType values carry N, P or K in the composition file"
            )

        self._template = ScheduleTemplate(events=events, scenario=self._reference_scenario())
        logger.info(
            "Fertilizer template: %d events (%s)",
            len(events),
            ", ".join(f"{e.vtype}@DVS{e.dvs:g}" for e in events),
        )
        return self._template

    def _nutrient_events(
        self,
        reference: list[tuple[str, float, float]],
        compositions: dict[str, FertilizerComposition],
    ) -> list[ScheduleEvent]:
        """The single straight-P and straight-K applications.

        Their timing is the first reference event that supplies the nutrient
        (whether as a compound or a straight carrier); with none, the earliest
        stage in the schedule.
        """
        earliest = min((dvs for _, dvs, _ in reference), default=0.001)
        built: list[ScheduleEvent] = []

        for nutrient, carrier_name, carries in (
            ("P", self._npk_config.p_fertilizer, lambda c: c.carries_p),
            ("K", self._npk_config.k_fertilizer, lambda c: c.carries_k),
        ):
            dvs = next(
                (d for vtype, d, _ in reference if carries(compositions[vtype])), None
            )
            if dvs is None:
                logger.info(
                    "Management reference applies no %s; scheduling the NPKGRIDS "
                    "%s rate at the schedule's first stage (DVS %g)",
                    nutrient, nutrient, earliest,
                )
                dvs = earliest

            carrier = compositions.get(carrier_name)
            if carrier is None:
                raise ValueError(
                    f"npk.{nutrient.lower()}_fertilizer {carrier_name!r} is not "
                    f"declared in the composition file (known: {sorted(compositions)})"
                )
            content = carrier.phosphorus if nutrient == "P" else carrier.potassium
            if content <= 0.0:
                raise ValueError(
                    f"npk.{nutrient.lower()}_fertilizer {carrier_name!r} carries "
                    f"no {nutrient} in the composition file"
                )
            built.append(
                ScheduleEvent(
                    vtype=carrier_name,
                    dvs=dvs,
                    nutrient=nutrient,
                    share=1.0,
                    content=content,
                )
            )
        return built

    def _n_events(
        self,
        reference: list[tuple[str, float, float]],
        compositions: dict[str, FertilizerComposition],
    ) -> list[ScheduleEvent]:
        """The N dressings, keeping the reference's timing and relative split.

        The split weights are the reference *nutrient* deliveries
        (``Amount x mineral N content``), not the raw amounts, so a schedule
        that mixes carriers of different strengths still splits the cell's N
        rate the way the reference does.
        """
        n_rows = [
            (vtype, dvs, amount)
            for vtype, dvs, amount in reference
            if compositions[vtype].mineral_n > 0.0
        ]
        if not n_rows:
            logger.warning(
                "Management reference applies no mineral N; the NPKGRIDS N rate "
                "will not be written"
            )
            return []

        carrier_name = self._npk_config.n_fertilizer
        carrier = compositions.get(carrier_name)
        if carrier is None:
            raise ValueError(
                f"npk.n_fertilizer {carrier_name!r} is not declared in the "
                f"composition file (known: {sorted(compositions)})"
            )
        if carrier.mineral_n <= 0.0:
            raise ValueError(
                f"npk.n_fertilizer {carrier_name!r} carries no mineral N in the "
                f"composition file"
            )

        configured = self._npk_config.n_split
        if configured is not None:
            if len(configured) != len(n_rows):
                raise ValueError(
                    f"npk.n_split has {len(configured)} entries but the "
                    f"management reference has {len(n_rows)} N events"
                )
            weights = np.asarray(configured, dtype="float64")
        else:
            weights = np.array(
                [amount * compositions[vtype].mineral_n for vtype, _, amount in n_rows],
                dtype="float64",
            )
        total = float(weights.sum())
        if total <= 0.0:
            raise ValueError("The management reference's N events deliver no N")
        shares = weights / total

        return [
            ScheduleEvent(
                vtype=carrier_name,
                dvs=dvs,
                nutrient="N",
                share=float(share),
                content=carrier.mineral_n,
            )
            for (_, dvs, _), share in zip(n_rows, shares)
        ]

    # ------------------------------------------------------------------ #
    # Frame construction
    # ------------------------------------------------------------------ #
    def _cell_rates(self, npk: xr.Dataset, cell_table: pd.DataFrame) -> dict[str, np.ndarray]:
        """Elemental nutrient rates (g element / m^2) for every table row.

        The NPK dataset is already on the target grid, so the cell table's
        ``row``/``col`` index it directly — one vectorised gather instead of a
        per-cell ``sel``, which matters at Europe's ~10^5 cells.
        """
        rows = cell_table["row"].to_numpy()
        cols = cell_table["col"].to_numpy()

        # Source variable -> (canonical nutrient, oxide->element factor).
        sources = {
            "N": ("N", 1.0),
            "P2O5": ("P", P2O5_TO_P),
            "K2O": ("K", K2O_TO_K),
            # A generic-raster NPK source already reports elemental P and K.
            "P": ("P", 1.0),
            "K": ("K", 1.0),
        }
        rates: dict[str, np.ndarray] = {}
        for variable, (nutrient, factor) in sources.items():
            if variable not in npk.data_vars or nutrient in rates:
                continue
            grid = np.asarray(npk[variable].values, dtype="float64")
            values = grid[rows, cols]
            # A negative rate is not physical; treat it as absent rather than
            # writing a negative application.
            values = np.where(values < 0.0, np.nan, values)
            rates[nutrient] = values * factor * _KG_HA_TO_G_M2
        return rates

    def _window_columns(self) -> tuple[str, str]:
        """The two planting-window column names, from the site config."""
        site = self._config.site
        return site.window_start_column, site.window_end_column

    def _extensions(
        self,
        irrigation: Optional[IrrigationClassification],
        window: Optional[SowingWindow],
    ) -> list[str]:
        """Columns this exporter adds after the reference's own, in order.

        SIMPLACE binds a CSV resource's columns by **position**, so this order
        is part of the file's contract with the solution: appending the window
        before ``vIRR`` would feed a day-of-year into the irrigation flag.
        """
        columns = []
        if irrigation is not None:
            columns.append(self._config.irrigation.column)
        if window is not None:
            columns.extend(self._window_columns())
        return columns

    def _columns(
        self,
        irrigation: Optional[IrrigationClassification],
        window: Optional[SowingWindow] = None,
    ) -> list[str]:
        """Output columns: the reference schedule, plus the extensions."""
        return list(_SCHEDULE_COLUMNS) + self._extensions(irrigation, window)

    def conform(self, frame: pd.DataFrame) -> pd.DataFrame:
        """Reference column order, with the extension columns kept on the end.

        The base class drops every column the reference does not declare, which
        is what the reference-driven schema demands. ``vIRR`` and the planting
        window are deliberate extensions rather than drift, so they are
        re-appended *after* the reference block — leaving the reference's own
        columns in their exact order.
        """
        site = self._config.site
        extensions = [
            column
            for column in (
                self._config.irrigation.column,
                site.window_start_column,
                site.window_end_column,
            )
            if column in frame.columns
        ]
        if not extensions:
            return super().conform(frame)
        conformed = super().conform(frame.drop(columns=extensions))
        for column in extensions:
            conformed[column] = frame[column].to_numpy()
        return conformed

    def build_frame(
        self,
        npk: xr.Dataset,
        cell_table: pd.DataFrame,
        irrigation: Optional[IrrigationClassification] = None,
        window: Optional[SowingWindow] = None,
    ) -> pd.DataFrame:
        """Build the long fertilizer schedule, one row per cell and event.

        Parameters
        ----------
        npk:
            The regridded NPK rates (``N`` / ``P2O5`` / ``K2O`` in kg/ha).
        cell_table:
            The exported cells, carrying ``SimplaceID``, ``row`` and ``col``.
        irrigation:
            Optional irrigated/rainfed classification. When given, its per-cell
            label is repeated onto every event row of that cell as the
            ``irrigation.column`` column.
        window:
            Optional planting window from the calendar. When given, its
            ``(start, end)`` day-of-year pair is repeated onto every event row
            of that cell, for a solution that sows by rule inside the window
            rather than on a fixed day.

        Returns
        -------
        pandas.DataFrame
            Columns ``location``, ``FertilizerScenario``, ``crop``, ``Event``,
            ``vType``, ``DVS``, ``Amount`` and, with a classification, ``vIRR``,
            then the window pair. Empty when no cell has a rate.
        """
        template = self.template()
        columns = self._columns(irrigation, window)
        if not npk.data_vars or cell_table.empty:
            logger.warning("No NPK layers or no exported cells; empty schedule")
            return pd.DataFrame(columns=columns)

        rates = self._cell_rates(npk, cell_table)
        missing = template.nutrients - set(rates)
        if missing:
            logger.warning(
                "NPK dataset carries no %s rate; those events are dropped from "
                "the schedule", ", ".join(sorted(missing))
            )
        events = [event for event in template.events if event.nutrient in rates]
        if not events:
            logger.warning("None of the schedule's nutrients are present in the NPK data")
            return pd.DataFrame(columns=columns)

        n_cells = len(cell_table)
        # (n_cells, n_events) so a ravel lines up with the repeated ids below.
        amounts = np.column_stack([event.amounts(rates[event.nutrient]) for event in events])

        frame = pd.DataFrame(
            {
                _LOCATION_COLUMN: np.repeat(
                    cell_table["SimplaceID"].to_numpy(dtype="int64"), len(events)
                ),
                "FertilizerScenario": template.scenario,
                "crop": self._npk_config.simplace_crop,
                "vType": np.tile([event.vtype for event in events], n_cells),
                "DVS": np.tile([event.dvs for event in events], n_cells),
                "Amount": amounts.ravel(),
            }
        )
        if irrigation is not None:
            # A per-location attribute, so it repeats across the cell's events.
            frame[self._config.irrigation.column] = np.repeat(
                irrigation.column(cell_table), len(events)
            )
        if window is not None:
            start_column, end_column = self._window_columns()
            start, end = window.columns(cell_table)
            frame[start_column] = np.repeat(start, len(events))
            frame[end_column] = np.repeat(end, len(events))

        # A cell with no NPKGRIDS rate is dropped, not filled with the reference
        # constants: an exported location must carry its own application.
        frame = frame[np.isfinite(frame["Amount"].to_numpy())]
        if frame.empty:
            logger.warning("No exported cell has an NPKGRIDS rate; empty schedule")
            return pd.DataFrame(columns=columns)

        frame["Amount"] = frame["Amount"].round(self._npk_config.amount_decimals)
        # Renumber per location so a cell missing one nutrient still has a
        # gap-free 1..n event sequence. The rows are already in per-cell blocks
        # ordered by DVS, so a plain cumulative count is the event index.
        frame["Event"] = frame.groupby(_LOCATION_COLUMN, sort=False).cumcount() + 1

        kept = int(frame[_LOCATION_COLUMN].nunique())
        logger.info(
            "Fertilizer schedule: %d of %d exported cells have NPKGRIDS rates "
            "(%d rows)", kept, n_cells, len(frame)
        )
        if irrigation is not None:
            column = self._config.irrigation.column
            irrigated = int(frame.groupby(_LOCATION_COLUMN, sort=False)[column].first().sum())
            logger.info(
                "%s: %d of %d scheduled cells irrigated (%s, share > %g)",
                column, irrigated, kept, irrigation.source, irrigation.threshold,
            )
        if window is not None:
            start_column, end_column = self._window_columns()
            per_cell = frame.groupby(_LOCATION_COLUMN, sort=False)[
                [start_column, end_column]
            ].first()
            logger.info(
                "Planting window: DOY %d-%d to %d-%d over %d scheduled cells "
                "(median length %d d), written as %s/%s",
                int(per_cell[start_column].min()), int(per_cell[end_column].min()),
                int(per_cell[start_column].max()), int(per_cell[end_column].max()),
                kept,
                int((per_cell[end_column] - per_cell[start_column]).median()),
                start_column, end_column,
            )
        return frame[columns]

    def export(
        self,
        npk: xr.Dataset,
        cell_table: pd.DataFrame,
        output_dir: str | Path,
        irrigation: Optional[IrrigationClassification] = None,
        window: Optional[SowingWindow] = None,
    ) -> Path:
        """Write ``fertilizer_<crop>.csv``; return its path."""
        frame = self.build_frame(npk, cell_table, irrigation=irrigation, window=window)
        out_path = (
            Path(output_dir) / "management" / f"fertilizer_{self._npk_config.simplace_crop}.csv"
        )
        return self.write_csv(frame, out_path)


class LongManagementExporter(ManagementExporter):
    """Export the fertilizer schedule in the EU SUSTAg long layout.

    The Brandenburg schedule is already one row per event, so what the long
    dialect changes is the *keying* rather than the shape:

    ==================  ==========================  =========================
    \\                   Brandenburg (wide default)  SUSTAg long
    ==================  ==========================  =========================
    irrigation          ``vIRR``, appended          ``vIRRIGATION``, a key
    fertilizer type     ``vType`` + composition     **absent**
    year                one schedule for all        ``Year``, one block each
    grouping            ``location, Event``         ``Location, ENZ, vCrop,
                                                     Year, Number``
    ==================  ==========================  =========================

    The missing ``vType`` is the trap. With no carrier column there is no
    carrier content to divide by, so ``Amount`` is the **nutrient** itself;
    writing product grams into a nutrient field is a silent factor-of-3.7 error
    for KAS. ``npk.long_amount_basis`` therefore has to say which, and
    ``product`` is refused on a dialect with no type column.
    """

    kind = "management (long)"

    def __init__(self, config: PipelineConfig, reference_path: str | Path | None) -> None:
        super().__init__(config, reference_path)
        self._long_reference = config.reference.management_file_long
        self._dialect: Optional[LongDialect] = None

    def fallback_spec(self) -> ReferenceSpec:
        """The selected dialect's own columns."""
        return ReferenceSpec(
            delimiter=self.dialect.delimiter,
            columns=self.dialect.column_names,
            missing_value=str(self._config.missing_value),
        )

    @property
    def spec(self) -> ReferenceSpec:
        """The long file's structure, from its own reference or the dialect.

        Deliberately *not* the base class' spec: ``self._reference_path`` still
        points at the wide schedule, which is what the schedule template is
        recovered from, and conforming to that file's columns would undo the
        whole layout.
        """
        if self._spec is None:
            self._spec = self.fallback_spec()
        return self._spec

    @property
    def dialect(self) -> LongDialect:
        """The long dialect this run writes."""
        if self._dialect is None:
            columns = (
                parse_reference_csv(
                    self._long_reference, default_missing=str(self._config.missing_value)
                ).columns
                if self._long_reference is not None
                and Path(self._long_reference).is_file()
                else []
            )
            self._dialect = select_dialect(
                columns, MANAGEMENT_DIALECTS, kind="management"
            )
        return self._dialect

    def conform(self, frame: pd.DataFrame) -> pd.DataFrame:
        """Order by the dialect. ``vIRR`` is a key column here, not an extra."""
        return BaseExporter.conform(self, frame)

    def build_frame(
        self,
        npk: xr.Dataset,
        cell_table: pd.DataFrame,
        irrigation: Optional[IrrigationClassification] = None,
        window: Optional[SowingWindow] = None,
    ) -> pd.DataFrame:
        """Build the long schedule, one row per cell, year and event.

        The planting window is **not** written: no long dialect declares a
        column for it, and the dialect decides this file's schema. Dropping it
        under a guessed name is the mis-map the layout module exists to prevent,
        so a long-driven run takes its window from the project file instead.
        """
        if window is not None:
            logger.warning(
                "The %r dialect declares no planting-window column, so the "
                "window is not written into the long schedule; a rule-based "
                "solution reading this file must take its window from the "
                "project file", self.dialect.name,
            )
        wide = super().build_frame(npk, cell_table, irrigation=irrigation)
        if wide.empty:
            return pd.DataFrame(columns=self.dialect.column_names)

        events = self._to_nutrient_basis(wide)
        events = self._expand_years(events)
        events["ENZ"] = self._environmental_zone(events)
        column = self._config.irrigation.column
        if column not in events:
            # The dialect declares an irrigation key, so a run without the
            # classification writes the rainfed value rather than a blank key.
            events[column] = 0
        events = events.rename(columns={column: "vIRR"}) if column != "vIRR" else events

        frame = self.dialect.build(events).reset_index(drop=True)
        logger.info(
            "Long fertilizer schedule: %d rows over %d locations and %d year(s) "
            "(dialect %r, %s basis)",
            len(frame), events["location"].nunique(), events["Year"].nunique(),
            self.dialect.name, self._npk_config.long_amount_basis,
        )
        return frame

    def _to_nutrient_basis(self, frame: pd.DataFrame) -> pd.DataFrame:
        """Convert ``Amount`` from product to nutrient grams, if asked to.

        Raises
        ------
        ValueError
            If ``product`` is configured on a dialect with no fertilizer-type
            column. The amounts would be product grams in a field the solution
            reads as a nutrient, which no downstream check would catch.
        """
        basis = self._npk_config.long_amount_basis
        has_type_column = any(
            c.name.lower() in ("vtype", "type", "fertilizertype")
            for c in self.dialect.columns
        )
        if basis == "product":
            if not has_type_column:
                raise ValueError(
                    f"npk.long_amount_basis: product needs a fertilizer-type "
                    f"column, and the {self.dialect.name!r} dialect has none "
                    f"({', '.join(self.dialect.column_names)}). Without a "
                    f"carrier the amount cannot be interpreted as a product; "
                    f"use long_amount_basis: nutrient."
                )
            return frame

        # Nutrient basis: undo the carrier division the wide schedule applies.
        template = {(e.vtype, e.dvs): e.content for e in self.template().events}
        contents = np.array(
            [
                template.get((str(vtype), float(dvs)), np.nan)
                for vtype, dvs in zip(frame["vType"], frame["DVS"])
            ]
        )
        out = frame.copy()
        out["Amount"] = (out["Amount"].to_numpy() * contents).round(
            self._npk_config.amount_decimals
        )
        return out

    def _expand_years(self, frame: pd.DataFrame) -> pd.DataFrame:
        """Repeat the schedule once per simulated year.

        The dialect keys on ``Year``, and the pipeline derives one schedule from
        a static NPKGRIDS rate, so every year gets the same applications. That
        is a property of the input, not of this exporter: NPKGRIDS publishes no
        time axis.
        """
        start = int(str(self._config.time.start)[:4])
        end = int(str(self._config.time.end)[:4])
        years = range(start, end + 1)
        expanded = pd.concat(
            [frame.assign(Year=year) for year in years], ignore_index=True
        )
        return expanded.sort_values(
            ["location", "Year", "Event"], kind="stable"
        ).reset_index(drop=True)

    def _environmental_zone(self, frame: pd.DataFrame) -> np.ndarray:
        """The ``ENZ`` key.

        The EnS v8 environmental zones are not a pipeline input, so this writes
        the missing sentinel rather than a plausible-looking zero: a wrong zone
        would select the wrong calibration in a solution that keys on it.
        """
        return np.full(len(frame), self._config.missing_value)

    def export(
        self,
        npk: xr.Dataset,
        cell_table: pd.DataFrame,
        output_dir: str | Path,
        irrigation: Optional[IrrigationClassification] = None,
        window: Optional[SowingWindow] = None,
    ) -> Path:
        """Write ``fertilizer_<crop>_long.csv``; return its path."""
        frame = self.build_frame(npk, cell_table, irrigation=irrigation, window=window)
        out_path = (
            Path(output_dir)
            / "management"
            / f"fertilizer_{self._npk_config.simplace_crop}_long.csv"
        )
        return self.write_csv(frame, out_path)
