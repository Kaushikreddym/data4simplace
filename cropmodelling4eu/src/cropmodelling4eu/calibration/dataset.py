"""The sampling pool, its weather cache, and the sampler that draws from it.

**The sampling unit is ``(observation_unit, harvest_year)``** — an ``adm_id``
-year for CyBench and CLMS, a station-year for PEP725 — not a bare cell. Those
references report on units, so the loss support and the sampling support are
the same object, and a unit is drawn whole: the loss simulates its cells and
area-weights them to the unit mean before comparing.

Two constraints shape everything here.

**A batch may carry one region's parameters and one calendar axis.**
``torchcrop.calibration.paths.rebuild_table`` assembles an ``[N, 2]`` table
from *scalar* ordinates, so a table parameter cannot be ``[B]``-shaped the way
a scalar field can — and every stage calibrates tables. Separately, the
simulation engine derives day-of-year from a single scalar ``start_doy`` for
the whole batch. Both are satisfied by batching within one ``(region, year)``,
which the region-and-year-stratified sampler produces anyway; the region's
anchor sowing day is the axis, and each cell still latches on its own ``idpl``.

**Weather caching is the biggest speed decision.** The production runner reads
each cell's 46-year gzip once; a calibration run revisits the same cells every
epoch. The pool's seasons are pre-extracted once into a memory-mapped
``float32 [n_rows, T, 8]`` array — and the pool must be restricted to
cell-years that carry an observation and the cache built per pool, not per
domain: 8 000 cells x 25 years x 330 d x 8 vars x 4 B is 210 GB.
"""

from __future__ import annotations

import json
import logging
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from pathlib import Path
from typing import ClassVar

import numpy as np
import pandas as pd

from cropmodelling4eu.calibration.config import DataConfig
from cropmodelling4eu.calibration.observations import ObservationSet
from cropmodelling4eu.calibration.regions import CalibrationRegions
from cropmodelling4eu.config import RunConfig
from cropmodelling4eu.export import ExportBundle
from cropmodelling4eu.export.weather import CHANNELS, load_windows, sowing_date

logger = logging.getLogger(__name__)

__all__ = [
    "CellYearDataset",
    "RegionBatch",
    "StratifiedFractionSampler",
    "WeatherCache",
    "build_pool",
    "collate_region_batch",
]

_INDEX_FILE = "pool_index.parquet"
_ARRAY_FILE = "pool_weather.f32"
_META_FILE = "pool_meta.json"


def build_pool(
    observations: ObservationSet,
    regions: CalibrationRegions,
    bundle: ExportBundle,
    settings: DataConfig,
    sowing: pd.DataFrame | None = None,
) -> pd.DataFrame:
    """The ``(unit, year, cell)`` rows a run will actually simulate.

    A unit's cells are subsampled to ``max_cells_per_unit`` by an even stride
    over the ascending id, which is a spatial stride on a row-major grid: the
    unit mean converges long before every cell is run and the cost is linear in
    what is kept.

    Args:
        sowing: ``(SimplaceID, year, sowing_doy)`` from a finished SIMPLACE
            run. **This is the sowing convention the production run is on**,
            and it is per season; the export's site calendar is one fixed date
            per cell. See ``DataConfig.sowing_file``. A (cell, season) the
            table does not cover is dropped rather than falling back, so one
            pool never mixes the two.

    Returns:
        ``unit``, ``unit_kind``, ``year``, ``region_id``, ``SimplaceID``,
        ``sowing_doy``, ``anchor_doy``.

    Raises:
        ValueError: If no observation unit reaches a runnable cell, or if the
            sowing table covers none of the pool.
    """
    of_unit = regions.of_unit()
    cells = regions.table.set_index("SimplaceID")
    runnable = set(bundle.ids.tolist())

    by_unit: dict[str, np.ndarray] = {}
    for unit, block in regions.table.dropna(subset=["adm_id"]).groupby("adm_id"):
        ids = np.sort(block["SimplaceID"].to_numpy(np.int64))
        ids = ids[np.isin(ids, list(runnable))] if len(runnable) else ids
        if ids.size == 0:
            continue
        step = max(1, int(np.ceil(ids.size / settings.max_cells_per_unit)))
        by_unit[str(unit)] = ids[::step][: settings.max_cells_per_unit]

    unit_years = observations.unit_years()
    rows: list[pd.DataFrame] = []
    for record in unit_years.itertuples():
        if record.unit_kind == "cell":
            cell = np.int64(record.unit)
            if cell not in runnable or cell not in cells.index:
                continue
            ids = np.array([cell], dtype=np.int64)
            region = int(cells.at[cell, "region_id"])
        else:
            ids = by_unit.get(str(record.unit), np.empty(0, np.int64))
            if ids.size == 0 or record.unit not in of_unit.index:
                continue
            region = int(of_unit.loc[record.unit])
        rows.append(
            pd.DataFrame(
                {
                    "unit": str(record.unit),
                    "unit_kind": record.unit_kind,
                    "year": int(record.year),
                    "region_id": region,
                    "SimplaceID": ids,
                }
            )
        )

    if not rows:
        raise ValueError(
            "no observation unit reaches a runnable cell — check that the "
            "regions were built on this export's cells"
        )
    pool = pd.concat(rows, ignore_index=True)

    if sowing is None:
        pool["sowing_doy"] = bundle.site.sowing_doy(
            pool["SimplaceID"].to_numpy(np.int64)
        )
        logger.warning(
            "sowing from the export's site calendar: one fixed date per cell "
            "for every season, so simulated emergence carries no interannual "
            "variation — and it is a median 22 d later than the rule-based "
            "date the production run uses. Pass data.sowing_file to calibrate "
            "on the same convention the CALIBRATION.md numbers are on"
        )
    else:
        table = sowing[["SimplaceID", "year", "sowing_doy"]].astype(
            {"SimplaceID": np.int64, "year": np.int64, "sowing_doy": np.int64}
        )
        before = len(pool)
        pool = pool.merge(table, on=["SimplaceID", "year"], how="inner")
        if pool.empty:
            raise ValueError(
                "the sowing table covers none of this pool's cells and seasons"
            )
        logger.info(
            "sowing from a simulated table: %d of %d rows kept, DOY %d-%d, "
            "per-cell interannual sd %.2f d",
            len(pool), before, pool["sowing_doy"].min(), pool["sowing_doy"].max(),
            pool.groupby("SimplaceID")["sowing_doy"].std().mean(),
        )

    # One anchor per region, not per cell: the batch's calendar axis has to be
    # shared, and the earliest sower in the region is the only anchor every
    # cell in it can latch inside. With a simulated table the minimum is over
    # (cell, season), so a region's anchor also covers its earliest *year*.
    pool["anchor_doy"] = pool.groupby("region_id")["sowing_doy"].transform("min")

    # The window has to reach from the anchor to the region's latest sower plus
    # a whole season; if that span eats into `window_days` the late sowers'
    # maturity is simply never reached, and the loss drops it silently.
    span = (pool["sowing_doy"] - pool["anchor_doy"]).max()
    logger.info(
        "widest anchor-to-latest-sower span in any region: %d d, leaving %d d "
        "of the %d-day window for the season itself",
        span, settings.window_days - span, settings.window_days,
    )

    logger.info(
        "pool: %d (unit, year, cell) rows — %d units, %d cell-years, %d regions",
        len(pool), pool["unit"].nunique(),
        pool[["SimplaceID", "year"]].drop_duplicates().shape[0],
        pool["region_id"].nunique(),
    )
    return pool


@dataclass(slots=True)
class WeatherCache:
    """A memory-mapped ``[n_rows, T, 8]`` season block, keyed by (cell, year).

    Attributes:
        index: ``SimplaceID``, ``year``, ``row``, ``start_date``, ``start_doy``.
        array: The memmap itself, in :data:`CHANNELS` order.
        window_days: ``T``.
    """

    index: pd.DataFrame
    array: np.memmap
    window_days: int

    #: What identifies a cached window. The **anchor** is part of the key, not
    #: just the cell and season: a cell can sit in the pool twice under
    #: different regions — once through its CyBench `adm_id` and once as its own
    #: PEP725 station cell — and those regions have different anchor sowing
    #: days. Keyed on ``(cell, year)`` alone, one of the two windows silently
    #: wins and the cell then disagrees with its batch-mates about which day
    #: the window opens, which `collate_region_batch` refuses.
    #: ``ClassVar``, not a field: ``slots=True`` would otherwise make this a
    #: fourth constructor argument rather than a constant.
    KEY: ClassVar[tuple[str, ...]] = ("SimplaceID", "year", "anchor_doy")

    @property
    def lookup(self) -> pd.Series:
        """``(SimplaceID, year, anchor_doy) -> row``."""
        return self.index.set_index(list(self.KEY))["row"]

    def block(self, rows: np.ndarray) -> np.ndarray:
        """``[B, T, 8]`` for a set of cache rows, as a real array."""
        return np.asarray(self.array[rows], dtype=np.float32)

    @classmethod
    def build(
        cls,
        pool: pd.DataFrame,
        config: RunConfig,
        settings: DataConfig,
        cache_dir: Path,
        workers: int = 16,
        overwrite: bool = False,
    ) -> "WeatherCache":
        """Extract every pooled cell-year's window once, to disk.

        The window is sowing - ``pre_days`` to + ``window_days``, which is the
        cheapest of the three memory mitigations and the only free one: the
        production run's 600+ day window costs autograd graph for six months of
        bare soil the loss never reads.
        """
        cache_dir = Path(cache_dir)
        cache_dir.mkdir(parents=True, exist_ok=True)
        index_path = cache_dir / _INDEX_FILE
        array_path = cache_dir / _ARRAY_FILE
        meta_path = cache_dir / _META_FILE

        wanted = (
            pool[list(cls.KEY)]
            .drop_duplicates()
            .sort_values(list(cls.KEY))
            .reset_index(drop=True)
        )
        total = settings.pre_days + settings.window_days

        if not overwrite and index_path.is_file() and meta_path.is_file():
            meta = json.loads(meta_path.read_text())
            index = pd.read_parquet(index_path)
            same = set(map(tuple, index[list(cls.KEY)].to_numpy())) >= set(
                map(tuple, wanted[list(cls.KEY)].to_numpy())
            )
            if same and meta["window_days"] == total:
                logger.info("weather cache reused from %s (%d rows)", cache_dir, len(index))
                return cls(
                    index,
                    np.memmap(array_path, dtype=np.float32, mode="r",
                              shape=(len(index), total, len(CHANNELS))),
                    total,
                )
            logger.info("%s does not cover this pool; rebuilding", cache_dir)

        # Grouped by (cell, anchor), so a cell needed under two anchors is read
        # once and sliced twice rather than one anchor overwriting the other.
        by_cell = {
            (int(cell), int(anchor)): block
            for (cell, anchor), block in wanted.groupby(["SimplaceID", "anchor_doy"])
        }
        export_dir = Path(config.paths.export_dir)

        def window_start(year: int, anchor: int) -> pd.Timestamp:
            """The window's first day, snapped to an exact day-of-year.

            The engine derives day-of-year from one scalar ``start_doy`` and
            wraps it every 365 days, so a batch must share that number exactly.
            Plain date arithmetic does not guarantee it: the same calendar date
            has a different ``dayofyear`` in a leap year, which is enough to
            make two seasons of one region disagree by a day. Snapping the
            start to ``anchor - pre_days`` makes it exact by construction.
            """
            nominal = (int(anchor) - settings.pre_days - 1) % 365 + 1
            start = sowing_date(year, int(anchor)) - pd.Timedelta(days=settings.pre_days)
            return start + pd.Timedelta(days=nominal - start.dayofyear)

        def windows(block: pd.DataFrame) -> dict[int, tuple[pd.Timestamp, pd.Timestamp]]:
            out = {}
            for year, anchor in zip(block["year"], block["anchor_doy"]):
                start = window_start(int(year), int(anchor))
                out[int(year)] = (start, start + pd.Timedelta(days=total - 1))
            return out

        def read(key: tuple[int, int]) -> tuple[tuple[int, int], dict[int, np.ndarray]]:
            cell, _anchor = key
            return key, load_windows(
                export_dir, cell, windows(by_cell[key]), config.grid
            )

        array = np.memmap(
            array_path, dtype=np.float32, mode="w+",
            shape=(len(wanted), total, len(CHANNELS)),
        )
        records: list[dict] = []
        row = 0
        with ThreadPoolExecutor(max_workers=workers) as pool_exec:
            for done, (key, blocks) in enumerate(pool_exec.map(read, by_cell), 1):
                cell, anchor = key
                for year, series in sorted(blocks.items()):
                    array[row] = series
                    records.append(
                        {
                            "SimplaceID": cell, "year": year, "anchor_doy": anchor,
                            "row": row, "start_date": window_start(year, anchor),
                            "start_doy": int(series[0, 0]),
                        }
                    )
                    row += 1
                if done % 500 == 0:
                    logger.info("weather cache: %d/%d cells", done, len(by_cell))

        array.flush()
        index = pd.DataFrame.from_records(records)
        if index.empty:
            raise ValueError(
                "no pooled cell-year has a complete weather window — check "
                "data.pre_days / window_days against the export's record"
            )
        index.to_parquet(index_path, index=False)
        meta_path.write_text(json.dumps({"window_days": total, "channels": list(CHANNELS)}))
        logger.info(
            "weather cache: %d of %d cell-years written to %s (%.1f GB)",
            len(index), len(wanted), array_path,
            len(index) * total * len(CHANNELS) * 4 / 1e9,
        )
        return cls(
            index,
            np.memmap(array_path, dtype=np.float32, mode="r",
                      shape=(len(wanted), total, len(CHANNELS))),
            total,
        )


@dataclass(frozen=True, slots=True)
class RegionBatch:
    """One forward pass: one region, and the ``(unit, year)`` observations in it.

    **One region, several years.** The region is forced — a table ordinate
    cannot be ``[B]``-shaped. The years are not: every window in a region opens
    on the same *day-of-year* (the region's anchor sowing day minus
    ``pre_days``), so seasons from different harvest years share one relative
    time axis and one scalar ``start_doy`` even though their calendar dates
    differ. Allowing them into one batch is what makes the batches full: with
    835 units over 33 regions and 8 seasons, a 20 % draw gives a region-year
    about five units — some 40 cells against a 192 limit — and LINTUL-5's day
    loop costs almost the same at 40 cells as at 192, so year-homogeneous
    batches would waste four fifths of every forward pass.

    ``start_doy`` is still checked, not assumed: a cell whose weather record is
    front-truncated would otherwise be dated against a window it does not have.

    Attributes:
        region_id: The region whose parameter set this batch materialises.
        ids: ``[B]`` ``SimplaceID`` per simulated cell.
        years: ``[B]`` harvest year per simulated cell. CO2 and the observation
            alignment are per season, so this is per cell, not per batch.
        weather: ``[B, T, 8]`` in :data:`CHANNELS` order.
        start_doy: Day-of-year of every window's first day, shared by the batch.
        start_dates: ``[n_units]`` calendar date each unit-year's window opens
            on — what turns an observed date into a day index on the shared
            axis.
        sowing_doy: ``[B]`` each cell's own sowing day-of-year.
        unit_index: ``[B]`` index into ``units`` — which unit-year each cell
            belongs to, i.e. the segment map the aggregate-then-compare loss
            reduces on.
        units: ``[n_units]`` observation unit per segment.
        unit_years: ``[n_units]`` harvest year per segment.
    """

    region_id: int
    ids: np.ndarray
    years: np.ndarray
    weather: np.ndarray
    start_doy: int
    start_dates: np.ndarray
    sowing_doy: np.ndarray
    unit_index: np.ndarray
    units: np.ndarray
    unit_years: np.ndarray

    @property
    def n_units(self) -> int:
        return len(self.units)


@dataclass(slots=True)
class CellYearDataset:
    """The pool plus its weather, addressed by ``(unit, year)``."""

    pool: pd.DataFrame
    weather: WeatherCache

    def __post_init__(self) -> None:
        lookup = self.weather.lookup
        keys = pd.MultiIndex.from_arrays(
            [self.pool[column] for column in self.weather.KEY]
        )
        self.pool = self.pool.assign(row=lookup.reindex(keys).to_numpy())
        dropped = int(self.pool["row"].isna().sum())
        if dropped:
            logger.info("%d pooled rows have no cached weather window", dropped)
        self.pool = self.pool[self.pool["row"].notna()].copy()
        self.pool["row"] = self.pool["row"].astype(int)

    def unit_years(self) -> pd.DataFrame:
        """``(unit, region_id, year, n_cells)`` — one row per drawable unit.

        ``n_cells`` is what the sampler packs batches against: the batch limit
        is in **cell-years**, since that is what a forward pass costs, and a
        unit contributing twelve cells is twelve times the memory of a PEP725
        station contributing one.
        """
        return (
            self.pool.groupby(
                ["unit", "unit_kind", "region_id", "year"], observed=True
            )
            .size()
            .rename("n_cells")
            .reset_index()
        )

    def batch(self, unit_years: pd.DataFrame) -> RegionBatch:
        """Assemble one :class:`RegionBatch` from drawn ``(unit, year)`` rows."""
        return collate_region_batch(self, unit_years)


def collate_region_batch(
    dataset: CellYearDataset, unit_years: pd.DataFrame
) -> RegionBatch:
    """Gather the cells of a drawn set of unit-years into one forward pass.

    Raises:
        ValueError: If the draw spans more than one region, or if its windows
            do not all open on the same day-of-year. Both are checked rather
            than assumed: the first is the table-ordinate constraint and the
            second is what a front-truncated weather record looks like.
    """
    regions = unit_years["region_id"].unique()
    if len(regions) != 1:
        raise ValueError(f"a batch must be one region; got {regions}")

    keys = set(map(tuple, unit_years[["unit", "year"]].to_numpy()))
    rows = dataset.pool[
        [
            (u, y) in keys
            for u, y in zip(dataset.pool["unit"], dataset.pool["year"])
        ]
    ].sort_values(["unit", "year"]).reset_index(drop=True)

    segment = list(zip(rows["unit"], rows["year"]))
    order = list(dict.fromkeys(segment))
    lookup = {key: i for i, key in enumerate(order)}
    unit_index = np.array([lookup[key] for key in segment], dtype=np.int64)

    index = dataset.weather.index.set_index("row")
    cache_rows = rows["row"].to_numpy()
    start_doys = index.loc[cache_rows, "start_doy"].unique()
    if len(start_doys) != 1:
        raise ValueError(
            f"batch cells disagree on the window's first day: {sorted(start_doys)}"
        )

    # One start date per segment, taken from that segment's first cell. Cells of
    # one unit-year share a window by construction; cells of different years do
    # not, which is exactly why this is per unit and not per batch.
    first_row = {key: None for key in order}
    for key, row in zip(segment, cache_rows):
        if first_row[key] is None:
            first_row[key] = row
    start_dates = np.array(
        [index.loc[first_row[key], "start_date"] for key in order],
        dtype="datetime64[ns]",
    )

    return RegionBatch(
        region_id=int(regions[0]),
        ids=rows["SimplaceID"].to_numpy(np.int64),
        years=rows["year"].to_numpy(np.int64),
        weather=dataset.weather.block(cache_rows),
        start_doy=int(start_doys[0]),
        start_dates=start_dates,
        sowing_doy=rows["sowing_doy"].to_numpy(np.int64),
        unit_index=unit_index,
        units=np.array([u for u, _ in order]),
        unit_years=np.array([y for _, y in order], dtype=np.int64),
    )


class StratifiedFractionSampler:
    """Draw a fraction of the pool per epoch, stratified by region and year.

    * **Without replacement within an epoch.** An unstratified draw can
      silently produce a "wet years only" epoch, and the parameters will chase
      it.
    * **Region- and year-homogeneous batches**, which the table-ordinate
      constraint and the shared calendar axis both force (see the module
      docstring).
    * **Deterministic per-epoch seeding**, so a killed run resumes the
      identical sequence.

    ``fraction`` is 0.20 by default. Over ~6 700 CLMS unit-years across 15-30
    regions that is ~45-90 unit-years per region per epoch — one batch at the
    low region count and one to two at the high end, so every region's gradient
    is estimated from a spread of years rather than from a single one.
    """

    def __init__(
        self,
        unit_years: pd.DataFrame,
        fraction: float = 0.20,
        batch_size: int = 192,
        strata: tuple[str, ...] = ("region_id", "year"),
        seed: int = 0,
    ) -> None:
        if not 0.0 < fraction <= 1.0:
            raise ValueError(f"fraction must be in (0, 1]; got {fraction}")
        missing = ({*strata, "n_cells"}) - set(unit_years.columns)
        if missing:
            raise KeyError(f"unit_years lacks {sorted(missing)}")
        self.unit_years = unit_years.reset_index(drop=True)
        self.fraction = fraction
        self.batch_size = batch_size
        self.strata = strata
        self.seed = seed

    def epoch(self, epoch: int) -> list[pd.DataFrame]:
        """The batches of one epoch, as drawn ``(unit, year)`` frames."""
        rng = np.random.default_rng([self.seed, epoch])
        drawn: list[pd.DataFrame] = []
        for _, block in self.unit_years.groupby(list(self.strata), observed=True):
            # At least one unit per stratum: a region-year with three units
            # would otherwise contribute nothing at a 20 % draw, and the
            # stratification would be undone by rounding.
            take = max(1, int(round(self.fraction * len(block))))
            picked = rng.choice(len(block), size=take, replace=False)
            drawn.append(block.iloc[np.sort(picked)])

        # Stratify the *draw* by region and year; pack the *batches* by region
        # only. The year stratification is what stops a "wet years only" epoch;
        # the region homogeneity is what the table ordinates force. Mixing
        # years inside a region costs nothing (they share a day-of-year axis,
        # see `RegionBatch`) and is what keeps a batch full.
        pooled = pd.concat(drawn, ignore_index=True) if drawn else self.unit_years
        batches = [
            chunk
            for _, block in pooled.groupby("region_id", observed=True, sort=True)
            for chunk in self._pack(block.sample(frac=1.0, random_state=epoch))
        ]
        rng.shuffle(batches)
        return batches

    def _pack(self, frame: pd.DataFrame) -> list[pd.DataFrame]:
        """Split one stratum into batches of at most ``batch_size`` cell-years.

        A single unit larger than the limit still goes through whole: the loss
        aggregates its cells before comparing, so splitting it would produce
        two partial means of one observation.
        """
        sizes = frame["n_cells"].to_numpy()
        cuts: list[pd.DataFrame] = []
        start, running = 0, 0
        for i, size in enumerate(sizes):
            if running and running + size > self.batch_size:
                cuts.append(frame.iloc[start:i])
                start, running = i, 0
            running += size
        cuts.append(frame.iloc[start:])
        return [c for c in cuts if len(c)]

    def __len__(self) -> int:
        return max(1, int(round(self.fraction * len(self.unit_years))))
