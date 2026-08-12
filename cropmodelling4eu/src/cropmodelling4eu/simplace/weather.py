"""Reconcile the export's weather units with a solution's.

SIMPLACE binds a CSV resource **by column position, not by name**: the export
writes ``Date, Precipitation, TempMin, …`` where the Brandenburg solution
declares ``CURRENTDATE, Rain, AirTemperatureMin, …`` and the run still works.
Nothing is misnamed, so a unit mismatch is invisible — and there is one:
``Irradiation`` is declared kJ m-2 d-1 where the export writes a daily-mean flux
in W m-2, a factor of 86.4 and the whole difference between 5 t/ha and 0.04.

Two ways out, selected by ``simplace.weather_conversion``:

``transform``
    Scale the column inside the solution's own SQL transformer
    (:meth:`SolutionDocument.scale_transform`), leaving the weather symlinked.
    :func:`declared_factors` translates a contract's factors onto the ids that
    SQL sees. Free, and the default.
``files``
    Write converted copies. Needed only when the contract changes the file's
    *shape* rather than its units — ``sustag_v2`` reorders columns and adds
    ``vprsd``/``dewp``. Costs a second copy of the weather (~15 GB for Europe).
"""

from __future__ import annotations

import gzip
import logging
from concurrent.futures import ProcessPoolExecutor
from dataclasses import dataclass
from functools import partial
from pathlib import Path

import numpy as np
import pandas as pd

from cropmodelling4eu.config import GridConfig
from cropmodelling4eu.export.cells import id_to_rowcol, weather_path

logger = logging.getLogger(__name__)

__all__ = [
    "WeatherConversion",
    "BRANDENBURG",
    "SUSTAG_V2",
    "build_weather_cache",
    "cached_weather_path",
    "declared_factors",
]

W_PER_M2_TO_KJ = 86.4
W_PER_M2_TO_MJ = 0.0864
M_PER_S_TO_KM_PER_DAY = 86.4

_SENTINELS = [-99.9, -99.0]


@dataclass(frozen=True, slots=True)
class WeatherConversion:
    """What one solution wants from the export's weather.

    ``columns`` is the contract: export column -> the position it must occupy,
    since position is what SIMPLACE reads. ``factors`` are applied on the way
    out. ``delimiter`` ``"\\t"`` matches an empty ``<divider>``, which SIMPLACE
    reads as whitespace.
    """

    name: str
    columns: tuple[str, ...]
    factors: dict[str, float]
    delimiter: str = "\t"
    gzip: bool = True
    date_format: str = "%Y-%m-%d"
    header: bool = True


BRANDENBURG = WeatherConversion(
    name="brandenburg",
    columns=("Date", "Precipitation", "TempMin", "TempMean", "TempMax", "Radiation"),
    factors={"Radiation": W_PER_M2_TO_KJ},
)

#: Comma separated, a leading scenario key pair, ``yyyyMMdd`` dates, radiation
#: in MJ m-2 d-1 and wind in km/day. ``vprsd`` is not in the export, so it is
#: derived; relative humidity is read twice, at tmin and at tmax.
SUSTAG_V2 = WeatherConversion(
    name="sustag_v2",
    columns=(
        "period", "gcm_rcp", "Date", "TempMax", "TempMin", "vprsd",
        "Windspeed", "Precipitation", "Radiation", "RelHumCalc", "RelHumCalc",
        "dewp",
    ),
    factors={"Radiation": W_PER_M2_TO_MJ, "Windspeed": M_PER_S_TO_KM_PER_DAY},
    delimiter=",",
    gzip=False,
    date_format="%Y%m%d",
)


def declared_factors(
    declared: list[str], conversion: WeatherConversion
) -> dict[str, float]:
    """Translate a contract's factors onto the solution's own column ids.

    ``factors`` is keyed on the *export's* names; a transform's SQL sees the
    ids the ``<resource>`` header declares. Only position relates the two —
    which is how SIMPLACE binds them — so zipping the orders both translates
    the keys and asserts the alignment the scheme rests on.
    """
    out = {}
    for position, column in enumerate(conversion.columns):
        if (factor := conversion.factors.get(column)) is None:
            continue
        if position >= len(declared):
            raise ValueError(
                f"{conversion.name!r} scales column {position} ({column!r}) but the "
                f"solution declares only {len(declared)}: {declared}"
            )
        out[declared[position]] = factor
    return out


def _saturation_vp(temp_c: np.ndarray) -> np.ndarray:
    """Tetens saturation vapour pressure [kPa]."""
    return 0.6108 * np.exp(17.27 * temp_c / (temp_c + 237.3))


def _derive(frame: pd.DataFrame) -> pd.DataFrame:
    """Add the columns a contract needs that the export does not carry."""
    out = frame.copy()
    if {"TempMean", "RelHumCalc"} <= set(out):
        out["vprsd"] = (
            _saturation_vp(out["TempMean"].to_numpy()) * out["RelHumCalc"].to_numpy() / 100.0
        ).round(4)
    if {"TempMean", "vprsd"} <= set(out):
        ratio = np.log(out["vprsd"].clip(lower=1e-4).to_numpy() / 0.6108)
        out["dewp"] = (237.3 * ratio / (17.27 - ratio)).round(2)
    return out


def convert_cell(
    source: Path,
    target: Path,
    conversion: WeatherConversion,
    scenario: tuple[int, str] = (0, "0_0"),
) -> int:
    """Convert one cell's weather file; return the rows written."""
    frame = _derive(
        pd.read_csv(source, sep="\t", parse_dates=["Date"], na_values=_SENTINELS)
    )
    frame["period"], frame["gcm_rcp"] = scenario

    missing = [c for c in conversion.columns if c not in frame]
    if missing:
        raise KeyError(
            f"{source.name} has no {missing}, which the {conversion.name!r} "
            f"contract needs (it has {list(frame.columns)})"
        )

    def values(column: str) -> np.ndarray:
        series = frame[column]
        if column == "Date":
            return series.dt.strftime(conversion.date_format).to_numpy()
        if (factor := conversion.factors.get(column)) is not None:
            return (series * factor).round(4).to_numpy()
        return series.to_numpy()

    # Keyed by position, not name: a contract may legitimately read one source
    # column twice (SUSTAg takes relative humidity at both tmin and tmax).
    out = pd.DataFrame(
        {f"c{i}": values(column) for i, column in enumerate(conversion.columns)}
    )
    target.parent.mkdir(parents=True, exist_ok=True)
    opener = gzip.open if conversion.gzip else open
    with opener(target, "wt", encoding="utf-8", newline="") as handle:
        out.to_csv(
            handle,
            sep=conversion.delimiter,
            index=False,
            header=list(conversion.columns) if conversion.header else False,
            na_rep="-99.9",
        )
    return len(out)


def cached_weather_path(
    cache_dir: Path,
    simplace_id: int,
    grid: GridConfig,
    conversion: WeatherConversion,
) -> Path:
    """Where a cell's converted file lives, nested by grid row like the tree."""
    row, _ = id_to_rowcol(np.asarray([simplace_id]), grid)
    target = cache_dir / str(int(row[0])) / weather_path(Path("."), simplace_id, grid).name
    return target if conversion.gzip else target.with_suffix("")


def _convert_one(
    export_dir: Path,
    cache_dir: Path,
    grid: GridConfig,
    conversion: WeatherConversion,
    scenario: tuple[int, str],
    simplace_id: int,
) -> bool | None:
    """``True`` converted, ``False`` already cached, ``None`` absent from the export.

    Module level, with the cell id last, so :func:`functools.partial` can bind
    the rest and hand it to a process pool.
    """
    source = weather_path(export_dir, simplace_id, grid)
    if not source.is_file():
        return None
    target = cached_weather_path(cache_dir, simplace_id, grid, conversion)
    if target.is_file() and target.stat().st_mtime >= source.stat().st_mtime:
        return False
    convert_cell(source, target, conversion, scenario)
    return True


def build_weather_cache(
    export_dir: Path,
    cache_dir: Path,
    ids: np.ndarray,
    grid: GridConfig,
    conversion: WeatherConversion = BRANDENBURG,
    scenario: tuple[int, str] = (0, "0_0"),
    workers: int = 16,
) -> tuple[int, int]:
    """Convert ``ids`` into a shared cache, skipping what it already holds.

    A converted file is a pure function of the export and the contract, so it
    belongs outside any one workspace: every run of the same contract reads the
    same bytes and the workspace holds symlinks rather than a second copy.
    Returns ``(converted, reused)``.

    ``workers`` are **processes** — the cost is pandas' CSV parse, which holds
    the GIL, so 16 threads measured 2.8x where 16 processes give ~6x.
    """
    work = partial(
        _convert_one, export_dir, Path(cache_dir), grid, conversion, scenario
    )
    cells = [int(value) for value in np.asarray(ids)]
    if workers > 1 and len(cells) > 1:
        with ProcessPoolExecutor(max_workers=workers) as pool:
            outcomes = list(pool.map(work, cells, chunksize=32))
    else:
        outcomes = [work(cell) for cell in cells]

    converted = outcomes.count(True)
    reused = outcomes.count(False)
    logger.info(
        "Weather cache %s (%r contract, %s): %d converted, %d reused",
        cache_dir, conversion.name,
        ", ".join(f"{k} x {v:g}" for k, v in conversion.factors.items()) or "no unit change",
        converted, reused,
    )
    return converted, reused
