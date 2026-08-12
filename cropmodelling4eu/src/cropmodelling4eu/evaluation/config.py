"""Paths, country sets and evaluation constants.

Every path can be overridden with an environment variable of the same name, so
the notebooks run unchanged against a different TorchCrop run::

    TORCHCROP_RUN_DIR=/.../torchcrop/potential jupyter lab

Nothing in this module reads the filesystem at import time except
:func:`available_countries`, which is called explicitly.
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Final, NamedTuple

__all__ = [
    "CALENDAR_STAGES",
    "CYBENCH_CALENDAR_TEMPLATE",
    "CYBENCH_CROP_DIR",
    "CYBENCH_CROP_MASK_TEMPLATE",
    "CYBENCH_POLYGON_DIR",
    "CYBENCH_ROOT",
    "CYBENCH_YIELD_TEMPLATE",
    "COUNTRY_NAMES",
    "CACHE_DIR",
    "CROP",
    "CalendarStage",
    "DAYS_IN_YEAR",
    "EU27",
    "EUROPE_BBOX",
    "EXCLUDED_COUNTRIES",
    "FIGURE_DIR",
    "GDHY_CROP",
    "GDHY_HARVEST_YEAR_OFFSET",
    "GDHY_ROOT",
    "GDHY_YIELD_TEMPLATE",
    "GRID_RES_DEG",
    "HARVEST_LAG_DAYS",
    "MIN_CELLS_PER_GRIDCELL",
    "MIN_WEIGHT_COVERAGE",
    "OUTPUT_DIR",
    "PhenologyStage",
    "SAGE_CROP",
    "SAGE_ROOT",
    "SAGE_TEMPLATE",
    "SCHENGEN_NON_EU",
    "SIM_PARQUET",
    "SIM_SOWING_DOY",
    "SNAP_KM",
    "STAGES",
    "TABLE_DIR",
    "TARGET_COUNTRIES",
    "TORCHCROP_RUN_DIR",
    "YIELD_MOISTURE_CONTENT",
    "available_countries",
    "ensure_output_dirs",
]


def _env_path(name: str, default: str) -> Path:
    """Path from environment variable ``name``, falling back to ``default``."""
    return Path(os.environ.get(name, default)).expanduser()


# --------------------------------------------------------------------------- #
# Inputs
# --------------------------------------------------------------------------- #

#: Crop these notebooks evaluate. Drives both the CyBench sub-tree and the
#: filename templates below.
CROP: Final[str] = "wheat"

#: A finished TorchCrop shard-combine (see ``submit/TORCHCROP.md``).
TORCHCROP_RUN_DIR: Final[Path] = _env_path(
    "TORCHCROP_RUN_DIR",
    "/data01/FDS/muduchuru/Data/SIMPLACE/torchcrop/winter_wheat_2000_2024",
)

#: One row per (cell, season). The gridded NetCDF twin is not used: the Parquet
#: carries the cell coordinates the country join needs, and reading it avoids a
#: regrid round-trip that would blur cells across national borders.
SIM_PARQUET: Final[Path] = _env_path(
    "SIM_PARQUET", str(TORCHCROP_RUN_DIR / "torchcrop_europe.parquet")
)

CYBENCH_ROOT: Final[Path] = _env_path(
    "CYBENCH_ROOT", "/data01/FDS/muduchuru/Data/Agri/cybench"
)

#: ``<CYBENCH_CROP_DIR>/<country>/<file>_<country>.csv``
CYBENCH_CROP_DIR: Final[Path] = _env_path(
    "CYBENCH_CROP_DIR", str(CYBENCH_ROOT / "cybench-data" / CROP)
)

#: ``<CYBENCH_POLYGON_DIR>/<country>/<country>.shp`` — the administrative units
#: the CyBench observations are reported on. Dissolving them per country gives
#: the national footprint used to assign 10 km cells, so both sides of every
#: comparison share one definition of "France".
CYBENCH_POLYGON_DIR: Final[Path] = _env_path(
    "CYBENCH_POLYGON_DIR", str(CYBENCH_ROOT / "polygons")
)

CYBENCH_YIELD_TEMPLATE: Final[str] = "yield_{crop}_{country}.csv"
CYBENCH_CALENDAR_TEMPLATE: Final[str] = "crop_calendar_{crop}_{country}.csv"
CYBENCH_CROP_MASK_TEMPLATE: Final[str] = "crop_mask_{crop}_{country}.csv"


# --------------------------------------------------------------------------- #
# Gridded references — GDHY yields and the SAGE crop calendar
# --------------------------------------------------------------------------- #
#
# Both are 0.5° global fields rather than national statistics, so they are
# compared per grid cell instead of per country (see :mod:`utils.grid`).

#: GDHY v1.2/v1.3 (Iizumi & Sakai 2020), one NetCDF per year under
#: ``<GDHY_ROOT>/<GDHY_CROP>/yield_<year>.nc4``. The variable is ``var``, in
#: t/ha, on a 0.5° grid with longitudes running 0-360.
GDHY_ROOT: Final[Path] = _env_path("GDHY_ROOT", "/data01/FDS/muduchuru/Data/Agri/GDHY")

#: Sub-directory of :data:`GDHY_ROOT`. ``wheat_winter`` is the product to pair
#: with a winter-wheat run; ``wheat`` merges winter and spring, and
#: ``wheat_spring`` is the other half of the split.
GDHY_CROP: Final[str] = "wheat_winter"

GDHY_YIELD_TEMPLATE: Final[str] = "yield_{year}.nc4"

#: Calendar years to add to a GDHY file year to get the **harvest** year the
#: TorchCrop rows are labelled with. GDHY dates a crop year by its harvest, the
#: same convention as the run, so the default is 0. It is a constant rather
#: than an assumption buried in a loader because the notebook tests it: §4 of
#: ``gdhy_yield_evaluation.ipynb`` correlates the two sides at offsets -1, 0
#: and +1 and reports which one wins.
GDHY_HARVEST_YEAR_OFFSET: Final[int] = 0

#: Sacks et al. (2010) SAGE crop calendar, ``<crop>.crop.calendar.fill.nc``.
#: A single climatological field — no year dimension — of planting and harvest
#: days-of-year on a 0.5° grid.
SAGE_ROOT: Final[Path] = _env_path(
    "SAGE_ROOT", "/data01/FDS/muduchuru/Data/Agri/SAGE_crop_calendar"
)

SAGE_CROP: Final[str] = "Wheat.Winter"
SAGE_TEMPLATE: Final[str] = "{crop}.crop.calendar.fill.nc"

#: Resolution of both gridded references, in degrees. The TorchCrop 10 km cells
#: are binned onto this grid rather than the references being interpolated onto
#: the 10 km one: aggregating the fine field is a mean over real values, while
#: interpolating the coarse one would invent detail and correlate neighbouring
#: comparison points even more strongly than they already are.
GRID_RES_DEG: Final[float] = 0.5

#: A 0.5° cell needs this many 10 km cells before it is paired. At European
#: latitudes a full 0.5° cell holds ~15-20 of them, so the default keeps cells
#: that are mostly sea or mostly non-cropland from carrying the same weight as
#: a fully covered inland one.
MIN_CELLS_PER_GRIDCELL: Final[int] = 3

#: ``(lon_min, lon_max, lat_min, lat_max)`` the global references are cropped
#: to before anything else. Matches ``utils.plots.EUROPE_EXTENT``.
EUROPE_BBOX: Final[tuple[float, float, float, float]] = (-11.0, 33.0, 34.0, 70.0)


# --------------------------------------------------------------------------- #
# Outputs
# --------------------------------------------------------------------------- #

OUTPUT_DIR: Final[Path] = _env_path(
    "EVALUATION_OUTPUT_DIR", str(Path(__file__).resolve().parents[1] / "outputs")
)
FIGURE_DIR: Final[Path] = OUTPUT_DIR / "figures"
TABLE_DIR: Final[Path] = OUTPUT_DIR / "tables"
CACHE_DIR: Final[Path] = OUTPUT_DIR / "cache"


def ensure_output_dirs() -> None:
    """Create the figure/table/cache tree. Idempotent."""
    for path in (FIGURE_DIR, TABLE_DIR, CACHE_DIR):
        path.mkdir(parents=True, exist_ok=True)


# --------------------------------------------------------------------------- #
# Country scope
# --------------------------------------------------------------------------- #

#: EU-27. ``EL`` is Greece: CyBench follows the Eurostat/NUTS convention, not
#: ISO 3166 (``GR``), and the polygon directories use the same code.
EU27: Final[tuple[str, ...]] = (
    "AT", "BE", "BG", "HR", "CY", "CZ", "DK", "EE", "FI", "FR", "DE", "EL",
    "HU", "IE", "IT", "LV", "LT", "LU", "MT", "NL", "PL", "PT", "RO", "SK",
    "SI", "ES", "SE",
)

#: Schengen members outside the EU.
SCHENGEN_NON_EU: Final[tuple[str, ...]] = ("CH", "IS", "LI", "NO")

#: Never evaluated, whatever the data holds.
EXCLUDED_COUNTRIES: Final[frozenset[str]] = frozenset({"UA", "RU"})

#: The requested scope: EU + Schengen, minus the exclusions. Which of these
#: actually carry data is resolved at runtime by :func:`available_countries` —
#: CyBench publishes no wheat for several of them.
TARGET_COUNTRIES: Final[tuple[str, ...]] = tuple(
    sorted((set(EU27) | set(SCHENGEN_NON_EU)) - EXCLUDED_COUNTRIES)
)

COUNTRY_NAMES: Final[dict[str, str]] = {
    "AT": "Austria", "BE": "Belgium", "BG": "Bulgaria", "CH": "Switzerland",
    "CY": "Cyprus", "CZ": "Czechia", "DE": "Germany", "DK": "Denmark",
    "EE": "Estonia", "EL": "Greece", "ES": "Spain", "FI": "Finland",
    "FR": "France", "HR": "Croatia", "HU": "Hungary", "IE": "Ireland",
    "IS": "Iceland", "IT": "Italy", "LI": "Liechtenstein", "LT": "Lithuania",
    "LU": "Luxembourg", "LV": "Latvia", "MT": "Malta", "NL": "Netherlands",
    "NO": "Norway", "PL": "Poland", "PT": "Portugal", "RO": "Romania",
    "SE": "Sweden", "SI": "Slovenia", "SK": "Slovakia",
}


def available_countries(
    countries: tuple[str, ...] = TARGET_COUNTRIES,
    crop: str = CROP,
    crop_dir: Path | None = None,
    polygon_dir: Path | None = None,
) -> list[str]:
    """The subset of ``countries`` with both a CyBench yield file and polygons.

    A country needs both to be evaluable: the yield file supplies the reference
    and the polygons decide which 10 km cells belong to it. Returned sorted, so
    every table and figure carries the same country order.
    """
    crop_dir = crop_dir or CYBENCH_CROP_DIR
    polygon_dir = polygon_dir or CYBENCH_POLYGON_DIR
    keep = []
    for code in countries:
        yield_csv = crop_dir / code / CYBENCH_YIELD_TEMPLATE.format(
            crop=crop, country=code
        )
        shapefile = polygon_dir / code / f"{code}.shp"
        if yield_csv.is_file() and shapefile.is_file():
            keep.append(code)
    return sorted(keep)


# --------------------------------------------------------------------------- #
# Evaluation constants
# --------------------------------------------------------------------------- #

#: The engine wraps its day-of-year at 365
#: (``cropmodelling4eu.torchcrop.run.run_batch``), and CyBench's fractional
#: ``sos``/``eos`` follow the same convention. Using 365.25 here would put a
#: slow drift into every circular difference.
DAYS_IN_YEAR: Final[float] = 365.0

#: Fallback sowing day-of-year, for a run made against an export with no site
#: table.
#:
#: **This is no longer necessarily the run's sowing date.** Since data4simplace
#: gained its site stage, the sowing date is a per-cell input read from
#: ``site.csv``, and the runs carry it through to the results as a
#: ``sowing_doy`` column. Prefer that column — :func:`utils.torchcrop.load_run`
#: uses it when present — and read this constant only as what an older run
#: assumed for the whole continent.
SIM_SOWING_DOY: Final[int] = 270

#: Days between maturity (``DVS = 2``) and harvest. LINTUL-5 terminates the
#: crop at maturity and models no drydown, so the default 0 makes the simulated
#: harvest date identical to maturity. Set it to a positive lag to test a
#: drydown assumption against CyBench ``eos``.
HARVEST_LAG_DAYS: Final[float] = 0.0

#: Grain moisture content of the CyBench statistics. **Unused by default** —
#: the notebooks compare raw values, so the simulated grain dry matter and the
#: reported market-moisture yields stay on their own bases and the resulting
#: offset shows up in Bias and MAPE. Pass ``moisture=YIELD_MOISTURE_CONTENT``
#: to :func:`utils.aggregate.to_dry_matter` to put both sides on dry matter.
YIELD_MOISTURE_CONTENT: Final[float] = 0.135

#: A 10 km cell centre this far outside a national polygon still counts as that
#: country: half a grid cell, so a cell whose centre falls just offshore but
#: whose body overlaps land is not silently dropped. Coastal countries lose a
#: fifth of their cells at 0 km.
SNAP_KM: Final[float] = 5.0

#: Below this share of regions carrying an area weight, a country-year average
#: falls back to the unweighted mean rather than weighting on a biased subset.
MIN_WEIGHT_COVERAGE: Final[float] = 0.5


class PhenologyStage(NamedTuple):
    """One simulated stage and the CyBench field it is compared against.

    Attributes:
        key: Identifier used in column names and file names.
        label: Human-readable name for figure titles and tables.
        sim_col: Day-of-year column produced by :mod:`utils.phenology`.
        obs_col: CyBench ``crop_calendar`` column.
        caveat: What the pairing does and does not mean. Printed by the
            notebook next to every result, because two of the three pairings
            are proxies rather than like-for-like.
    """

    key: str
    label: str
    sim_col: str
    obs_col: str
    caveat: str


#: The stages that both datasets can support.
#:
#: Flowering is absent: `torchcrop_run.py` discards the DVS trajectory after
#: dating the fertilizer schedule, so no anthesis date reaches the Parquet, and
#: CyBench's crop calendar carries only a season start and end. Neither side
#: has it, so it is not modelled here.
STAGES: Final[tuple[PhenologyStage, ...]] = (
    PhenologyStage(
        key="sowing",
        label="Sowing",
        sim_col="sowing_doy",
        obs_col="sos",
        caveat=(
            "Weak pairing. The simulated date is a constant model input "
            f"(DOY {SIM_SOWING_DOY}), not a prediction, so its only degree of "
            "freedom is a single continental offset. CyBench 'sos' is a "
            "start-of-season, and it switches convention with climate: ~DOY "
            "330-360 in ES/EL/PT/southern IT (autumn sowing) but ~DOY 40-75 in "
            "DE/PL/EE/LV (post-winter green-up). Read the northern residuals "
            "as a convention difference, not a model error."
        ),
    ),
    PhenologyStage(
        key="maturity",
        label="Maturity",
        sim_col="maturity_doy",
        obs_col="eos",
        caveat=(
            "Strongest pairing. Simulated maturity is the first day at "
            "DVS >= 2; CyBench 'eos' is the end of season, which for winter "
            "wheat falls at harvest (~DOY 205-225). The residual absorbs the "
            "maturity-to-harvest interval as well as any model error."
        ),
    ),
    PhenologyStage(
        key="harvest",
        label="Harvest",
        sim_col="harvest_doy",
        obs_col="eos",
        caveat=(
            "Identical to maturity while HARVEST_LAG_DAYS = 0: LINTUL-5 stops "
            "the crop at DVS = 2 and models no drydown, so it predicts no "
            "harvest date of its own. Raise the lag to test a drydown "
            "assumption; the row is kept so the assumption stays visible."
        ),
    ),
)


class CalendarStage(NamedTuple):
    """One simulated date and the SAGE crop-calendar field it is paired with.

    The SAGE calendar publishes a *window* around each date as well as the date
    itself, so unlike :class:`PhenologyStage` every stage here carries the two
    window columns. Whether the simulated date falls inside that window is a
    more forgiving — and more meaningful — test than a bias in days: SAGE's
    planting window is 10 to 100 days wide, and a date 12 days from the
    midpoint of a 64-day window is not late.

    Attributes:
        key: Identifier used in column names and file names.
        label: Human-readable name for figure titles and tables.
        sim_col: Day-of-year column produced by :mod:`utils.torchcrop`.
        obs_col: SAGE date column.
        start_col: SAGE window start.
        end_col: SAGE window end.
        caveat: What the pairing does and does not mean.
    """

    key: str
    label: str
    sim_col: str
    obs_col: str
    start_col: str
    end_col: str
    caveat: str


#: The stages the SAGE calendar can support. Flowering is absent from both
#: sides, exactly as in :data:`STAGES`.
CALENDAR_STAGES: Final[tuple[CalendarStage, ...]] = (
    CalendarStage(
        key="sowing",
        label="Sowing",
        sim_col="sowing_doy",
        obs_col="plant",
        start_col="plant_start",
        end_col="plant_end",
        caveat=(
            "A like-for-like pairing of a constant against a field. SAGE "
            f"'plant' is a planting date, which is what DOY {SIM_SOWING_DOY} "
            "is meant to be — unlike CyBench 'sos', which switches between an "
            "autumn sowing and a spring green-up with climate. The bias "
            "therefore measures the run's single sowing latch honestly, and "
            "the residual field is exactly the sowing calendar the export is "
            "missing."
        ),
    ),
    CalendarStage(
        key="maturity",
        label="Maturity / harvest",
        sim_col="maturity_doy",
        obs_col="harvest",
        start_col="harvest_start",
        end_col="harvest_end",
        caveat=(
            "Simulated maturity is the first day at DVS >= 2; SAGE 'harvest' "
            "is the date the crop is taken off the field, which is at or "
            "after maturity. The residual absorbs that interval, so a small "
            "positive bias is not necessarily an error."
        ),
    ),
)
