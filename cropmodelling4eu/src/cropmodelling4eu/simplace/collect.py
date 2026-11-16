"""Collect SIMPLACE's per-cell yearly outputs into the run schema.

SIMPLACE writes one CSV per location — ``<out>/yearly/<vLocationID>_yearly.csv``
— with one row per harvest. This concatenates them and renames the columns into
the **same schema the torchcrop runner writes**, so one evaluation code path
serves both models and a figure can carry them side by side.

Two conversions and one absence are worth knowing:

* **t/ha to g/m².** SIMPLACE reports ``Yield_t_ha``; the shared schema is
  g/m², and 1 t/ha = 100 g/m².
* **kg/ha to g/m².** ``inputChemN_kg_ha`` divided by 10.
* **The yearly file carries no location column.** Its header starts at
  ``Year``, so the cell identity lives only in the filename — which is why the
  file stem is parsed rather than a column read.
"""

from __future__ import annotations

import logging
import re
from pathlib import Path

import numpy as np
import pandas as pd
import xarray as xr

from cropmodelling4eu.config import GridConfig
from cropmodelling4eu.daily import DAILY_KEYS, DailySink
from cropmodelling4eu.export.cells import id_to_lonlat

logger = logging.getLogger(__name__)

__all__ = ["COLUMN_MAP", "DAILY_FILE", "collect_daily", "collect_run", "daily_frame", "read_yearly", "to_grid"]

#: ``<vLocationID>_yearly.csv``.
_YEARLY_NAME = re.compile(r"^(\d+)_yearly\.csv$")

#: ``<vLocationID>_daily.csv``, the solution's ``CropDaily`` output.
_DAILY_NAME = re.compile(r"^(\d+)_daily\.csv$")

#: The collected daily table, beside ``simplace_europe.parquet``.
DAILY_FILE = "simplace_daily.parquet"

#: ``CropDaily`` column -> the wide daily layout's (torchcrop's DAILY_VARIABLES
#: names; the solution already writes AGB/WSO in t/ha).
DAILY_COLUMNS = {
    "DevStage": "DVS", "LAI": "LAI", "AGB": "AGB", "WSO": "WSO",
    "TRANRF": "TRANRF", "NNI": "NNI", "TRAN": "TRAN",
}

#: SIMPLACE column -> the shared schema's, with the factor applied on the way.
COLUMN_MAP: dict[str, tuple[str, float]] = {
    "Year": ("year", 1.0),
    "Yield_t_ha": ("yield_g_m2", 100.0),
    "Yield_translocated_t_ha": ("adjusted_yield_g_m2", 100.0),
    "AGBiomass_t_ha": ("biomass_g_m2", 100.0),
    "maxLAI": ("max_lai", 1.0),
    "DevStage": ("final_dvs", 1.0),
    # EU_SUSTAg_data4simplace.sol.xml's own WLOutput columns. Unlike
    # Brandenburg, these are already g/m2 (Yield_noHS/Biomass_noHS run
    # 300-2500 -- t/ha at that magnitude would be an agronomic impossibility),
    # so the factor is 1.0, not 100.0. Yield_HS_H_TCan is the heat-stress
    # -adjusted figure, filling the same "adjusted" slot Brandenburg's
    # translocated yield does, not a translocation adjustment.
    "Yield_noHS": ("yield_g_m2", 1.0),
    "Yield_HS_H_TCan": ("adjusted_yield_g_m2", 1.0),
    "Biomass_noHS": ("biomass_g_m2", 1.0),
    "LAImax": ("max_lai", 1.0),
    # Two spellings of the sowing date, and they do not mean the same thing.
    # Brandenburg's `PlantingDOY` echoes the constant `vIDPL`; the rule-based
    # SUSTAg solution writes the *realized* `SowingDate.SowingDOY`. A run that
    # carries both is a rule-based solution still reading a fixed vIDPL
    # somewhere, so the realized date wins (it is read second).
    "PlantingDOY": ("sowing_doy", 1.0),
    "SowingDOY": ("sowing_doy", 1.0),
    "ForcedSow": ("sowing_forced", 1.0),
    # The window the date was chosen inside. It has two possible sources -- the
    # fertilizer schedule's SAGE pair, else the project file -- so a sowing date
    # cannot be interpreted without it.
    "SowWindowStart": ("sowing_window_start", 1.0),
    "SowWindowEnd": ("sowing_window_end", 1.0),
    "MaturityDOY": ("maturity_doy", 1.0),
    "AnthesisDOY": ("anthesis_doy", 1.0),
    "EmergenceDOY": ("emergence_doy", 1.0),
    "inputChemN_kg_ha": ("n_applied_g_m2", 0.1),
    "TRANRF": ("tranrf_mean", 1.0),
    "NNI": ("nni_mean", 1.0),
    "cropNuptake_kg_ha": ("n_uptake_kg_ha", 1.0),
    "Rain": ("rain_mm", 1.0),
    "ActualEvapotranspiration": ("aet_mm", 1.0),
    "PotentialEvapotranspiration": ("pet_mm", 1.0),
}

#: Variables carried onto the gridded NetCDF, matching the torchcrop maps.
GRID_VARS = [
    "yield_g_m2", "biomass_g_m2", "max_lai", "final_dvs", "days_to_maturity",
    "n_applied_g_m2", "tranrf_mean", "nni_mean",
]


def read_yearly(out_dir: str | Path) -> pd.DataFrame:
    """Concatenate every ``<id>_yearly.csv`` under ``out_dir``.

    Searched recursively: a SIMPLACE run writes into a timestamped
    sub-directory per invocation, so an array job leaves several.

    Raises
    ------
    FileNotFoundError
        If no yearly output exists — which is what an entirely failed run
        looks like, and is worth separating from "ran and produced nothing".
    """
    out_dir = Path(out_dir)
    files = sorted(out_dir.rglob("*_yearly.csv"))
    if not files:
        raise FileNotFoundError(
            f"No *_yearly.csv under {out_dir}. Either no task has finished, or "
            f"the solution's output interface points somewhere else."
        )

    frames = []
    skipped = 0
    for path in files:
        match = _YEARLY_NAME.match(path.name)
        if match is None:
            skipped += 1
            continue
        try:
            # SIMPLACE writes its outputs semicolon-separated whatever the
            # inputs used, so the separator is sniffed rather than assumed.
            frame = pd.read_csv(path, sep=None, engine="python")
        except (pd.errors.EmptyDataError, pd.errors.ParserError):
            # A cell that never harvested writes a header-only file.
            skipped += 1
            continue
        if frame.empty:
            skipped += 1
            continue
        # The output carries `projectid`, but the filename is the authoritative
        # key: it is what the solution's output interface was named from.
        frame["SimplaceID"] = int(match.group(1))
        frames.append(frame)

    if not frames:
        raise FileNotFoundError(
            f"{len(files)} yearly file(s) under {out_dir}, none with any rows"
        )
    combined = pd.concat(frames, ignore_index=True)
    logger.info(
        "Read %d yearly files (%d empty/skipped): %d rows, %d cells",
        len(files), skipped, len(combined), combined["SimplaceID"].nunique(),
    )
    return combined


def to_run_schema(frame: pd.DataFrame, grid: GridConfig) -> pd.DataFrame:
    """Rename and convert SIMPLACE's columns into the shared run schema."""
    out = pd.DataFrame({"SimplaceID": frame["SimplaceID"].astype(np.int64)})
    for source, (target, factor) in COLUMN_MAP.items():
        if source not in frame.columns:
            continue
        values = pd.to_numeric(frame[source], errors="coerce")
        out[target] = values * factor if factor != 1.0 else values

    # Per *target*, not per source: a target with two accepted spellings is
    # only missing when neither was written.
    missing = sorted(
        {target for _, (target, _) in COLUMN_MAP.items()} - set(out.columns)
    )
    if missing:
        logger.warning(
            "SIMPLACE output carries no %s; those columns are absent from the "
            "collected run", ", ".join(missing),
        )

    if "maturity_doy" not in out.columns and "CURRENT.DATE" in frame.columns:
        # EU_SUSTAg_data4simplace.sol.xml's WLOutputs fires
        # rule="HarvestManagement.DoHarvest", so unlike Brandenburg (which
        # writes an explicit MaturityDOY column) this row's own date *is* the
        # maturity/harvest date -- there is just no separate DOY column
        # naming it. LINTUL-5 models no drydown, so maturity == harvest here
        # too, same as the comment below.
        dates = pd.to_datetime(frame["CURRENT.DATE"], format="%d.%m.%Y", errors="coerce")
        out["maturity_doy"] = dates.dt.dayofyear.astype("float64")

    # A stage the crop never reached is written as 0 -- the Phenology
    # component's initial value -- and 0 is not a day of the year. Left alone
    # it would enter a circular mean as 31 December and drag the unit's date
    # across New Year, which is exactly the failure the circular arithmetic
    # exists to avoid. NaN is the honest value; every metric already drops it.
    for column in ("sowing_doy", "emergence_doy", "anthesis_doy", "maturity_doy"):
        if column in out.columns:
            out[column] = out[column].where(out[column] > 0)

    lon, lat = id_to_lonlat(out["SimplaceID"].to_numpy(), grid)
    out["lon"] = lon.astype(np.float32)
    out["lat"] = lat.astype(np.float32)

    # Days from sowing to each stage, wrapped for a season crossing New Year.
    for stage, target in (("maturity_doy", "days_to_maturity"),
                          ("emergence_doy", "days_to_emergence")):
        if {stage, "sowing_doy"} <= set(out.columns):
            span = out[stage] - out["sowing_doy"]
            out[target] = span.where(span > 0, span + 365.0)

    if "yield_g_m2" in out.columns:
        # Both models' schemas carry t/ha alongside g/m2, so a yield reads the
        # same way whichever runner produced the file.
        out["yield_t_ha"] = out["yield_g_m2"] / 100.0
    if "year" in out.columns:
        out["year"] = out["year"].astype("int16")
    out = out.sort_values(["SimplaceID", "year"], kind="stable").reset_index(drop=True)
    out["model"] = "simplace"
    return out


def to_grid(frame: pd.DataFrame, grid: GridConfig) -> xr.Dataset:
    """Scatter the per-(cell, year) rows onto the ``(year, lat, lon)`` grid."""
    from cropmodelling4eu.export.cells import id_to_rowcol

    years = np.sort(frame["year"].unique())
    year_index = pd.Series(np.arange(len(years)), index=years)
    row, col = id_to_rowcol(frame["SimplaceID"].to_numpy(), grid)
    yr = year_index.reindex(frame["year"].to_numpy()).to_numpy()

    res, n_lat, n_lon = grid.resolution_deg, grid.n_lat, grid.n_lon
    data = {}
    for name in GRID_VARS:
        if name not in frame.columns:
            continue
        cube = np.full((len(years), n_lat, n_lon), np.nan, dtype=np.float32)
        cube[yr, row, col] = frame[name].to_numpy(np.float32)
        data[name] = (("year", "lat", "lon"), cube)

    dataset = xr.Dataset(
        data,
        coords={
            "year": years,
            "lat": grid.max_lat - res / 2.0 - np.arange(n_lat) * res,
            "lon": grid.min_lon + res / 2.0 + np.arange(n_lon) * res,
        },
    )
    # The climatology carries `_clim`, not `_mean`: `tranrf_mean` and `nni_mean`
    # are already growing-season means, so `_mean` would be ambiguous between
    # the 3-D per-year cube and its 2-D climatology.
    mean = dataset.mean("year", skipna=True)
    for name, array in mean.data_vars.items():
        dataset[f"{name}_clim"] = array
    dataset["n_years"] = dataset[GRID_VARS[0]].notnull().sum("year")
    dataset.attrs["description"] = "SIMPLACE Lintul5+SLIM, data4simplace Europe export"
    dataset["lat"].attrs.update(units="degrees_north", standard_name="latitude")
    dataset["lon"].attrs.update(units="degrees_east", standard_name="longitude")
    return dataset


#: Written beside the collected Parquet; the handoff to a torchcrop run that
#: is to sow on SIMPLACE's dates rather than on the export's.
SOWING_FILE = "sowing_from_simplace.csv"


def sowing_table(frame: pd.DataFrame) -> pd.DataFrame:
    """SIMPLACE's *simulated* sowing, as ``(SimplaceID, year, sowing_doy)``.

    Under the rule-based solution the sowing date is an output — the first day
    inside the planting window on which a weather rule fired — so handing it to
    torchcrop removes the single largest difference between the two runs.
    Cells that never sowed are absent rather than defaulted, so a consumer
    either has SIMPLACE's date or knows it does not.

    The exported ``sowing_doy`` is one day earlier than the day the crop
    actually starts growing in SIMPLACE: ``brandenburg_rulesow.sol.xml``'s
    component order makes ``DefaultManagement`` read the *previous* day's
    ``SowingRule.DoSow``, a one-day latch documented in that solution file.
    Measured on the German smoke test, torchcrop given the raw exported date
    starts growing exactly one day ahead of SIMPLACE in 60/60 sampled
    cell-years; adding a day here collapses that to ordinary day-to-day noise
    (0 in 51/60, +-1 in the rest).
    """
    columns = ["SimplaceID", "year", "sowing_doy"]
    if any(column not in frame.columns for column in columns):
        raise KeyError(f"the collected run has no {columns}; it has {list(frame.columns)}")
    return (
        frame[columns]
        .dropna()
        .astype({"SimplaceID": "int64", "year": "int32", "sowing_doy": "int16"})
        .assign(sowing_doy=lambda d: d["sowing_doy"] + 1)
        .drop_duplicates(["SimplaceID", "year"])
        .sort_values(["year", "SimplaceID"])
    )


def collect_run(
    out_dir: str | Path,
    target_dir: str | Path,
    grid: GridConfig,
    write_netcdf: bool = True,
) -> Path:
    """Collect a finished run into ``simplace_europe.parquet`` (+ NetCDF).

    Also writes :data:`SOWING_FILE`, so a torchcrop run chained behind this one
    has the sowing dates without needing to open the Parquet or know its schema.

    Returns
    -------
    pathlib.Path
        The Parquet, whose schema matches the torchcrop runner's so both feed
        one evaluation.
    """
    frame = to_run_schema(read_yearly(out_dir), grid)
    target_dir = Path(target_dir)
    target_dir.mkdir(parents=True, exist_ok=True)

    sowing = sowing_table(frame)
    sowing.to_csv(target_dir / SOWING_FILE, index=False)
    logger.info(
        "Wrote %s (%d cell-seasons, DOY %d-%d)", target_dir / SOWING_FILE,
        len(sowing), sowing["sowing_doy"].min(), sowing["sowing_doy"].max(),
    )

    parquet = target_dir / "simplace_europe.parquet"
    frame.to_parquet(parquet, index=False, compression="zstd")
    logger.info("Wrote %s (%d rows, %d cells)", parquet, len(frame), frame["SimplaceID"].nunique())

    if write_netcdf and "year" in frame.columns:
        dataset = to_grid(frame, grid)
        netcdf = target_dir / "simplace_europe_grid.nc"
        dataset.to_netcdf(
            netcdf, encoding={v: {"zlib": True, "complevel": 4} for v in dataset.data_vars}
        )
        logger.info("Wrote %s", netcdf)
    return parquet


def daily_frame(raw: pd.DataFrame, simplace_id: int) -> pd.DataFrame:
    """One cell's ``CropDaily`` rows in the wide daily layout (:mod:`cropmodelling4eu.daily`).

    Trimmed to ``0 < DVS < 2`` as torchcrop's are. A season is a contiguous run
    of such days (a calendar gap or a new ``SowingDOY`` starts the next one),
    labelled with the year its last day falls in. ``das`` counts from the
    realized sowing date, the latest ``SowingDOY`` on or before the row's date.
    """
    date = pd.to_datetime(raw["DATE"], format="%d.%m.%Y")
    dvs = raw["DevStage"].to_numpy(float)
    crop = (dvs > 0.0) & (dvs < 2.0)
    if not crop.any():
        return pd.DataFrame(columns=[*DAILY_KEYS, *DAILY_COLUMNS.values()])
    raw, date = raw.loc[crop], date[crop]
    sow_doy = raw["SowingDOY"].to_numpy(np.int64)
    new_season = np.r_[True, (np.diff(date.to_numpy()) != np.timedelta64(1, "D"))
                       | (np.diff(sow_doy) != 0)]
    season = np.cumsum(new_season)
    harvest_year = pd.Series(date.dt.year.to_numpy()).groupby(season).transform("max")

    doy = date.dt.dayofyear.to_numpy(np.int64)
    sow_year = date.dt.year.to_numpy(np.int64) - (doy < sow_doy)
    sown = pd.to_datetime({"year": sow_year, "month": 1, "day": 1}) + pd.to_timedelta(
        sow_doy - 1, unit="D"
    )

    present = {k: v for k, v in DAILY_COLUMNS.items() if k in raw.columns}
    return pd.DataFrame(
        {
            "SimplaceID": np.full(len(raw), simplace_id, dtype=np.int64),
            "year": harvest_year.to_numpy(np.int16),
            "date": date.to_numpy("datetime64[ns]"),
            "doy": doy.astype(np.int16),
            "das": ((date.to_numpy() - sown.to_numpy()) // np.timedelta64(1, "D")).astype(np.int16),
        }
        | {name: raw[column].to_numpy(np.float32) for column, name in present.items()}
    )


def collect_daily(out_dir: str | Path, target_dir: str | Path) -> Path | None:
    """Stream every ``<id>_daily.csv`` under ``out_dir`` into :data:`DAILY_FILE`.

    Cell by cell through :class:`~cropmodelling4eu.daily.DailySink`: a CMIP6
    region holds ~10^8 daily rows, which never need to be in memory at once.
    Returns ``None`` when the run wrote no daily output.
    """
    files = sorted(Path(out_dir).rglob("*_daily.csv"))
    files = [(f, _DAILY_NAME.match(f.name)) for f in files]
    files = [(f, int(m.group(1))) for f, m in files if m is not None]
    if not files:
        return None
    sink = DailySink(Path(target_dir) / DAILY_FILE)
    for path, simplace_id in files:
        try:
            raw = pd.read_csv(path, sep=";")
        except pd.errors.EmptyDataError:
            continue
        sink(daily_frame(raw, simplace_id))
    written = sink.close()
    logger.info("Wrote %s (%d daily rows from %d files)", written, sink.rows, len(files))
    return written
