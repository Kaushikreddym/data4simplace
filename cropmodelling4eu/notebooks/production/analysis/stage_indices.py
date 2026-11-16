"""Stage × variable weather indices for the Step 3 feature ablations.

Twelve daily-weather variables are summarised in five growth stages of every
season, under three window definitions, from the same MSWX files as
:mod:`extremes`:

=================  ===========================================================
Window variant     Stage boundaries
=================  ===========================================================
``calendar``       Fixed months everywhere: autumn Oct–Nov, winter Dec–Feb,
                   stem Mar–Apr, flowering May, grain fill Jun–Jul
``phenology``      The cell's climatological torchcrop phenology (as in
                   :mod:`extremes`)
``simulated``      That season's own torchcrop sowing, anthesis and maturity;
                   the climatology where the run has no date
=================  ===========================================================

Phenology stages, in days after sowing (``A`` anthesis, ``M`` maturity):
autumn ``[0, 60)``, winter ``[60, A − 60)``, stem elongation ``[A − 60, A − 7)``,
flowering ``[A − 7, A + 7)``, grain fill ``[A + 7, M)``.

Two derived daily series are added to what the export carries:

* **VPD** (kPa) from the min/max saturation pressure and ``RelHumCalc``.
* **Soil moisture**, a one-layer bucket of :data:`BUCKET_MM` holding capacity
  filled by precipitation and emptied by Hargreaves ET₀, scaled down linearly
  below half full. It is a proxy for timing and persistence of water deficit,
  not a soil-specific water balance; the crop models carry that.
"""

from __future__ import annotations

import logging
import os
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np
import pandas as pd
import xarray as xr

from extremes import EXPORT_DIR, N_LON, DRY_DAY_MM, EDD_BASE_C, MISSING, hargreaves_et0, weather_path

logger = logging.getLogger(__name__)

STAGES: tuple[str, ...] = ("autumn", "winter", "stem", "flowering", "grainfill")
VARIABLES: tuple[str, ...] = ("tmean", "tmax", "edd", "frost", "prec", "cdd", "wet",
                              "waterlog", "sm", "vpd", "rad", "cwb")
VARIANTS: tuple[str, ...] = ("calendar", "phenology", "simulated")
COLUMNS: list[str] = [f"{v}_{s}" for s in STAGES for v in VARIABLES] + ["winterkill"]

BUCKET_MM = 150.0
FROST_C = -2.0
WINTERKILL_C = -15.0
WET_DAY_MM = 10.0
WATERLOG_FILL = 0.95
CALENDAR: dict[str, tuple[tuple[int, int, int], tuple[int, int, int]]] = {
    "autumn": ((-1, 10, 1), (-1, 12, 1)),
    "winter": ((-1, 12, 1), (0, 3, 1)),
    "stem": ((0, 3, 1), (0, 5, 1)),
    "flowering": ((0, 5, 1), (0, 6, 1)),
    "grainfill": ((0, 6, 1), (0, 8, 1)),
}


def _es(t: np.ndarray) -> np.ndarray:
    return 0.6108 * np.exp(17.27 * t / (t + 237.3))


def _bucket(prec: np.ndarray, et0: np.ndarray) -> np.ndarray:
    store, out = BUCKET_MM, np.empty(prec.size)
    for i, (p, e) in enumerate(zip(prec, np.nan_to_num(et0))):
        store = min(store + p, BUCKET_MM)
        store = max(store - e * min(1.0, store / (0.5 * BUCKET_MM)), 0.0)
        out[i] = store / BUCKET_MM
    return out


def _longest_run(mask: np.ndarray) -> int:
    if not mask.any():
        return 0
    edges = np.flatnonzero(np.diff(np.r_[0, mask.astype(np.int8), 0]))
    return int((edges[1::2] - edges[::2]).max())


def phenology_slices(sow: int, anthesis: int, maturity: int) -> dict[str, slice]:
    """Stage slices, as day offsets from the start of the daily record."""
    winter_end = max(anthesis - 60, 61)
    stem_end = max(anthesis - 7, winter_end + 1)
    flower_end = max(anthesis + 7, stem_end + 1)
    bounds = {"autumn": (0, 60), "winter": (60, winter_end), "stem": (winter_end, stem_end),
              "flowering": (stem_end, flower_end),
              "grainfill": (flower_end, max(maturity, flower_end + 1))}
    return {k: slice(sow + a, sow + b) for k, (a, b) in bounds.items()}


def calendar_slices(t0: np.datetime64, year: int) -> dict[str, slice]:
    def offset(dy: int, month: int, day: int) -> int:
        return int((np.datetime64(f"{year + dy}-{month:02d}-{day:02d}") - t0).astype(int))
    return {k: slice(offset(*a), offset(*b)) for k, (a, b) in CALENDAR.items()}


def _stage_stats(d: dict[str, np.ndarray], s: slice) -> list[float]:
    p = d["prec"][s]
    return [np.nanmean(d["tmean"][s]), np.nanmean(d["tmax"][s]), np.nansum(d["edd"][s]),
            float(np.sum(d["tmin"][s] <= FROST_C)), p.sum(), float(_longest_run(p < DRY_DAY_MM)),
            float(np.sum(p >= WET_DAY_MM)), float(np.sum(d["sm"][s] >= WATERLOG_FILL)),
            np.nanmean(d["sm"][s]), np.nanmean(d["vpd"][s]), np.nanmean(d["rad"][s]),
            np.nansum(d["cwb"][s])]


def cell_stage_indices(simplace_id: int, lat: float, climatology: tuple[int, int, int],
                       simulated: pd.DataFrame, years: tuple[int, int],
                       export_dir: Path = EXPORT_DIR) -> dict[str, pd.DataFrame]:
    """``{variant: frame}`` with one row per harvest year and :data:`COLUMNS`."""
    daily = pd.read_csv(weather_path(simplace_id, export_dir), sep="\t", usecols=[
        "Date", "Precipitation", "TempMin", "TempMean", "TempMax", "Radiation", "RelHumCalc"])
    dates = pd.to_datetime(daily["Date"]).to_numpy()
    raw = daily.drop(columns="Date").to_numpy(dtype=float)
    raw[raw <= MISSING + 1e-6] = np.nan
    prec, tmin, tmean, tmax, rad, rh = raw.T
    prec = np.nan_to_num(prec)
    doy = pd.DatetimeIndex(dates).dayofyear.to_numpy()
    et0 = hargreaves_et0(doy, lat, tmin, tmax, tmean)
    d = {"prec": prec, "tmin": tmin, "tmean": tmean, "tmax": tmax, "rad": rad * 0.0864,
         "edd": np.clip(tmax - EDD_BASE_C, 0.0, None), "cwb": prec - et0, "sm": _bucket(prec, et0),
         "vpd": np.clip((_es(tmax) + _es(tmin)) / 2.0 - _es(tmean) * rh / 100.0, 0.0, None)}

    t0 = dates[0].astype("datetime64[D]")
    sow_c, a_c, m_c = climatology
    rows: dict[str, list] = {v: [] for v in VARIANTS}
    for year in range(years[0], years[1] + 1):
        start = int((np.datetime64(f"{year - 1}-01-01") - t0).astype(int))
        sim = simulated.loc[year] if year in simulated.index else None
        windows = {
            "calendar": calendar_slices(t0, year),
            "phenology": phenology_slices(start + sow_c - 1, a_c, m_c),
            "simulated": phenology_slices(start + int(sim["sowing_doy"]) - 1,
                                          int(sim["days_to_anthesis"]), int(sim["days_to_maturity"]))
            if sim is not None and sim.notna().all() else phenology_slices(start + sow_c - 1, a_c, m_c),
        }
        for variant, slices in windows.items():
            if min(s.start for s in slices.values()) < 0 or max(s.stop for s in slices.values()) > len(dates):
                continue
            values = [x for stage in STAGES for x in _stage_stats(d, slices[stage])]
            values.append(float(np.sum(tmin[slices["winter"]] <= WINTERKILL_C)))
            rows[variant].append([year, *values])
    return {v: pd.DataFrame(r, columns=["year", *COLUMNS]).assign(SimplaceID=simplace_id)
            for v, r in rows.items()}


def _chunk(args) -> dict[str, pd.DataFrame]:
    cells, simulated, years, export_dir = args
    out: dict[str, list] = {v: [] for v in VARIANTS}
    for c in cells.itertuples(index=False):
        sim = simulated.get(c.SimplaceID, pd.DataFrame())
        try:
            frames = cell_stage_indices(c.SimplaceID, c.lat, (c.sowing_doy, c.days_to_anthesis,
                                        c.days_to_maturity), sim, years, export_dir)
        except FileNotFoundError:
            continue
        for v, f in frames.items():
            out[v].append(f)
    return {v: pd.concat(f, ignore_index=True) for v, f in out.items() if f}


def compute(windows: pd.DataFrame, torchcrop_parquet: Path, years: tuple[int, int], out_dir: Path,
            export_dir: Path = EXPORT_DIR, workers: int | None = None, chunk_size: int = 250) -> None:
    """Write ``stage_indices_<variant>.parquet`` for every cell and harvest year."""
    run = pd.read_parquet(torchcrop_parquet, columns=[
        "SimplaceID", "year", "sowing_doy", "days_to_anthesis", "days_to_maturity"])
    run = run[run["year"].between(*years)]
    by_cell = {k: g.set_index("year")[["sowing_doy", "days_to_anthesis", "days_to_maturity"]]
               for k, g in run.groupby("SimplaceID")}
    chunks = []
    for i in range(0, len(windows), chunk_size):
        cells = windows.iloc[i:i + chunk_size]
        chunks.append((cells, {k: by_cell[k] for k in cells["SimplaceID"] if k in by_cell},
                       years, export_dir))
    with ProcessPoolExecutor(workers or min(64, os.cpu_count() or 1)) as pool:
        parts = list(pool.map(_chunk, chunks))
    out_dir.mkdir(parents=True, exist_ok=True)
    for variant in VARIANTS:
        frame = pd.concat([p[variant] for p in parts if variant in p], ignore_index=True)
        frame = frame[["SimplaceID", "year", *COLUMNS]].astype({"year": "int16", **{c: "float32" for c in COLUMNS}})
        frame.to_parquet(out_dir / f"stage_indices_{variant}.parquet", index=False)


def cropland_area(export_dir: Path = EXPORT_DIR) -> pd.Series:
    """Cropland area (km²) per ``SimplaceID`` from the soil stage's uncertainty tiles."""
    frames = []
    for path in sorted((export_dir / "soil" / "netcdf_tiles").glob("intermediate_soil_uncertainty*.nc")):
        with xr.open_dataset(path) as ds:
            frames.append(ds["total_area_km2"].to_dataframe().reset_index())
    cells = pd.concat(frames, ignore_index=True).dropna(subset=["total_area_km2"])
    row = np.rint((72.0 - 0.05 - cells["lat"]) / 0.1).astype(int)
    col = np.rint((cells["lon"] + 17.0 - 0.05) / 0.1).astype(int)
    return pd.Series(cells["total_area_km2"].to_numpy(), index=row * N_LON + col + 1,
                     name="cropland_km2").groupby(level=0).max()


def unit_means(cells: pd.DataFrame, cell_to_adm: pd.DataFrame, columns: list[str],
               weights: pd.Series | None = None) -> pd.DataFrame:
    """Unit-year means of ``columns``, cropland-area weighted when ``weights`` is given."""
    frame = cells.merge(cell_to_adm[["SimplaceID", "adm_id"]], on="SimplaceID").dropna(subset=["adm_id"])
    w = (frame["SimplaceID"].map(weights).fillna(0.0) if weights is not None
         else pd.Series(1.0, index=frame.index)).to_numpy()
    values = frame[columns].to_numpy(dtype=float)
    present = np.isfinite(values)
    num = pd.DataFrame(np.where(present, values * w[:, None], 0.0), columns=columns)
    den = pd.DataFrame(present * w[:, None], columns=columns)
    keys = [frame["adm_id"].to_numpy(), frame["year"].to_numpy()]
    out = num.groupby(keys).sum() / den.groupby(keys).sum().replace(0.0, np.nan)
    out.index.names = ["adm_id", "year"]
    return out.reset_index().astype({"adm_id": "string", "year": int})


if __name__ == "__main__":
    import argparse

    from extremes import phenology_windows

    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--run", type=Path, required=True)
    parser.add_argument("--first", type=int, default=2000)
    parser.add_argument("--last", type=int, default=2021)
    parser.add_argument("--workers", type=int)
    a = parser.parse_args()
    logging.basicConfig(level=logging.INFO)
    tc = a.run / "torchcrop" / "torchcrop_europe.parquet"
    compute(phenology_windows(tc), tc, (a.first, a.last),
            a.run / "analysis" / "diagnostics" / "stage_indices", workers=a.workers)
