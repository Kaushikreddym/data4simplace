"""Run torchcrop (LINTUL-5) over the SIMPLACE Europe export, one shard per job.

Library + CLI behind ``submit/torchcrop_array.sh``. Carries the same logic as
``notebooks/torchcrop_europe_run.ipynb`` — see that notebook for the derivation
of every unit conversion and for the list of inputs the export cannot supply.

The work splits on two axes:

* **Shards** (``--shard`` / ``--n-shards``) — one SLURM array task per shard.
  Cells are dealt round-robin so every shard spans the whole domain and the
  tasks finish together; a contiguous split would give one task all of Iberia
  and another all of Scandinavia, with very different weather-file sizes.
* **Batches** (``--batch-size``) — within a shard, cells run through the model
  in chunks. LINTUL-5 is a Python loop over days, so a batch of 2 000 cells
  costs barely more wall time than a batch of 20; the limit is that
  ``ModelOutput`` retains every per-day state, rate and diagnostic, so peak
  memory scales with ``batch_size × window length``.
* **Seasons** (``--start-year`` / ``--end-year``) — one simulation per harvest
  year, over the window `season_window` defines. Consecutive harvest years
  overlap on the calendar; each is an independent run with its own spin-up.

Each shard writes one Parquet file of per-(cell, year) summaries and never
touches another shard's output, so a failed task is re-runnable on its own.
"""

from __future__ import annotations

import argparse
import logging
import os
import re
import sys
import time
import xml.etree.ElementTree as ET
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from torchcrop import (
    CropParameters,
    Lintul5Model,
    SiteParameters,
    SoilParameters,
    WeatherDriver,
)

logger = logging.getLogger("torchcrop_run")

# --------------------------------------------------------------------------- #
# Paths & grid
# --------------------------------------------------------------------------- #

DEFAULT_DATA_DIR = Path(
    os.environ.get("TC_DATA_DIR", "/data01/FDS/muduchuru/Data/SIMPLACE/europe_torchcrop")
)
DEFAULT_COMPOSITION_XML = Path(
    os.environ.get(
        "TC_COMPOSITION_XML",
        "/beegfs/muduchuru/simplace/Brandenburg_1KM_winter_wheat/data/management"
        "/fertilizer_composition.xml",
    )
)

#: Target grid, from ``_work/config_run.yaml`` -> ``grid:``.
GRID_MIN_LON, GRID_MAX_LON = -17.0, 52.0
GRID_MIN_LAT, GRID_MAX_LAT = 34.0, 72.0
GRID_RES = 0.1
N_LON = int(round((GRID_MAX_LON - GRID_MIN_LON) / GRID_RES))   # 690
N_LAT = int(round((GRID_MAX_LAT - GRID_MIN_LAT) / GRID_RES))   # 380

SENTINEL = -99.9
#: MSWX ``rsds`` is a daily-mean flux in W m-2 and is exported unconverted;
#: LINTUL wants MJ m-2 d-1.
W_PER_M2_TO_MJ = 86_400.0 / 1e6

#: Simulation window per harvest year — see `season_window`. The spin-up runs
#: the water balance without a crop; the window as a whole is long enough that
#: no season is cut short, because the model stops itself at ``DVS = 2``.
SPINUP_MONTHS = 6
SIM_YEARS = 2
#: Days of weather a window must still hold after its sowing date to be run.
MIN_DAYS_AFTER_SOWING = 365

#: Soil-layer bottoms of the SIMPLACE profile written by ``soil_export.py`` [m].
LAYER_BOTTOMS_M = np.array([0.1, 0.3, 0.5, 0.7, 1.0, 2.0])
LAYER_TOPS_M = np.concatenate([[0.0], LAYER_BOTTOMS_M[:-1]])
N_LAYERS = LAYER_BOTTOMS_M.size

#: Fills for torchcrop inputs the European export does not carry. Every entry
#: here is a modelling assumption, not data — see §10 of the notebook.
ASSUMPTIONS = {
    # Used only when the weather export carries no Windspeed (all -99.9);
    # `load_weather_seasons` prefers the real column whenever it is present.
    "wind_m_s": 2.0,
    # Default for `--sowing-doy`, which drives both the model's sowing latch
    # and the simulation window. No sowing-date field is exported.
    "sowing_doy": 270,
    "altitude_m": 0.0,
    "crairc": 0.07,
    "ksub": 100.0,
    "runfr": 0.0,
    "cfev": 2.0,
    "cn_ratio": 12.0,
    "mineralisable_frac": 0.003,
    "rtnmins": 0.025,
    "nmini_cap_g_m2": 15.0,
    "nminti_scale": 1.0,
}

#: Global-mean CO2 [ppm]; no CO2 field is exported, so the season year picks one.
CO2_BY_YEAR = pd.Series(
    {1980: 338.8, 1990: 354.4, 2000: 369.6, 2010: 389.9, 2020: 414.2, 2024: 422.7}
)


def co2_for_year(year: int) -> float:
    """Linearly interpolated global-mean CO2 for a calendar year [ppm]."""
    idx = np.arange(CO2_BY_YEAR.index.min(), CO2_BY_YEAR.index.max() + 1)
    series = CO2_BY_YEAR.reindex(idx).interpolate()
    return float(series.reindex([year]).ffill().bfill().iloc[0])


# --------------------------------------------------------------------------- #
# Grid <-> SimplaceID <-> weather file
# --------------------------------------------------------------------------- #


def id_to_rowcol(simplace_id) -> tuple[np.ndarray, np.ndarray]:
    """``SimplaceID`` (1-based, row-major from the NW corner) -> (row, col)."""
    zero = np.asarray(simplace_id, dtype=np.int64) - 1
    return zero // N_LON, zero % N_LON


def id_to_lonlat(simplace_id) -> tuple[np.ndarray, np.ndarray]:
    """``SimplaceID`` -> cell-centre (lon, lat) in EPSG:4326."""
    row, col = id_to_rowcol(simplace_id)
    return (
        GRID_MIN_LON + GRID_RES / 2.0 + col * GRID_RES,
        GRID_MAX_LAT - GRID_RES / 2.0 - row * GRID_RES,
    )


def weather_path(data_dir: Path, simplace_id: int) -> Path:
    """Path of the gzipped weather file for a ``SimplaceID``."""
    row, col = id_to_rowcol(int(simplace_id))
    return data_dir / "weather" / f"daily_mean_RES1_C{int(col)}R{int(row)}.csv.gz"


_WX_NAME = re.compile(r"daily_mean_RES1_C(\d+)R(\d+)\.csv\.gz$")


def weather_ids(data_dir: Path) -> np.ndarray:
    """Every ``SimplaceID`` that has a weather file on disk."""
    ids = [
        int(m.group(2)) * N_LON + int(m.group(1)) + 1
        for name in (data_dir / "weather").iterdir()
        if (m := _WX_NAME.match(name.name))
    ]
    return np.sort(np.asarray(ids, dtype=np.int64))


# --------------------------------------------------------------------------- #
# Weather
# --------------------------------------------------------------------------- #


def saturation_vp(temp_c: np.ndarray) -> np.ndarray:
    """Tetens saturation vapour pressure [kPa] for air temperature [°C]."""
    return 0.6108 * np.exp(17.27 * temp_c / (temp_c + 237.3))


def sowing_date(harvest_year: int, sowing_doy: int) -> pd.Timestamp:
    """Calendar date of the sowing that is harvested in ``harvest_year``.

    A sowing day-of-year in the second half of the year is an autumn sowing, so
    the season straddles New Year and the crop goes in the ground the year
    *before* harvest (winter wheat, ``sowing_doy = 270``). An earlier DOY is a
    spring sowing, harvested in the same calendar year.
    """
    year = harvest_year - 1 if sowing_doy > 182 else harvest_year
    return pd.Timestamp(year, 1, 1) + pd.Timedelta(days=sowing_doy - 1)


def season_window(
    harvest_year: int,
    sowing_doy: int,
    spinup_months: int = SPINUP_MONTHS,
    sim_years: int = SIM_YEARS,
) -> tuple[pd.Timestamp, pd.Timestamp]:
    """Simulation window for the season harvested in ``harvest_year``.

    The window opens ``spinup_months`` before sowing and runs ``sim_years``
    from there:

    * The **spin-up** carries no crop — `Lintul5Model` latches sowing on
      ``doy >= site.idpl`` — so its only job is to let the water balance
      relax out of the ``soilwater_init`` profile before the crop needs it.
    * The **remainder** has to be long enough for the crop to finish. It is
      deliberately generous rather than cut at a nominal harvest date: the
      model self-terminates at ``DVS = 2`` (every growth rate is gated on
      ``dvs < 2``), so days past maturity change no reported quantity, whereas
      a window that ends early truncates a late season in a cold year.

    Windows of consecutive harvest years therefore overlap by a year. That is
    intended — each season is an independent run, not a continuous rotation.
    """
    start = sowing_date(harvest_year, sowing_doy) - pd.DateOffset(months=spinup_months)
    return start, start + pd.DateOffset(years=sim_years) - pd.Timedelta(days=1)


def load_weather_seasons(
    data_dir: Path, simplace_id: int, years: list[int], sowing_doy: int
) -> dict[int, np.ndarray]:
    """Read one cell's weather file once and slice out every requested season.

    A gzip stream has no random access, so the whole 46-year file is decoded
    whichever season is wanted; decoding it once and slicing all years out of
    it is the difference between one pass and ``len(years)`` passes.

    A harvest year is dropped when the file cannot cover its window's crop
    phase — the record starts after sowing, or ends less than
    ``MIN_DAYS_AFTER_SOWING`` past it. Without that check the first and last
    requested years would run on a silently truncated window: a front-truncated
    one misses its own sowing entirely and the latch fires on the *next*
    autumn, reporting a season one year off.

    Returns:
        ``{harvest_year: [T, 8]}`` arrays in `WeatherDriver` channel order
        (doy, davtmp, tmin, tmax, irrad, rain, vp, wind).
    """
    raw = pd.read_csv(
        weather_path(data_dir, simplace_id),
        sep="\t",
        parse_dates=["Date"],
        usecols=["Date", "Precipitation", "TempMin", "TempMean", "TempMax",
                 "Radiation", "Windspeed", "RelHumCalc"],
        na_values=[SENTINEL, -99.0],
    ).set_index("Date")

    # Exports made before `sfcwind` reached the weather exporter carry -99.9 in
    # every Windspeed row, which read_csv turned into NaN above. Fall back to
    # the assumed constant only then, so a re-exported file is picked up with
    # no flag to set.
    has_wind = raw["Windspeed"].notna().any()

    out: dict[int, np.ndarray] = {}
    for year in years:
        start, end = season_window(year, sowing_doy)
        block = raw.loc[start:end]
        sowing = sowing_date(year, sowing_doy)
        if block.empty or block.index[0] > sowing or (
            block.index[-1] - sowing
        ).days < MIN_DAYS_AFTER_SOWING:
            continue
        tmin, tmax = block["TempMin"].to_numpy(), block["TempMax"].to_numpy()
        frame = pd.DataFrame(
            {
                "doy": block.index.dayofyear.to_numpy(float),
                "davtmp": (tmin + tmax) / 2.0,
                "tmin": tmin,
                "tmax": tmax,
                "irrad": block["Radiation"].to_numpy() * W_PER_M2_TO_MJ,
                "rain": np.nan_to_num(block["Precipitation"].to_numpy(), nan=0.0),
                "vp": saturation_vp(block["TempMean"].to_numpy())
                * block["RelHumCalc"].to_numpy() / 100.0,
                "wind": (block["Windspeed"].to_numpy() if has_wind
                         else ASSUMPTIONS["wind_m_s"]),
            }
        ).interpolate(limit_direction="both")
        out[year] = frame.to_numpy(dtype=np.float32)
    return out


def load_weather_block(
    data_dir: Path, ids: np.ndarray, years: list[int], sowing_doy: int, workers: int
) -> dict[int, tuple[np.ndarray, int]]:
    """Weather for a batch of cells, per year, as ``{year: ([B, T, 8], start_doy)}``.

    Reads are I/O-bound on gzip decompression, which releases the GIL, so a
    thread pool scales here without the pickling cost of processes.

    Seasons are trimmed to the shortest series in the batch so every row of the
    returned array shares one time axis, and ``start_doy`` is read back off that
    axis (channel 0 is the day-of-year) rather than from the nominal window, so
    a window the record truncates at the front still hands `Lintul5Model` the
    day its sowing latch actually opens on.
    """
    with ThreadPoolExecutor(max_workers=workers) as pool:
        per_cell = list(
            pool.map(
                lambda sid: load_weather_seasons(data_dir, sid, years, sowing_doy), ids
            )
        )

    blocks: dict[int, tuple[np.ndarray, int]] = {}
    for year in years:
        arrays = [c[year] for c in per_cell if year in c]
        if len(arrays) != len(ids):
            logger.warning(
                "year %d: only %d/%d cells have a usable window; skipping the year",
                year, len(arrays), len(ids),
            )
            continue
        # Trimming equalises the tails; a batch whose cells start on different
        # days cannot share one time axis at all, and one start_doy would be
        # wrong for all but the first row.
        start_doys = {int(a[0, 0]) for a in arrays}
        if len(start_doys) != 1:
            logger.warning(
                "year %d: cells disagree on the first day (%s); skipping the year",
                year, sorted(start_doys),
            )
            continue
        n_days = min(a.shape[0] for a in arrays)
        blocks[year] = (np.stack([a[:n_days] for a in arrays]), start_doys.pop())
    return blocks


# --------------------------------------------------------------------------- #
# Soil & site
# --------------------------------------------------------------------------- #


def overlap_thickness(top_m: float, bottom_m: float) -> np.ndarray:
    """Thickness [m] of each soil layer inside the window ``[top_m, bottom_m]``."""
    return np.clip(
        np.minimum(LAYER_BOTTOMS_M, bottom_m) - np.maximum(LAYER_TOPS_M, top_m),
        0.0, None,
    )


def _layer_values(frame: pd.DataFrame, stem: str) -> np.ndarray:
    """``[n_cells, 6]`` array of ``<stem>_1 .. <stem>_6``."""
    return frame[[f"{stem}_{i}" for i in range(1, N_LAYERS + 1)]].to_numpy(float)


def depth_mean(frame: pd.DataFrame, stem: str, top_m: float, bottom_m: float) -> np.ndarray:
    """Thickness-weighted mean of a layered property over a depth window."""
    w = overlap_thickness(top_m, bottom_m)
    return _layer_values(frame, stem) @ w / w.sum()


def depth_sum(frame: pd.DataFrame, stem: str, top_m: float, bottom_m: float) -> np.ndarray:
    """Depth-integrated stock: per-layer values are already per-layer totals."""
    w = overlap_thickness(top_m, bottom_m) / (LAYER_BOTTOMS_M - LAYER_TOPS_M)
    return _layer_values(frame, stem) @ w


def build_soil_params(
    soil_df: pd.DataFrame,
    ids: np.ndarray,
    rootzone_m: float,
    profile_bottom_m: float,
    dtype=torch.float32,
) -> SoilParameters:
    """Collapse the 6-layer SIMPLACE profile into LINTUL-5 bucket parameters."""
    f = soil_df.loc[ids]
    root, deep = (0.0, rootzone_m), (rootzone_m, profile_bottom_m)

    wcwp = depth_mean(f, "soilwater_wp", *root)
    wcfc = depth_mean(f, "soilwater_fc", *root)
    wcst = depth_mean(f, "soilwater_sat", *root)
    wcad = np.minimum(depth_mean(f, "soilwater_res", *root), wcwp * 0.999)
    wci = np.clip(depth_mean(f, "soilwater_init", *root), wcwp, wcst)
    wci_lower = np.clip(
        depth_mean(f, "soilwater_init", *deep),
        depth_mean(f, "soilwater_wp", *deep),
        depth_mean(f, "soilwater_sat", *deep),
    )

    # Mineral N over the rooting zone: kg N/ha -> g N/m2.
    nminti = (
        depth_sum(f, "ammonium", *root) + depth_sum(f, "nitrate", *root)
    ) / 10.0 * ASSUMPTIONS["nminti_scale"]

    # Mineralisable organic N from SOC: g C/kg x kg/dm3 -> g C/dm3, then
    # x 100 dm3/m2 per 0.1 m of depth -> g C/m2, then / (C:N).
    c_stock = (
        _layer_values(f, "carbon")
        * _layer_values(f, "bulkdensity")
        * (overlap_thickness(*root) * 10.0)
        * 100.0
    ).sum(axis=1)
    nmini = np.clip(
        c_stock / ASSUMPTIONS["cn_ratio"] * ASSUMPTIONS["mineralisable_frac"],
        1.0, ASSUMPTIONS["nmini_cap_g_m2"],
    )

    n = len(ids)
    const = lambda v: torch.full((n,), float(v), dtype=dtype)
    t = lambda a: torch.as_tensor(np.asarray(a, float), dtype=dtype)
    defaults = SoilParameters()

    return SoilParameters(
        wcad=t(wcad), wcwp=t(wcwp), wcfc=t(wcfc), wcst=t(wcst),
        wci=t(wci), wci_lower=t(wci_lower),
        crairc=const(ASSUMPTIONS["crairc"]),
        ksub=const(ASSUMPTIONS["ksub"]),
        runfr=const(ASSUMPTIONS["runfr"]),
        cfev=const(ASSUMPTIONS["cfev"]),
        rdmso=const(rootzone_m),
        irri=torch.zeros(n, dtype=dtype),
        rtnmins=const(ASSUMPTIONS["rtnmins"]),
        nmini=t(nmini),
        nminti=t(nminti),
        # No European P/K data exists — torchcrop defaults, so iopt=4 is not
        # data-driven. See the notebook's gap table.
        pmini=const(defaults.pmini.item()),
        kmini=const(defaults.kmini.item()),
    )


def build_site_params(
    ids: np.ndarray, year: int, sowing_doy: int, dtype=torch.float32
) -> SiteParameters:
    """Latitude from the grid; altitude and CO2 from ASSUMPTIONS / the year.

    ``idpl`` must be the same ``sowing_doy`` `season_window` was built from:
    the window places the spin-up ahead of that day and the model's latch opens
    on it.
    """
    _, lat = id_to_lonlat(ids)
    n = len(ids)
    const = lambda v: torch.full((n,), float(v), dtype=dtype)
    return SiteParameters(
        latitude=torch.as_tensor(lat, dtype=dtype),
        altitude=const(ASSUMPTIONS["altitude_m"]),
        co2=const(co2_for_year(year)),
        plant_at_sowing=torch.ones(n, dtype=dtype),
        idpl=const(sowing_doy),
    )


# --------------------------------------------------------------------------- #
# Management
# --------------------------------------------------------------------------- #


def load_composition(path: Path) -> pd.DataFrame:
    """Fertilizer type -> elemental N, P, K content [g element / g product]."""
    root = ET.parse(path).getroot()
    rows = {}
    for fert in root.findall("fertilizer"):
        params = {p.get("id"): p.text.strip() for p in fert.findall("parameter")}
        rows[params["Fertilizertype"]] = {
            "N": float(params["NitrateAndAmmonium"]),
            "P": float(params["Phosphorus"]),
            "K": float(params["Potassium"]),
        }
    return pd.DataFrame(rows).T


def build_fertilizer_plans(
    fert_df: pd.DataFrame, composition: pd.DataFrame
) -> dict[int, np.ndarray]:
    """Per-cell event arrays ``[n_events, 4]`` of (DVS, N, P, K) in g m-2.

    Sorted by DVS so `fertilizer_from_dvs` can place them with one
    ``searchsorted`` against the (monotone) development-stage trajectory.
    """
    events = fert_df.join(composition, on="vType")
    unknown = events.loc[events["N"].isna(), "vType"].unique()
    if unknown.size:
        raise KeyError(f"products absent from the composition file: {list(unknown)}")

    for element in ("N", "P", "K"):
        events[f"{element}_g_m2"] = events["Amount"] * events[element]

    events = events.sort_values(["location", "DVS"])
    cols = ["DVS", "N_g_m2", "P_g_m2", "K_g_m2"]
    return {
        int(loc): grp[cols].to_numpy(np.float64)
        for loc, grp in events.groupby("location", sort=False)
    }


def fertilizer_from_dvs(
    ids: np.ndarray,
    dvs: torch.Tensor,
    plans: dict[int, np.ndarray],
    dtype=torch.float32,
) -> torch.Tensor:
    """Place each cell's DVS-keyed doses onto calendar days -> ``[B, T, 3]``.

    Nothing on disk maps a DVS to a date, and the mapping is weather-dependent,
    so it comes from the model's own first (unfertilised) pass. Phenology is
    temperature-driven, so the unfertilised trajectory is the right schedule.

    Args:
        ids: ``SimplaceID`` per batch row.
        dvs: ``[B, T+1]`` development-stage trajectory; the leading entry is the
            pre-sowing state and is dropped.
        plans: output of `build_fertilizer_plans`.

    Returns:
        Daily N, P, K in g element m-2 d-1, last axis ordered (N, P, K).
    """
    track = dvs.detach().cpu().numpy()[:, 1:]
    n_batch, n_days = track.shape
    out = np.zeros((n_batch, n_days, 3))

    for b, sid in enumerate(ids):
        plan = plans.get(int(sid))
        if plan is None:                            # no plan -> unfertilised
            continue
        # DVS accumulates monotonically, so the first day at or past each
        # event's stage is a binary search rather than a scan.
        day = np.searchsorted(track[b], plan[:, 0], side="left")
        reached = day < n_days
        if not reached.any():
            continue
        np.add.at(out[b], day[reached], plan[reached, 1:])
    return torch.as_tensor(out, dtype=dtype)


# --------------------------------------------------------------------------- #
# Simulation
# --------------------------------------------------------------------------- #


def _state_trajectory(output, name: str) -> torch.Tensor:
    """``[B, T+1]`` trajectory of a `ModelState` field."""
    return torch.stack([getattr(s, name) for s in output.states], dim=1)


def _diagnostic_trajectory(output, name: str) -> torch.Tensor:
    """``[B, T]`` trajectory of a `DiagnosticState` field."""
    return torch.stack([getattr(d, name) for d in output.diagnostics], dim=1)


def run_batch(
    ids: np.ndarray,
    weather_np: np.ndarray,
    year: int,
    start_doy: int,
    sowing_doy: int,
    soil_df: pd.DataFrame,
    plans: dict[int, np.ndarray],
    irrigated: pd.Series,
    crop: str,
    iopt: int,
    rootzone_m: float,
    profile_bottom_m: float,
    device: torch.device,
) -> pd.DataFrame:
    """Two-pass run of one batch of cells for one season; per-cell summaries.

    Pass 1 is unfertilised and exists only to date the DVS-keyed fertilizer
    schedule; pass 2 is the run that is reported.

    The window (`season_window`) runs well past maturity, which costs integrator
    steps but changes no summary below: every crop rate is gated on ``dvs < 2``,
    so yield, biomass and LAI freeze at maturity, and the stress means are taken
    over ``0 < DVS < 2`` — which also excludes the pre-sowing spin-up.
    """
    crop_params = CropParameters(crop_name=crop)
    crop_params.iopt = torch.tensor(float(iopt))

    soil_params = build_soil_params(
        soil_df, ids, rootzone_m, profile_bottom_m
    )
    soil_params.irri = torch.as_tensor(
        irrigated.reindex(ids).fillna(0).to_numpy(float), dtype=torch.float32
    )
    site_params = build_site_params(ids, year, sowing_doy)
    weather = WeatherDriver(torch.as_tensor(weather_np, dtype=torch.float32))

    crop_params = crop_params.to(device=device)
    soil_params = soil_params.to(device=device)
    site_params = site_params.to(device=device)
    weather = weather.to(device=device)

    model = Lintul5Model(crop_params, soil_params, site_params).eval().to(device)

    with torch.no_grad():
        pass1 = model(weather, start_doy=start_doy)
        fertilizer = fertilizer_from_dvs(ids, pass1.dvs, plans).to(device)
        del pass1
        out = model(weather, start_doy=start_doy, fertilizer=fertilizer)

    dvs = out.dvs[:, 1:]
    applied = fertilizer.sum(dim=1).cpu().numpy()
    # Stress indices are only meaningful while a canopy exists; average them
    # over the days the crop is actually growing (0 < DVS < 2).
    growing = ((dvs > 0.0) & (dvs < 2.0)).float()
    n_growing = growing.sum(dim=1).clamp(min=1.0)

    def growing_mean(name: str) -> np.ndarray:
        return ((_diagnostic_trajectory(out, name) * growing).sum(dim=1)
                / n_growing).cpu().numpy()

    lon, lat = id_to_lonlat(ids)
    # Maturity is reported in days since sowing, not since the start of the
    # window — the window opens on a spin-up the crop is not in. The sowing day
    # is rebuilt from the engine's own wrapped day-of-year (`engine.run`), which
    # is what the latch compares against, rather than the weather's doy channel.
    engine_doy = (start_doy - 1 + np.arange(dvs.shape[1])) % 365 + 1
    sown = engine_doy >= sowing_doy
    sow_index = int(sown.argmax()) if sown.any() else 0
    mature = (dvs >= 2.0)
    days_to_maturity = np.where(
        mature.any(dim=1).cpu().numpy(),
        mature.float().argmax(dim=1).cpu().numpy() - sow_index,
        np.nan,
    )

    frame = pd.DataFrame(
        {
            "SimplaceID": ids.astype(np.int64),
            "year": np.int16(year),
            "lon": lon.astype(np.float32),
            "lat": lat.astype(np.float32),
            "yield_g_m2": out.yield_.cpu().numpy(),
            "adjusted_yield_g_m2": out.adjusted_yield.cpu().numpy(),
            "biomass_g_m2": out.biomass[:, -1].cpu().numpy(),
            "max_lai": out.lai.max(dim=1).values.cpu().numpy(),
            "final_dvs": out.dvs[:, -1].cpu().numpy(),
            "days_to_maturity": days_to_maturity,
            "n_applied_g_m2": applied[:, 0],
            "p_applied_g_m2": applied[:, 1],
            "k_applied_g_m2": applied[:, 2],
            "tranrf_mean": growing_mean("tranrf"),
            "nni_mean": growing_mean("nni"),
            "heat_stress_factor": out.heat_stress_factor.cpu().numpy(),
            "irri": soil_params.irri.cpu().numpy().astype(np.int8),
        }
    )
    del out, fertilizer, model
    return frame.astype({c: np.float32 for c in frame.columns
                         if frame[c].dtype == np.float64})


# --------------------------------------------------------------------------- #
# Shard driver
# --------------------------------------------------------------------------- #


def resolve_cells(data_dir: Path) -> tuple[np.ndarray, pd.DataFrame, pd.DataFrame]:
    """Cells runnable end-to-end, plus the soil and fertilizer tables.

    Runnable = has weather **and** soil **and** a fertilizer plan, so no cell
    silently falls back to a default schedule.
    """
    soil_df = pd.read_csv(data_dir / "soil" / "soil.csv").set_index("location")
    fert_df = pd.read_csv(data_dir / "management" / "fertilizer_winter_wheat.csv")
    ids = np.intersect1d(
        np.intersect1d(weather_ids(data_dir), soil_df.index.to_numpy()),
        fert_df["location"].unique(),
    )
    return ids, soil_df, fert_df


def shard_cells(ids: np.ndarray, shard: int, n_shards: int) -> np.ndarray:
    """Round-robin slice of ``ids`` for one task.

    Round-robin, not contiguous: ``SimplaceID`` runs north-to-south, so a
    contiguous split would hand one task all of Scandinavia and another all of
    Iberia — very different weather-file sizes and very different wall times.
    """
    return ids[shard::n_shards]


def run_shard(args: argparse.Namespace) -> Path:
    """Run one shard over every requested year and write its Parquet output."""
    data_dir = Path(args.data_dir)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / f"torchcrop_shard_{args.shard:03d}.parquet"

    if out_path.exists() and not args.overwrite:
        logger.info("shard %d already done: %s", args.shard, out_path)
        return out_path

    device = torch.device(args.device)
    logger.info("resolving the runnable cell set")
    all_ids, soil_df, fert_df = resolve_cells(data_dir)
    ids = shard_cells(all_ids, args.shard, args.n_shards)
    if args.max_cells and len(ids) > args.max_cells:
        # Even subsample, not the head: SimplaceID runs north-to-south, so
        # truncating would give a smoke test a narrow Scandinavian band — and
        # the map layout is driven by the data's bounding box, so a biased
        # subset silently exercises the wrong geometry.
        ids = ids[:: int(np.ceil(len(ids) / args.max_cells))][: args.max_cells]

    plans = build_fertilizer_plans(fert_df, load_composition(Path(args.composition)))
    irrigated = fert_df.groupby("location")["vIRR"].max()
    del fert_df

    years = list(range(args.start_year, args.end_year + 1))
    n_batches = int(np.ceil(len(ids) / args.batch_size))
    sowing = sowing_date(years[0], args.sowing_doy)
    window = season_window(years[0], args.sowing_doy)
    if window[0].dayofyear > sowing.dayofyear:
        # The latch is `doy >= idpl` on a day-of-year, so a spin-up that crosses
        # New Year before sowing already satisfies it on day 0: the crop would
        # go in at the start of the window and get no spin-up at all.
        logger.warning(
            "sowing DOY %d leaves a spin-up that crosses New Year (window opens "
            "on DOY %d); the model will sow on the first day of the window",
            args.sowing_doy, window[0].dayofyear,
        )
    logger.info(
        "shard %d/%d: %d cells (of %d), %d years, %d batches of %d, device=%s",
        args.shard, args.n_shards, len(ids), len(all_ids), len(years),
        n_batches, args.batch_size, device,
    )
    logger.info(
        "window: %s .. %s (%d days), sowing DOY %d on %s, %d months of spin-up",
        window[0].date(), window[1].date(), (window[1] - window[0]).days + 1,
        args.sowing_doy, sowing.date(), SPINUP_MONTHS,
    )

    results: list[pd.DataFrame] = []
    t_start = time.time()
    for b, start in enumerate(range(0, len(ids), args.batch_size)):
        batch = ids[start : start + args.batch_size]
        t0 = time.time()
        blocks = load_weather_block(
            data_dir, batch, years, args.sowing_doy, args.io_workers
        )
        t_io = time.time() - t0

        for year, (weather_np, start_doy) in blocks.items():
            results.append(
                run_batch(
                    batch, weather_np, year, start_doy, args.sowing_doy, soil_df,
                    plans, irrigated, args.crop, args.iopt, args.rootzone_m,
                    args.profile_bottom_m, device,
                )
            )
        elapsed = time.time() - t_start
        logger.info(
            "batch %d/%d (%d cells): io %.1fs, total %.1fs, elapsed %.1f min, "
            "eta %.1f min",
            b + 1, n_batches, len(batch), t_io, time.time() - t0, elapsed / 60,
            elapsed / (b + 1) * (n_batches - b - 1) / 60,
        )

    if not results:
        raise RuntimeError(f"shard {args.shard} produced no results")

    frame = pd.concat(results, ignore_index=True)
    frame.to_parquet(out_path, index=False, compression="zstd")
    logger.info(
        "shard %d wrote %d rows to %s in %.1f min",
        args.shard, len(frame), out_path, (time.time() - t_start) / 60,
    )
    return out_path


def build_parser() -> argparse.ArgumentParser:
    """CLI for one array task."""
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--shard", type=int, required=True, help="0-based task index")
    p.add_argument("--n-shards", type=int, default=10)
    p.add_argument("--data-dir", default=str(DEFAULT_DATA_DIR))
    p.add_argument("--out-dir", required=True)
    p.add_argument("--composition", default=str(DEFAULT_COMPOSITION_XML))
    p.add_argument("--start-year", type=int, default=2000, help="first harvest year")
    p.add_argument("--end-year", type=int, default=2024, help="last harvest year")
    p.add_argument("--crop", default="wheat")
    p.add_argument("--sowing-doy", type=int, default=ASSUMPTIONS["sowing_doy"],
                   help="day-of-year of sowing. Sets both the model's sowing "
                        "latch and the simulation window, which opens "
                        f"{SPINUP_MONTHS} months before it")
    p.add_argument("--iopt", type=int, default=3, choices=[1, 2, 3, 4],
                   help="1 potential | 2 water | 3 water+N | 4 water+NPK")
    p.add_argument("--batch-size", type=int, default=1024)
    p.add_argument("--io-workers", type=int, default=8)
    p.add_argument("--rootzone-m", type=float, default=1.0)
    p.add_argument("--profile-bottom-m", type=float, default=2.0)
    p.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    p.add_argument("--torch-threads", type=int, default=4,
                   help="intra-op threads. The model works on [B]-shaped "
                        "tensors, which are too small to repay many threads — "
                        "the cores are better spent on --io-workers")
    p.add_argument("--max-cells", type=int, default=0,
                   help="cap cells per shard (smoke tests)")
    p.add_argument("--overwrite", action="store_true")
    return p


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)-7s %(message)s",
        datefmt="%H:%M:%S",
        stream=sys.stdout,
    )
    if args.torch_threads > 0:
        torch.set_num_threads(args.torch_threads)
    run_shard(args)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
