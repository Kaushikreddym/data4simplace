"""Run torchcrop (LINTUL-5) over the export, one shard per job.

Library and ``python -m`` entry point behind ``submit/torchcrop_array.sh``.
Everything about *reading* the export lives in
:mod:`cropmodelling4eu.export`; this module is only the modelling.

The work splits on three axes:

* **Shards** (``--shard`` / ``--n-shards``) — one SLURM array task each. Cells
  are dealt round-robin so every shard spans the whole domain and the tasks
  finish together; a contiguous split would give one task all of Iberia and
  another all of Scandinavia, with very different weather-file sizes.
* **Sowing groups** — within a shard, cells are grouped by harvest year (and,
  without a simulated-sowing table, by calendar sowing day-of-year too, since
  that stays fixed across seasons — see :func:`group_by_sowing`). A batch
  shares one calendar time axis, anchored to the *earliest* sowing
  day-of-year the batch contains, but ``idpl`` is per-cell: the model already
  latches sowing element-wise (``doy >= site.idpl``), so cells sowing on
  different days still share a batch rather than fragmenting one-by-one; see
  :func:`sowing_plan`.
* **Batches** (``--batch-size``) — within a group, cells run through the model
  in chunks. LINTUL-5 is a Python loop over days, so a batch of 2 000 cells
  costs barely more wall time than a batch of 20; the limit is that
  ``ModelOutput`` retains every per-day state, so peak memory scales with
  ``batch_size × window length``.

Each shard writes one Parquet of per-(cell, season) summaries and never touches
another shard's output, so a failed task is re-runnable on its own.
"""

from __future__ import annotations

import argparse
import copy
import logging
import sys
import time
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

from cropmodelling4eu.config import RunConfig, load_config
from cropmodelling4eu.export import ExportBundle, resolve_export
from cropmodelling4eu.export.cells import id_to_lonlat
from cropmodelling4eu.export.soil import SoilProfiles
from cropmodelling4eu.export.weather import load_season_block, season_window, sowing_date
from cropmodelling4eu.torchcrop.workspace import load_crop_parameters

logger = logging.getLogger("cropmodelling4eu.torchcrop.run")

#: Fills for torchcrop inputs the export does not carry. Every entry is a
#: modelling assumption, not data. Sowing date, altitude and CO2 used to be
#: here too; they now come from ``site.csv`` when the export has one.
ASSUMPTIONS = {
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

#: Simulation window length per harvest year, in years from the window's start.
SIM_YEARS = 2


# --------------------------------------------------------------------------- #
# Sowing groups
# --------------------------------------------------------------------------- #


def group_by_sowing(ids: np.ndarray, sowing_doy: np.ndarray) -> dict[int, np.ndarray]:
    """Split a shard's cells by (calendar) sowing day-of-year.

    Used only for the site-calendar path, where a cell's sowing day-of-year is
    fixed across every season, so grouping by it also groups by nothing else —
    every group already shares one calendar time axis. Grouping costs nothing
    in practice: a 0.5 degree calendar takes a handful of distinct values over
    any one shard, so the groups stay large enough to keep the batches full.

    See :func:`sowing_plan` for the simulated-sowing-table path, where a
    cell's date varies year to year and grouping by day-of-year the same way
    would fragment every batch down to one or two cells.

    Returns:
        ``{sowing_doy: ids}``, ascending by day-of-year.
    """
    doys = np.asarray(sowing_doy, dtype=int)
    return {
        int(doy): np.asarray(ids)[doys == doy] for doy in np.unique(doys)
    }


def sowing_plan(
    ids: np.ndarray,
    years: list[int],
    site_sowing_doy: np.ndarray,
    sowing: pd.DataFrame | None = None,
) -> list[tuple[list[int], np.ndarray, np.ndarray]]:
    """``[(years, ids, sowing_doy)]`` — the units one batch may be built from.

    ``sowing_doy`` is always an array aligned with ``ids``, even from the
    site-calendar branch (one repeated value per group), so every downstream
    caller treats sowing day-of-year as per-cell with no scalar special case.

    Without ``sowing`` a group is :func:`group_by_sowing`'s cells, spanning
    every season — the calendar gives a cell one date for every year. With it
    — a table of another model's *simulated* dates, which move year to year —
    grouping is by **year alone**. SIMPLACE's rule-based sowing rarely agrees
    across cells, so grouping by ``(year, sowing_doy)`` the way the site
    calendar does would fragment a batch down to one or two cells and throw
    away the model's batching speedup (LINTUL-5's Python day-loop pays a
    per-batch dispatch cost that only amortises at scale — see the
    ``batch_size`` note in ``config.yaml``). A batch built this way still
    shares one calendar time axis — :func:`run_batch` and :func:`daily_batch`
    anchor it to the *earliest* sowing day-of-year the batch contains, wide
    enough to cover every member's own spin-up — but each cell latches on its
    own ``idpl`` inside that shared window, since ``Lintul5Model`` already
    compares ``doy >= site.idpl`` element-wise rather than batch-wide.

    A (cell, season) the table does not cover is dropped. Falling back to the
    site calendar there would mix two sowing conventions inside one output
    with nothing in the file to say which rows came from which.
    """
    if sowing is None:
        return [
            (list(years), group, np.full(group.size, doy, dtype=np.int64))
            for doy, group in group_by_sowing(ids, site_sowing_doy).items()
        ]

    table = sowing[sowing["SimplaceID"].isin(ids) & sowing["year"].isin(list(years))]
    if table.empty:
        raise ValueError(
            "the sowing table covers none of this run's cells and seasons"
        )
    missing = len(ids) * len(years) - len(table)
    if missing:
        logger.warning(
            "%d of %d (cell, season) pairs have no simulated sowing date and "
            "are dropped", missing, len(ids) * len(years),
        )
    return [
        (
            [int(year)],
            rows["SimplaceID"].to_numpy(np.int64),
            rows["sowing_doy"].to_numpy(np.int64),
        )
        for year, rows in table.sort_values("SimplaceID").groupby("year")
    ]


# --------------------------------------------------------------------------- #
# Soil & site
# --------------------------------------------------------------------------- #


def build_soil_params(
    soil: SoilProfiles,
    rootzone_m: float,
    profile_bottom_m: float,
    dtype=torch.float32,
) -> SoilParameters:
    """Collapse a layered SIMPLACE profile into LINTUL-5 bucket parameters.

    ``soil`` must already be restricted to the batch's cells, in their order.
    Depth windows are integrated over the profile's **own** layer geometry, read
    from the export rather than assumed, so a file carrying SoilGrids' native
    horizons is handled as correctly as one on the SIMPLACE layers.
    """
    root, deep = (0.0, rootzone_m), (rootzone_m, profile_bottom_m)

    wcwp = soil.depth_mean("soilwater_wp", *root)
    wcfc = soil.depth_mean("soilwater_fc", *root)
    wcst = soil.depth_mean("soilwater_sat", *root)
    wcad = np.minimum(soil.depth_mean("soilwater_res", *root), wcwp * 0.999)

    # Air-filled headroom above field capacity, capped at half of what the
    # soil actually has (wcst - wcfc) rather than ASSUMPTIONS["crairc"]
    # outright. wcst here is SoilGrids' wv0010 "drained upper limit", not
    # true porosity (see soil.py), so it sits far closer to wcfc than a real
    # saturation point would: for 28 of 30 German smoke cells wcst - wcfc was
    # already below the fixed 0.07 default, which put smair = wcst - crairc
    # *below* field capacity -- ordinary, undrained-but-not-waterlogged soil
    # moisture then read as oxygen shortage (RWET < 1) from day one. Capping
    # at half the headroom keeps smair strictly above wcfc for every soil.
    crairc = np.minimum(ASSUMPTIONS["crairc"], 0.5 * (wcst - wcfc))
    wci = np.clip(soil.depth_mean("soilwater_init", *root), wcwp, wcst)
    wci_lower = np.clip(
        soil.depth_mean("soilwater_init", *deep),
        soil.depth_mean("soilwater_wp", *deep),
        soil.depth_mean("soilwater_sat", *deep),
    )

    # Mineral N over the rooting zone: kg N/ha -> g N/m2.
    nminti = (
        (soil.depth_sum("ammonium", *root) + soil.depth_sum("nitrate", *root))
        / 10.0
        * ASSUMPTIONS["nminti_scale"]
    )

    # Mineralisable organic N from SOC: g C/kg x kg/dm3 -> g C/dm3, then
    # x 100 dm3/m2 per 0.1 m of depth -> g C/m2, then / (C:N).
    c_stock = (
        soil.values["carbon"]
        * soil.values["bulkdensity"]
        * (soil.overlap_thickness(*root) * 10.0)
        * 100.0
    ).sum(axis=1)
    nmini = np.clip(
        c_stock / ASSUMPTIONS["cn_ratio"] * ASSUMPTIONS["mineralisable_frac"],
        1.0,
        ASSUMPTIONS["nmini_cap_g_m2"],
    )

    n = soil.ids.size
    const = lambda v: torch.full((n,), float(v), dtype=dtype)  # noqa: E731
    t = lambda a: torch.as_tensor(np.asarray(a, float), dtype=dtype)  # noqa: E731
    defaults = SoilParameters()

    return SoilParameters(
        wcad=t(wcad), wcwp=t(wcwp), wcfc=t(wcfc), wcst=t(wcst),
        wci=t(wci), wci_lower=t(wci_lower),
        crairc=t(crairc),
        ksub=const(ASSUMPTIONS["ksub"]),
        runfr=const(ASSUMPTIONS["runfr"]),
        cfev=const(ASSUMPTIONS["cfev"]),
        rdmso=const(rootzone_m),
        irri=torch.ones(n, dtype=dtype),
        rtnmins=const(ASSUMPTIONS["rtnmins"]),
        nmini=t(nmini),
        nminti=t(nminti),
        # No European soil P/K product exists, so these stay at torchcrop's
        # defaults and iopt=4 is not data-driven.
        pmini=const(defaults.pmini.item()),
        kmini=const(defaults.kmini.item()),
    )


def build_site_params(
    ids: np.ndarray,
    year: int,
    sowing_doy: np.ndarray | int,
    bundle: ExportBundle,
    dtype=torch.float32,
) -> SiteParameters:
    """Latitude from the grid, altitude and CO2 from the export.

    ``idpl`` is per-cell (aligned with ``ids``) -- a scalar ``sowing_doy`` is
    broadcast, so a single shared date still works. The window only has to
    open early enough for the *earliest* sower in the batch, since
    ``Lintul5Model`` compares ``doy >= site.idpl`` element-wise and each cell
    latches on its own value inside the shared window.
    """
    _, lat = id_to_lonlat(ids, bundle.config.grid)
    altitude = bundle.site.altitude_m(ids)
    co2 = float(bundle.co2.get(year, bundle.co2.iloc[-1]))
    n = ids.size
    idpl = np.broadcast_to(np.asarray(sowing_doy, dtype=float), (n,))
    return SiteParameters(
        latitude=torch.as_tensor(lat, dtype=dtype),
        altitude=torch.as_tensor(altitude, dtype=dtype),
        co2=torch.full((n,), co2, dtype=dtype),
        plant_at_sowing=torch.ones(n, dtype=dtype),
        idpl=torch.as_tensor(idpl.copy(), dtype=dtype),
    )


# --------------------------------------------------------------------------- #
# Fertilizer
# --------------------------------------------------------------------------- #


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
        dvs: ``[B, T+1]`` development-stage trajectory; the leading entry is
            the pre-sowing state and is dropped.
        plans: ``{SimplaceID: [n_events, 4]}`` of ``(DVS, N, P, K)`` in g/m².

    Returns:
        Daily N, P, K in g element m-2 d-1, last axis ordered (N, P, K).
    """
    track = dvs.detach().cpu().numpy()[:, 1:]
    n_batch, n_days = track.shape
    out = np.zeros((n_batch, n_days, 3))

    for b, simplace_id in enumerate(ids):
        plan = plans.get(int(simplace_id))
        if plan is None:  # no plan -> unfertilised
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


def _diagnostic_trajectory(output, name: str) -> torch.Tensor:
    """``[B, T]`` trajectory of a ``DiagnosticState`` field."""
    return torch.stack([getattr(d, name) for d in output.diagnostics], dim=1)


def _simulate(
    ids: np.ndarray,
    weather_np: np.ndarray,
    year: int,
    start_doy: int,
    sowing_doy: np.ndarray,
    bundle: ExportBundle,
    device: torch.device,
    crop_params: "CropParameters | None" = None,
):
    """The two-pass run itself: ``(output, fertilizer, irrigated)``.

    Pass 1 is unfertilised and exists only to date the DVS-keyed fertilizer
    schedule; pass 2 is the run that is reported. Both :func:`run_batch` and
    :func:`daily_batch` go through here, so a daily trajectory is a *view of
    the same simulation* the summary row came from rather than a second run
    that could differ from it.
    """
    settings = bundle.config.torchcrop
    # A caller-supplied set is copied, not mutated: the same object is reused
    # across every batch of a run, and iopt is written on it below.
    crop_params = (
        CropParameters(crop_name=bundle.config.season.crop)
        if crop_params is None else copy.deepcopy(crop_params)
    )
    crop_params.iopt = torch.tensor(float(settings.iopt))

    soil_params = build_soil_params(
        bundle.soil.select(ids), settings.rootzone_m, settings.profile_bottom_m
    )
    soil_params.irri = torch.as_tensor(
        bundle.irrigated.reindex(ids).fillna(0).to_numpy(float), dtype=torch.float32
    )
    if settings.iopt == 1:
        # Potential production: every cell is auto-irrigated (mode 1, see
        # _irrigation_demand), not just the ones the export classifies as
        # irrigated, so water never limits growth.
        soil_params.irri = torch.ones_like(soil_params.irri)
    site_params = build_site_params(ids, year, sowing_doy, bundle)
    weather = WeatherDriver(torch.as_tensor(weather_np, dtype=torch.float32))

    crop_params = crop_params.to(device=device)
    soil_params = soil_params.to(device=device)
    site_params = site_params.to(device=device)
    weather = weather.to(device=device)

    model = Lintul5Model(crop_params, soil_params, site_params).eval().to(device)

    with torch.no_grad():
        pass1 = model(weather, start_doy=start_doy)
        fertilizer = fertilizer_from_dvs(ids, pass1.dvs, bundle.plans).to(device)
        del pass1
        out = model(weather, start_doy=start_doy, fertilizer=fertilizer)

    irrigated = soil_params.irri.cpu().numpy().astype(np.int8)
    del model, soil_params, site_params, weather
    return out, fertilizer, irrigated


def _summary_frame(
    ids: np.ndarray,
    out,
    fertilizer: torch.Tensor,
    irrigated: np.ndarray,
    year: int,
    start_doy: int,
    sowing_doy: np.ndarray,
    bundle: ExportBundle,
) -> pd.DataFrame:
    """Per-cell summary of an already-simulated batch; see :func:`run_batch`.

    Split out of ``run_batch`` so :func:`run_shard` can build the summary and
    the daily trajectory (:func:`_daily_frame`) from one ``_simulate`` call
    instead of two — a daily trajectory is then genuinely *the same run* the
    summary row came from, and the model forward pass is not paid twice.
    """
    dvs = out.dvs[:, 1:]
    applied = fertilizer.sum(dim=1).cpu().numpy()
    # Stress indices are only meaningful while a canopy exists; average them
    # over the days the crop is actually growing (0 < DVS < 2).
    growing = ((dvs > 0.0) & (dvs < 2.0)).float()
    n_growing = growing.sum(dim=1).clamp(min=1.0)

    def growing_mean(name: str) -> np.ndarray:
        return (
            (_diagnostic_trajectory(out, name) * growing).sum(dim=1) / n_growing
        ).cpu().numpy()

    lon, lat = id_to_lonlat(ids, bundle.config.grid)
    # Maturity is reported in days since sowing, not since the start of the
    # window -- the window opens on a spin-up the crop is not in. The sowing day
    # is rebuilt from the engine's own wrapped day-of-year, which is what the
    # latch compares against, rather than from the weather's doy channel.
    # Per-cell: each row can cross its own sowing_doy on a different engine day.
    engine_doy = (start_doy - 1 + np.arange(dvs.shape[1])) % 365 + 1
    sown = engine_doy[None, :] >= sowing_doy[:, None]
    sow_index = np.where(sown.any(axis=1), sown.argmax(axis=1), 0)
    mature = dvs >= 2.0
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
            "sowing_doy": sowing_doy.astype(np.int16),
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
            "irri": irrigated,
        }
    )
    return frame.astype(
        {c: np.float32 for c in frame.columns if frame[c].dtype == np.float64}
    )


def run_batch(
    ids: np.ndarray,
    weather_np: np.ndarray,
    year: int,
    start_doy: int,
    sowing_doy: np.ndarray,
    bundle: ExportBundle,
    device: torch.device,
    crop_params: "CropParameters | None" = None,
) -> pd.DataFrame:
    """Per-cell summaries for one batch of cells and one season.

    ``sowing_doy`` is per-cell (aligned with ``ids``): a batch built from a
    simulated-sowing table shares a calendar window but not a sowing date, so
    "days since sowing" is computed row-by-row below rather than off one
    shared index.

    The window runs well past maturity, which costs integrator steps but
    changes no summary below: every crop rate is gated on ``dvs < 2``, so
    yield, biomass and LAI freeze at maturity, and the stress means are taken
    over ``0 < DVS < 2`` — which also excludes the pre-sowing spin-up.
    """
    sowing_doy = np.asarray(sowing_doy)
    out, fertilizer, irrigated = _simulate(
        ids, weather_np, year, start_doy, sowing_doy, bundle, device, crop_params
    )
    frame = _summary_frame(
        ids, out, fertilizer, irrigated, year, start_doy, sowing_doy, bundle
    )
    del out, fertilizer
    return frame


#: Daily series :func:`daily_batch` can return, and where each comes from. The
#: state trajectories carry one extra leading entry (the initial condition),
#: which is dropped so every series is indexed by the day that produced it.
DAILY_VARIABLES: dict[str, str] = {
    "LAI": "state",
    "DVS": "state",
    "AGB": "state",
    "TRANRF": "diagnostic",
    "NNI": "diagnostic",
}

#: ``ModelOutput`` attribute and unit scale for each ``"state"`` variable.
#: ``AGB`` is converted from torchcrop's native g/m2 to t/ha (x0.01) -- the
#: same factor SIMPLACE's own ``AGBiomass_t_ha`` rule applies to
#: ``Biomass.sTAGB`` -- so the two models' daily series need no further
#: conversion at the point they are compared.
_STATE_FIELD = {"LAI": "lai", "DVS": "dvs", "AGB": "biomass"}
_STATE_SCALE = {"LAI": 1.0, "DVS": 1.0, "AGB": 0.01}


def _daily_frame(
    ids: np.ndarray,
    out,
    year: int,
    start_doy: int,
    sowing_doy: np.ndarray,
    bundle: ExportBundle,
    variables: tuple[str, ...] = ("LAI", "AGB", "NNI", "TRANRF"),
) -> pd.DataFrame:
    """Per-**day** trajectories of an already-simulated batch; see :func:`daily_batch`.

    Same simulation as :func:`_summary_frame`; what differs is only what is
    kept. A summary cannot say *when* a crop was short of water or nitrogen,
    which is the whole question in a drought year — ``tranrf_mean`` of 0.6 is
    a season half-stressed throughout and a season shut down for six weeks in
    June, and those are different crops.

    Rows are trimmed to the **crop phase** (``0 < DVS < 2``), so the 6-month
    spin-up and the days after maturity — where a stress index is either
    undefined or frozen — never reach a plot.

    Returns
    -------
    DataFrame
        ``model, SimplaceID, year, date, doy, das, variable, value``, where
        ``das`` is days after each row's own sowing latch (``sowing_doy`` is
        per-cell; the shared calendar axis is anchored to the batch's
        earliest sower, see :func:`sowing_plan`).
    """
    unknown = set(variables) - set(DAILY_VARIABLES)
    if unknown:
        raise ValueError(f"unknown daily variable(s): {sorted(unknown)}")

    sowing_doy = np.asarray(sowing_doy)

    def trajectory(name: str) -> np.ndarray:
        if DAILY_VARIABLES[name] == "diagnostic":
            return _diagnostic_trajectory(out, name.lower()).cpu().numpy()
        field = getattr(out, _STATE_FIELD[name])[:, 1:] * _STATE_SCALE[name]
        return field.cpu().numpy()

    dvs = out.dvs[:, 1:].cpu().numpy()
    n_days = dvs.shape[1]
    # Step i is driven by weather row i and produces states[i + 1] and
    # diagnostics[i], so both describe the day `window start + i`. The window
    # start is the one load_seasons sliced on (anchored to the batch's
    # earliest sowing_doy, matching load_season_block), so this is the real
    # calendar axis -- leap days included, unlike the engine's wrapped
    # 365-day count.
    anchor_doy = int(sowing_doy.min())
    dates = pd.date_range(
        season_window(year, anchor_doy, bundle.config.torchcrop.spinup_months)[0],
        periods=n_days, freq="D",
    )
    # One sowing date per row, so "days after sowing" is a [B, T] matrix, not
    # a single axis shared by the whole batch.
    sown = np.array(
        [sowing_date(year, int(doy)) for doy in sowing_doy], dtype="datetime64[ns]"
    )
    das = (
        (dates.to_numpy()[None, :] - sown[:, None]) / np.timedelta64(1, "D")
    ).astype(np.int16)

    growing = (dvs > 0.0) & (dvs < 2.0)
    cell_index, day_index = np.nonzero(growing)
    if cell_index.size == 0:
        return pd.DataFrame(
            columns=["model", "SimplaceID", "year", "date", "doy", "das",
                     "variable", "value"]
        )

    base = pd.DataFrame(
        {
            "model": "torchcrop",
            "SimplaceID": ids.astype(np.int64)[cell_index],
            "year": np.int16(year),
            "date": dates.to_numpy()[day_index],
            "doy": dates.dayofyear.to_numpy()[day_index].astype(np.int16),
            "das": das[cell_index, day_index],
        }
    )
    frame = pd.concat(
        [
            base.assign(variable=name,
                        value=trajectory(name)[cell_index, day_index].astype(np.float32))
            for name in variables
        ],
        ignore_index=True,
    )
    return frame


def daily_batch(
    ids: np.ndarray,
    weather_np: np.ndarray,
    year: int,
    start_doy: int,
    sowing_doy: np.ndarray,
    bundle: ExportBundle,
    device: torch.device,
    variables: tuple[str, ...] = ("LAI", "AGB", "NNI", "TRANRF"),
    crop_params: "CropParameters | None" = None,
) -> pd.DataFrame:
    """Per-**day** trajectories for one batch and one season; see :func:`_daily_frame`."""
    sowing_doy = np.asarray(sowing_doy)
    out, _fertilizer, _irrigated = _simulate(
        ids, weather_np, year, start_doy, sowing_doy, bundle, device, crop_params
    )
    frame = _daily_frame(ids, out, year, start_doy, sowing_doy, bundle, variables)
    del out
    return frame


def run_cells(
    config: RunConfig,
    ids: np.ndarray,
    years: list[int],
    mode: str = "summary",
    variables: tuple[str, ...] = ("LAI", "AGB", "NNI", "TRANRF"),
    bundle: ExportBundle | None = None,
    sowing: pd.DataFrame | None = None,
    crop_params: "CropParameters | None" = None,
) -> pd.DataFrame | tuple[pd.DataFrame, pd.DataFrame]:
    """Run an explicit cell list over explicit seasons; summaries or trajectories.

    The cell list is what makes this comparable with SIMPLACE: a shard deals
    cells round-robin out of the whole runnable set, whereas an evaluation
    needs *these* cells. Cells are grouped exactly as :func:`run_shard` does —
    see :func:`sowing_plan` — so a comparison run pays the same per-batch cost
    a production shard does rather than fragmenting on simulated sowing dates.

    Parameters
    ----------
    mode:
        ``summary`` for one row per (cell, season) — the shard schema —
        ``daily`` for the long trajectory table, or ``both`` for
        ``(summary, daily)`` sharing one ``_simulate`` call per batch (see
        :func:`run_shard`'s own ``daily=True`` path) instead of running the
        cells twice.
    sowing:
        Optional ``(SimplaceID, year, sowing_doy)`` table replacing the site
        calendar, so a run can be driven by another model's **simulated**
        sowing date rather than the export's input one. Grouping is by year
        (see :func:`sowing_plan`); a (cell, season) the table does not cover
        is dropped rather than quietly falling back to the site value, which
        would mix two sowing conventions in one output.
    crop_params:
        Optional parameter set (see
        :func:`cropmodelling4eu.torchcrop.params.crop_parameters_from_simplace`)
        replacing torchcrop's bundled preset.
    """
    if mode not in ("summary", "daily", "both"):
        raise ValueError(f"mode must be 'summary', 'daily' or 'both', not {mode!r}")
    settings = config.torchcrop
    if settings.torch_threads > 0:
        torch.set_num_threads(settings.torch_threads)
    if bundle is None:
        bundle = resolve_export(config, require_management=settings.iopt != 1)

    ids = np.intersect1d(np.asarray(ids, dtype=np.int64), bundle.ids)
    if ids.size == 0:
        raise ValueError("none of the requested cells are runnable in this export")

    plan = sowing_plan(ids, list(years), bundle.site.sowing_doy(ids), sowing)
    n_batches = sum(
        int(np.ceil(group.size / settings.batch_size)) for _, group, _ in plan
    )
    if sowing is not None:
        all_doy = np.concatenate([doy for _, _, doy in plan])
        logger.info(
            "sowing overridden from a table: %d cell-season group(s), %d "
            "batch(es), DOY %d-%d",
            len(plan), n_batches, all_doy.min(), all_doy.max(),
        )

    device = torch.device(settings.device)
    frames: list[pd.DataFrame] = []
    daily_frames: list[pd.DataFrame] = []
    t_start = time.time()
    done = 0
    for group_years, group, doy in plan:
        for start in range(0, group.size, settings.batch_size):
            batch = group[start : start + settings.batch_size]
            batch_doy = doy[start : start + settings.batch_size]
            # The shared calendar window only has to open early enough for
            # the earliest sower in the batch; each cell latches on its own
            # idpl inside it (see sowing_plan).
            anchor_doy = int(batch_doy.min())
            blocks = load_season_block(
                bundle.export_dir, batch, group_years, anchor_doy, config.grid,
                settings.io_workers, settings.spinup_months,
                settings.min_days_after_sowing,
            )
            for year, (weather, start_doy) in blocks.items():
                if mode == "both":
                    out, fertilizer, irrigated = _simulate(
                        batch, weather, year, start_doy, batch_doy, bundle,
                        device, crop_params,
                    )
                    frames.append(_summary_frame(
                        batch, out, fertilizer, irrigated, year, start_doy,
                        batch_doy, bundle,
                    ))
                    daily_frames.append(_daily_frame(
                        batch, out, year, start_doy, batch_doy, bundle, variables,
                    ))
                    del out, fertilizer
                else:
                    frames.append(
                        run_batch(batch, weather, year, start_doy, batch_doy, bundle,
                                  device, crop_params)
                        if mode == "summary" else
                        daily_batch(batch, weather, year, start_doy, batch_doy, bundle,
                                    device, variables, crop_params)
                    )
            done += 1
            elapsed = time.time() - t_start
            logger.info(
                "batch %d/%d (DOY %d-%d, %d cells): elapsed %.1f min, eta %.1f min",
                done, n_batches, batch_doy.min(), batch_doy.max(), batch.size,
                elapsed / 60, elapsed / done * (n_batches - done) / 60,
            )

    if not frames:
        raise SystemExit("no (cell, season) produced a result")
    summary = pd.concat(frames, ignore_index=True)
    if mode == "both":
        return summary, pd.concat(daily_frames, ignore_index=True)
    return summary


def run_cells_daily(
    config: RunConfig,
    ids: np.ndarray,
    years: list[int],
    variables: tuple[str, ...] = ("LAI", "AGB", "NNI", "TRANRF"),
    bundle: ExportBundle | None = None,
    crop_params: "CropParameters | None" = None,
) -> pd.DataFrame:
    """Daily trajectories for an explicit cell list; :func:`run_cells` in daily mode."""
    return run_cells(config, ids, years, "daily", variables, bundle, crop_params=crop_params)


# --------------------------------------------------------------------------- #
# Shard driver
# --------------------------------------------------------------------------- #


def run_shard(
    config: RunConfig,
    shard: int,
    n_shards: int,
    out_dir: Path,
    max_cells: int = 0,
    overwrite: bool = False,
    sowing: pd.DataFrame | None = None,
    crop_file: Path | None = None,
    daily: bool = False,
    daily_variables: tuple[str, ...] = ("LAI", "AGB", "NNI", "TRANRF"),
) -> Path:
    """Run one shard over every requested season and write its Parquet.

    ``sowing`` replaces the export's calendar with another model's simulated
    dates — what ``submit/submit_cropmodelling.sh`` hands over from a finished
    SIMPLACE run. It changes the batching as well as the dates: see
    :func:`sowing_plan`.

    ``daily`` additionally writes ``torchcrop_daily_shard_<n>.parquet`` beside
    the summary, sharing the same ``_simulate`` call per batch (see
    :func:`_summary_frame` / :func:`_daily_frame`) rather than re-running the
    model, so the trajectory is exactly the run the summary row came from and
    costs no extra forward pass.
    """
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / f"torchcrop_shard_{shard:03d}.parquet"
    daily_path = out_dir / f"torchcrop_daily_shard_{shard:03d}.parquet"
    if out_path.exists() and (not daily or daily_path.exists()) and not overwrite:
        logger.info("shard %d already done: %s", shard, out_path)
        return out_path

    settings = config.torchcrop
    device = torch.device(settings.device)

    # The workspace's crop file when there is one, so the parameters a shard
    # runs with are the ones on disk that can be opened and checked.
    crop_params = load_crop_parameters(crop_file, config.season.crop)

    # A potential-yield run applies no fertilizer, so it needs no schedule.
    bundle = resolve_export(config, require_management=settings.iopt != 1)
    ids = bundle.shard(shard, n_shards)
    if max_cells and ids.size > max_cells:
        # Even subsample, not the head: SimplaceID runs north-to-south, so
        # truncating gives a smoke test a narrow Scandinavian band.
        ids = ids[:: int(np.ceil(ids.size / max_cells))][:max_cells]

    years = config.season.years
    plan = sowing_plan(ids, years, bundle.site.sowing_doy(ids), sowing)
    n_batches = sum(
        int(np.ceil(group.size / settings.batch_size)) for _, group, _ in plan
    )
    all_doy = np.concatenate([doy for _, _, doy in plan])
    logger.info(
        "shard %d/%d: %d cells (of %d), %d years, %d group(s), %d batch(es), "
        "DOY %d-%d, sowing from %s, device=%s",
        shard, n_shards, ids.size, bundle.ids.size, len(years), len(plan),
        n_batches, all_doy.min(), all_doy.max(),
        "a simulated table" if sowing is not None else "the site calendar", device,
    )
    logger.info(bundle.site.summarise())

    results: list[pd.DataFrame] = []
    daily_results: list[pd.DataFrame] = []
    t_start = time.time()
    done = 0

    for group_years, group, doy in plan:
        for start in range(0, group.size, settings.batch_size):
            batch = group[start : start + settings.batch_size]
            batch_doy = doy[start : start + settings.batch_size]
            # The shared calendar window only has to open early enough for
            # the earliest sower in the batch; each cell latches on its own
            # idpl inside it (see sowing_plan).
            anchor_doy = int(batch_doy.min())
            window = season_window(
                group_years[0], anchor_doy, settings.spinup_months, SIM_YEARS
            )
            if window[0].dayofyear > sowing_date(group_years[0], anchor_doy).dayofyear:
                # The latch is `doy >= idpl` on a day-of-year, so a spin-up that
                # crosses New Year before sowing already satisfies it on day 0: the
                # crop would go in on the first day of the window with no spin-up.
                logger.warning(
                    "sowing DOY %d leaves a spin-up that crosses New Year (window "
                    "opens on DOY %d); the model will sow on the first day of the "
                    "window", anchor_doy, window[0].dayofyear,
                )

            t0 = time.time()
            blocks = load_season_block(
                bundle.export_dir, batch, group_years, anchor_doy, config.grid,
                settings.io_workers, settings.spinup_months,
                settings.min_days_after_sowing,
            )
            t_io = time.time() - t0

            for year, (weather_np, start_doy) in blocks.items():
                if daily:
                    out, fertilizer, irrigated = _simulate(
                        batch, weather_np, year, start_doy, batch_doy, bundle,
                        device, crop_params,
                    )
                    results.append(_summary_frame(
                        batch, out, fertilizer, irrigated, year, start_doy,
                        batch_doy, bundle,
                    ))
                    daily_results.append(_daily_frame(
                        batch, out, year, start_doy, batch_doy, bundle,
                        daily_variables,
                    ))
                    del out, fertilizer
                else:
                    results.append(
                        run_batch(batch, weather_np, year, start_doy, batch_doy, bundle,
                                  device, crop_params)
                    )
            done += 1
            elapsed = time.time() - t_start
            logger.info(
                "batch %d/%d (DOY %d-%d, %d cells): io %.1fs, total %.1fs, "
                "elapsed %.1f min, eta %.1f min",
                done, n_batches, batch_doy.min(), batch_doy.max(), batch.size,
                t_io, time.time() - t0,
                elapsed / 60, elapsed / done * (n_batches - done) / 60,
            )

    if not results:
        raise RuntimeError(f"shard {shard} produced no results")

    frame = pd.concat(results, ignore_index=True)
    frame.to_parquet(out_path, index=False, compression="zstd")
    logger.info(
        "shard %d wrote %d rows to %s in %.1f min",
        shard, len(frame), out_path, (time.time() - t_start) / 60,
    )

    if daily:
        daily_frame = pd.concat(daily_results, ignore_index=True)
        daily_frame.to_parquet(daily_path, index=False, compression="zstd")
        logger.info(
            "shard %d wrote %d daily rows to %s", shard, len(daily_frame), daily_path,
        )
    return out_path


def _config_from_args(args: argparse.Namespace) -> RunConfig:
    """The run config, from a file or built around a bare export directory."""
    if args.config is not None:
        config = load_config(args.config)
    elif args.export_dir is not None:
        config = RunConfig.from_export(args.export_dir)
    else:
        raise SystemExit("Pass either --config or --export-dir")

    updates: dict = {}
    season = {}
    if args.start_year is not None:
        season["start_year"] = args.start_year
    if args.end_year is not None:
        season["end_year"] = args.end_year
    if args.crop is not None:
        season["crop"] = args.crop
    if season:
        updates["season"] = config.season.model_copy(update=season)

    torch_updates = {
        key: value
        for key, value in (
            ("iopt", args.iopt),
            ("batch_size", args.batch_size),
            ("io_workers", args.io_workers),
            ("device", args.device),
        )
        if value is not None
    }
    if torch_updates:
        updates["torchcrop"] = config.torchcrop.model_copy(update=torch_updates)

    if args.composition is not None:
        updates["paths"] = config.paths.model_copy(
            update={"composition_file": args.composition}
        )
    return config.model_copy(update=updates) if updates else config


def build_parser() -> argparse.ArgumentParser:
    """CLI for one array task."""
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--shard", type=int, required=True, help="0-based task index")
    p.add_argument("--n-shards", type=int, default=10)
    p.add_argument("--config", type=Path, help="Run configuration YAML.")
    p.add_argument("--export-dir", type=Path, help="Export directory (instead of --config).")
    p.add_argument("--out-dir", required=True)
    p.add_argument("--composition", type=Path)
    p.add_argument("--start-year", type=int, help="first harvest year")
    p.add_argument("--end-year", type=int, help="last harvest year")
    p.add_argument("--crop", default=None)
    p.add_argument("--iopt", type=int, choices=[1, 2, 3, 4],
                   help="1 potential | 2 water | 3 water+N | 4 water+NPK")
    p.add_argument("--batch-size", type=int)
    p.add_argument("--io-workers", type=int)
    p.add_argument("--device")
    p.add_argument("--torch-threads", type=int, default=None,
                   help="intra-op threads. The model works on [B]-shaped "
                        "tensors, too small to repay many threads -- the cores "
                        "are better spent on --io-workers")
    p.add_argument("--max-cells", type=int, default=0,
                   help="cap cells per shard (smoke tests)")
    p.add_argument("--crop-file", type=Path,
                   help="crop parameter YAML from the run workspace "
                        "(crop_<crop>.yaml); without it torchcrop's bundled "
                        "preset is used")
    p.add_argument("--sowing-file", type=Path,
                   help="CSV of (SimplaceID, year, sowing_doy) to sow on "
                        "instead of the export's calendar -- what a finished "
                        "SIMPLACE run writes as sowing_from_simplace.csv")
    p.add_argument("--daily", action="store_true",
                   help="also write torchcrop_daily_shard_<n>.parquet -- the "
                        "same simulation's day-by-day trajectory, at no extra "
                        "model cost")
    p.add_argument("--daily-variables", nargs="+", choices=list(DAILY_VARIABLES),
                   default=["LAI", "AGB", "NNI", "TRANRF"],
                   help="which series --daily writes; default LAI, AGB, NNI, "
                        "TRANRF (DVS is also available)")
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
    config = _config_from_args(args)
    threads = args.torch_threads
    if threads is None:
        threads = config.torchcrop.torch_threads
    if threads > 0:
        torch.set_num_threads(threads)

    run_shard(
        config,
        shard=args.shard,
        n_shards=args.n_shards,
        out_dir=Path(args.out_dir),
        max_cells=args.max_cells,
        overwrite=args.overwrite,
        sowing=pd.read_csv(args.sowing_file) if args.sowing_file else None,
        crop_file=args.crop_file,
        daily=args.daily,
        daily_variables=tuple(args.daily_variables),
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
