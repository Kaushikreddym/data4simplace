"""Run torchcrop (LINTUL-5) over the export, one shard per job.

Library and ``python -m`` entry point behind ``submit/utils/torchcrop_array.sh``.
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
from collections import deque
from collections.abc import Iterator
from itertools import islice
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
from cropmodelling4eu.daily import DAILY_KEYS, DailySink
from cropmodelling4eu.export import ExportBundle, resolve_export
from cropmodelling4eu.export.cells import id_to_lonlat
from cropmodelling4eu.export.soil import SoilProfiles
from cropmodelling4eu.export.weather import (
    RowBlockReader,
    load_row_block,
    load_season_block,
    season_window,
    sowing_date,
)
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


def pack_seasons(
    plan: list[tuple[list[int], np.ndarray, np.ndarray]], batch_size: int
) -> list[tuple[np.ndarray, np.ndarray, np.ndarray]]:
    """Repack a one-season-per-group plan into ``[(ids, years, sowing_doy)]`` batches.

    The model costs ~12 s per call almost regardless of its size (150 cells
    13.5 s, 2048 cells 17 s), so a region of 59 cells run one season at a time
    pays 164 near-empty calls for a CMIP6 window. Rows are ``(cell, season)``
    pairs instead, filled up to ``batch_size`` across seasons.

    Rows are ordered by sowing day-of-year before chunking: a batch's window
    opens before its *earliest* sower, so neighbouring dates keep every row's
    spin-up close to its own -- closer, for a large region, than one season's
    2048 cells spread across the whole sowing range.
    """
    ids = np.concatenate([group for _, group, _ in plan])
    years = np.concatenate([np.full(g.size, y[0], dtype=np.int64) for y, g, _ in plan])
    doy = np.concatenate([d for _, _, d in plan])
    order = np.lexsort((ids, years, doy))
    ids, years, doy = ids[order], years[order], doy[order]
    return [
        (ids[k : k + batch_size], years[k : k + batch_size], doy[k : k + batch_size])
        for k in range(0, ids.size, batch_size)
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
    year: int | np.ndarray,
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

    ``year`` is per-cell too, for the same reason: a production batch is one
    season, but a calibration batch is one *region* and may hold several (see
    :mod:`cropmodelling4eu.calibration.dataset`). CO2 rose 90 ppm across the
    export's own window, so a shared value across seasons would be a real
    error rather than a rounding one.
    """
    _, lat = id_to_lonlat(ids, bundle.config.grid)
    altitude = bundle.site.altitude_m(ids)
    n = ids.size
    years = np.broadcast_to(np.asarray(year, dtype=np.int64), (n,))
    last = float(bundle.co2.iloc[-1])
    co2 = np.array([float(bundle.co2.get(int(y), last)) for y in years])
    idpl = np.broadcast_to(np.asarray(sowing_doy, dtype=float), (n,))
    return SiteParameters(
        latitude=torch.as_tensor(lat, dtype=dtype),
        altitude=torch.as_tensor(altitude, dtype=dtype),
        co2=torch.as_tensor(co2, dtype=dtype),
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


def crossing_day(x: torch.Tensor, threshold) -> torch.Tensor:
    """Fractional day index at which ``x`` first reaches ``threshold``.

    ``x`` is a ``[B, T]`` monotonically non-decreasing accumulator (``tsump``,
    ``tsum``, ``dvs`` before maturity). The crossing is found by linear
    interpolation between the last step below the threshold and the first at or
    above it, which is exact and differentiable in both the trajectory and the
    threshold — the construction torchcrop's own phenology-calibration example
    uses.

    Preferred over ``argmax``, which quantises to whole days and carries no
    gradient, and over a sigmoid-sum date, which is exact only in the sharp
    limit and biased at any finite sharpness. The quantity is reported in days,
    so a smoothing bias is a bias in the answer.

    Args:
        x: ``[B, T]`` accumulator, non-decreasing along ``T``.
        threshold: Scalar or ``[B]`` value to cross.

    Returns:
        ``[B]`` fractional step index, **NaN where the threshold is never
        reached** — a row that never emerged has no emergence day, and a
        clamped last-day answer would be indistinguishable from a real one.
    """
    thr = torch.as_tensor(threshold, dtype=x.dtype, device=x.device).expand(x.shape[0])
    below = x < thr.unsqueeze(1)
    t0 = (below.sum(dim=1) - 1).clamp(0, x.shape[1] - 2)
    x0 = x.gather(1, t0.unsqueeze(1)).squeeze(1)
    x1 = x.gather(1, (t0 + 1).unsqueeze(1)).squeeze(1)
    frac = ((thr - x0) / (x1 - x0).clamp(min=1e-9)).clamp(0.0, 1.0)
    day = t0.to(x.dtype) + frac
    return day.where(~below.all(dim=1), torch.full_like(day, float("nan")))


def _simulate(
    ids: np.ndarray,
    weather_np: np.ndarray,
    year: int | np.ndarray,
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
    # The emergence threshold travels with the output because it is the only
    # thing needed to date emergence and it must be the value this batch
    # actually ran with -- `crop_params` may be the caller's workspace set
    # rather than the crop name's defaults, so rebuilding it downstream would
    # silently date emergence against a different tsumem.
    tsumem = float(crop_params.tsumem)
    del model, soil_params, site_params, weather
    return out, fertilizer, irrigated, tsumem


def _summary_frame(
    ids: np.ndarray,
    out,
    fertilizer: torch.Tensor,
    irrigated: np.ndarray,
    year: int | np.ndarray,
    start_doy: int,
    sowing_doy: np.ndarray,
    bundle: ExportBundle,
    tsumem: float,
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
    # Emergence is dated on the thermal-time clock, not on DVS: DVS is pinned
    # at 0 until the crop emerges (`dvs_rate` is gated by `emerged`), so it
    # carries no emergence signal, and its first non-zero step is one day late
    # because the rate on day t produces the state on day t + 1.
    # `tsump` is unaffected by `tsumem`, so this is the only column in the
    # frame whose value depends on where that threshold sits.
    tsump = torch.stack([s.tsump for s in out.states[1:]], dim=1)
    days_to_emergence = crossing_day(tsump, tsumem).cpu().numpy() - sow_index
    # Anthesis is DVS 1.0 on the same interpolated crossing. It is the only
    # observation that separates the pre- from the post-anthesis thermal sum:
    # without it `tsum1` and `tsum2` are identified only by their sum (see
    # CALIBRATION.md's identifiability rule). SIMPLACE has always written
    # `Phenology.AnthesisDOY`; this is torchcrop's side of the same column.
    days_to_anthesis = crossing_day(dvs, 1.0).cpu().numpy() - sow_index

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
            "days_to_emergence": days_to_emergence,
            "days_to_anthesis": days_to_anthesis,
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


def row_blocks(
    batches: list[tuple[np.ndarray, np.ndarray, np.ndarray]],
    bundle: ExportBundle,
    config: RunConfig,
    records: dict[int, pd.DataFrame],
) -> Iterator[tuple[tuple[np.ndarray, np.ndarray, np.ndarray], list]]:
    """Each packed batch with its :func:`load_row_block` weather.

    With ``torchcrop.io_processes`` the weather is read in worker processes,
    ``io_prefetch`` batches ahead, so it overlaps the model running on the
    batch before; ``records`` is then unused (each worker caches its own cells).
    """
    settings = config.torchcrop
    if settings.io_processes == 0:
        for batch_ids, batch_years, batch_doy in batches:
            yield (batch_ids, batch_years, batch_doy), load_row_block(
                bundle.export_dir, batch_ids, batch_years, int(batch_doy.min()),
                config.grid, records, settings.io_workers, settings.spinup_months,
                settings.min_days_after_sowing,
            )
        return
    with RowBlockReader(bundle.export_dir, config.grid, settings.io_processes,
                        settings.spinup_months, settings.min_days_after_sowing) as reader:
        def submit(batch):
            return batch, reader.submit(batch[0], batch[1], int(batch[2].min()))

        remaining = iter(batches)
        pending = deque(submit(b) for b in islice(remaining, settings.io_prefetch))
        while pending:
            batch, block = pending.popleft()
            following = next(remaining, None)
            if following is not None:
                pending.append(submit(following))
            yield batch, block.result()


def run_batch(
    ids: np.ndarray,
    weather_np: np.ndarray,
    year: int | np.ndarray,
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
    out, fertilizer, irrigated, tsumem = _simulate(
        ids, weather_np, year, start_doy, sowing_doy, bundle, device, crop_params
    )
    frame = _summary_frame(
        ids, out, fertilizer, irrigated, year, start_doy, sowing_doy, bundle, tsumem
    )
    del out, fertilizer
    return frame


#: Daily series a trajectory can carry, and where each comes from. The state
#: trajectories carry one extra leading entry (the initial condition), which is
#: dropped so every series is indexed by the day that produced it.
#:
#: ``state`` fields are ``ModelOutput`` tensors, ``snapshot`` fields are read
#: off the per-day ``ModelState`` list (the engine stacks only LAI/DVS/biomass),
#: ``diagnostic`` fields off the per-day ``DiagnosticState`` list.
DAILY_VARIABLES: dict[str, str] = {
    "LAI": "state",
    "DVS": "state",
    "AGB": "state",
    "WSO": "snapshot",
    "WA": "snapshot",
    "TRANRF": "diagnostic",
    "NNI": "diagnostic",
    "TRAN": "diagnostic",
    "CO2_FACTOR": "diagnostic",
    "LEAF_HEAT_FACTOR": "diagnostic",
}

#: Source attribute for every variable whose name is not simply its lower case.
#: ``AGB`` and ``WSO`` are converted from torchcrop's native g/m2 to t/ha
#: (x0.01) -- the factor SIMPLACE's own ``AGBiomass_t_ha`` rule applies to
#: ``Biomass.sTAGB`` -- so the two models' daily series need no further
#: conversion at the point they are compared. ``WA`` is mm in the rooted zone,
#: ``TRAN`` mm/d.
_DAILY_FIELD = {"AGB": "biomass"}
_DAILY_SCALE = {"AGB": 0.01, "WSO": 0.01}

def _trajectory(out, name: str) -> np.ndarray:
    """``[B, T]`` series of one :data:`DAILY_VARIABLES` entry, day-indexed."""
    field = _DAILY_FIELD.get(name, name.lower())
    match DAILY_VARIABLES[name]:
        case "state":
            series = getattr(out, field)[:, 1:]
        case "snapshot":
            series = torch.stack([getattr(s, field) for s in out.states[1:]], dim=1)
        case "diagnostic":
            series = _diagnostic_trajectory(out, field)
    return (series * _DAILY_SCALE.get(name, 1.0)).detach().cpu().numpy()


def _daily_wide(
    ids: np.ndarray,
    out,
    year: int,
    start_doy: int,
    sowing_doy: np.ndarray,
    bundle: ExportBundle,
    variables: tuple[str, ...] = ("LAI", "AGB", "NNI", "TRANRF"),
) -> pd.DataFrame:
    """Per-**day** trajectories of an already-simulated batch, one row per cell-day.

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
        :data:`DAILY_KEYS` then one float32 column per variable, where ``das``
        is days after each row's own sowing latch (``sowing_doy`` is per-cell;
        the shared calendar axis is anchored to the batch's earliest sower, see
        :func:`sowing_plan`).
    """
    unknown = set(variables) - set(DAILY_VARIABLES)
    if unknown:
        raise ValueError(f"unknown daily variable(s): {sorted(unknown)}")

    sowing_doy = np.asarray(sowing_doy)
    dvs = out.dvs[:, 1:].detach().cpu().numpy()
    n_days = dvs.shape[1]
    # Step i is driven by weather row i and produces states[i + 1] and
    # diagnostics[i], so both describe the day `window start + i`. The window
    # start is the one load_seasons sliced on (anchored to the batch's
    # earliest sowing_doy, matching load_season_block), so this is the real
    # calendar axis -- leap days included, unlike the engine's wrapped
    # 365-day count.
    dates = pd.date_range(
        season_window(year, int(sowing_doy.min()), bundle.config.torchcrop.spinup_months)[0],
        periods=n_days, freq="D",
    )
    sown = np.array(
        [sowing_date(year, int(doy)) for doy in sowing_doy], dtype="datetime64[ns]"
    )
    das = (
        (dates.to_numpy()[None, :] - sown[:, None]) / np.timedelta64(1, "D")
    ).astype(np.int16)

    cell_index, day_index = np.nonzero((dvs > 0.0) & (dvs < 2.0))
    return pd.DataFrame(
        {
            "SimplaceID": ids.astype(np.int64)[cell_index],
            "year": np.full(cell_index.size, year, dtype=np.int16),
            "date": dates.to_numpy()[day_index],
            "doy": dates.dayofyear.to_numpy()[day_index].astype(np.int16),
            "das": das[cell_index, day_index],
        }
        | {
            name: _trajectory(out, name)[cell_index, day_index].astype(np.float32)
            for name in variables
        }
    )


def _daily_frame(
    ids: np.ndarray,
    out,
    year: int,
    start_doy: int,
    sowing_doy: np.ndarray,
    bundle: ExportBundle,
    variables: tuple[str, ...] = ("LAI", "AGB", "NNI", "TRANRF"),
) -> pd.DataFrame:
    """:func:`_daily_wide` in the long layout the in-memory comparisons use.

    Returns
    -------
    DataFrame
        ``model, SimplaceID, year, date, doy, das, variable, value``,
        variable-major.
    """
    wide = _daily_wide(ids, out, year, start_doy, sowing_doy, bundle, variables)
    long = wide.melt(id_vars=list(DAILY_KEYS), value_vars=list(variables),
                     var_name="variable", value_name="value")
    long.insert(0, "model", "torchcrop")
    return long


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
    out, _fertilizer, _irrigated, _tsumem = _simulate(
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
    daily_sink: "DailySink | None" = None,
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
    daily_sink:
        With ``mode="summary"``, every batch's **wide** trajectory
        (:func:`_daily_wide`) is handed to this callable as it is produced
        instead of being collected, so a region-sized run never holds its
        daily table in memory. Only the summary is returned.
    """
    if mode not in ("summary", "daily", "both"):
        raise ValueError(f"mode must be 'summary', 'daily' or 'both', not {mode!r}")
    if daily_sink is not None and mode != "summary":
        raise ValueError("daily_sink streams the trajectory; use it with mode='summary'")
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
    # A sowing table batches one season at a time, so each cell's record is
    # decoded once here and every season sliced from it (see load_season_block).
    records: dict[int, pd.DataFrame] | None = {} if sowing is not None else None
    t_start = time.time()
    done = 0

    if sowing is not None and mode == "summary" and daily_sink is None:
        # Summary rows carry their own year; a daily frame shares one date
        # axis per batch, so only the summary path packs across seasons.
        batches = pack_seasons(plan, settings.batch_size)
        logger.info("packed into %d batch(es) across seasons", len(batches))
        for (batch_ids, batch_years, batch_doy), blocks in row_blocks(
            batches, bundle, config, records,
        ):
            for rows, weather, start_doy in blocks:
                frames.append(run_batch(
                    batch_ids[rows], weather, batch_years[rows], start_doy,
                    batch_doy[rows], bundle, device, crop_params,
                ))
            done += 1
            elapsed = time.time() - t_start
            logger.info(
                "batch %d/%d (DOY %d-%d, %d rows, years %d-%d): elapsed %.1f min, "
                "eta %.1f min",
                done, len(batches), batch_doy.min(), batch_doy.max(), batch_ids.size,
                batch_years.min(), batch_years.max(), elapsed / 60,
                elapsed / done * (len(batches) - done) / 60,
            )
        if not frames:
            raise SystemExit("no (cell, season) produced a result")
        return pd.concat(frames, ignore_index=True)
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
                settings.min_days_after_sowing, records,
            )
            for year, (weather, start_doy) in blocks.items():
                if mode == "both" or daily_sink is not None:
                    out, fertilizer, irrigated, tsumem = _simulate(
                        batch, weather, year, start_doy, batch_doy, bundle,
                        device, crop_params,
                    )
                    frames.append(_summary_frame(
                        batch, out, fertilizer, irrigated, year, start_doy,
                        batch_doy, bundle, tsumem,
                    ))
                    if daily_sink is not None:
                        daily_sink(_daily_wide(
                            batch, out, year, start_doy, batch_doy, bundle, variables,
                        ))
                    else:
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


def region_crop_map(
    ids: np.ndarray,
    regions_file: Path,
    region_crop_dir: Path,
    fallback: "CropParameters",
) -> dict[int, tuple[np.ndarray, "CropParameters"]]:
    """``ids`` split by calibration region, each with its own crop parameters.

    ``regions_file`` is a calibration run's own ``regions.parquet``
    (``SimplaceID``, ``region_id``); ``region_crop_dir`` its ``<stage>/crops/``
    (see :func:`cropmodelling4eu.torchcrop.params.write_crop_yaml_from_parameters`).
    Two things fall back to ``fallback`` rather than failing a production run
    outright, and both are logged by what they cover: a region with no
    ``crop_region_<NN>.yaml`` there is uncalibrated the same way a region with
    no observations is (see ``CalibrationResult.blocked``), and a cell the
    table does not place in any region at all is one this calibration run
    never saw. ``-1`` keys that second group.

    Returns:
        ``{region_id: (cell_ids, crop_params)}``, every value of ``ids``
        covered exactly once.
    """
    table = pd.read_parquet(Path(regions_file))[["SimplaceID", "region_id"]]
    table = table.drop_duplicates("SimplaceID").set_index("SimplaceID")["region_id"]

    region_of = table.reindex(ids)
    covered = region_of.dropna()
    groups: dict[int, np.ndarray] = {
        int(region_id): sub.index.to_numpy(np.int64)
        for region_id, sub in covered.groupby(covered)
    }
    uncovered = region_of.index[region_of.isna()].to_numpy(np.int64)
    if uncovered.size:
        groups[-1] = uncovered
        logger.warning(
            "%d cell(s) are not in %s and run on the fallback crop",
            uncovered.size, regions_file,
        )

    out: dict[int, tuple[np.ndarray, "CropParameters"]] = {}
    missing_regions: list[int] = []
    for region_id, region_ids in groups.items():
        if region_id == -1:
            out[region_id] = (region_ids, fallback)
            continue
        path = Path(region_crop_dir) / f"crop_region_{region_id:02d}.yaml"
        if path.is_file():
            out[region_id] = (region_ids, CropParameters(config_file=str(path)))
        else:
            out[region_id] = (region_ids, fallback)
            missing_regions.append(region_id)
    if missing_regions:
        logger.warning(
            "%d region(s) have no calibrated crop file under %s and run on "
            "the fallback crop: %s", len(missing_regions), region_crop_dir,
            sorted(missing_regions),
        )
    return out


#: A region costs one model call per season however few cells it holds -- a
#: 1-cell region took 4.7 min over 25 seasons, a 471-cell one 9.5 min -- so in
#: shard-balancing terms every region slice carries this many cells of overhead.
_REGION_OVERHEAD_CELLS = 450


def region_shard(ids: np.ndarray, regions_file: Path, shard: int, n_shards: int) -> np.ndarray:
    """``ids`` of whole calibration regions for one task, balanced by cost.

    :func:`~cropmodelling4eu.export.cells.shard_cells` deals cells round-robin,
    which in region mode hands every shard a slice of *every* region: each
    shard then runs ``n_regions × n_seasons`` small batches, paying the per-call
    overhead ``n_shards`` times over. Keeping regions whole runs each region's
    seasons once. A region larger than one shard's share is split round-robin
    so a single large region cannot hold up the array, and the slices go to
    shards longest-first onto the least-loaded one. The assignment depends only
    on ``ids`` and the regions table, so every task computes the same one.
    """
    if not 0 <= shard < n_shards:
        raise ValueError(f"shard {shard} out of range 0..{n_shards - 1}")
    if ids.size < n_shards:
        raise ValueError(f"{ids.size} cells cannot fill {n_shards} shards")

    table = pd.read_parquet(Path(regions_file))[["SimplaceID", "region_id"]]
    table = table.drop_duplicates("SimplaceID").set_index("SimplaceID")["region_id"]
    region_of = table.reindex(ids).fillna(-1).astype(np.int64)

    share = ids.size / n_shards
    units = [
        group[k::parts]
        for _, sub in region_of.groupby(region_of)
        for group in [sub.index.to_numpy(np.int64)]
        for parts in [int(np.ceil(group.size / share))]
        for k in range(parts)
    ]
    while len(units) < n_shards:
        largest = units.pop(max(range(len(units)), key=lambda i: units[i].size))
        units += [largest[0::2], largest[1::2]]

    load = np.zeros(n_shards)
    mine: list[np.ndarray] = []
    for unit in sorted(units, key=lambda u: (-u.size, u[0])):
        target = int(np.argmin(load))
        load[target] += unit.size + _REGION_OVERHEAD_CELLS
        if target == shard:
            mine.append(unit)
    return np.sort(np.concatenate(mine))


def run_shard(
    config: RunConfig,
    shard: int,
    n_shards: int,
    out_dir: Path,
    max_cells: int = 0,
    overwrite: bool = False,
    sowing: pd.DataFrame | None = None,
    crop_file: Path | None = None,
    region_crop_dir: Path | None = None,
    regions_file: Path | None = None,
    daily: bool = False,
    daily_variables: tuple[str, ...] = ("LAI", "AGB", "NNI", "TRANRF"),
    shard_by: str = "cells",
) -> Path:
    """Run one shard over every requested season and write its Parquet.

    ``sowing`` replaces the export's calendar with another model's simulated
    dates — what ``submit/baseline/submit_cropmodelling.sh`` hands over from a finished
    SIMPLACE run. It changes the batching as well as the dates: see
    :func:`sowing_plan`.

    ``daily`` additionally writes ``torchcrop_daily_shard_<n>.parquet`` beside
    the summary, sharing the same ``_simulate`` call per batch (see
    :func:`_summary_frame` / :func:`_daily_wide`) rather than re-running the
    model, so the trajectory is exactly the run the summary row came from and
    costs no extra forward pass. The file is in the **wide** layout, one row
    per cell-day (:data:`DAILY_KEYS` + one column per variable), streamed to
    disk batch by batch through :class:`DailySink`.

    ``region_crop_dir`` (with ``regions_file``) runs a calibration's per-region
    crop fit instead of the single ``crop_file`` for the whole domain: the
    shard's cells are split by :func:`region_crop_map` and each region's slice
    goes through :func:`run_cells` with that region's own ``CropParameters`` —
    the same per-region-batch invariant
    :mod:`cropmodelling4eu.calibration.dataset` calibrates under, since a
    torchcrop ``CropParameters`` is one object applied uniformly to whatever
    batch it is given, not a per-cell field. ``crop_file`` is then what an
    uncalibrated region or an unplaced cell falls back to, same as when it is
    the only crop argument given.

    ``shard_by="regions"`` (region mode only) gives each shard whole regions
    via :func:`region_shard` instead of a round-robin slice of every region.
    Every task of one array must use the same ``shard_by``, or cells are run
    twice or not at all.
    """
    if bool(region_crop_dir) != bool(regions_file):
        raise ValueError("region_crop_dir and regions_file must be given together")
    if shard_by not in ("cells", "regions"):
        raise ValueError(f"shard_by must be 'cells' or 'regions', not {shard_by!r}")
    if shard_by == "regions" and regions_file is None:
        raise ValueError("shard_by='regions' needs regions_file")

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
    # runs with are the ones on disk that can be opened and checked. In region
    # mode this is the fallback for whatever region_crop_map cannot place.
    crop_params = load_crop_parameters(crop_file, config.season.crop)

    # A potential-yield run applies no fertilizer, so it needs no schedule.
    bundle = resolve_export(config, require_management=settings.iopt != 1)
    ids = (
        region_shard(bundle.ids, regions_file, shard, n_shards)
        if shard_by == "regions" else bundle.shard(shard, n_shards)
    )
    if max_cells and ids.size > max_cells:
        # Even subsample, not the head: SimplaceID runs north-to-south, so
        # truncating gives a smoke test a narrow Scandinavian band.
        ids = ids[:: int(np.ceil(ids.size / max_cells))][:max_cells]

    years = config.season.years
    results: list[pd.DataFrame] = []
    sink = DailySink(daily_path) if daily else None
    t_start = time.time()

    if region_crop_dir is not None:
        groups = region_crop_map(ids, regions_file, region_crop_dir, crop_params)
        logger.info(
            "shard %d/%d: %d cells (of %d) over %d region(s), %d years, "
            "sowing from %s, device=%s",
            shard, n_shards, ids.size, bundle.ids.size, len(groups), len(years),
            "a simulated table" if sowing is not None else "the site calendar", device,
        )
        logger.info(bundle.site.summarise())
        for region_id, (region_ids, region_params) in sorted(groups.items()):
            label = "unplaced cells" if region_id == -1 else f"region {region_id:02d}"
            logger.info("shard %d: %s, %d cells", shard, label, region_ids.size)
            results.append(run_cells(
                config, region_ids, years, variables=daily_variables,
                bundle=bundle, sowing=sowing, crop_params=region_params,
                daily_sink=sink,
            ))

        frame = pd.concat(results, ignore_index=True)
        frame.to_parquet(out_path, index=False, compression="zstd")
        logger.info(
            "shard %d wrote %d rows to %s in %.1f min",
            shard, len(frame), out_path, (time.time() - t_start) / 60,
        )
        if sink is not None:
            logger.info("shard %d wrote %d daily rows to %s", shard, sink.rows, sink.close())
        return out_path

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

    done = 0
    records: dict[int, pd.DataFrame] | None = {} if sowing is not None else None

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
                settings.min_days_after_sowing, records,
            )
            t_io = time.time() - t0

            for year, (weather_np, start_doy) in blocks.items():
                if daily:
                    out, fertilizer, irrigated, tsumem = _simulate(
                        batch, weather_np, year, start_doy, batch_doy, bundle,
                        device, crop_params,
                    )
                    results.append(_summary_frame(
                        batch, out, fertilizer, irrigated, year, start_doy,
                        batch_doy, bundle, tsumem,
                    ))
                    sink(_daily_wide(
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

    if sink is not None:
        logger.info("shard %d wrote %d daily rows to %s", shard, sink.rows, sink.close())
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
            ("io_processes", args.io_processes),
            ("device", args.device),
        )
        if value is not None
    }
    if torch_updates:
        updates["torchcrop"] = config.torchcrop.model_copy(update=torch_updates)

    paths = {
        key: value
        for key, value in (("composition_file", args.composition), ("cells_file", args.cells_file))
        if value is not None
    }
    if paths:
        updates["paths"] = config.paths.model_copy(update=paths)
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
    p.add_argument("--io-processes", type=int,
                   help="weather-reader processes, reading packed batches ahead "
                        "of the model (0: in-process; a GPU wants the spare cores)")
    p.add_argument("--device")
    p.add_argument("--torch-threads", type=int, default=None,
                   help="intra-op threads. The model works on [B]-shaped "
                        "tensors, too small to repay many threads -- the cores "
                        "are better spent on --io-workers")
    p.add_argument("--cells-file", type=Path,
                   help="CSV with a SimplaceID column restricting the run's domain.")
    p.add_argument("--max-cells", type=int, default=0,
                   help="cap cells per shard (smoke tests)")
    p.add_argument("--crop-file", type=Path,
                   help="crop parameter YAML from the run workspace "
                        "(crop_<crop>.yaml); without it torchcrop's bundled "
                        "preset is used. In --region-crop-dir mode this is "
                        "the fallback for an uncalibrated region or a cell "
                        "the region table does not cover.")
    p.add_argument("--region-crop-dir", type=Path,
                   help="a calibration's <stage>/crops/ directory "
                        "(crop_region_<NN>.yaml per region) -- run each "
                        "region's cells on its own calibrated crop instead "
                        "of one crop_file for the whole domain. Needs "
                        "--regions-file too.")
    p.add_argument("--regions-file", type=Path,
                   help="the calibration's regions.parquet (SimplaceID, "
                        "region_id) --region-crop-dir splits cells by.")
    p.add_argument("--sowing-file", type=Path,
                   help="CSV of (SimplaceID, year, sowing_doy) to sow on "
                        "instead of the export's calendar -- what a finished "
                        "SIMPLACE run writes as sowing_from_simplace.csv")
    p.add_argument("--daily", action="store_true",
                   help="also write torchcrop_daily_shard_<n>.parquet -- the "
                        "same simulation's day-by-day trajectory, one row per "
                        "cell-day, at no extra model cost")
    p.add_argument("--daily-variables", nargs="+", choices=list(DAILY_VARIABLES),
                   default=["LAI", "AGB", "NNI", "TRANRF"],
                   help="which series --daily writes; default LAI, AGB, NNI, "
                        "TRANRF (see DAILY_VARIABLES for the rest)")
    p.add_argument("--shard-by", choices=["cells", "regions"], default="cells",
                   help="cells: round-robin over all cells; regions: whole "
                        "calibration regions per shard (needs --regions-file), "
                        "one model call per region-season instead of one per "
                        "shard-region-season. Every task of an array must agree.")
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
        region_crop_dir=args.region_crop_dir,
        regions_file=args.regions_file,
        daily=args.daily,
        daily_variables=tuple(args.daily_variables),
        shard_by=args.shard_by,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
