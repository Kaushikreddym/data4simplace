"""Run torchcrop (LINTUL-5) over the export, one shard per job.

Library and ``python -m`` entry point behind ``submit/torchcrop_array.sh``.
Everything about *reading* the export lives in
:mod:`cropmodelling4eu.export`; this module is only the modelling.

The work splits on three axes:

* **Shards** (``--shard`` / ``--n-shards``) — one SLURM array task each. Cells
  are dealt round-robin so every shard spans the whole domain and the tasks
  finish together; a contiguous split would give one task all of Iberia and
  another all of Scandinavia, with very different weather-file sizes.
* **Sowing groups** — within a shard, cells are grouped by their sowing
  day-of-year *before* batching. A batch shares one time axis and one
  ``idpl`` latch, so mixing sowing dates inside a batch is not a rounding
  question but a correctness one; see :func:`group_by_sowing`.
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
    """Split a shard's cells by sowing day-of-year.

    A batch must share one time axis: `load_season_block` trims every cell's
    season to a common length and hands the model a single ``start_doy``, and
    `Lintul5Model` latches sowing on one ``idpl`` for the whole batch. With a
    per-cell calendar, batching without grouping first would either skip nearly
    every batch (the cells disagree on their first day) or silently sow a
    Spanish cell on a German date.

    Grouping costs nothing in practice: a 0.5 degree calendar takes a handful
    of distinct values over any one shard, so the groups stay large enough to
    keep the batches full.

    Returns:
        ``{sowing_doy: ids}``, ascending by day-of-year.
    """
    doys = np.asarray(sowing_doy, dtype=int)
    return {
        int(doy): np.asarray(ids)[doys == doy] for doy in np.unique(doys)
    }


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
        crairc=const(ASSUMPTIONS["crairc"]),
        ksub=const(ASSUMPTIONS["ksub"]),
        runfr=const(ASSUMPTIONS["runfr"]),
        cfev=const(ASSUMPTIONS["cfev"]),
        rdmso=const(rootzone_m),
        irri=torch.zeros(n, dtype=dtype),
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
    sowing_doy: int,
    bundle: ExportBundle,
    dtype=torch.float32,
) -> SiteParameters:
    """Latitude from the grid, altitude and CO2 from the export.

    ``idpl`` must be the same ``sowing_doy`` the window was built from: the
    window places the spin-up ahead of that day and the model's latch opens on
    it. Grouping by sowing day-of-year is what makes one value correct for the
    whole batch.
    """
    _, lat = id_to_lonlat(ids, bundle.config.grid)
    altitude = bundle.site.altitude_m(ids)
    co2 = float(bundle.co2.get(year, bundle.co2.iloc[-1]))
    n = ids.size
    return SiteParameters(
        latitude=torch.as_tensor(lat, dtype=dtype),
        altitude=torch.as_tensor(altitude, dtype=dtype),
        co2=torch.full((n,), co2, dtype=dtype),
        plant_at_sowing=torch.ones(n, dtype=dtype),
        idpl=torch.full((n,), float(sowing_doy), dtype=dtype),
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


def run_batch(
    ids: np.ndarray,
    weather_np: np.ndarray,
    year: int,
    start_doy: int,
    sowing_doy: int,
    bundle: ExportBundle,
    device: torch.device,
) -> pd.DataFrame:
    """Two-pass run of one batch of cells for one season; per-cell summaries.

    Pass 1 is unfertilised and exists only to date the DVS-keyed fertilizer
    schedule; pass 2 is the run that is reported.

    The window runs well past maturity, which costs integrator steps but
    changes no summary below: every crop rate is gated on ``dvs < 2``, so
    yield, biomass and LAI freeze at maturity, and the stress means are taken
    over ``0 < DVS < 2`` — which also excludes the pre-sowing spin-up.
    """
    settings = bundle.config.torchcrop
    crop_params = CropParameters(crop_name=bundle.config.season.crop)
    crop_params.iopt = torch.tensor(float(settings.iopt))

    soil_params = build_soil_params(
        bundle.soil.select(ids), settings.rootzone_m, settings.profile_bottom_m
    )
    soil_params.irri = torch.as_tensor(
        bundle.irrigated.reindex(ids).fillna(0).to_numpy(float), dtype=torch.float32
    )
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
    engine_doy = (start_doy - 1 + np.arange(dvs.shape[1])) % 365 + 1
    sown = engine_doy >= sowing_doy
    sow_index = int(sown.argmax()) if sown.any() else 0
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
            "sowing_doy": np.int16(sowing_doy),
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
    return frame.astype(
        {c: np.float32 for c in frame.columns if frame[c].dtype == np.float64}
    )


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
) -> Path:
    """Run one shard over every requested season and write its Parquet."""
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / f"torchcrop_shard_{shard:03d}.parquet"
    if out_path.exists() and not overwrite:
        logger.info("shard %d already done: %s", shard, out_path)
        return out_path

    settings = config.torchcrop
    device = torch.device(settings.device)

    # A potential-yield run applies no fertilizer, so it needs no schedule.
    bundle = resolve_export(config, require_management=settings.iopt != 1)
    ids = bundle.shard(shard, n_shards)
    if max_cells and ids.size > max_cells:
        # Even subsample, not the head: SimplaceID runs north-to-south, so
        # truncating gives a smoke test a narrow Scandinavian band.
        ids = ids[:: int(np.ceil(ids.size / max_cells))][:max_cells]

    years = config.season.years
    groups = group_by_sowing(ids, bundle.site.sowing_doy(ids))
    logger.info(
        "shard %d/%d: %d cells (of %d), %d years, %d sowing group(s) %s, device=%s",
        shard, n_shards, ids.size, bundle.ids.size, len(years),
        len(groups), sorted(groups), device,
    )
    logger.info(bundle.site.summarise())

    results: list[pd.DataFrame] = []
    t_start = time.time()
    n_batches = sum(
        int(np.ceil(group.size / settings.batch_size)) for group in groups.values()
    )
    done = 0

    for doy, group in groups.items():
        window = season_window(years[0], doy, settings.spinup_months, SIM_YEARS)
        if window[0].dayofyear > sowing_date(years[0], doy).dayofyear:
            # The latch is `doy >= idpl` on a day-of-year, so a spin-up that
            # crosses New Year before sowing already satisfies it on day 0: the
            # crop would go in on the first day of the window with no spin-up.
            logger.warning(
                "sowing DOY %d leaves a spin-up that crosses New Year (window "
                "opens on DOY %d); the model will sow on the first day of the "
                "window", doy, window[0].dayofyear,
            )

        for start in range(0, group.size, settings.batch_size):
            batch = group[start : start + settings.batch_size]
            t0 = time.time()
            blocks = load_season_block(
                bundle.export_dir, batch, years, doy, config.grid,
                settings.io_workers, settings.spinup_months,
                settings.min_days_after_sowing,
            )
            t_io = time.time() - t0

            for year, (weather_np, start_doy) in blocks.items():
                results.append(
                    run_batch(batch, weather_np, year, start_doy, doy, bundle, device)
                )
            done += 1
            elapsed = time.time() - t_start
            logger.info(
                "batch %d/%d (DOY %d, %d cells): io %.1fs, total %.1fs, "
                "elapsed %.1f min, eta %.1f min",
                done, n_batches, doy, batch.size, t_io, time.time() - t0,
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
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
