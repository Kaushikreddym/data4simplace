"""Independent stress test: SIMPLACE and torchcrop on one crop, one sowing date.

    ./submit/submit_stresstest.py --dry-run     # audit the parameters, run nothing
    ./submit/submit_stresstest.py               # both scenarios, both seasons
    ./submit/submit_stresstest.py --scenarios limited --years 2003

This is not the smoke test with different settings. The smoke test asks *how
close to observations* each model gets and accepts that they differ in every
respect. This asks the narrower question the smoke test cannot answer — **given
the same crop, the same sowing date and no spin-up, do the two models agree?** —
by removing, one at a time, every difference that is not the model itself:

======================  ========================================================
Confound                How it is removed
======================  ========================================================
Crop parameters         torchcrop is loaded from SIMPLACE's own ``crop.xml``
                        (``--crop-params simplace``, the default). Run with the
                        shipped presets instead and 21 of 72 parameters differ,
                        including ``TSUM1`` (1623 vs 1050) and the leaf death
                        rates (4x) — the two would be growing different crops.
                        The audit runs either way and is written into the
                        workspace config.
Sowing date             SIMPLACE runs first and its **simulated** ``PlantingDOY``
                        (the rule-based solution decides it per cell and season)
                        is what torchcrop's latch is then set to. Not the site
                        table's date, which neither model would have used.
Spin-up                 Both start at sowing: torchcrop with
                        ``spinup_months: 0``, SIMPLACE with a window opening
                        ``--presow-days`` before the planting window. Both
                        therefore begin from the export's initial soil water
                        rather than from their own relaxed state.
Irrigation              **Removed from both.** The solution has no irrigation
                        module — ``vIRR`` appears in it once, as a column it
                        never reads — so torchcrop is forced rainfed too rather
                        than being the only model that can irrigate.
CO2                     **Removed from both**, held at the concentration where
                        the crop file's own response curve is 1.0 (360 ppm).
                        Left alone the solution uses a hard-coded 380 ppm for
                        every year while torchcrop reads the export's annual
                        series, so CO2 would differ both between the models and
                        between the seasons being compared.
Fertilizer              Not adjusted — **checked**. Both read the export's one
                        schedule, by different routes, and the run reports
                        whether the nitrogen they actually applied agrees per
                        cell and season (``fertilizer_check.csv``).
Nutrient limitation     Two scenarios, run by both models: ``potential``
                        (IOPT=1) and ``limited`` (IOPT=3).
======================  ========================================================

**The config is the input, not a report.** ``<out>/torchcrop/config.yaml`` is
written first, loaded back, and the resulting ``RunConfig`` is what both halves
are driven by; SIMPLACE's simulated sowing lands beside it as
``sowing_from_simplace.csv`` and is read back from there. So the two files in
that workspace are exactly what ran, and editing them and re-running with
``--reuse-simplace`` re-runs the torchcrop half as edited.

**What remains, recorded in the written config rather than papered over:**

1. *IOPT=1 is not potential production* — in either model, and they agree about
   that. ``vIOPT`` feeds one component of the solution (NPK demand and uptake)
   and torchcrop's ``iopt`` appears only in ``nutrient_demand.py``, while both
   apply water stress to growth unconditionally. The ``potential`` scenario is
   therefore nutrient-unlimited but still water-limited on both sides:
   symmetric, but not what the name suggests.
2. SIMPLACE integrates soil C/N and a layered water balance that torchcrop
   approximates with a bucket, so "no spin-up" starts them from initial states
   that are equivalent by construction but not identical.
"""

from __future__ import annotations

import argparse
import logging
import sys
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
import yaml

from cropmodelling4eu.config import RunConfig, load_config
from cropmodelling4eu.export import resolve_export
from cropmodelling4eu.export.weather import sowing_date
from cropmodelling4eu.simplace import (
    build_workspace,
    collect_run,
    run_lines,
    write_run_scripts,
)
from cropmodelling4eu.simplace.solution import read_solution
from cropmodelling4eu.torchcrop import params as crop_params_audit
from cropmodelling4eu.torchcrop.run import run_cells
from cropmodelling4eu.torchcrop.workspace import crop_file_name

logger = logging.getLogger("stresstest")

DEFAULT_CELLS = Path(
    "/data01/FDS/muduchuru/Data/SIMPLACE/cropmodelling4eu/de_smoke/de_cells.csv"
)

#: Where SIMPLACE's simulated sowing is handed to torchcrop, beside the config.
SOWING_FILE = "sowing_from_simplace.csv"

#: SIMPLACE's crop, written as a torchcrop preset and loaded through
#: ``CropParameters(config_file=...)`` -- so the crop torchcrop ran is a file in
#: the workspace, diffable against the bundled wheat.yaml. Named by
#: :func:`~cropmodelling4eu.torchcrop.workspace.crop_file_name`, the same
#: convention a production run's workspace uses, so one script reads either.
#: This ``torchcrop/`` directory *is* the run's working directory: its config,
#: its crop, its sowing dates and its outputs, flat.


@dataclass(frozen=True, slots=True)
class Scenario:
    """One nutrient setting, applied to both models. Both are rainfed."""

    name: str
    iopt: int
    note: str


SCENARIOS: dict[str, Scenario] = {
    "potential": Scenario(
        "potential", 1,
        "IOPT=1 -- nutrients non-limiting. Still water-limited on both "
        "sides, so this is not potential production: see known_asymmetries",
    ),
    "limited": Scenario(
        "limited", 3,
        "IOPT=3 -- water- and N-limited; the scenario the Europe runs use",
    ),
}

#: CO2 is held at the concentration where the crop file's own response curve
#: (``COTableCo2`` / ``COTableFactor``) is exactly 1.0, so the response is
#: *removed* rather than merely harmonised. Both models otherwise disagree
#: silently: the solution hard-codes ``vCO2 = 380`` for every year and cell
#: while torchcrop reads the export's annual series.
NEUTRAL_CO2_PPM = 360.0


# --------------------------------------------------------------------------- #
# SIMPLACE
# --------------------------------------------------------------------------- #


def simplace_config(
    config: RunConfig, year: int, scenario: Scenario, window_open_doy: int,
    presow_days: int,
) -> RunConfig:
    """The run config for one season, starting at that season's sowing window.

    The window opens ``presow_days`` before the planting window rather than in
    a previous year: that is what "no spin-up" means for SIMPLACE, whose
    ``start_date`` is otherwise years ahead of the crop.
    """
    start = sowing_date(year, window_open_doy) - pd.Timedelta(days=presow_days)
    return config.model_copy(
        update={
            "run_name": f"stress_{scenario.name}_{year}",
            "season": config.season.model_copy(
                update={"start_year": year, "end_year": year}
            ),
            "simplace": config.simplace.model_copy(
                update={
                    "iopt": scenario.iopt,
                    "start_date": start.strftime("%Y-%m-%d"),
                    "end_date": f"{year}-12-31",
                    "lines_per_task": 100_000,
                }
            ),
        }
    )


def run_simplace(
    config: RunConfig, ids: np.ndarray, root: Path, reuse: bool, co2_ppm: float
) -> pd.DataFrame:
    """Build, run and collect one SIMPLACE season; returns the yearly table."""
    parquet = root / "simplace_europe.parquet"
    if reuse and parquet.is_file():
        logger.info("reusing %s", parquet)
        return pd.read_parquet(parquet)

    bundle = resolve_export(config, require_management=config.simplace.iopt != 1)
    workspace = build_workspace(config, bundle, root=root, cells=ids, validate=True)

    # CO2 is a solution variable, not a config field, so it is patched in the
    # built workspace rather than in the pipeline every other run shares.
    document = read_solution(workspace.solution)
    changed = document.set_variables({"vCO2": co2_ppm})
    document.write(workspace.solution)
    logger.info("CO2 fixed at %.0f ppm (%s)", co2_ppm, ", ".join(changed) or "no vCO2")

    write_run_scripts(workspace, config)
    code = run_lines(workspace, config, 1, workspace.n_lines)
    if code != 0:
        raise SystemExit(f"SIMPLACE exited {code} for {config.run_name}")
    collect_run(workspace.out_dir, root, config.grid)
    return pd.read_parquet(parquet)


def sowing_from_simplace(simplace: pd.DataFrame) -> pd.DataFrame:
    """SIMPLACE's simulated sowing, as the table :func:`run_cells` takes.

    ``sowing_doy`` in the collected run is the day the solution's rules fired,
    which is the whole point: torchcrop is then latched to a date SIMPLACE
    produced rather than to the one the export proposed.

    That exported date is one day earlier than the day the crop actually
    starts growing in SIMPLACE, a one-day latch from the rule-based solution's
    component order (see the matching adjustment and its measurement in
    :func:`cropmodelling4eu.simplace.collect.sowing_table`). Compensated for
    here too, so the same latch does not reappear via the stresstest's own
    path to :data:`SOWING_FILE`.
    """
    table = (
        simplace[["SimplaceID", "year", "sowing_doy"]]
        .dropna()
        .astype({"SimplaceID": np.int64, "year": int, "sowing_doy": int})
    )
    table["sowing_doy"] = table["sowing_doy"] + 1
    return table.drop_duplicates(["SimplaceID", "year"])


# --------------------------------------------------------------------------- #
# torchcrop
# --------------------------------------------------------------------------- #


def run_torchcrop(
    config: RunConfig,
    ids: np.ndarray,
    years: list[int],
    scenario: Scenario,
    sowing: pd.DataFrame,
    crop_parameters,
    co2_ppm: float,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Summaries and daily trajectories for one scenario, no spin-up.

    Three inputs are removed rather than matched, because matching them is not
    possible: irrigation (the solution has no module to irrigate with), the CO2
    series (the solution takes one number) and the spin-up.
    """
    settings = config.torchcrop.model_copy(
        update={"iopt": scenario.iopt, "spinup_months": 0}
    )
    config = config.model_copy(update={"torchcrop": settings})
    bundle = resolve_export(config, require_management=scenario.iopt != 1)
    # Rainfed everywhere: SIMPLACE cannot irrigate, so an irrigated torchcrop
    # cell would be the one cell the two models are not running the same site.
    bundle.irrigated = pd.Series(0.0, index=bundle.ids)
    bundle.co2 = pd.Series(co2_ppm, index=bundle.co2.index)

    shared = dict(bundle=bundle, sowing=sowing, crop_params=crop_parameters)
    summary = run_cells(config, ids, years, "summary", **shared)
    daily = run_cells(config, ids, years, "daily", **shared)
    return summary.assign(scenario=scenario.name), daily.assign(scenario=scenario.name)


def check_fertilizer(simplace: pd.DataFrame, torchcrop: pd.DataFrame) -> pd.DataFrame:
    """Do both models apply the same nitrogen to the same cell and season?

    They read one schedule — the export's ``fertilizer_<crop>.csv`` — but by
    different routes: SIMPLACE takes the product amounts and the carrier
    contents from ``fertilizer_composition.xml`` inside the solution, while
    torchcrop's ``fertilizer_from_dvs`` converts them to nutrient rates before
    the run and keys them on its own pass-1 DVS trajectory. Two conversions of
    one file is exactly where a silent factor hides, so the run asserts the
    agreement instead of assuming it.
    """
    columns = ["scenario", "SimplaceID", "year", "n_applied_g_m2"]
    joined = simplace[columns].merge(
        torchcrop[columns], on=["scenario", "SimplaceID", "year"],
        suffixes=("_simplace", "_torchcrop"),
    )
    joined["difference"] = (
        joined["n_applied_g_m2_simplace"] - joined["n_applied_g_m2_torchcrop"]
    )
    worst = joined["difference"].abs().max()
    agree = int((joined["difference"].abs() < 1e-3).sum())
    (logger.info if agree == len(joined) else logger.warning)(
        "fertilizer N: %d/%d cell-seasons agree within 1e-3 g/m2 "
        "(largest difference %.4f, mean applied %.3f g N/m2)",
        agree, len(joined), worst, joined["n_applied_g_m2_simplace"].mean(),
    )
    return joined


# --------------------------------------------------------------------------- #
# Driver
# --------------------------------------------------------------------------- #


def load_stresstest_config(path: Path) -> tuple[RunConfig, dict]:
    """Read the workspace config back as ``(RunConfig, stresstest block)``.

    The file is one document holding both, and ``RunConfig`` forbids unknown
    keys, so the ``stresstest`` block is popped before validation. That block
    is provenance — the audit, the scenarios, the sowing source — and lives
    beside the settings rather than in a second file that can drift from them.
    """
    payload = yaml.safe_load(Path(path).read_text())
    stresstest = payload.pop("stresstest", {})
    return RunConfig.model_validate(payload), stresstest


def write_workspace_config(
    path: Path, config: RunConfig, args: argparse.Namespace,
    scenarios: list[Scenario], comparison: pd.DataFrame, ids: np.ndarray,
) -> None:
    """The config the torchcrop side is then *driven by*, in its own workspace.

    Not a report of the run: it is written first, loaded back by
    :func:`load_stresstest_config`, and the resulting ``RunConfig`` is what the
    torchcrop runs use. Editing it and re-running with ``--reuse-simplace``
    therefore re-runs torchcrop exactly as edited, and the file on disk cannot
    describe settings that were not the ones applied.

    It also carries the parameter audit and the asymmetries the design could
    not remove, so the outputs beside it can be read months later without
    re-deriving any of it.
    """
    differing = comparison[comparison["status"] != "same"]
    payload = {
        "stresstest": {
            "purpose": (
                "SIMPLACE vs torchcrop with the same crop parameters, the same "
                "simulated sowing date and no spin-up"
            ),
            "created_from": str(args.config),
            "seasons": [int(y) for y in args.years],
            "cells": {"file": str(args.cells), "n": int(ids.size)},
            "spinup": {
                "torchcrop_spinup_months": 0,
                "simplace_presow_days": args.presow_days,
                "note": "both models start at the season's sowing window, from "
                        "the export's initial soil water",
            },
            "sowing": {
                "source": "simplace_simulated_idpl",
                "file": str(path.parent / SOWING_FILE),
                "note": "SIMPLACE runs first; its rule-based PlantingDOY per "
                        "(cell, season) is written to that file and read back "
                        "as torchcrop's idpl latch",
            },
            "removed_inputs": {
                "irrigation": "every cell rainfed in both models (vIRR is not "
                              "read by the solution and is forced to 0 in "
                              "torchcrop), so neither model irrigates",
                "co2": f"held at {args.co2_ppm:.0f} ppm in both models -- the "
                       "concentration at which the crop file's own response "
                       "curve is 1.0, so the CO2 response is removed rather "
                       "than harmonised. The solution's vCO2 is patched in the "
                       "built workspace; torchcrop's annual series is replaced "
                       "by the same constant",
                "spinup": "no spin-up on either side",
            },
            "scenarios": {
                s.name: {"iopt": s.iopt, "note": s.note} for s in scenarios
            },
            "crop_parameters": {
                "torchcrop_source": args.crop_params,
                "file": (str(path.parent / crop_file_name(config.season.crop))
                         if args.crop_params == "simplace" else None),
                "simplace_crop_xml": str(args.crop_xml),
                "simplace_seeds_xml": str(args.seeds_xml),
                "simplace_management_xml": str(args.management_xml),
                "note": "the file is a complete torchcrop preset built from "
                        "crop.xml plus the NRF/PRF/KRF recovery fractions of "
                        "management.xml, loaded with "
                        "CropParameters(config_file=...). seeds.xml is recorded "
                        "in it but not mapped: torchcrop has no SeedsToSprouts "
                        "counterpart",
                "audit": {
                    status: int(n)
                    for status, n in comparison["status"].value_counts().items()
                },
                "differing": [
                    {"parameter": row["parameter"], "kind": row["kind"],
                     "simplace": row["simplace"], "torchcrop": row["torchcrop"]}
                    for _, row in differing.iterrows()
                ],
            },
            "known_asymmetries": [
                "IOPT=1 is not potential production in either model, and the "
                "two agree about that. vIOPT feeds exactly one component of "
                "the solution -- NPK demand and uptake -- while "
                "LintulWaterStress.TRANRF is wired into biomass growth "
                "unconditionally; torchcrop's iopt likewise appears only in "
                "nutrient_demand.py. So the 'potential' scenario is "
                "nutrient-unlimited but still water-limited on both sides, "
                "which is symmetric but is not what the name suggests. With "
                "irrigation removed the two scenarios differ by ~2-8% in "
                "either model; the apparent large IOPT effect in an earlier "
                "run was irrigation, not IOPT.",
                "SIMPLACE integrates soil C/N and a layered water balance where "
                "torchcrop uses a two-bucket approximation, so 'no spin-up' "
                "starts them from equivalent but not identical states.",
            ],
        },
    }
    # The rest of the file is a complete, valid RunConfig — dumped from the
    # object rather than assembled field by field, so a field added to the
    # model later cannot go missing here and silently take its default.
    payload |= (
        config.model_copy(
            update={
                "run_name": "stresstest",
                "season": config.season.model_copy(
                    update={"start_year": min(args.years), "end_year": max(args.years)}
                ),
                "torchcrop": config.torchcrop.model_copy(
                    update={"spinup_months": 0}
                ),
            }
        ).model_dump(mode="json")
    )
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w") as handle:
        yaml.safe_dump(payload, handle, sort_keys=False, default_flow_style=False)
    logger.info("wrote %s", path)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--config", type=Path,
                        default=Path(__file__).resolve().parents[1] / "config.yaml")
    parser.add_argument("--cells", type=Path, default=DEFAULT_CELLS)
    parser.add_argument("--out-dir", type=Path)
    parser.add_argument("--years", type=int, nargs="+", default=[2003, 2005])
    parser.add_argument("--scenarios", nargs="+", default=list(SCENARIOS),
                        choices=list(SCENARIOS))
    parser.add_argument("--crop-params", choices=("simplace", "torchcrop"),
                        default="simplace",
                        help="which crop parameters torchcrop runs with")
    parser.add_argument("--crop-xml", type=Path,
                        help="SIMPLACE crop.xml; default is the template's")
    parser.add_argument("--seeds-xml", type=Path,
                        help="SIMPLACE seeds.xml; recorded, not mapped")
    parser.add_argument("--management-xml", type=Path,
                        help="SIMPLACE management.xml, for the NRF/PRF/KRF "
                             "recovery fractions; default is the template's")
    parser.add_argument("--presow-days", type=int, default=0,
                        help="days SIMPLACE runs before the planting window opens")
    parser.add_argument("--co2-ppm", type=float, default=NEUTRAL_CO2_PPM,
                        help="CO2 both models are held at; the default is where "
                             "the crop file's response curve equals 1.0")
    parser.add_argument("--reuse-simplace", action="store_true",
                        help="skip a SIMPLACE run whose Parquet already exists")
    parser.add_argument("--dry-run", action="store_true",
                        help="audit the parameters and write the config; run nothing")
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)-7s %(message)s",
                        datefmt="%H:%M:%S")
    config = load_config(args.config)
    args.out_dir = args.out_dir or config.paths.output_dir / "stresstest"
    template_data = Path(config.simplace.template_dir) / "data"
    args.crop_xml = args.crop_xml or template_data / "crop" / "crop.xml"
    args.seeds_xml = args.seeds_xml or template_data / "crop" / "seeds.xml"
    args.management_xml = (
        args.management_xml or template_data / "management" / "management.xml"
    )
    ids = pd.read_csv(args.cells)["SimplaceID"].to_numpy(np.int64)
    scenarios = [SCENARIOS[name] for name in args.scenarios]

    # --- 1. The audit, before anything is run -------------------------------
    comparison = crop_params_audit.compare_crop_parameters(
        args.crop_xml, crop_name=config.season.crop,
        management_xml=args.management_xml,
    )
    print("\n" + crop_params_audit.summarise(comparison) + "\n")
    args.out_dir.mkdir(parents=True, exist_ok=True)
    comparison.to_csv(args.out_dir / "crop_parameter_audit.csv", index=False)

    crop_parameters = None
    if args.crop_params == "simplace":
        crop_parameters = crop_params_audit.crop_parameters_from_simplace(
            args.crop_xml, crop_name=config.season.crop,
            seeds_xml=args.seeds_xml, management_xml=args.management_xml,
            yaml_path=args.out_dir / "torchcrop" / crop_file_name(config.season.crop),
        )
    elif (comparison["status"] != "same").any():
        logger.warning(
            "--crop-params torchcrop: the two models will run %d different "
            "parameters, so a disagreement below is not attributable to the model",
            int((comparison["status"] != "same").sum()),
        )

    # --- 2. The workspace config, which is then what the runs are driven by --
    config_path = args.out_dir / "torchcrop" / "config.yaml"
    write_workspace_config(config_path, config, args, scenarios, comparison, ids)
    config, provenance = load_stresstest_config(config_path)
    logger.info("driving both models from %s", config_path)
    if args.dry_run:
        print(f"--dry-run: audit and config written under {args.out_dir}; nothing run.")
        return 0

    # --- 3. Per scenario: SIMPLACE, then torchcrop on its sowing dates -------
    years = [int(y) for y in provenance["seasons"]]
    window_open = int(
        pd.read_csv(Path(config.paths.export_dir) / "site" / "site.csv")
        .set_index("location").loc[ids, "sowing_start_doy"].min()
    )
    simplace_runs, torchcrop_runs, torchcrop_daily = [], [], []
    for scenario in scenarios:
        for year in years:
            logger.info("=== %s %d: SIMPLACE ===", scenario.name, year)
            frame = run_simplace(
                simplace_config(config, year, scenario, window_open, args.presow_days),
                ids,
                args.out_dir / "simplace" / scenario.name / str(year),
                args.reuse_simplace,
                args.co2_ppm,
            )
            simplace_runs.append(frame.assign(scenario=scenario.name))

        # Through the file, not through a variable: the sowing dates torchcrop
        # ran on are then inspectable beside its outputs, and re-running the
        # torchcrop half alone reads exactly what SIMPLACE produced.
        simplace = pd.concat(simplace_runs[-len(years):], ignore_index=True)
        sowing_path = config_path.parent / SOWING_FILE
        sowing_from_simplace(simplace).to_csv(sowing_path, index=False)
        sowing = pd.read_csv(sowing_path)
        logger.info(
            "=== %s: torchcrop on SIMPLACE's sowing from %s (DOY %d-%d) ===",
            scenario.name, sowing_path.name,
            sowing["sowing_doy"].min(), sowing["sowing_doy"].max(),
        )
        summary, daily = run_torchcrop(
            config, ids, years, scenario, sowing, crop_parameters, args.co2_ppm
        )
        torchcrop_runs.append(summary)
        torchcrop_daily.append(daily)

    # --- 4. Outputs ----------------------------------------------------------
    out = args.out_dir
    simplace = pd.concat(simplace_runs, ignore_index=True)
    torchcrop = pd.concat(torchcrop_runs, ignore_index=True)
    daily = pd.concat(torchcrop_daily, ignore_index=True)
    fertilizer = check_fertilizer(simplace, torchcrop)
    fertilizer.to_csv(out / "fertilizer_check.csv", index=False)
    simplace.to_parquet(out / "stresstest_simplace.parquet", index=False)
    torchcrop.to_parquet(out / "torchcrop" / "stresstest_torchcrop.parquet", index=False)
    daily.to_parquet(out / "torchcrop" / "stresstest_torchcrop_daily.parquet", index=False)

    paired = (
        simplace.assign(model="simplace")[
            ["model", "scenario", "SimplaceID", "year", "yield_t_ha", "sowing_doy",
             "max_lai", "maturity_doy"]
        ]
        .pipe(lambda f: pd.concat([
            f,
            torchcrop.assign(model="torchcrop", yield_t_ha=torchcrop["yield_g_m2"] / 100.0)[
                ["model", "scenario", "SimplaceID", "year", "yield_t_ha", "sowing_doy",
                 "max_lai"]
            ].assign(maturity_doy=np.nan),
        ], ignore_index=True))
    )
    table = (
        paired.groupby(["scenario", "year", "model"])
        .agg(n=("SimplaceID", "size"), yield_t_ha=("yield_t_ha", "mean"),
             max_lai=("max_lai", "mean"), sowing_doy=("sowing_doy", "mean"))
        .round(2)
    )
    table.to_csv(out / "stresstest_comparison.csv")

    print("\n" + "=" * 78)
    print("STRESS TEST — mean over cells")
    print("=" * 78)
    print(table.to_string())
    print(f"\nOutputs   : {out}")
    print(f"Config    : {out / 'torchcrop' / 'config.yaml'}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
