"""Build the two evaluation notebooks from cell lists.

Run once: ``python _build_notebooks.py``. Kept in the repo so the notebooks can
be regenerated from a diffable source rather than hand-edited JSON.
"""

from __future__ import annotations

import json
from pathlib import Path

HERE = Path(__file__).resolve().parent


_counter = {"n": 0}


def _next_id() -> str:
    """Stable per-notebook cell id, so a rebuild produces a clean diff."""
    _counter["n"] += 1
    return f"cell-{_counter['n']:03d}"


def md(text: str) -> dict:
    return {
        "cell_type": "markdown", "id": _next_id(), "metadata": {},
        "source": text.strip("\n").splitlines(keepends=True),
    }


def code(text: str) -> dict:
    return {
        "cell_type": "code", "id": _next_id(), "execution_count": None,
        "metadata": {}, "outputs": [],
        "source": text.strip("\n").splitlines(keepends=True),
    }


def write(name: str, cells: list[dict]) -> None:
    nb = {
        "cells": cells,
        "metadata": {
            "kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
            "language_info": {"name": "python", "version": "3.10"},
        },
        "nbformat": 4,
        "nbformat_minor": 5,
    }
    path = HERE / name
    path.write_text(json.dumps(nb, indent=1, ensure_ascii=False) + "\n")
    print(f"wrote {path} ({len(cells)} cells)")


# --------------------------------------------------------------------------- #
# Shared preamble
# --------------------------------------------------------------------------- #

SETUP = '''
import logging
import sys
from pathlib import Path

# The notebooks live beside utils/, so they import it without installation.
sys.path.insert(0, str(Path.cwd()))

import numpy as np
import pandas as pd

from utils import aggregate, config, cybench, doy, metrics, plots, regions, torchcrop
from utils.style import use_style

logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s",
                    force=True)
logging.getLogger("matplotlib").setLevel(logging.WARNING)

PALETTE = use_style("light")
config.ensure_output_dirs()

pd.set_option("display.max_rows", 60)
pd.set_option("display.width", 140)

print(f"TorchCrop run : {config.TORCHCROP_RUN_DIR}")
print(f"CyBench root  : {config.CYBENCH_ROOT}")
print(f"Outputs       : {config.OUTPUT_DIR}")
'''

SCOPE = '''
COUNTRIES = config.available_countries()

scope = pd.DataFrame({
    "code": list(config.TARGET_COUNTRIES),
    "name": [config.COUNTRY_NAMES.get(c, c) for c in config.TARGET_COUNTRIES],
    "eu27": [c in config.EU27 for c in config.TARGET_COUNTRIES],
    "schengen_non_eu": [c in config.SCHENGEN_NON_EU for c in config.TARGET_COUNTRIES],
    "in_cybench": [c in COUNTRIES for c in config.TARGET_COUNTRIES],
}).set_index("code")

print(f"{len(COUNTRIES)} of {len(config.TARGET_COUNTRIES)} target countries are evaluable")
print("missing:", ", ".join(scope.index[~scope["in_cybench"]]))
scope
'''

CELLS = '''
cells = torchcrop.simulation_cells(sim)
cells = regions.assign_cells_to_countries(
    cells, COUNTRIES, cache=regions.default_cache_path(len(COUNTRIES))
)

sim = sim.merge(cells[["SimplaceID", "country", "snapped"]], on="SimplaceID", how="left")
scoped = sim[sim["country"].notna()].copy()

print(f"{cells['country'].notna().sum():,} of {len(cells):,} cells inside the "
      f"CyBench footprint ({cells['snapped'].sum():,} matched by the {config.SNAP_KM} km snap)")
print(f"{len(scoped):,} of {len(sim):,} simulated cell-seasons kept")

cells_per_country = (
    cells.dropna(subset=["country"]).groupby("country").size().rename("cells").to_frame()
)
cells_per_country["share_%"] = (
    100 * cells_per_country["cells"] / cells_per_country["cells"].sum()
).round(1)
cells_per_country.T
'''


# --------------------------------------------------------------------------- #
# Notebook 1 — yield
# --------------------------------------------------------------------------- #

yield_cells = [
    md("""
# Winter wheat yield — TorchCrop against CyBench

**What this notebook does.** It aggregates the 10 km TorchCrop winter-wheat
yield simulation to national averages, compares them with the CyBench
sub-national yield statistics aggregated the same way, and reports RMSE, MAE,
Bias, MAPE, R² and Pearson r per country and pooled.

**Scope.** EU-27 and Schengen, excluding Ukraine and Russia. CyBench publishes
wheat for 23 of those 31 countries; the rest are listed in §1 and dropped.

**Three conventions worth knowing before reading any number:**

1. **Error is `simulated − observed`.** A negative bias means the model is low.
2. **No moisture conversion is applied.** TorchCrop's LINTUL-5 reports grain
   **dry matter**; CyBench reports the national statistics at market moisture
   (~13.5 % for wheat). Both sides are compared as published, so a systematic
   offset of roughly that size is expected on top of any model error. To put
   both on dry matter instead, call `aggregate.to_dry_matter` on the
   observations — everything downstream is unchanged.
3. **The simulated national mean is an unweighted mean of cropland cells.** The
   SIMPLACE export carries no per-cell wheat area, so no weight can be applied
   without inventing one. The observed national mean *is* area-weighted, by
   `harvest_area`, wherever enough regions report it.

All reusable code is in [`utils/`](utils/); this notebook is the workflow only.
"""),
    md("## 0. Setup"),
    code(SETUP),
    md("""
## 1. Scope — which countries are evaluable

A country needs a CyBench yield file *and* CyBench polygons: the file supplies
the reference, the polygons decide which 10 km cells belong to it.
"""),
    code(SCOPE),
    md("""
## 2. Load the TorchCrop simulation

One row per (cell, season). `year` is the **harvest** year — winter wheat is
sown in the preceding autumn.
"""),
    code('''
sim = torchcrop.load_simulation(columns=[
    "SimplaceID", "year", "lon", "lat", "yield_t_ha", "biomass_g_m2",
    "max_lai", "days_to_maturity", "tranrf_mean", "nni_mean",
    "heat_stress_factor", "n_applied_g_m2", "irri",
])

print(f"{len(sim):,} cell-seasons, {sim['SimplaceID'].nunique():,} cells, "
      f"{sim['year'].min()}-{sim['year'].max()}")
sim[["yield_t_ha", "biomass_g_m2", "max_lai", "days_to_maturity"]].describe().T
'''),
    md("""
## 3. Assign each 10 km cell to a country

The national footprints are the CyBench administrative polygons dissolved per
country, so both sides of the comparison cover the same ground. Cells outside
that footprint — the UK, Norway, Switzerland, the western Balkans, North
Africa — are dropped.
"""),
    code(CELLS),
    md("""
## 4. Aggregate the simulation to national means
"""),
    code('''
sim_country = aggregate.aggregate_simulated(scoped, {"yield_t_ha": False})
sim_country.head()
'''),
    md("""
## 5. Load and aggregate the CyBench observations

Weighted by `harvest_area` wherever at least half a country's regions report
one, which makes the national value `Σ production / Σ area` rather than the
mean of regional yields. Where it does not, the aggregation falls back to the
unweighted mean and records that in `obs_method` — Germany reports no area for
three quarters of its rows.
"""),
    code('''
obs = cybench.load_yield(COUNTRIES)
obs_country = aggregate.aggregate_observed_yield(obs)

print(obs_country["obs_method"].value_counts().to_string())
obs_country.head()
'''),
    md("""
## 6. Pair the two sides

An inner join on `(country, year)`. What it drops is logged: simulated years
CyBench does not cover (2021–2024 for most countries) and observed years before
the run starts (pre-2000).
"""),
    code('''
paired = aggregate.pair_observations(sim_country, obs_country, ["country", "year"])
paired["residual"] = paired["yield_t_ha"] - paired["obs_yield"]

coverage = (
    paired.groupby("country")
    .agg(n_years=("year", "size"), first=("year", "min"), last=("year", "max"),
         cells=("n_cells", "median"), regions=("n_regions", "median"))
    .sort_values("n_years")
)
print(f"{len(paired)} paired country-years across {paired['country'].nunique()} countries")
coverage.T
'''),
    md("""
## 7. Metrics

`R²` is the coefficient of determination `1 − SS_res/SS_tot`, **not** the square
of Pearson r — it goes negative when the simulation predicts worse than the
observed mean, which is the informative case for a process model carrying a
systematic offset. Pearson r is reported separately, so a country whose
interannual pattern is right but whose level is wrong is still visible.
"""),
    code('''
pooled = metrics.yield_metrics(paired["obs_yield"], paired["yield_t_ha"])
print("Pooled over every country-year:")
for key in metrics.YIELD_METRIC_ORDER:
    print(f"  {metrics.METRIC_LABELS[key]:>16s}  {pooled[key]:>8.2f}")
'''),
    code('''
by_country = metrics.metrics_by_group(paired, "obs_yield", "yield_t_ha")
ranked = metrics.rank_countries(by_country, by="rmse")

ranked.drop(columns=["sparse", "pearson_p"]).to_csv(
    config.TABLE_DIR / "yield_metrics_by_country.csv", float_format="%.3f"
)
plots.metric_table(
    ranked, ("rank", *metrics.YIELD_METRIC_ORDER),
    caption="Country yield skill, ranked by RMSE (best first). "
            "Bias and R² are in t/ha and dimensionless; countries with fewer "
            "than three paired years are listed last, unranked.",
)
'''),
    md("""
### Ranked by correlation instead

RMSE ranks by usability. Pearson r ranks by whether the model tracks the *shape*
of the interannual variation, which is a different question and gives a
different order.
"""),
    code('''
plots.metric_table(
    metrics.rank_countries(by_country, by="pearson_r", ascending=False),
    ("rank", "n", "pearson_r", "pearson_p", "bias", "rmse"),
    gradient_on=("pearson_r",),
    caption="The same countries ranked by interannual correlation.",
)
'''),
    md("""
## 8. Figures

### 8.1 Observed against simulated

Every country-year in one cloud. With 23 countries on screen colour cannot
carry identity — the categorical palette separates at most eight hues — so the
points share one hue and the countries are separated by panel in §8.2 instead.
The three largest under-predictions are highlighted.
"""),
    code('''
worst = ranked.drop(index="ALL").nsmallest(3, "bias").index.tolist()

fig = plots.scatter_one_to_one(
    paired, "obs_yield", "yield_t_ha", stats=pooled,
    title="National winter wheat yield, 2000-2020",
    xlabel="CyBench observed yield (t ha$^{-1}$)",
    ylabel="TorchCrop simulated yield (t ha$^{-1}$)",
    highlight=worst,
)
plots.save(fig, "yield_01_scatter_one_to_one")
fig
'''),
    md("""
### 8.2 One panel per country

Shared axes across every panel, so a country's distance from the diagonal is
comparable at a glance. Panels are ordered by RMSE, best first.
"""),
    code('''
fig = plots.scatter_small_multiples(
    paired, "obs_yield", "yield_t_ha", metrics=by_country,
    order=[c for c in ranked.index if c != "ALL"],
    title="Observed against simulated yield, by country (ordered by RMSE)",
    xlabel="CyBench observed yield (t ha$^{-1}$)",
    ylabel="TorchCrop simulated yield (t ha$^{-1}$)",
)
plots.save(fig, "yield_02_scatter_small_multiples")
fig
'''),
    md("""
### 8.3 Country-wise bias

The fill carries the **sign** only — the bar's length already encodes
magnitude. The whisker is ± RMSE, which is always at least |bias|; where it is
much larger, the country's error is scatter rather than offset.
"""),
    code('''
fig = plots.bias_bars(
    ranked, value_col="bias", error_col="rmse",
    title="Mean yield bias by country (simulated - observed)",
    xlabel="Bias (t ha$^{-1}$)",
)
plots.save(fig, "yield_03_bias_by_country")
fig
'''),
    md("""
### 8.4 Residuals

Three views, because they fail differently: against the observation (a slope
error shows as a trend), against time (drift or a bad year), and as a
distribution (shift versus spread).
"""),
    code('''
fig = plots.residual_panels(
    paired, "residual", "obs_yield",
    title="Yield residuals",
    ylabel="Residual (t ha$^{-1}$)",
    xlabel_obs="CyBench observed yield (t ha$^{-1}$)",
)
plots.save(fig, "yield_04_residuals")
fig
'''),
    md("""
### 8.5 Time series by country
"""),
    code('''
fig = plots.timeseries_small_multiples(
    paired, "obs_yield", "yield_t_ha",
    order=[c for c in ranked.index if c != "ALL"],
    title="National yield through time",
    ylabel="Yield (t ha$^{-1}$)",
)
plots.save(fig, "yield_05_timeseries")
fig
'''),
    md("""
### 8.6 Maps

The native 10 km field first — the country means above are averages over this —
then the country bias.
"""),
    code('''
polygons = regions.load_country_polygons(COUNTRIES)

cell_mean = (
    scoped.groupby(["SimplaceID", "lon", "lat"], as_index=False)["yield_t_ha"].mean()
)

fig = plots.cell_map(
    cell_mean, "yield_t_ha",
    title=f"Simulated mean winter wheat yield, {paired['year'].min()}-{paired['year'].max()}",
    cbar_label="Yield (t ha$^{-1}$)", overlay=polygons,
)
plots.save(fig, "yield_06_map_simulated_cells")
fig
'''),
    code('''
fig = plots.country_choropleth(
    polygons, ranked["bias"],
    title="Mean yield bias (simulated - observed)",
    cbar_label="Bias (t ha$^{-1}$)", diverging=True,
)
plots.save(fig, "yield_07_map_bias")
fig
'''),
    md("""
## 9. Diagnostics — where the failure is, and what it looks like

The bias is not a uniform offset: it is near zero across the Baltic and central
Europe and catastrophic around the North Sea. That pattern is not something the
metrics explain on their own, so this section pulls the run's own state
variables alongside the bias to say what the model is doing in the countries it
gets wrong.
"""),
    code('''
state = (
    scoped.groupby("country")
    .agg(sim_yield=("yield_t_ha", "mean"),
         max_lai=("max_lai", "mean"),
         biomass=("biomass_g_m2", "mean"),
         days_to_maturity=("days_to_maturity", "mean"),
         water_stress=("tranrf_mean", "mean"),
         n_index=("nni_mean", "mean"),
         heat=("heat_stress_factor", "mean"),
         failed_share=("yield_t_ha", lambda s: float((s < 0.5).mean())))
)
state["obs_yield"] = ranked["obs_mean"]
state["bias"] = ranked["bias"]

print("Correlation of country bias with the run's own state variables:")
print(state.corr(numeric_only=True)["bias"].drop("bias").round(2).sort_values().to_string())

state.sort_values("bias").round(3)
'''),
    md("""
`failed_share` is the fraction of cell-seasons yielding under 0.5 t ha⁻¹ — an
effective crop failure. Read it next to `max_lai`: where the simulated canopy
never closes, there is no yield to speak of, and the country's bias is that
failure rather than a calibration offset.
"""),
    code('''
fig = plots.cell_map(
    scoped.groupby(["SimplaceID", "lon", "lat"], as_index=False)["max_lai"].mean(),
    "max_lai",
    title="Simulated maximum leaf area index (mean over seasons)",
    cbar_label="Max LAI (m$^2$ m$^{-2}$)", overlay=polygons,
)
plots.save(fig, "yield_08_map_max_lai")
fig
'''),
    md("""
## 10. Summary
"""),
    code('''
summary = pd.Series({
    "countries": paired["country"].nunique(),
    "country-years": len(paired),
    "years": f"{paired['year'].min()}-{paired['year'].max()}",
    "10 km cells": int(cells["country"].notna().sum()),
    "observed mean (t/ha)": round(pooled["obs_mean"], 2),
    "simulated mean (t/ha)": round(pooled["sim_mean"], 2),
    "bias (t/ha)": round(pooled["bias"], 2),
    "RMSE (t/ha)": round(pooled["rmse"], 2),
    "MAE (t/ha)": round(pooled["mae"], 2),
    "MAPE (%)": round(pooled["mape"], 1),
    "R2": round(pooled["r2"], 2),
    "Pearson r": round(pooled["pearson_r"], 2),
}, name="value")

paired.to_csv(config.TABLE_DIR / "yield_paired_country_year.csv",
              index=False, float_format="%.4f")
state.to_csv(config.TABLE_DIR / "yield_diagnostics_by_country.csv",
             float_format="%.4f")
print(f"tables written to {config.TABLE_DIR}")
print(f"figures written to {config.FIGURE_DIR}")
summary.to_frame()
'''),
    md("""
### What the numbers mean, and what they do not

* **The pooled skill is dominated by a level error, not by noise.** The bias is
  a large share of the RMSE, and the residual trends with the observation — the
  model reproduces less of the high-yielding end than of the low.
* **Part of that offset is a units difference, not model error.** No moisture
  conversion was applied (see the header); the simulated dry matter is expected
  to sit ~13.5 % below a market-moisture statistic before any model error is
  counted. That accounts for a fraction of the offset, not for the countries
  simulating near-total crop failure.
* **The maritime north-west is a distinct failure, not a worse calibration.**
  §9 shows the simulated canopy never developing there, so those countries'
  metrics measure a model failure, not a yield gap. Excluding them changes the
  pooled numbers substantially — quote the per-country table, not the pooled
  row, for anything downstream.
* **The observed side has its own limits.** CyBench's national averages come
  from sub-national statistics with uneven area reporting; `obs_method` records
  where the weighting fell back.
"""),
]


# --------------------------------------------------------------------------- #
# Notebook 2 — phenology
# --------------------------------------------------------------------------- #

phenology_cells = [
    md("""
# Winter wheat phenology — TorchCrop against CyBench

**What this notebook does.** It derives the simulated stage dates from the
10 km TorchCrop run, averages them to national dates, and compares them with
the CyBench crop calendar. RMSE, MAE, Bias and Pearson r are reported per stage
and per country.

## What each side actually contains

Read this before the results — two of the three stage pairings are proxies, and
the fourth requested stage does not exist on either side.

| Stage | TorchCrop | CyBench | Comparable? |
|---|---|---|---|
| Sowing | **constant input**, DOY 270 for every cell | `sos` | weakly — see below |
| Flowering | **not in the run** | **not in the calendar** | no |
| Maturity | first day at `DVS ≥ 2` | `eos` | yes, best pairing |
| Harvest | not modelled; = maturity + `HARVEST_LAG_DAYS` (0) | `eos` | as maturity |

* **Sowing is not a prediction.** The SIMPLACE export carries no sowing
  calendar, so `torchcrop_run.py` sows every cell from Crete to Lapland on
  DOY 270. Its only degree of freedom is one continental offset.
* **CyBench `sos` is not a sowing date, and it changes meaning with climate.**
  It is ~DOY 330–360 in Spain, Greece and Portugal (autumn sowing) but ~DOY
  40–75 in Germany, Poland and the Baltics (post-winter green-up). The northern
  residuals are a convention difference, not a model error; §7 separates the
  two regimes so this is visible rather than buried in a mean.
* **Flowering is absent from both.** `run_batch` discards the DVS trajectory
  once it has dated the fertilizer schedule, so no anthesis date reaches the
  Parquet, and the CyBench calendar carries only a season start and end. It is
  therefore not evaluated here. Emitting the `DVS ≥ 1` crossing in
  `submit/torchcrop_run.py` and re-running the shards would supply the
  simulated side; the reference side would still need another source.
* **Harvest is maturity.** LINTUL-5 stops the crop at `DVS = 2` and models no
  drydown. The row is kept because the assumption should be visible, not
  because it adds information; raise `config.HARVEST_LAG_DAYS` to test a lag.

## Two conventions

1. **Error is `simulated − observed`**, so a positive bias means the model is
   **late**.
2. **Every date statistic is circular.** A day-of-year lives on a circle:
   means go through `doy.circular_mean_doy`, errors through
   `doy.doy_difference`, and correlations are computed on both sides unwrapped
   about the observed circular mean. Without that, Spain's regional `sos` of
   DOY 363 and DOY 0.7 average to July.

All reusable code is in [`utils/`](utils/); this notebook is the workflow only.
"""),
    md("## 0. Setup"),
    code(SETUP),
    md("## 1. Scope — which countries are evaluable"),
    code(SCOPE),
    md("""
## 2. Load the simulation and derive the stage dates

`days_to_maturity` counts from the sowing latch, so maturity is
`sowing_doy + days_to_maturity` wrapped back into the year: a crop sown on
DOY 270 and maturing 272 days later matures on DOY 177 of the following
calendar year, which is the harvest year the row is labelled with.
"""),
    code('''
sim = torchcrop.load_simulation(columns=[
    "SimplaceID", "year", "lon", "lat", "days_to_maturity", "final_dvs", "yield_t_ha",
])

print("stages derived:", ", ".join(torchcrop.PHENOLOGY_COLUMNS))
for stage in config.STAGES:
    print(f"  {stage.label:9s} <- {stage.sim_col:14s} vs CyBench '{stage.obs_col}'")

sim[list(torchcrop.PHENOLOGY_COLUMNS)].describe().T
'''),
    md("## 3. Assign each 10 km cell to a country"),
    code(CELLS),
    md("""
## 4. Aggregate the simulation to national dates

Every stage column is flagged circular, so the national date is a circular
mean. `season_length_days` is a **duration**, not a date, so it stays linear.
"""),
    code('''
STAGE_COLS = {stage.sim_col: True for stage in config.STAGES}
STAGE_COLS["season_length_days"] = False

sim_country = aggregate.aggregate_simulated(scoped, STAGE_COLS)
sim_country.head()
'''),
    md("""
## 5. Load and aggregate the CyBench crop calendar

The calendar is **static** — one `sos`/`eos` per administrative unit, with no
year dimension. Every simulated season is therefore compared against the same
reference date, which means interannual scatter in the simulation can only be
penalised, never rewarded. Regions are weighted by `crop_area` from the CyBench
crop mask, so a 200 ha alpine district does not outvote a 200 000 ha plain.
"""),
    code('''
calendar = cybench.load_calendar(COUNTRIES)
crop_mask = cybench.load_crop_mask(COUNTRIES)
obs_country = aggregate.aggregate_observed_calendar(calendar, crop_mask)

obs_country["sos_date"] = obs_country["sos"].map(lambda d: doy.doy_to_month_day(d))
obs_country["eos_date"] = obs_country["eos"].map(lambda d: doy.doy_to_month_day(d))
obs_country.set_index("country").round(1)
'''),
    md("""
### The two `sos` conventions

The split is the reason a single "sowing" metric would be meaningless. A
country whose observed `sos` falls in the autumn is reporting a sowing date; one
whose `sos` falls after New Year is reporting a green-up.
"""),
    code('''
AUTUMN_SOS_THRESHOLD = 200.0  # DOY; above it, sos is an autumn date

obs_country["sos_regime"] = np.where(
    obs_country["sos"] >= AUTUMN_SOS_THRESHOLD, "autumn sowing", "post-winter green-up"
)
regimes = obs_country.groupby("sos_regime")["country"].apply(list)
for regime, members in regimes.items():
    print(f"{regime:22s} ({len(members):2d}): {', '.join(sorted(members))}")
'''),
    md("""
## 6. Pair the two sides

The join is on `country` alone: the reference has no year, so each simulated
country-year is paired against its country's climatological calendar.
"""),
    code('''
paired = aggregate.pair_observations(sim_country, obs_country, ["country"])

for stage in config.STAGES:
    paired[f"{stage.key}_error"] = doy.doy_difference(
        paired[stage.sim_col], paired[stage.obs_col]
    )

print(f"{len(paired)} country-years across {paired['country'].nunique()} countries")
paired.head()
'''),
    md("""
## 7. Metrics by stage

Pooled across every country-year. Two correlations are undefined here and come
back as NaN, both for the same reason — a constant has no variance:

* **Sowing**, because the *simulated* date is the DOY 270 latch.
* **Every per-country row** in the next section, because the *observed* date is
  a single climatological value repeated across that country's years.

The pooled maturity `pearson_r` is therefore a **spatial** correlation across
the 23 countries, not an interannual one.
"""),
    code('''
pooled = {}
for stage in config.STAGES:
    pooled[stage.label] = metrics.phenology_metrics(
        paired[stage.obs_col], paired[stage.sim_col]
    )

pooled_table = pd.DataFrame(pooled).T[list(metrics.PHENOLOGY_METRIC_ORDER)]
pooled_table["observed date"] = pooled_table["obs_mean"].map(doy.doy_to_month_day)
pooled_table["simulated date"] = pooled_table["sim_mean"].map(doy.doy_to_month_day)

pooled_table.round(2)
'''),
    code('''
for stage in config.STAGES:
    print(f"--- {stage.label} ---")
    print(stage.caveat)
    print()
'''),
    md("""
### Per country and stage

One table per stage, and a wide summary across all three.
"""),
    code('''
stage_metrics = {
    stage.key: metrics.metrics_by_group(
        paired, stage.obs_col, stage.sim_col, metric_fn=metrics.phenology_metrics
    )
    for stage in config.STAGES
}

summary = pd.concat(
    {stage.label: stage_metrics[stage.key][["n", "bias", "mae", "rmse", "pearson_r"]]
     for stage in config.STAGES},
    axis=1,
)
summary.to_csv(config.TABLE_DIR / "phenology_metrics_by_country_stage.csv",
               float_format="%.3f")
summary.round(1)
'''),
    md("""
> **The per-country `pearson_r` column is NaN by construction, not by
> accident.** The CyBench calendar has no year dimension, so within one country
> the observed date is the same value in every paired year — a constant has no
> variance to correlate against. The column is kept rather than hidden so the
> limitation is visible in the table itself.
>
> The **pooled** `pearson_r` in §7 is therefore a *spatial* correlation across
> the 23 countries, not an interannual one: it says the model orders the
> countries' seasons correctly. Nothing here tests whether it tracks a warm
> year against a cold one — that would need a year-resolved reference.
"""),
    code('''
maturity_ranked = metrics.rank_countries(stage_metrics["maturity"], by="rmse")

plots.metric_table(
    maturity_ranked, ("rank", *metrics.PHENOLOGY_METRIC_ORDER), precision=1,
    caption="Maturity (simulated DVS >= 2) against CyBench 'eos', ranked by "
            "RMSE in days. obs_mean/sim_mean are days-of-year; bias, MAE and "
            "RMSE are in days, positive meaning the model is late.",
)
'''),
    md("""
### Sowing, split by `sos` convention

Pooling the two regimes would report a mean of two incompatible quantities.
Split, the autumn-sowing countries give a usable check on the DOY 270 latch;
the green-up countries do not, and their column is here only to make the size of
the convention gap explicit.
"""),
    code('''
# `sos_regime` is already on `paired`: it was added to obs_country before the
# join, so it came across with the rest of the reference columns.
sowing_rows = []
for regime, block in paired.groupby("sos_regime"):
    row = metrics.phenology_metrics(block["sos"], block["sowing_doy"])
    row["countries"] = block["country"].nunique()
    row["regime"] = regime
    sowing_rows.append(row)

sowing_table = pd.DataFrame(sowing_rows).set_index("regime")
sowing_table["observed date"] = sowing_table["obs_mean"].map(doy.doy_to_month_day)
sowing_table["simulated date"] = sowing_table["sim_mean"].map(doy.doy_to_month_day)
sowing_table[["countries", "n", "observed date", "simulated date", "bias", "mae", "rmse"]].round(1)
'''),
    md("""
## 8. Figures

### 8.1 Observed against simulated, one panel per stage

Both axes are days-of-year unwrapped about the observed circular mean, then
relabelled as calendar dates — a scatter axis is linear, but the underlying
quantity is not.
"""),
    code('''
for stage in config.STAGES:
    fig = plots.scatter_one_to_one(
        paired, stage.obs_col, stage.sim_col, circular=True,
        stats=pooled[stage.label],
        metric_keys=("n", "bias", "rmse", "pearson_r"),
        title=f"{stage.label}: simulated against CyBench '{stage.obs_col}'",
        xlabel=f"CyBench {stage.obs_col}",
        ylabel=f"TorchCrop {stage.label.lower()}",
    )
    plots.save(fig, f"phenology_01_scatter_{stage.key}")
    display(fig)
'''),
    md("""
### 8.2 Per country, for the maturity pairing

Only maturity gets the small-multiple treatment: it is the one pairing where
both sides mean the same thing, so it is the one where a per-country panel says
something about the model.
"""),
    code('''
fig = plots.scatter_small_multiples(
    paired, "eos", "maturity_doy", metrics=stage_metrics["maturity"],
    order=[c for c in maturity_ranked.index if c != "ALL"],
    circular=True, annotate=("bias", "rmse"),
    title="Maturity against CyBench 'eos', by country (ordered by RMSE)",
    xlabel="CyBench end of season",
    ylabel="TorchCrop maturity",
)
plots.save(fig, "phenology_02_maturity_small_multiples")
fig
'''),
    md("""
### 8.3 Country-wise bias, one figure per stage

Positive (warm) means the model is **late**.
"""),
    code('''
for stage in config.STAGES:
    fig = plots.bias_bars(
        metrics.rank_countries(stage_metrics[stage.key], by="bias"),
        value_col="bias", error_col="rmse",
        title=f"{stage.label} bias by country (simulated - observed)",
        xlabel="Bias (days; positive = simulated late)",
    )
    plots.save(fig, f"phenology_03_bias_{stage.key}")
    display(fig)
'''),
    md("""
### 8.4 Maps
"""),
    code('''
polygons = regions.load_country_polygons(COUNTRIES)

for stage in config.STAGES:
    fig = plots.country_choropleth(
        polygons, stage_metrics[stage.key]["bias"],
        title=f"{stage.label} bias (simulated - observed)",
        cbar_label="Bias (days; positive = late)", diverging=True,
    )
    plots.save(fig, f"phenology_04_map_bias_{stage.key}")
    display(fig)
'''),
    code('''
cell_maturity = (
    scoped.groupby(["SimplaceID", "lon", "lat"], as_index=False)
    .agg(days_to_maturity=("days_to_maturity", "mean"))
)

fig = plots.cell_map(
    cell_maturity, "days_to_maturity",
    title=f"Simulated season length, sowing (DOY {config.SIM_SOWING_DOY}) to maturity",
    cbar_label="Days", overlay=polygons,
)
plots.save(fig, "phenology_05_map_season_length")
fig
'''),
    md("""
## 9. Summary
"""),
    code('''
paired.to_csv(config.TABLE_DIR / "phenology_paired_country_year.csv",
              index=False, float_format="%.4f")
pooled_table.to_csv(config.TABLE_DIR / "phenology_metrics_pooled.csv",
                    float_format="%.3f")
print(f"tables written to {config.TABLE_DIR}")
print(f"figures written to {config.FIGURE_DIR}")

pooled_table[["n", "observed date", "simulated date", "bias", "mae", "rmse",
              "pearson_r"]].round(1)
'''),
    md("""
### What the numbers mean, and what they do not

* **Maturity is the only result to quote.** Both sides mean the same thing
  there, and the correlation across countries is real: the model orders the
  countries' seasons correctly even where it places them wrongly in absolute
  terms.
* **The maturity bias is not purely a model error.** CyBench `eos` is an end of
  season, which falls at or after harvest; simulated maturity is `DVS = 2`. Part
  of the gap is the maturity-to-harvest interval that LINTUL-5 does not model.
* **Sowing measures the DOY 270 assumption, not the model.** For the autumn-`sos`
  countries the bias is a real statement about that constant. For the
  green-up-`sos` countries it is a statement about the CyBench convention and
  should not be read as skill.
* **Harvest carries no information beyond maturity** while
  `config.HARVEST_LAG_DAYS` is 0.
* **The reference is climatological.** With no year dimension on `sos`/`eos`,
  none of these metrics test whether the model tracks a warm or a cold season —
  only whether it places the average season correctly.
"""),
]


# --------------------------------------------------------------------------- #
# Shared preamble — gridded notebooks
# --------------------------------------------------------------------------- #

GRID_SETUP = '''
import logging
import sys
from pathlib import Path

# The notebooks live beside utils/, so they import it without installation.
sys.path.insert(0, str(Path.cwd()))

import numpy as np
import pandas as pd

from utils import config, doy, grid, metrics, plots, torchcrop
from utils.style import use_style

logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s",
                    force=True)
logging.getLogger("matplotlib").setLevel(logging.WARNING)

PALETTE = use_style("light")
config.ensure_output_dirs()

pd.set_option("display.max_rows", 60)
pd.set_option("display.width", 140)

print(f"TorchCrop run : {config.TORCHCROP_RUN_DIR}")
print(f"Outputs       : {config.OUTPUT_DIR}")
print(f"Grid          : {config.GRID_RES_DEG}° over {config.EUROPE_BBOX}")
'''

DIAGNOSTIC_STATE_AGG = '''
state = (
    sim.groupby("grid_id", as_index=False)
    .agg(lon=("lon", "first"), lat=("lat", "first"),
         max_lai=("max_lai", "mean"), biomass=("biomass_g_m2", "mean"),
         days_to_maturity=("days_to_maturity", "mean"),
         water_stress=("tranrf_mean", "mean"), n_index=("nni_mean", "mean"),
         heat=("heat_stress_factor", "mean"),
         failed_share=("yield_t_ha", lambda s: float((s < 0.5).mean())))
)
'''


# --------------------------------------------------------------------------- #
# Notebook 3 — GDHY gridded yield
# --------------------------------------------------------------------------- #

gdhy_cells = [
    md("""
# Winter wheat yield — TorchCrop against GDHY (gridded)

**What this notebook does.** It bins the 10 km TorchCrop winter-wheat yield
simulation onto the 0.5° grid of GDHY (Iizumi & Sakai 2020), pairs the two
fields per grid cell and year, and reports the same skill metrics as
[`yield_evaluation.ipynb`](yield_evaluation.ipynb) — but spatially resolved
rather than collapsed to 23 national means.

**Why a second yield notebook.** CyBench evaluates at the level a country runs
its statistics service at; GDHY evaluates at the level the model actually
predicts. The two ask different questions and can disagree: a country-level
bias can hide entirely opposite errors in its western and eastern halves, and
this notebook is where that would show up.

**Three conventions worth knowing before reading any number:**

1. **Error is `simulated − observed`.** A negative bias means the model is low.
2. **GDHY is not an observation of the grid cell**, it downscales national and
   sub-national statistics with a satellite vegetation-index proxy and a crop
   mask. A cell's *level* is largely its country's statistic; what varies
   between neighbouring cells is mostly the proxy and the mask, so a spatial
   correlation partly measures agreement with the mask, and neighbouring cells
   are not independent samples. §7 separates the spatial and interannual
   signal for exactly this reason.
3. **No moisture conversion is applied**, for the same reason as the CyBench
   notebook: GDHY inherits the moisture basis of the national statistics behind
   it, TorchCrop reports grain dry matter, and both are compared as published.

All reusable code is in [`utils/`](utils/); this notebook is the workflow only.
"""),
    md("## 0. Setup"),
    code(GRID_SETUP),
    md("""
## 1. Load the TorchCrop simulation and bin it onto the GDHY grid

Cells are averaged into whichever 0.5° cell their centre falls in
(`utils.grid.bin_cells`), unweighted — the export carries no per-cell wheat
area, the same limitation as the country-level notebook. A 0.5° cell is kept
only once it holds at least `config.MIN_CELLS_PER_GRIDCELL` simulated 10 km
cells, so a cell that is mostly sea or mostly non-cropland does not carry the
same weight as a fully covered inland one.
"""),
    code('''
sim = torchcrop.load_simulation(columns=[
    "SimplaceID", "year", "lon", "lat", "yield_t_ha", "biomass_g_m2", "max_lai",
    "days_to_maturity", "tranrf_mean", "nni_mean", "heat_stress_factor", "irri",
])
sim = grid.crop_to_bbox(sim, config.EUROPE_BBOX)
# Tagged onto every 10 km row (not just the binned means) so the diagnostics
# in §8 can group the run's own state variables by 0.5° cell too.
sim["grid_id"] = grid.grid_cell_id(*grid.snap_to_grid(sim["lon"], sim["lat"]))

sim_grid = grid.bin_cells(sim, {"yield_t_ha": False}, by=["year"])
print(f"{len(sim):,} cell-seasons on the 10 km grid -> "
      f"{len(sim_grid):,} cell-years on the {config.GRID_RES_DEG}° grid, "
      f"{sim_grid['grid_id'].nunique():,} distinct 0.5° cells")
sim_grid.describe().T
'''),
    md("""
## 2. Load GDHY

One NetCDF per year, cropped to the European bounding box and stacked. `year`
is the file's own label; GDHY dates a crop by its harvest year, the same
convention `torchcrop.load_simulation` uses, so
`config.GDHY_HARVEST_YEAR_OFFSET` is 0 by default — checked in §3, not assumed.
"""),
    code('''
from utils import gdhy

print(f"GDHY crop: {config.GDHY_CROP} ({config.GDHY_ROOT})")
published = gdhy.available_years(crop=config.GDHY_CROP)
print(f"published years: {published[0]}-{published[-1]}")

obs_grid = gdhy.load_gdhy(crop=config.GDHY_CROP)
print(f"{len(obs_grid):,} cell-years, {obs_grid['grid_id'].nunique():,} cells, "
      f"{obs_grid['year'].min()}-{obs_grid['year'].max()}")
obs_grid["obs_yield"].describe()
'''),
    md("""
## 3. Check the harvest-year convention

`torchcrop.load_simulation` labels a row by the calendar year the crop was
**harvested** in — a crop sown in autumn 1999 and maturing in mid-2000 is
`year = 2000`. If GDHY's file year meant something else (e.g. the sowing year),
pairing at offset 0 would silently compare each season against the wrong
year's weather. Pairing at three offsets and comparing the pooled RMSE is a
crude but honest check: the correct offset should not be worse than its
neighbours.
"""),
    code('''
offset_check = []
for offset in (-1, 0, 1):
    shifted = obs_grid.assign(year=(obs_grid["year"] + offset).astype("int16"))
    trial = grid.pair_gridded(sim_grid, shifted, ["grid_id", "year"])
    m = metrics.yield_metrics(trial["obs_yield"], trial["yield_t_ha"])
    offset_check.append({"offset": offset, "n": m["n"], "rmse": m["rmse"],
                         "bias": m["bias"], "pearson_r": m["pearson_r"]})

offset_table = pd.DataFrame(offset_check).set_index("offset")
print(f"config.GDHY_HARVEST_YEAR_OFFSET = {config.GDHY_HARVEST_YEAR_OFFSET}")
offset_table.round(3)
'''),
    md("""
> Read this table for **stability, not a winner.** The correlation is
> dominated by the spatial pattern common to all three offsets (§7 shows
> exactly how much), so it barely moves. What would flag a wrong convention is
> a large RMSE or bias jump at the configured offset relative to its
> neighbours; a small, smooth change across all three is the expected shape
> when the offset is right and the residual signal is weak.
"""),
    md("""
## 4. Pair the two sides
"""),
    code('''
paired = grid.pair_gridded(sim_grid, obs_grid, ["grid_id", "year"])
paired["residual"] = paired["yield_t_ha"] - paired["obs_yield"]

print(f"{len(paired):,} paired cell-years across {paired['grid_id'].nunique():,} "
      f"0.5° cells, {paired['year'].min()}-{paired['year'].max()}")
paired.head()
'''),
    md("""
## 5. Pooled metrics

Pooled over every cell-year — dominated by the spatial pattern, as noted above.
`n` here is cell-years, not the 412 country-years of the CyBench notebook, so
the two RMSEs are not read against the same denominator.
"""),
    code('''
pooled = metrics.yield_metrics(paired["obs_yield"], paired["yield_t_ha"])
print("Pooled over every 0.5-degree cell-year:")
for key in metrics.YIELD_METRIC_ORDER:
    print(f"  {metrics.METRIC_LABELS[key]:>16s}  {pooled[key]:>8.2f}")
'''),
    md("""
## 6. Figures — the raw field
"""),
    code('''
fig = plots.scatter_density(
    paired, "obs_yield", "yield_t_ha", stats=pooled,
    title=f"Gridded winter wheat yield, {paired['year'].min()}-{paired['year'].max()}",
    xlabel="GDHY observed yield (t ha$^{-1}$)",
    ylabel="TorchCrop simulated yield (t ha$^{-1}$)",
)
plots.save(fig, "gdhy_01_scatter_density")
fig
'''),
    code('''
cell_bias = paired.groupby(["grid_id", "lon", "lat"], as_index=False)["residual"].mean()

fig = plots.cell_map(
    cell_bias, "residual",
    title="Mean yield bias by 0.5° cell (simulated - observed)",
    cbar_label="Bias (t ha$^{-1}$)", diverging=True,
)
plots.save(fig, "gdhy_02_map_bias")
fig
'''),
    md("""
This is the map the country-level notebook cannot draw. Compare it with
`outputs/figures/yield_07_map_bias.pdf` — if the two agree at the coastline
where a CyBench country boundary sits, the country-level bias was not hiding a
finer structure; if they disagree, this map is the more honest one.
"""),
    code('''
fig = plots.timeseries_pair(
    paired, "obs_yield", "yield_t_ha",
    title="Domain-mean winter wheat yield through time",
    ylabel="Yield (t ha$^{-1}$)",
)
plots.save(fig, "gdhy_03_timeseries")
fig
'''),
    code('''
fig = plots.residual_panels(
    paired, "residual", "obs_yield",
    title="Yield residuals (gridded)",
    ylabel="Residual (t ha$^{-1}$)",
    xlabel_obs="GDHY observed yield (t ha$^{-1}$)",
)
plots.save(fig, "gdhy_04_residuals")
fig
'''),
    md("""
## 7. Spatial signal against interannual signal

Splitting a cell's yield into its long-term mean (spatial) and its deviation
from that mean in a given year (interannual, via `metrics.to_anomalies`)
answers the question §2's caveat raises directly: how much of the correlation
above is "the model puts high-yielding cells where GDHY does" against "the
model's good and bad years line up with GDHY's".
"""),
    code('''
anom = metrics.to_anomalies(paired, ["obs_yield", "yield_t_ha"])

spatial_mean = paired.groupby("grid_id")[["obs_yield", "yield_t_ha"]].mean()
spatial = metrics.yield_metrics(spatial_mean["obs_yield"], spatial_mean["yield_t_ha"])
interannual = metrics.yield_metrics(anom["obs_yield_anom"], anom["yield_t_ha_anom"])

decomposition = pd.DataFrame(
    {"Spatial (cell means)": spatial, "Interannual (anomalies)": interannual}
).T[list(metrics.YIELD_METRIC_ORDER)]
decomposition.round(3)
'''),
    md("""
> If the spatial row's `pearson_r` is far from zero and the interannual row's
> is close to it, the model is reproducing *where* GDHY expects high yields but
> not *when* — its year-to-year variation is close to noise relative to GDHY's.
> That is a materially different statement from the pooled r in §5, which
> cannot tell the two apart.
"""),
    code('''
cell_r = (
    anom.groupby("grid_id")
    .apply(lambda g: metrics.yield_metrics(g["obs_yield"], g["yield_t_ha"])["pearson_r"]
           if len(g) >= 5 else np.nan, include_groups=False)
    .rename("interannual_r")
)
cell_pos = paired.groupby(["grid_id", "lon", "lat"], as_index=False).size().drop(columns="size")
cell_pos["interannual_r"] = cell_pos["grid_id"].map(cell_r)

share_positive = float((cell_r.dropna() > 0).mean())
print(f"{cell_r.notna().sum()} cells with >=5 paired years; "
      f"{100 * share_positive:.0f}% have a positive interannual correlation")

fig = plots.cell_map(
    cell_pos.dropna(subset=["interannual_r"]), "interannual_r",
    title="Interannual correlation per cell (>= 5 paired years)",
    cbar_label="Pearson r", diverging=True, vmax=1.0,
)
plots.save(fig, "gdhy_05_map_interannual_r")
fig
'''),
    md("""
## 8. Diagnostics — the run's own state next to the bias

The same diagnostic the country-level notebook runs in its §9, regridded: the
run's state variables, averaged per 0.5° cell, correlated against that cell's
bias.
"""),
    code(DIAGNOSTIC_STATE_AGG + '''
state["obs_yield"] = state["grid_id"].map(spatial_mean["obs_yield"])
state["sim_yield"] = state["grid_id"].map(spatial_mean["yield_t_ha"])
state["bias"] = state["sim_yield"] - state["obs_yield"]
state = state.dropna(subset=["bias"])

print("Correlation of cell bias with the run's own state variables:")
print(state.corr(numeric_only=True)["bias"].drop(columns=[], errors="ignore")
      .drop("bias").round(2).sort_values().to_string())

fig = plots.cell_map(
    state, "max_lai",
    title="Simulated maximum leaf area index (mean over seasons)",
    cbar_label="Max LAI (m$^2$ m$^{-2}$)",
)
plots.save(fig, "gdhy_06_map_max_lai")
fig
'''),
    md("""
## 9. Summary
"""),
    code('''
summary = pd.Series({
    "0.5-degree cells": paired["grid_id"].nunique(),
    "cell-years": len(paired),
    "years": f"{paired['year'].min()}-{paired['year'].max()}",
    "observed mean (t/ha)": round(pooled["obs_mean"], 2),
    "simulated mean (t/ha)": round(pooled["sim_mean"], 2),
    "bias (t/ha)": round(pooled["bias"], 2),
    "RMSE (t/ha)": round(pooled["rmse"], 2),
    "pooled Pearson r": round(pooled["pearson_r"], 2),
    "spatial Pearson r": round(spatial["pearson_r"], 2),
    "interannual Pearson r": round(interannual["pearson_r"], 2),
}, name="value")

paired.to_csv(config.TABLE_DIR / "gdhy_paired_cell_year.csv",
              index=False, float_format="%.4f")
decomposition.to_csv(config.TABLE_DIR / "gdhy_spatial_interannual.csv",
                     float_format="%.4f")
state.to_csv(config.TABLE_DIR / "gdhy_diagnostics_by_cell.csv", float_format="%.4f")
print(f"tables written to {config.TABLE_DIR}")
print(f"figures written to {config.FIGURE_DIR}")
summary.to_frame()
'''),
    md("""
### What the numbers mean, and what they do not

* **The gridded bias map is the finer-grained twin of `yield_07_map_bias`.**
  Where the two agree, the country-level number was not hiding structure the
  country boundary happens to cut through; where they disagree, trust this one.
* **The pooled correlation is not a skill score.** §7's spatial/interannual
  split is the number to quote for "does the model track a good year" — the
  pooled figure mixes that question with "does the model know where wheat
  yields well", which GDHY answers largely from its own crop mask.
* **A single p-value for the pooled fit would be fiction.** Tens of thousands
  of cell-years are not independent samples; two neighbouring cells share most
  of their signal. No p-value is reported here for that reason — read the
  effect sizes only.
* **The moisture offset from the CyBench notebook still applies.** No
  conversion is applied here either, so part of any negative bias is expected
  before model error is counted.
"""),
]


# --------------------------------------------------------------------------- #
# Notebook 4 — SAGE gridded crop calendar
# --------------------------------------------------------------------------- #

sage_cells = [
    md("""
# Winter wheat phenology — TorchCrop against the SAGE crop calendar (gridded)

**What this notebook does.** It bins the simulated sowing and maturity dates
onto the 0.5° grid of the Sacks et al. (2010) SAGE crop calendar and compares
them per cell — the gridded counterpart of
[`phenology_evaluation.ipynb`](phenology_evaluation.ipynb), with one important
difference: **SAGE reports a planting date**, not a green-up proxy that
changes meaning with climate the way CyBench's `sos` does. This is therefore
the more direct test of `config.SIM_SOWING_DOY`, the constant every cell in
the run is sown on.

## What each side contains

| Stage | TorchCrop | SAGE | Comparable? |
|---|---|---|---|
| Sowing | **constant input**, DOY 270 for every cell | `plant`, with a `plant.start`-`plant.end` window | yes — a genuine test of the constant |
| Flowering | **not in the run** | not in the calendar | no |
| Maturity | first day at `DVS ≥ 2` | `harvest`, with a `harvest.start`-`harvest.end` window | yes, with a harvest-after-maturity offset expected |

* **SAGE is a climatology.** One planting and harvest date per cell, assembled
  around 1990-2000 from agricultural-census and extension-service reports —
  no year dimension, so nothing here tests interannual skill, only whether the
  model places the *average* season in the right week.
* **A third of the domain is filled, not reported.** Where no reporting unit
  covers a cell, the `.fill` product extrapolates from the nearest one; those
  cells carry their donor's date exactly and are flagged `filled = True`
  throughout this notebook so they can be read separately.
* **A date comes with a window**, 10 to 100 days wide in Europe. Whether the
  simulated date falls **inside** that window is a more forgiving and more
  informative test than a bias in days, computed with `doy.doy_in_window`.

All reusable code is in [`utils/`](utils/); this notebook is the workflow only.
"""),
    md("## 0. Setup"),
    code(GRID_SETUP),
    md("""
## 1. Load the simulation and derive the stage dates, binned onto the SAGE grid
"""),
    code('''
sim = torchcrop.load_simulation(columns=[
    "SimplaceID", "year", "lon", "lat", "days_to_maturity", "yield_t_ha",
    "biomass_g_m2", "max_lai", "tranrf_mean", "nni_mean", "heat_stress_factor",
])
sim = grid.crop_to_bbox(sim, config.EUROPE_BBOX)
# Tagged onto every 10 km row (not just the binned means) so the diagnostics
# in §7 can group the run's own state variables by 0.5° cell too.
sim["grid_id"] = grid.grid_cell_id(*grid.snap_to_grid(sim["lon"], sim["lat"]))

sim_grid = grid.bin_cells(
    sim, {"sowing_doy": True, "maturity_doy": True, "season_length_days": False},
)
# `season_length_days` exists on both sides once paired with the SAGE calendar
# (its own `tot.days`); rename the simulated one now rather than disambiguate
# a pandas merge suffix later.
sim_grid = sim_grid.rename(columns={"season_length_days": "sim_season_length_days"})

print(f"{len(sim):,} cell-seasons on the 10 km grid -> "
      f"{len(sim_grid):,} 0.5° cells, "
      f"{sim_grid['n_cells'].median():.0f} simulated cells per cell (median)")
sim_grid.describe().T
'''),
    md("""
## 2. Load the SAGE calendar
"""),
    code('''
from utils import sage

calendar = sage.load_sage_calendar(crop=config.SAGE_CROP)
print(f"SAGE crop: {config.SAGE_CROP} ({config.SAGE_ROOT})")
print(f"{len(calendar):,} cells in the domain, "
      f"{100 * calendar['filled'].mean():.0f}% filled from a neighbouring unit")
calendar.describe().T
'''),
    md("""
## 3. Pair the two sides

A plain join on `grid_id` — SAGE carries no year, so every simulated season in
a cell is compared against the same climatological date, exactly as CyBench's
`sos`/`eos` are in the country-level notebook.
"""),
    code('''
paired = grid.pair_gridded(sim_grid, calendar, ["grid_id"])

for stage in config.CALENDAR_STAGES:
    paired[f"{stage.key}_error"] = doy.doy_difference(
        paired[stage.sim_col], paired[stage.obs_col]
    )
    paired[f"{stage.key}_in_window"] = doy.doy_in_window(
        paired[stage.sim_col], paired[stage.start_col], paired[stage.end_col]
    )

print(f"{len(paired):,} paired 0.5° cells, "
      f"{100 * paired['filled'].mean():.0f}% filled")
paired.head()
'''),
    md("""
## 4. Metrics by stage

Reported once pooled and once restricted to the reported (non-filled) cells —
the filled third of the domain repeats a neighbour's exact date, which can
only make a local bias look smoother than it is.
"""),
    code('''
pooled = {}
reported = paired[~paired["filled"]]
for stage in config.CALENDAR_STAGES:
    pooled[stage.label] = metrics.phenology_metrics(paired[stage.obs_col], paired[stage.sim_col])
    pooled[f"{stage.label} (reported only)"] = metrics.phenology_metrics(
        reported[stage.obs_col], reported[stage.sim_col]
    )

pooled_table = pd.DataFrame(pooled).T[list(metrics.PHENOLOGY_METRIC_ORDER)]
pooled_table["observed date"] = pooled_table["obs_mean"].map(doy.doy_to_month_day)
pooled_table["simulated date"] = pooled_table["sim_mean"].map(doy.doy_to_month_day)
pooled_table.round(2)
'''),
    code('''
for stage in config.CALENDAR_STAGES:
    print(f"--- {stage.label} ---")
    print(stage.caveat)
    print()
'''),
    md("""
### Window containment

The share of cells whose simulated date falls inside the SAGE window, split
into "too early", "inside" and "too late" — the categorical answer the bias
figures below cannot give on their own.
"""),
    code('''
window_summary = {}
for stage in config.CALENDAR_STAGES:
    position = doy.window_position(paired[stage.sim_col], paired[stage.start_col],
                                   paired[stage.end_col])
    category = np.select(
        [position < 0, position == 0, position > 0],
        ["before window", "in window", "after window"],
        default="not paired",
    )
    paired[f"{stage.key}_position"] = category
    counts = pd.Series(category).value_counts(normalize=True) * 100
    window_summary[stage.label] = counts

pd.DataFrame(window_summary).round(1).fillna(0.0)
'''),
    md("""
## 5. Figures

### 5.1 Observed against simulated, one panel per stage
"""),
    code('''
for stage in config.CALENDAR_STAGES:
    fig = plots.scatter_density(
        paired, stage.obs_col, stage.sim_col, circular=True,
        stats=pooled[stage.label], metric_keys=("n", "bias", "rmse", "pearson_r"),
        title=f"{stage.label}: simulated against SAGE '{stage.obs_col}'",
        xlabel=f"SAGE {stage.obs_col}", ylabel=f"TorchCrop {stage.label.lower()}",
    )
    plots.save(fig, f"sage_01_scatter_{stage.key}")
    display(fig)
'''),
    md("""
### 5.2 Bias maps
"""),
    code('''
for stage in config.CALENDAR_STAGES:
    stage_paired = paired.dropna(subset=[f"{stage.key}_error"])
    fig = plots.cell_map(
        stage_paired, f"{stage.key}_error",
        title=f"{stage.label} bias (simulated - observed)",
        cbar_label="Bias (days; positive = late)", diverging=True,
    )
    plots.save(fig, f"sage_02_map_bias_{stage.key}")
    display(fig)
'''),
    md("""
### 5.3 Window containment maps

Discrete fills rather than a colour ramp: "inside the window" has no useful
distance to "before" or "after" it.
"""),
    code('''
for stage in config.CALENDAR_STAGES:
    fig = plots.category_map(
        paired, f"{stage.key}_position",
        categories=["before window", "in window", "after window"],
        title=f"{stage.label}: simulated date against the SAGE window",
        legend_title=f"vs. {stage.start_col}-{stage.end_col}",
    )
    plots.save(fig, f"sage_03_map_window_{stage.key}")
    display(fig)
'''),
    md("""
### 5.4 Season length

`sim_season_length_days` is sowing-to-maturity; SAGE's `season_length_days` is
planting-to-harvest. The two are not the same interval — this compares their
*durations*, not the dates that bound them, so it survives the DOY 270 sowing
offset entirely.
"""),
    code('''
season_pooled = metrics.yield_metrics(
    paired["season_length_days"], paired["sim_season_length_days"]
)
fig = plots.scatter_density(
    paired, "season_length_days", "sim_season_length_days", stats=season_pooled,
    metric_keys=("n", "bias", "rmse", "pearson_r"),
    title="Season length: simulated (sowing-to-maturity) against SAGE (planting-to-harvest)",
    xlabel="SAGE season length (days)", ylabel="TorchCrop season length (days)",
)
plots.save(fig, "sage_04_scatter_season_length")
fig
'''),
    md("""
## 6. Error decomposition — how much of the maturity error is the sowing offset

The run's maturity date is `sowing_doy + days_to_maturity`. If sowing is early
by `X` days and the simulated season length is otherwise right, maturity is
early by about `X` days too — a sowing artefact, not a growth-rate error. This
splits the maturity bias into the two additive pieces so the difference is
visible rather than absorbed into one number.
"""),
    code('''
# `paired` is already one row per grid_id (SAGE carries no year), so this is a
# rename, not an aggregation.
decomposition = pd.DataFrame({
    "Sowing offset": paired["sowing_error"].to_numpy(),
    "Season-length residual": (paired["maturity_error"] - paired["sowing_error"]).to_numpy(),
}, index=paired["grid_id"].to_numpy())
decomposition.index.name = "grid_id"
decomposition["Maturity error (total)"] = (
    decomposition["Sowing offset"] + decomposition["Season-length residual"]
)

print("Mean absolute contribution to the maturity error:")
print(decomposition[["Sowing offset", "Season-length residual"]].abs().mean().round(2).to_string())

fig = plots.component_bars(
    decomposition.sample(min(40, len(decomposition)), random_state=0)
                .sort_values("Maturity error (total)"),
    ["Sowing offset", "Season-length residual"],
    title="Maturity-error decomposition, 40 sampled cells (sorted by total error)",
    xlabel="Days (positive = late)",
)
plots.save(fig, "sage_05_bars_decomposition")
fig
'''),
    md("""
## 7. Diagnostics — the run's own state next to the bias
"""),
    code(DIAGNOSTIC_STATE_AGG + '''
state["maturity_bias"] = state["grid_id"].map(paired.set_index("grid_id")["maturity_error"])
state["sowing_bias"] = state["grid_id"].map(paired.set_index("grid_id")["sowing_error"])
state = state.dropna(subset=["maturity_bias"])

print("Correlation of cell maturity bias with the run's own state variables:")
print(state.corr(numeric_only=True)["maturity_bias"]
      .drop(["maturity_bias", "sowing_bias"]).round(2).sort_values().to_string())
state.round(2).head()
'''),
    md("""
## 8. Summary
"""),
    code('''
summary = pd.Series({
    "0.5-degree cells": paired["grid_id"].nunique(),
    "filled share (%)": round(100 * paired["filled"].mean(), 1),
    "sowing bias (days)": round(pooled["Sowing"]["bias"], 1),
    "sowing RMSE (days)": round(pooled["Sowing"]["rmse"], 1),
    "sowing in-window (%)": round(
        100 * paired["sowing_in_window"].mean(), 1),
    "maturity bias (days)": round(pooled["Maturity / harvest"]["bias"], 1),
    "maturity RMSE (days)": round(pooled["Maturity / harvest"]["rmse"], 1),
    "maturity Pearson r": round(pooled["Maturity / harvest"]["pearson_r"], 2),
    "season-length bias (days)": round(season_pooled["bias"], 1),
    "season-length RMSE (days)": round(season_pooled["rmse"], 1),
}, name="value")

paired.to_csv(config.TABLE_DIR / "sage_paired_cell.csv", index=False, float_format="%.4f")
pooled_table.to_csv(config.TABLE_DIR / "sage_metrics_pooled.csv", float_format="%.3f")
state.to_csv(config.TABLE_DIR / "sage_diagnostics_by_cell.csv", float_format="%.4f")
print(f"tables written to {config.TABLE_DIR}")
print(f"figures written to {config.FIGURE_DIR}")
summary.to_frame()
'''),
    md("""
### What the numbers mean, and what they do not

* **Sowing is the genuine result here, unlike in the CyBench notebook.** SAGE
  `plant` does not switch convention with climate the way CyBench `sos` does,
  so this bias is a direct, continent-wide statement about the DOY 270 latch —
  and §5.3/§6 say where and by how much a real sowing calendar would have to
  differ from that constant.
* **The window test and the bias are complementary, not redundant.** A cell can
  be "in window" with a large bias if the window is wide, and "out of window"
  with a small one if the window is narrow — read both.
* **Maturity still carries the CyBench notebook's caveat**: SAGE `harvest` is
  at or after `DVS = 2`, so part of the bias is that interval, not model error.
  §6's decomposition only separates the sowing contribution from the rest; it
  does not remove the harvest-after-maturity gap.
* **A third of the domain is a repeated neighbour's date, not an independent
  observation.** The `filled` split in §4 is there so a result is not quoted
  from a part of the map that is smoother than the underlying data.
* **The reference is climatological**, exactly as CyBench's calendar is:
  nothing here tests interannual timing, only the average placement of the
  season.
"""),
]


if __name__ == "__main__":
    write("yield_evaluation.ipynb", yield_cells)
    write("phenology_evaluation.ipynb", phenology_cells)
    write("gdhy_yield_evaluation.ipynb", gdhy_cells)
    write("sage_calendar_evaluation.ipynb", sage_cells)
