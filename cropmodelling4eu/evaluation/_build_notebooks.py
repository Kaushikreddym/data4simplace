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

# cropmodelling4eu is installed (pip install -e .), so the evaluation
# library is imported like any other package rather than off sys.path.

import numpy as np
import pandas as pd

from cropmodelling4eu.evaluation import aggregate, config, cybench, doy, metrics, plots, regions, torchcrop
from cropmodelling4eu.evaluation.style import use_style

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
  calendar, so `cropmodelling4eu.torchcrop.run` sows every cell from Crete to Lapland on
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
  `cropmodelling4eu.torchcrop.run` and re-running the shards would supply the
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

# cropmodelling4eu is installed (pip install -e .), so the evaluation
# library is imported like any other package rather than off sys.path.

import numpy as np
import pandas as pd

from cropmodelling4eu.evaluation import config, doy, grid, metrics, plots, torchcrop
from cropmodelling4eu.evaluation.style import use_style

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
from cropmodelling4eu.evaluation import gdhy

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
from cropmodelling4eu.evaluation import sage

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



# --------------------------------------------------------------------------- #
# Germany smoke test: SIMPLACE vs torchcrop, against CyBench and PEP725
# --------------------------------------------------------------------------- #

germany_cells = [
    md("""
# Germany smoke test — SIMPLACE vs torchcrop vs observations

~30 cropland cells across Germany, each in a distinct CyBench NUTS-3 region,
run by **both models over the same cells**, each swept over the same three
`vIOPT`/`iopt` settings (1 potential, 2 water-limited, 3 water+N — 4 is
dropped, since no European soil P or K layer exists). Every figure below
therefore carries six series, and a same-IOPT pair (`simplace_iopt<n>` vs
`torchcrop_iopt<n>`) differs only in the model. Colour alone does not stay
legible at six overlapping lines, so every plot also varies marker and
linestyle by series — the same combination in every figure, so a series is
identifiable without re-reading each legend.

Two references, and they are not the same kind of thing:

* **CyBench** — an administrative yield statistic over a whole NUTS-3 region,
  at market moisture (~13.5 %). A simulated cell is one point inside it, and
  the models report grain **dry matter**, so ~15 % of the reference is water
  that no correction is applied for here.
* **PEP725** — BBCH stage dates observed by volunteers at a station, matched to
  the nearest simulated cell within 25 km.

Neither pairing is exact, and the spread that comes from the mismatch is not
model error. Produce the inputs with:

```bash
./submit/submit_cropmodelling.sh --smoke
```

which, once per swept IOPT (`TC_SMOKE_IOPTS`, default `1 2 3`), builds and
runs its own SIMPLACE solution — there is no CLI override for `vIOPT` the way
torchcrop has `--iopt`, so unlike torchcrop this is a separate build per
value — hands its simulated sowing dates to torchcrop at that same IOPT, and
prepares `torchcrop/workspace/crop_wheat.yaml` — the crop parameters torchcrop
actually ran, checkable directly against the SIMPLACE template's
`data/crop/crop.xml`, `data/crop/seeds.xml` and `data/management/management.xml`
(`torchcrop/workspace/crop_parameter_audit.csv` does that comparison already,
parameter by parameter).
"""),
    code(SETUP),
    code('''
from cropmodelling4eu.evaluation import germany

SMOKE = Path("/data01/FDS/muduchuru/Data/SIMPLACE/cropmodelling4eu/de_smoke")

# Both models sweep all three usable IOPT settings over the same cells and
# seasons -- submit_cropmodelling.sh --smoke builds one SIMPLACE run
# (simplace_iopt<n>/) and one torchcrop run (de_torchcrop_iopt<n>.parquet)
# per value. A same-IOPT pair then differs only in the model, and the spread
# across IOPT on one side is the nutrient-limitation effect on its own, not a
# confound with the model.
IOPTS = [1, 2, 3]

cells = pd.read_csv(SMOKE / "de_cells.csv")
runs = germany.load_runs(
    **{f"simplace_iopt{i}": SMOKE / f"simplace_iopt{i}" / "simplace_europe.parquet"
       for i in IOPTS},
    **{f"torchcrop_iopt{i}": SMOKE / "torchcrop" / f"de_torchcrop_iopt{i}.parquet"
       for i in IOPTS},
)
lo, hi = germany.common_window(runs)
runs = {name: f[f["year"].between(lo, hi)] for name, f in runs.items()}

print(f"{len(cells)} cells, {cells['adm_id'].nunique()} NUTS-3 regions, "
      f"{cells['lon'].min():.2f}-{cells['lon'].max():.2f} E, "
      f"{cells['lat'].min():.2f}-{cells['lat'].max():.2f} N")
print(f"common window: {lo}-{hi}")
for name, frame in runs.items():
    print(f"  {name:16s} {len(frame):5d} rows, {frame['SimplaceID'].nunique()} cells")
'''),
    md("""
## 0. What torchcrop actually ran

`torchcrop/workspace/` is written by `submit_cropmodelling.sh --smoke` before
either model runs. `crop_<crop>.yaml` is not a description of the run's crop
parameters, it is the file torchcrop loaded them from
(`CropParameters(config_file=...)`) — open it beside the SIMPLACE template's
`data/crop/crop.xml`, `data/crop/seeds.xml` and `data/management/management.xml`
for the same comparison this cell prints.
"""),
    code('''
audit = pd.read_csv(SMOKE / "torchcrop" / "workspace" / "crop_parameter_audit.csv")
print(audit["status"].value_counts().to_string())
display(audit[audit["status"] != "same"])
'''),
    md("""
## 1. Where the cells are

One cell per region, chosen nearest the centre of a 2-D lon/lat bin. Binning in
**two** dimensions matters: `SimplaceID` runs west-to-east within a row, so
taking the first cell of each latitude band walks the western border and
samples a climate gradient rather than the country.
"""),
    code('''
fig, ax = germany.plot_cells(cells, palette=PALETTE)
fig.savefig(config.FIGURE_DIR / "de_smoke_cells.png", dpi=150, bbox_inches="tight")
'''),
    md("""
## 2. Yield vs CyBench

Paired on `(NUTS-3 region, harvest year)`. Bias is `simulated − observed`, so a
negative bias is an under-prediction — and remember ~15 % of the reference is
moisture the models do not report.
"""),
    code('''
yields = germany.validate_yield(runs, cells)
display(yields[["model", "n", "regions", "years", "sim_mean", "obs_mean",
                "bias", "rmse", "mae", "r"]].round(2))
yields.to_csv(config.TABLE_DIR / "de_smoke_yield_vs_cybench.csv", index=False)
'''),
    code('''
fig, axes = germany.plot_yield(runs, cells, palette=PALETTE)
fig.savefig(config.FIGURE_DIR / "de_smoke_yield.png", dpi=150, bbox_inches="tight")
'''),
    md("""
## 3. Phenology vs PEP725

One observed date per `(cell, year)`: the median over the stations matched to
that cell, which is robust to a single mis-typed entry.

**Three rows are not model error and must not be read as such:**

* **Sowing is a constant input** in both runs, so its "bias" is one continental
  offset with no degrees of freedom — `r` is undefined because a constant has
  no variance. It still matters, because it propagates into every later stage.
* **Soft dough (BBCH 85) precedes full ripeness (BBCH 89)**, which is what the
  models report as maturity, so a positive bias of a couple of weeks is the
  expected sign.
* **Emergence** depends on the solution's `vTSUMEM`, not on the export.

Heading and harvest are the genuine tests: neither is an input.
"""),
    code('''
observed = germany.load_pep725(cells, (lo, hi))
phenology = germany.validate_phenology(runs, observed)
display(phenology[["stage", "model", "n", "sim_mean", "obs_mean",
                   "bias", "rmse", "mae", "r"]].round(1))
phenology.to_csv(config.TABLE_DIR / "de_smoke_phenology_vs_pep725.csv", index=False)

for stage, caveat in phenology[["stage", "caveat"]].drop_duplicates().values:
    print(f"  {stage:24s} {caveat}")
'''),
    code('''
fig, axes = germany.plot_phenology(phenology, palette=PALETTE)
fig.savefig(config.FIGURE_DIR / "de_smoke_phenology.png", dpi=150, bbox_inches="tight")
'''),
    md("""
## 4. Inside two seasons: 2003 and 2005

A season mean cannot say *when* a crop was short of water or nitrogen, and that
is the whole question in a drought year: `TRANRF` averaging 0.6 is a season
half-stressed throughout and a season shut down for six weeks in June, and
those are different crops. So both models are read out **daily** over two
contrasting seasons — **2003**, the European drought, and **2005**, an ordinary
year beside it.

Both series are now build artifacts read off disk, the same way the yields
above are — neither is re-derived by this notebook:

* **SIMPLACE** writes `out/daily/<id>_daily.csv` as it runs, once per swept
  IOPT (`simplace_iopt<n>/out/daily/`, its own solution build per value).
  `LAI`, `AGB`, `TRANRF` and `NNI` are in it only because the solution's
  `Daily_crop_growth` output declares them — the same rules `LintulYearly`
  averages or maxes over the season, so the yearly value is a summary of
  exactly this series. A run from before that was added carries `LAI` alone,
  and `load_simplace_daily` says so rather than dropping the rest silently.
* **torchcrop** keeps every per-day state in memory, so `submit_cropmodelling.sh
  --smoke` / `submit_torchcrop.sh --smoke` read the summary and the daily
  trajectory off *one* simulation per swept IOPT (`run_cells_torchcrop.py
  --daily-out`, `run_cells(..., mode="both")`) rather than running the cells
  twice, and write `torchcrop/de_torchcrop_daily_iopt<1,2,3>.parquet`. These
  trajectories are therefore not just comparable to but *literally* the same
  simulations the yields above came from, on the same crop file and (in the
  chained smoke test) the same SIMPLACE-simulated sowing dates, IOPT for
  IOPT. `AGB` is `out.biomass`, converted from its native g/m² to t/ha
  (×0.01) so it lands in the same unit as SIMPLACE's `AGBiomass_t_ha` with
  nothing left for a reader to convert.

Each file's own `model` column reads `simplace`/`torchcrop` regardless of
IOPT — both are relabelled `{model}_iopt<n>` below, the same keys the summary
`runs` dict above uses, so one series stays one series across every figure in
this notebook. Rows are trimmed to the crop phase (`0 < DVS < 2`) on both
sides, so the spin-up and the frozen days past maturity never reach a plot.
"""),
    code('''
SEASONS = [2003, 2005]
ids = cells["SimplaceID"].to_numpy()

# One frame per swept IOPT on each side, relabelled so daily_envelope's
# groupby("model") tells all six apart instead of averaging any of them
# together.
simplace_daily = [
    germany.load_simplace_daily(SMOKE / f"simplace_iopt{i}" / "out", ids=ids,
                                 years=SEASONS, model=f"simplace_iopt{i}")
    for i in IOPTS
]
torchcrop_daily = []
for i in IOPTS:
    frame = pd.read_parquet(SMOKE / "torchcrop" / f"de_torchcrop_daily_iopt{i}.parquet")
    frame = frame[frame["year"].isin(SEASONS)].assign(model=f"torchcrop_iopt{i}")
    torchcrop_daily.append(frame)

daily = pd.concat([*simplace_daily, *torchcrop_daily], ignore_index=True)

print(f"{len(daily):,} rows, {daily['SimplaceID'].nunique()} cells, "
      f"{daily['variable'].nunique()} variables")
'''),
    code('''
# The season in one number per variable, so the curves below can be checked
# against something: peak canopy, and how far each stress index fell.
season = (
    daily.groupby(["variable", "year", "model"])["value"]
    .agg(mean="mean", peak="max", trough="min")
    .round(3)
    .unstack("model")
)
display(season)
season.to_csv(config.TABLE_DIR / "de_smoke_daily_2003_2005.csv")
'''),
    code('''
fig, axes = germany.plot_daily(daily, years=SEASONS, palette=PALETTE)
fig.savefig(config.FIGURE_DIR / "de_smoke_daily_2003_2005.png", dpi=150,
            bbox_inches="tight")
'''),
    md("""
Reading the four rows:

* **LAI** is the level check. Both models should build a canopy through spring
  and lose it after anthesis; a peak that never reaches ~4-6 m² m⁻² over German
  winter wheat is a growth problem, not a timing one.
* **AGB** is the integral LAI drives — a canopy gap that looks small in LAI
  compounds daily into a much larger gap in accumulated biomass, so this row is
  usually the more legible of the two for "is one model just growing a smaller
  crop". It should climb monotonically and flatten at maturity; a fall implies
  a reported loss the daily series can date.
* **TRANRF** is where 2003 has to show itself. A model that responds to water
  drops here in the weeks around anthesis in 2003 and does not in 2005. A
  *chronic* offset through winter is a different thing entirely — it is
  transpiration being reduced when there is barely a canopy to transpire, which
  is a bug in the water balance, not a drought.
* **NNI** is gated on `iopt` in both models now, so the gating is a visible
  curve rather than something to take on faith on one side and assumed on the
  other: `iopt1`/`iopt2` never see a nitrogen limit (NNI pinned at 1 for
  `iopt1`; `iopt2` is water-only, so its NNI series is also flat), on
  `simplace` and `torchcrop` alike, and only the `iopt3` pair can crash
  mid-season — a fall there says the fertilizer schedule ran out, not that
  the weather was hostile. A crash on one side's `iopt3` and not the other's
  is a genuine disagreement about the same setting, not a confound with which
  IOPT each model happened to run.

Two caveats on the x-axis. `das` is days after each model's **own** sowing —
SIMPLACE sows on a rule and torchcrop latches the site table's DOY, so day 0 is
not the same calendar day for both, and aggregating on the calendar date
instead would smear a timing difference into an apparent level difference. The
band is the interquartile range **across the 30 cells**, not an uncertainty:
cells from the Rhine to Mecklenburg do not share a season, and one model's band
being far wider than the other's is itself a result. The three series on each
side share cells and seasons (and, for torchcrop, the same crop file) and
differ only in `iopt`, so any spread within one side **is** the
nutrient-limitation effect rather than a confound with something else.
"""),
    md("""
## 5. Reading it

A short checklist, because the two references fail in different directions:

* **Yield level** is the weakest claim here — a regional statistic against one
  cell, on two different moisture bases. Treat a bias inside ~1 t/ha as
  "unbiased at national level" and nothing finer.
* **Interannual `r`** is the stronger claim: it needs no moisture correction and
  no area weighting, since both sides vary about their own mean. A run with a
  fixed sowing date, one CO₂ value and no cultivar calibration should not be
  expected to manage much above ~0.4.
* **A drought year is the useful single test.** 2003 is the one in this window;
  a model that misses it is not responding to water at all.
* **Heading and harvest bias** are the cleanest results the smoke test
  produces. If either drifts by more than ~2 weeks after a change to the
  export, the change is the first suspect.
"""),
]

# --------------------------------------------------------------------------- #
# Stress test: SIMPLACE vs torchcrop with every input matched
# --------------------------------------------------------------------------- #

stresstest_cells = [
    md("""
# Stress test — SIMPLACE against torchcrop, same crop, same day, no spin-up

The smoke test asks *how close to observations* each model gets and accepts
that they differ in every respect. This asks the narrower question that one
cannot answer: **with every difference that is not the model itself removed, do
the two agree?** The run is produced by
[`submit/submit_stresstest.py`](../submit/submit_stresstest.py), which removes
them one at a time:

| Confound | How it is removed |
|---|---|
| Crop parameters | torchcrop is loaded from SIMPLACE's own `crop.xml` |
| Sowing date | SIMPLACE runs first; torchcrop is latched to its **simulated** `PlantingDOY` |
| Spin-up | Both start at sowing, from the export's initial soil water |
| Irrigation | Removed from both — the solution has no irrigation module |
| CO₂ | Held at 360 ppm, where the crop file's own response curve is 1.0 |
| Fertilizer | Not adjusted — **checked**, in §3 |

**There is no observation in this notebook.** Neither model is a reference, so
every number below is a difference between two simulations of the same site and
season, signed `torchcrop − simplace`. A disagreement says the two models are
not the same model; it does not say which is right. For that, read
[`germany_smoke_evaluation.ipynb`](germany_smoke_evaluation.ipynb).

```bash
./submit/submit_stresstest.py                 # both scenarios, both seasons
./submit/submit_stresstest.py --dry-run       # audit the parameters, run nothing
```
"""),
    md("## 0. Setup"),
    code(SETUP),
    code('''
from cropmodelling4eu.evaluation import stresstest as st

ROOT = st.DEFAULT_ROOT

run = st.load_run(ROOT)
paired = st.pair(run)
provenance = st.load_provenance(ROOT)

print(f"root      : {ROOT}")
print(f"cells     : {run['SimplaceID'].nunique()}")
print(f"seasons   : {sorted(run['year'].unique())}")
print(f"scenarios : {sorted(run['scenario'].unique())}")
print(f"paired    : {len(paired)} cell-seasons both models ran")
'''),
    md("""
## 1. What this run actually removed

Read from `torchcrop/config.yaml`, which is not a report: the script writes it,
loads it back, and the `RunConfig` in it is what both halves were driven by. So
this is the run's input, and editing it and re-running with `--reuse-simplace`
re-runs the torchcrop half exactly as edited.
"""),
    code('''
print(provenance.get("purpose", "no provenance block found"), "\\n")
for name, note in provenance.get("removed_inputs", {}).items():
    print(f"  {name:12s} {note}")
print()
for name, scenario in provenance.get("scenarios", {}).items():
    print(f"  {name:12s} iopt={scenario['iopt']}  {scenario['note']}")
'''),
    md("""
### The asymmetries the design could **not** remove

These are recorded in the config rather than papered over, and both bound how
far §4 can be read.
"""),
    code('''
for note in provenance.get("known_asymmetries", []):
    print("* " + note + "\\n")
'''),
    md("""
## 2. Crop parameters — are the two models growing the same crop?

The audit runs before anything else. With `--crop-params simplace` (the
default) torchcrop is *built from* `crop.xml`, so the remaining differences are
parameters SIMPLACE has no counterpart for, or a mapping that could not be
made. Run with the shipped presets instead and 21 of 72 differ — including
`TSUM1` (1623 vs 1050) — and no disagreement below is attributable to the model.
"""),
    code('''
audit = st.load_audit(ROOT)
print(audit["status"].value_counts().to_string())
print(f"\\ntorchcrop crop parameters: {provenance.get('crop_parameters', {}).get('torchcrop_source')}")
display(audit[audit["status"] != "same"][["parameter", "kind", "simplace", "torchcrop", "status"]])
'''),
    md("""
## 3. Two checks that must pass before anything else is read

**The sowing latch.** torchcrop is set to SIMPLACE's simulated `PlantingDOY`
per cell and season. Any row here means it fell back to the export's calendar
instead, and those cells are then two different seasons rather than two models.

**The fertilizer.** Both read one schedule — the export's
`fertilizer_<crop>.csv` — but by different routes: SIMPLACE takes product
amounts and carrier contents from `fertilizer_composition.xml`, torchcrop
converts them to nutrient rates before the run. Two conversions of one file is
exactly where a silent factor hides.
"""),
    code('''
mismatched = st.check_latches(paired)
fertilizer = st.load_fertilizer_check(ROOT)

print(f"sowing latch : {len(mismatched)} of {len(paired)} cell-seasons differ  "
      f"({'PASS' if mismatched.empty else 'FAIL'})")
print(f"fertilizer N : largest |difference| {fertilizer['difference'].abs().max():.2e} g N/m², "
      f"mean applied {fertilizer['n_applied_g_m2_simplace'].mean():.2f} g N/m²  "
      f"({'PASS' if fertilizer['difference'].abs().max() < 1e-3 else 'FAIL'})")
display(mismatched.head())
'''),
    md("""
## 4. Agreement, quantity by quantity

Ordered from the result to the diagnostics that explain it. `bias` and `rmse`
are in the quantity's own unit; `ratio` is `torchcrop / simplace` on the means,
which is what travels between a 4 t/ha cell and a 9 t/ha one.

`r` is not a skill score here. Both sides vary across cells and seasons for
their own reasons, so a low `r` with a small bias means the two models disagree
about *which* cells are good ones — often a more serious finding than a level
offset, and invisible in the mean.
"""),
    code('''
scores = st.agreement(paired)
display(scores.round(3))
scores.to_csv(config.TABLE_DIR / "stresstest_agreement.csv", index=False,
              float_format="%.4f")

for variable in st.COMPARED:
    print(f"  {variable.label:14s} {variable.note}")
'''),
    code('''
fig, axes = st.plot_agreement(paired, palette=PALETTE)
fig.savefig(config.FIGURE_DIR / "stresstest_agreement.png", dpi=150, bbox_inches="tight")
'''),
    md("""
## 5. Which cells disagree

A scatter hides *which* cells the disagreement sits in; this does not. Read the
shape, not the individual rules: a long rule on one cell with none on its
neighbours is a site problem (a soil, a failed establishment), while a fan that
widens with the SIMPLACE value is a systematic difference in the response.
"""),
    code('''
fig, axes = st.plot_cell_pairs(paired, "yield_t_ha", palette=PALETTE)
fig.savefig(config.FIGURE_DIR / "stresstest_cell_yields.png", dpi=150, bbox_inches="tight")
'''),
    code('''
worst = (
    paired.assign(ratio=paired["yield_ratio"])
    .sort_values("ratio")
    [["scenario", "SimplaceID", "year", "lon", "lat", "yield_t_ha_simplace",
      "yield_t_ha_torchcrop", "ratio", "max_lai_simplace", "max_lai_torchcrop",
      "tranrf_mean_simplace", "tranrf_mean_torchcrop"]]
)
print("The ten cell-seasons where torchcrop is furthest below SIMPLACE:")
display(worst.head(10).round(2))
paired.to_csv(config.TABLE_DIR / "stresstest_paired_cells.csv", index=False,
              float_format="%.4f")
'''),
    md("""
## 6. Where the divergence comes from

The yield ratio against the difference in each state variable. Which one it
tracks is the answer the tables above cannot give:

* it falls with **Δ peak LAI** → a canopy that never built, so the difference is
  in growth or establishment;
* it falls with **Δ TRANRF** → the water balances closed the stomata at
  different times, which is the bucket-vs-layered asymmetry the config records;
* it falls with **Δ season length** → the phenology integrated different
  temperatures despite the shared `TSUM1`;
* it tracks **nothing** → partitioning, and the season means do not resolve it —
  go to the daily trajectories in §7.
"""),
    code('''
fig, axes = st.plot_divergence(paired, palette=PALETTE)
fig.savefig(config.FIGURE_DIR / "stresstest_divergence.png", dpi=150, bbox_inches="tight")
'''),
    md("""
## 7. Inside the season

A season mean cannot say *when* the two models parted, and that is the whole
question: `TRANRF` averaging 0.8 is a season mildly stressed throughout and a
season shut down for three weeks in June, and those are different crops.

Unlike the smoke test's version of this figure, `das` here is the **same
calendar day in both models** — that is what the sowing latch of §3 buys. A
horizontal offset between the curves is therefore a difference in development
rate, not in the day the crop went in. The band is the interquartile range
**across cells**, not an uncertainty; one model's band being far wider than the
other's is itself a result.
"""),
    code('''
SCENARIO = "limited"
daily = st.load_daily(ROOT, scenarios=[SCENARIO])
season = (
    daily.groupby(["variable", "year", "model"])["value"]
    .agg(mean="mean", peak="max", trough="min")
    .round(3)
    .unstack("model")
)
display(season)
season.to_csv(config.TABLE_DIR / "stresstest_daily_season.csv")
'''),
    code('''
fig, axes = st.plot_daily(daily, SCENARIO, palette=PALETTE)
fig.savefig(config.FIGURE_DIR / "stresstest_daily.png", dpi=150, bbox_inches="tight")
'''),
    md("""
## 8. The nutrient-limitation response

`potential` (IOPT=1) minus `limited` (IOPT=3), taken **within** each model. The
question is not whether the two agree in level — §4 answered that — but whether
they respond to `iopt` by a similar amount.

Two things to keep in mind. IOPT=1 is not potential production in either model:
both apply water stress to growth unconditionally, so this is a
nutrient-unlimited but still water-limited run on both sides. And a model that
ran only one scenario is dropped from the figure with a warning rather than
plotted against its own missing half.
"""),
    code('''
effect = st.scenario_effect(run, "yield_t_ha")
if effect.empty:
    print("only one scenario is present in this run — nothing to compare")
else:
    display(
        effect.groupby("model")[["effect", "effect_percent"]]
        .describe().round(2).T
    )
    fig, axes = st.plot_scenario_effect(effect, palette=PALETTE)
    fig.savefig(config.FIGURE_DIR / "stresstest_scenario_effect.png", dpi=150,
                bbox_inches="tight")
'''),
    md("""
## 9. Summary
"""),
    code('''
overview = st.summary(paired)
overview.to_frame().to_csv(config.TABLE_DIR / "stresstest_summary.csv")
print(f"tables written to {config.TABLE_DIR}")
print(f"figures written to {config.FIGURE_DIR}")
overview.to_frame()
'''),
    md("""
### Reading it

* **The checks in §3 are pass/fail, not diagnostics.** A sowing latch mismatch
  or a fertilizer difference above ~1e-3 g N/m² invalidates everything after
  it; fix the run before reading §4.
* **A level offset and a rank disagreement are different findings.** A ratio of
  0.8 with `r ≈ 0.9` is one model consistently lower — a calibration
  difference. A ratio near 1 with `r ≈ 0.1` is the two models disagreeing about
  which cells are good, which no bias correction repairs.
* **Follow the chain, not the yield.** Yield is the last quantity in it: peak
  LAI explains biomass, biomass and partitioning explain yield, and §6 says
  which link the run broke.
* **Cells where torchcrop collapses to a fraction of SIMPLACE** (the last row of
  the summary) are worth reading individually before any pooled statistic —
  a handful of failed establishments moves a mean far more than a systematic
  offset does.
* **This notebook cannot say which model is right.** It says whether the two are
  the same model given the same inputs, and where they stop being one.
"""),
]


# --------------------------------------------------------------------------- #
# Notebook 7 — full continental run, both models
# --------------------------------------------------------------------------- #

FULL_RUN_SETUP = '''
import logging
import sys
from pathlib import Path

# cropmodelling4eu is installed (pip install -e .), so the evaluation
# library is imported like any other package rather than off sys.path.

import numpy as np
import pandas as pd

from cropmodelling4eu.evaluation import aggregate, config, cybench, doy, fullrun, metrics, plots, regions, torchcrop
from cropmodelling4eu.evaluation.style import use_style

logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s",
                    force=True)
logging.getLogger("matplotlib").setLevel(logging.WARNING)

PALETTE = use_style("light")
config.ensure_output_dirs()

pd.set_option("display.max_rows", 60)
pd.set_option("display.width", 140)

print(f"SIMPLACE run  : {config.SIMPLACE_PARQUET}",
      "(found)" if config.SIMPLACE_PARQUET.is_file() else "(not finished yet)")
print(f"TorchCrop run : {config.SIM_PARQUET}",
      "(found)" if config.SIM_PARQUET.is_file() else "(not finished yet)")
print(f"CyBench root  : {config.CYBENCH_ROOT}")
print(f"Outputs       : {config.OUTPUT_DIR}")
'''

full_run_cells = [
    md("""
# Full continental run — SIMPLACE and TorchCrop, together

**What this notebook does.** It reads each model's own `winter_wheat_2000_2024`
production run exactly as
[`submit/submit_cropmodelling.sh`](../submit/submit_cropmodelling.sh) writes
it — `cm4eu simplace collect`'s `simplace_europe.parquet` and the shard
combine's `torchcrop_europe.parquet` — compares both against the CyBench
national yield and phenology statistics (§4-5), and then compares the two
models directly against each other over the whole domain (§6).

**How this differs from the other two-model notebooks.**
`germany_smoke_evaluation.ipynb` matches 30 cells and hands SIMPLACE's own
simulated sowing dates to torchcrop by construction; `stresstest_evaluation.ipynb`
forces every non-model input equal (crop parameters, spin-up, irrigation,
CO₂). Neither one is what ships. This notebook reads whatever each pipeline
actually produced for the full run — a different sowing convention on each
side unless the run was submitted through the chain (SIMPLACE's simulated
dates handed to torchcrop), a different year range, a cell set that need not
match exactly — because the question here is what the delivered output says,
not an idealised comparison.

**Either run can be missing, and the notebook says so rather than failing.**
`fullrun.load_runs` skips a model with no Parquet yet and logs a warning; §2
reports which models loaded. §4-5 run per model over whichever are present;
§6, which needs both, prints a note and stops there if only one is.

Four conventions carried over from the single-model notebooks:

1. **Error is `simulated − observed`** against CyBench (§4-5); `torchcrop −
   simplace` in the direct comparison (§6), where neither side is a
   reference — see `evaluation.fullrun.pair_models`.
2. **No moisture conversion.** Both models report grain dry matter; CyBench
   reports market moisture (~13.5 %), so part of any negative bias against it
   is that unit difference rather than model error.
3. **The simulated national mean is unweighted** over cropland cells — the
   export carries no per-cell wheat area to weight by.
4. **Day-of-year statistics are circular** (`doy.circular_mean_doy`,
   `doy.doy_difference`), for the reason `phenology_evaluation.ipynb`'s header
   gives: Spain's regional `sos` runs from DOY 0.7 to 363.

All reusable code is in
[`../src/cropmodelling4eu/evaluation/`](../src/cropmodelling4eu/evaluation/);
this notebook is the workflow only.
"""),
    md("## 0. Setup"),
    code(FULL_RUN_SETUP),
    md("""
## 1. Scope — which countries are evaluable

A country needs a CyBench yield file *and* CyBench polygons: the file supplies
the reference, the polygons decide which 10 km cells belong to it.
"""),
    code(SCOPE),
    md("""
## 2. Load both runs

One row per (cell, season) on each side. `fullrun.load_runs` reads each with
its own model's rules — SIMPLACE's collector already writes the dates it
simulated, torchcrop's `days_to_maturity` is turned into dates by
`torchcrop.add_phenology_columns` — and tags every row with `model`. A model
with no Parquet yet is skipped with a warning, not raised on.
"""),
    code('''
runs = fullrun.load_runs()
print(f"\\nmodels loaded: {', '.join(runs)}")
for model, frame in runs.items():
    print(f"{model:10s}: {len(frame):>9,} cell-seasons, "
          f"{frame['SimplaceID'].nunique():>6,} cells, "
          f"{frame['year'].min()}-{frame['year'].max()}")

pd.concat(
    {model: frame[["yield_t_ha", "days_to_maturity"]].describe()
     for model, frame in runs.items()},
    axis=1,
)
'''),
    md("""
## 3. Assign each 10 km cell to a country

Built from the **union** of both models' cells: both pipelines read the same
10 km export, so they share one grid, and one country join covers whichever
model (or both) is present. Cells outside the CyBench footprint — the UK,
Norway, Switzerland, the western Balkans, North Africa — are dropped from
every model.
"""),
    code('''
all_cells = torchcrop.simulation_cells(
    pd.concat([frame[["SimplaceID", "lon", "lat"]] for frame in runs.values()],
              ignore_index=True)
)
all_cells = regions.assign_cells_to_countries(
    all_cells, COUNTRIES, cache=regions.default_cache_path(len(COUNTRIES))
)
print(f"{all_cells['country'].notna().sum():,} of {len(all_cells):,} cells "
      f"inside the CyBench footprint ({all_cells['snapped'].sum():,} matched "
      f"by the {config.SNAP_KM} km snap)")

scoped = {}
for model, frame in runs.items():
    tagged = frame.merge(all_cells[["SimplaceID", "country", "snapped"]],
                          on="SimplaceID", how="left")
    scoped[model] = tagged[tagged["country"].notna()].copy()
    print(f"{model:10s}: {len(scoped[model]):>9,} of {len(frame):>9,} "
          f"cell-seasons kept")

cells_per_country = (
    all_cells.dropna(subset=["country"]).groupby("country").size()
    .rename("cells").to_frame()
)
cells_per_country.T
'''),
    md("""
## 4. National yield — both models against CyBench

Same pipeline as `yield_evaluation.ipynb` §4-8, run once per loaded model. The
CyBench side (§4.2) does not depend on the model, so it is aggregated once and
paired against each.
"""),
    md("### 4.1 Aggregate each model to national means, and pair against CyBench"),
    code('''
obs = cybench.load_yield(COUNTRIES)
obs_country = aggregate.aggregate_observed_yield(obs)
print(obs_country["obs_method"].value_counts().to_string())

yield_paired = []
for model, frame in scoped.items():
    sim_country = aggregate.aggregate_simulated(frame, {"yield_t_ha": False})
    paired = aggregate.pair_observations(sim_country, obs_country, ["country", "year"])
    paired["model"] = model
    paired["residual"] = paired["yield_t_ha"] - paired["obs_yield"]
    yield_paired.append(paired)
yield_paired = pd.concat(yield_paired, ignore_index=True)

print(f"\\n{len(yield_paired)} paired (model, country, year) rows across "
      f"{yield_paired['country'].nunique()} countries")
yield_paired.head()
'''),
    md("### 4.2 Pooled skill, both models"),
    code('''
yield_pooled = {
    model: metrics.yield_metrics(block["obs_yield"], block["yield_t_ha"])
    for model, block in yield_paired.groupby("model")
}
yield_pooled_table = pd.DataFrame(yield_pooled).T[list(metrics.YIELD_METRIC_ORDER)]
yield_pooled_table.round(3)
'''),
    md("""
### 4.3 Per-country skill, both models

One RMSE/Bias/Pearson r table per model, side by side, so a country either
model struggles with is visible at a glance rather than in two separate
notebooks.
"""),
    code('''
yield_by_country = {
    model: metrics.metrics_by_group(block, "obs_yield", "yield_t_ha")
    for model, block in yield_paired.groupby("model")
}
yield_comparison = pd.concat(
    {model: table[["n", "bias", "rmse", "pearson_r"]]
     for model, table in yield_by_country.items()},
    axis=1,
)
yield_comparison.to_csv(
    config.TABLE_DIR / "full_run_yield_metrics_by_country.csv", float_format="%.3f"
)
yield_comparison.round(2)
'''),
    md("### 4.4 Figures, one per model"),
    code('''
for model, block in yield_paired.groupby("model"):
    fig = plots.scatter_one_to_one(
        block, "obs_yield", "yield_t_ha", stats=yield_pooled[model],
        title=f"{model}: national winter wheat yield, "
              f"{block['year'].min()}-{block['year'].max()}",
        xlabel="CyBench observed yield (t ha$^{-1}$)",
        ylabel=f"{model} simulated yield (t ha$^{-1}$)",
    )
    plots.save(fig, f"full_run_01_yield_scatter_{model}")
    display(fig)
'''),
    code('''
for model, table in yield_by_country.items():
    ranked = metrics.rank_countries(table, by="rmse")
    fig = plots.bias_bars(
        ranked, value_col="bias", error_col="rmse",
        title=f"{model}: mean yield bias by country (simulated - observed)",
        xlabel="Bias (t ha$^{-1}$)",
    )
    plots.save(fig, f"full_run_02_yield_bias_{model}")
    display(fig)
'''),
    md("""
### 4.5 Figures — spatial maps, both models

Where §4.3 gives a per-country number, these put it on the map: the cell-level
mean shows the field each national average is drawn from, and the choropleth
repeats §4.3's bias per country in the shape a reader recognises faster than a
table.
"""),
    code('''
polygons = regions.load_country_polygons(COUNTRIES)

for model, frame in scoped.items():
    cell_mean = frame.groupby(["SimplaceID", "lon", "lat"], as_index=False)["yield_t_ha"].mean()
    fig = plots.cell_map(
        cell_mean, "yield_t_ha",
        title=f"{model}: mean simulated winter wheat yield, "
              f"{frame['year'].min()}-{frame['year'].max()}",
        cbar_label="Yield (t ha$^{-1}$)",
    )
    plots.save(fig, f"full_run_06_yield_map_{model}")
    display(fig)
'''),
    code('''
for model, table in yield_by_country.items():
    fig = plots.country_choropleth(
        polygons, table["bias"],
        title=f"{model}: mean yield bias by country (simulated - observed)",
        cbar_label="Bias (t ha$^{-1}$)", diverging=True,
    )
    plots.save(fig, f"full_run_07_yield_bias_map_{model}")
    display(fig)
'''),
    md("""
## 5. Phenology — both models against CyBench

Same three stages as `phenology_evaluation.ipynb` — sowing, maturity, harvest
— for whichever models are loaded. Read `config.STAGES[*].caveat` before the
sowing and harvest numbers: sowing is a weak pairing whenever a model's
`sowing_doy` is a constant rather than a per-cell result, and harvest repeats
maturity while `HARVEST_LAG_DAYS = 0`.
"""),
    code('''
for stage in config.STAGES:
    print(f"{stage.label:9s} <- {stage.sim_col:14s} vs CyBench '{stage.obs_col}'")
    print(f"  {stage.caveat}\\n")
'''),
    md("### 5.1 Aggregate each model's stage dates, and pair against the CyBench calendar"),
    code('''
STAGE_COLS = {stage.sim_col: True for stage in config.STAGES}

calendar = cybench.load_calendar(COUNTRIES)
crop_mask = cybench.load_crop_mask(COUNTRIES)
obs_calendar = aggregate.aggregate_observed_calendar(calendar, crop_mask)

phen_paired = []
for model, frame in scoped.items():
    available = {c: circ for c, circ in STAGE_COLS.items() if c in frame.columns}
    sim_country = aggregate.aggregate_simulated(frame, available)
    paired = aggregate.pair_observations(sim_country, obs_calendar, ["country"])
    paired["model"] = model
    phen_paired.append(paired)
phen_paired = pd.concat(phen_paired, ignore_index=True, sort=False)

print(f"{len(phen_paired)} (model, country) rows across "
      f"{phen_paired['country'].nunique()} countries")
phen_paired.head()
'''),
    md("### 5.2 Pooled skill by stage, both models"),
    code('''
phen_pooled_rows = []
for stage in config.STAGES:
    if stage.sim_col not in phen_paired.columns:
        continue
    for model, block in phen_paired.groupby("model"):
        row = metrics.phenology_metrics(block[stage.obs_col], block[stage.sim_col])
        row.update(stage=stage.label, model=model)
        phen_pooled_rows.append(row)

phen_pooled_table = (
    pd.DataFrame(phen_pooled_rows).set_index(["stage", "model"])
    [list(metrics.PHENOLOGY_METRIC_ORDER)]
)
phen_pooled_table.round(1)
'''),
    md("""
### 5.3 Maturity bias by country, both models

Maturity is "the one clean pairing" (see the caveat printed in §5): both sides
mean the same thing, so it is the stage where a per-country figure says
something about the model rather than about a convention mismatch.
"""),
    code('''
for model, block in phen_paired.groupby("model"):
    stage_metrics = metrics.metrics_by_group(
        block, "eos", "maturity_doy", metric_fn=metrics.phenology_metrics
    )
    ranked = metrics.rank_countries(stage_metrics, by="rmse")
    fig = plots.bias_bars(
        ranked, value_col="bias", error_col="rmse",
        title=f"{model}: maturity bias by country (simulated - observed)",
        xlabel="Bias (days; positive = simulated late)",
    )
    plots.save(fig, f"full_run_03_maturity_bias_{model}")
    display(fig)
'''),
    md("""
### 5.4 Figure — spatial maturity bias, both models

The same country bias as §5.3, on the map — useful here because the sowing
convention mismatch (the §5 caveat) is regional: southern European countries
share an autumn-sowing convention with the model, so where the bias
concentrates says more than the ranked bar chart does.
"""),
    code('''
for model, block in phen_paired.groupby("model"):
    stage_metrics = metrics.metrics_by_group(
        block, "eos", "maturity_doy", metric_fn=metrics.phenology_metrics
    )
    fig = plots.country_choropleth(
        polygons, stage_metrics["bias"],
        title=f"{model}: mean maturity bias by country (simulated - observed)",
        cbar_label="Bias (days; positive = simulated late)", diverging=True,
    )
    plots.save(fig, f"full_run_08_maturity_bias_map_{model}")
    display(fig)
'''),
    md("""
## 6. SIMPLACE against TorchCrop, directly

**No observation in this section — every number is a difference between the
two models**, signed `torchcrop − simplace` (`fullrun.pair_models`), the same
convention `stresstest_evaluation.ipynb` uses and for the same reason: calling
one side "simulated" and the other "observed" would claim a reference neither
run supports. This section needs both models and is skipped otherwise.
"""),
    code('''
BOTH_MODELS = len(runs) == 2
if not BOTH_MODELS:
    print("Only one model has a finished run -- skipping the direct comparison. "
          "Re-run this notebook once the other has been collected.")
else:
    model_pairs = fullrun.pair_models(runs)
    print(f"{len(model_pairs):,} (cell, year) pairs common to both runs "
          f"(of {len(runs['simplace']):,} simplace, {len(runs['torchcrop']):,} torchcrop)")
    model_pairs.head()
'''),
    md("""
### 6.1 Do the two runs share a sowing date?

Zero everywhere is what `submit_cropmodelling.sh`'s chained run is *for* —
SIMPLACE's simulated date handed straight to torchcrop's shard array. Anything
else means torchcrop sowed from its own convention instead (a per-cell site
table date, or the DOY 270 constant for a run made before the site stage), and
every later stage in this section is comparing two different seasons.
"""),
    code('''
if BOTH_MODELS:
    mismatch = (model_pairs["sowing_doy_delta"] != 0).mean()
    print(f"{mismatch:.1%} of paired cell-years sow on a different day.")
    display(model_pairs["sowing_doy_delta"].describe().to_frame())
'''),
    md("### 6.2 Agreement, pooled over every paired cell-season"),
    code('''
if BOTH_MODELS:
    rows = []
    for key, label, is_date in [
        ("yield_t_ha", "Yield (t/ha)", False),
        ("biomass_g_m2", "Biomass (g/m2)", False),
        ("max_lai", "Peak LAI (m2/m2)", False),
        ("maturity_doy", "Maturity (DOY)", True),
    ]:
        simplace_col, torchcrop_col = f"{key}_simplace", f"{key}_torchcrop"
        if simplace_col not in model_pairs:
            continue
        fn = metrics.phenology_metrics if is_date else metrics.yield_metrics
        rows.append(fn(model_pairs[simplace_col], model_pairs[torchcrop_col]) | {"variable": label})
    agreement = pd.DataFrame(rows).set_index("variable")
    agreement.to_csv(config.TABLE_DIR / "full_run_model_agreement.csv", float_format="%.3f")
    agreement.round(3)
'''),
    md("""
### 6.3 Figure — yield agreement

The 1:1 line is the only reference drawn: there is no observation to regress
on here, so a fitted line would suggest a skill this run cannot measure.
"""),
    code('''
if BOTH_MODELS:
    stats = metrics.yield_metrics(
        model_pairs["yield_t_ha_simplace"], model_pairs["yield_t_ha_torchcrop"]
    )
    fig = plots.scatter_one_to_one(
        model_pairs, "yield_t_ha_simplace", "yield_t_ha_torchcrop", stats=stats,
        title="Yield: torchcrop against SIMPLACE, same cell and season",
        xlabel="SIMPLACE yield (t ha$^{-1}$)",
        ylabel="TorchCrop yield (t ha$^{-1}$)",
    )
    plots.save(fig, "full_run_04_model_agreement_yield")
    display(fig)
'''),
    md("""
### 6.4 Figure — maturity agreement
"""),
    code('''
if BOTH_MODELS and "maturity_doy_simplace" in model_pairs:
    stats = metrics.phenology_metrics(
        model_pairs["maturity_doy_simplace"], model_pairs["maturity_doy_torchcrop"]
    )
    fig = plots.scatter_one_to_one(
        model_pairs, "maturity_doy_simplace", "maturity_doy_torchcrop",
        stats=stats, circular=True,
        title="Maturity: torchcrop against SIMPLACE, same cell and season",
        xlabel="SIMPLACE maturity (DOY)",
        ylabel="TorchCrop maturity (DOY)",
    )
    plots.save(fig, "full_run_05_model_agreement_maturity")
    display(fig)
'''),
    md("""
### 6.5 Figure — spatial map of model disagreement

Signed `torchcrop - simplace` per cell, mean over the paired years — the
spatial counterpart of §6.2's pooled numbers, and the only way to see whether
the disagreement is a uniform offset or concentrated in particular regions.
The §6.1 sowing mismatch, in particular, need not affect every cell the same
way.
"""),
    code('''
if BOTH_MODELS:
    cell_delta = model_pairs.groupby(
        ["SimplaceID", "lon", "lat"], as_index=False
    )[["yield_t_ha_delta", "maturity_doy_delta"]].mean()

    fig = plots.cell_map(
        cell_delta, "yield_t_ha_delta",
        cmap=plots.DIVERGING_CMAP, norm=plots.diverging_norm(cell_delta["yield_t_ha_delta"]),
        title="Yield disagreement: torchcrop - simplace, mean over paired years",
        cbar_label="Yield delta (t ha$^{-1}$)",
    )
    plots.save(fig, "full_run_09_model_agreement_yield_map")
    display(fig)
'''),
    code('''
if BOTH_MODELS and "maturity_doy_delta" in model_pairs:
    fig = plots.cell_map(
        cell_delta, "maturity_doy_delta",
        cmap=plots.DIVERGING_CMAP, norm=plots.diverging_norm(cell_delta["maturity_doy_delta"]),
        title="Maturity disagreement: torchcrop - simplace, mean over paired years",
        cbar_label="Maturity delta (days)",
    )
    plots.save(fig, "full_run_10_model_agreement_maturity_map")
    display(fig)
'''),
    md("""
## 7. Summary
"""),
    code('''
yield_paired.to_csv(config.TABLE_DIR / "full_run_yield_paired.csv",
                    index=False, float_format="%.4f")
phen_paired.to_csv(config.TABLE_DIR / "full_run_phenology_paired.csv",
                   index=False, float_format="%.4f")
if BOTH_MODELS:
    model_pairs.to_csv(config.TABLE_DIR / "full_run_model_pairs.csv",
                       index=False, float_format="%.4f")
print(f"tables written to {config.TABLE_DIR}")
print(f"figures written to {config.FIGURE_DIR}")

summary = pd.DataFrame({
    "yield RMSE (t/ha)": {m: round(v["rmse"], 2) for m, v in yield_pooled.items()},
    "yield Bias (t/ha)": {m: round(v["bias"], 2) for m, v in yield_pooled.items()},
    "yield Pearson r": {m: round(v["pearson_r"], 2) for m, v in yield_pooled.items()},
})
summary
'''),
    md("""
### What this notebook can and cannot say

* **§4-5 score each model against an external reference**, exactly as the
  single-model notebooks do — a country either model gets wrong is a genuine
  finding there.
* **§6 cannot say which model is right.** It says whether the two production
  runs agree, and §6.1 says how much of any disagreement is simply a different
  sowing convention rather than a model difference — read it before the rest
  of the section.
* **This is the delivered pipeline, not a controlled experiment.** Unlike
  `stresstest_evaluation.ipynb`, nothing here has been forced equal, so a
  disagreement in §6 can come from the sowing date, the crop parameters, the
  year range, or the model physics — this notebook does not separate them.
  `stresstest_evaluation.ipynb` is where that separation is done.
* **Re-run once both arrays finish.** `submit/submit_cropmodelling.sh --status`
  reports what each stage has completed; §2's model list is this notebook's
  own record of what it actually saw.
"""),
]


if __name__ == "__main__":
    write("yield_evaluation.ipynb", yield_cells)
    write("phenology_evaluation.ipynb", phenology_cells)
    write("gdhy_yield_evaluation.ipynb", gdhy_cells)
    write("sage_calendar_evaluation.ipynb", sage_cells)
    write("germany_smoke_evaluation.ipynb", germany_cells)
    write("stresstest_evaluation.ipynb", stresstest_cells)
    write("full_run_evaluation.ipynb", full_run_cells)
