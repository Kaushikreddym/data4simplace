"""Generate the CMIP6 exploratory notebooks.

Written from source, like ``../build_notebooks.py``, so the cells stay diffable. Run
``features.py`` once (05 reads its cache), then this, then execute the ``.ipynb`` files.
Findings are written after a first execution and live in ``findings.py``.
"""

from __future__ import annotations

from pathlib import Path

import nbformat as nbf

from findings import FINDINGS

HERE = Path(__file__).parent
MD = lambda text: nbf.v4.new_markdown_cell(text.strip("\n"))
CODE = lambda text: nbf.v4.new_code_cell(text.strip("\n"))

CAVEATS = """
**Reading these numbers**

* **Scope.** Torchcrop only, on calibrated parameters, for 4 models × 5 experiments, one run per
  model-experiment. Differences between experiments of one model include internal variability.
* **Cells.** The 59,304 cells every run simulated, weighted by cropland area. Each forcing has its own
  frame, and all 5,572 dropped cells lie outside the box the frames share.
* **Fixed crop system.** Cultivar, fertiliser and sowing (per-cell median of the MSWX SIMPLACE dates)
  do not change, so nothing here includes adaptation and sowing cannot respond to the forcing.
* **Wind** is an MSWX climatology in every run, so wind carries no scenario signal.
* **`hist-aer` holds CO₂ at 1850 (284 ppm)**; the other experiments follow their own CO₂. A contrast with
  `hist-aer` therefore mixes the climate effect with the CO₂ effect on the crop.
* **`yield_t_ha` excludes heat sterility.** `adjusted_yield_t_ha` = yield × (1 − heat stress). The
  calibration and evaluation use `yield_t_ha`, so it is the primary series here (see 04's sensitivity test).
* **No season 2015.** Historical runs end at harvest 2014, and SSP runs start with sowing in 2015.
"""


def setup(name: str, *lines: str) -> str:
    body = "\n".join(lines)
    return f"""
import logging
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import xarray as xr
from IPython.display import display

sys.path.insert(0, str(Path.cwd()))
import scenarios as sc
from cropmodelling4eu.evaluation.plots import save
from cropmodelling4eu.evaluation.style import SEQUENTIAL_CMAP, DIVERGING_CMAP, use_style

logging.basicConfig(level=logging.WARNING, force=True)
PALETTE = use_style("light")
plt.rcParams["figure.dpi"] = 90
FIGURES, TABLES = sc.table_dir("{name}")
cells = sc.load_cells()
y = sc.cube("yield_t_ha")
EXP_COLOURS = {{"historical": PALETTE["primary"], "hist-GHG": PALETTE["series_4"], "hist-aer": PALETTE["series_3"],
               "ssp126": PALETTE["series_1"], "ssp370": PALETTE["series_2"]}}
MODEL_COLOURS = dict(zip(sc.MODELS, (PALETTE[f"series_{{i}}"] for i in (1, 2, 3, 5))))
SPAN = lambda s: f"{{s[0]}}–{{s[1]}}"
{body}
print(f"{{cells.n}} cells; outputs in {{FIGURES.parent}}")
"""


def notebook(file: str, *cells: nbf.NotebookNode) -> None:
    nb = nbf.v4.new_notebook(cells=list(cells), metadata={"kernelspec": {
        "display_name": "Python 3", "language": "python", "name": "python3"}})
    finding = FINDINGS.get(file)
    if finding:
        nb["cells"].insert(1, MD(finding))
    nbf.write(nb, HERE / f"{file}.ipynb")


# ------------------------------------------------------------------------- spatial
notebook("01_spatial_differences",
MD(f"""
# CMIP6 torchcrop runs — spatial differences

Where does simulated winter-wheat yield differ, between the runs and from today's climate? Changes
are per model first, then averaged. Hatching marks cells where fewer than 3 of 4 models share the sign
of the multi-model mean, so an unhatched cell means models agree on sign, not that the change is large.

{CAVEATS}
"""),
MD("## 0. Setup"),
CODE(setup("spatial")),
MD("## 1. Which cells are compared\n\nThe table lists, per run, the cells with no yield in any year. The map shows what the common mask removes."),
CODE("""
inv = sc.run_inventory()
inv["no yield"] = inv["null"].map(len)
display(inv.pivot(index="model", columns="experiment", values="no yield")[list(sc.EXPERIMENTS)])

held = set.intersection(*inv["cells"])
lost = pd.Index(sorted(set.union(*inv["null"])))
index = lost.to_numpy() - 1
lost_frame = pd.DataFrame({"lat": sc.MAX_LAT - sc.RES / 2 - (index // sc.N_LON) * sc.RES,
                           "lon": sc.MIN_LON + sc.RES / 2 + (index % sc.N_LON) * sc.RES}, index=lost)
box = sc.COMMON_BOX
inside = lost_frame["lat"].between(*box["lat"]) & lost_frame["lon"].between(*box["lon"])
mswx_ids = set(sc.mswx_cube().cell.values) | set(pd.read_parquet(sc.MSWX_RUN / "torchcrop" / "torchcrop_europe.parquet", columns=["SimplaceID"])["SimplaceID"].unique())
print(f"Run cells {len(held):,}; common {cells.n:,}; dropped {len(lost):,}, of which inside the shared frame box: {int(inside.sum())}")
print(f"MSWX production cells not in any CMIP6 run: {len(mswx_ids - held):,}")
print(f"Cropland in the common set: {cells.weight.sum():,.0f} km²; in CyBench countries: {cells.frame.loc[cells.frame.adm_country.notna(), 'cropland_km2'].sum():,.0f} km²")

fig, ax = sc.map_axes(1, 1, 5.5, 5.0)
ax = ax[0, 0]
sc.draw_map(ax, np.log10(cells.weight + 0.1), cmap=SEQUENTIAL_CMAP, title="Common cells (log₁₀ cropland km²) and dropped cells (red)")
import cartopy.crs as ccrs
ax.scatter(lost_frame["lon"], lost_frame["lat"], s=0.4, c="tab:red", linewidths=0, transform=ccrs.PlateCarree(), zorder=4)
save(fig, "mask", FIGURES)
plt.show()
"""),
MD("## 2. Are the runs anchored to the MSWX run?\n\nThe forcing is bias-corrected to MSWX, so `historical` should reproduce the MSWX run's **mean state** over the 15 shared seasons (2000–2014). A free-running model has its own weather sequence, so individual years and interannual correlation are not comparable."),
CODE("""
SHARED = (2000, 2014)
mswx = sc.clim(sc.mswx_cube(), SHARED)
diff = sc.clim(y.sel(experiment="historical"), SHARED) - mswx
rel = 100.0 * diff / mswx.where(mswx > 0)
weights = cells.weight / cells.weight.sum()
rows = []
for m in sc.MODELS:
    sim = sc.clim(y.sel(experiment="historical", model=m), SHARED).values
    ok = np.isfinite(sim) & np.isfinite(mswx.values)
    mx, my = np.average(mswx.values[ok], weights=weights[ok]), np.average(sim[ok], weights=weights[ok])
    cov = np.average((mswx.values[ok] - mx) * (sim[ok] - my), weights=weights[ok])
    r = cov / np.sqrt(np.average((mswx.values[ok] - mx) ** 2, weights=weights[ok]) * np.average((sim[ok] - my) ** 2, weights=weights[ok]))
    rows.append({"model": m, "MSWX (t/ha)": mx, "historical (t/ha)": my, "bias (t/ha)": my - mx, "bias (%)": 100 * (my / mx - 1),
                 "RMSE (t/ha)": np.sqrt(np.average((sim[ok] - mswx.values[ok]) ** 2, weights=weights[ok])), "pattern r": r})
anchor = pd.DataFrame(rows).set_index("model")
anchor.to_csv(TABLES / "historical_vs_mswx.csv")
display(anchor.round(3))

fig, axes = sc.map_axes(1, 5, 2.9, 3.2)
scale = sc.diverging(diff)
for ax, m in zip(axes[0], sc.MODELS):
    mesh = sc.draw_map(ax, diff.sel(model=m).values, title=m, **scale)
mesh = sc.draw_map(axes[0, 4], diff.mean("model").values, title="Multi-model mean", **scale)
fig.colorbar(mesh, ax=axes[0], shrink=0.7, label=f"historical − MSWX, {SPAN(SHARED)} (t/ha)")
save(fig, "historical_minus_mswx", FIGURES)
plt.show()
"""),
MD("## 3. Baseline climate of the yield, 1985–2014\n\nMulti-model mean, the spread between models, and each model's departure from that mean."),
CODE("""
base = sc.clim(y.sel(experiment="historical"), sc.BASELINE)
fig, axes = sc.map_axes(1, 2, 4.2, 4.0)
fig.colorbar(sc.draw_map(axes[0, 0], base.mean("model").values, cmap=SEQUENTIAL_CMAP, vmin=0, vmax=float(base.mean("model").quantile(0.99)), title="Multi-model mean"), ax=axes[0, 0], shrink=0.7, label="t/ha")
fig.colorbar(sc.draw_map(axes[0, 1], base.std("model").values, cmap=SEQUENTIAL_CMAP, vmin=0, vmax=float(base.std("model").quantile(0.99)), title="Between-model standard deviation"), ax=axes[0, 1], shrink=0.7, label="t/ha")
save(fig, "baseline_climatology", FIGURES)
plt.show()
display(sc.regional_mean(base).to_pandas().T.rename(index=sc.REGION_NAMES).round(2))
"""),
MD("## 4. Future change against 1985–2014\n\nChange relative to each model's own baseline. Hatching marks cells where models disagree on the sign. Regional % values are the ratio of regional means, not the mean of per-cell %, which low-yield cells would dominate."),
CODE("""
SCENARIOS = [(e, s) for e in sc.FUTURE for s in (sc.NEAR, sc.FAR)]
pct = {(e, s): sc.relative_change(y, e, s) for e, s in SCENARIOS}
blank = 100 * cells.weight[(sc.clim(y.sel(experiment="historical"), sc.BASELINE) < sc.MIN_BASE_YIELD).any("model").values].sum() / cells.weight.sum()
print(f"Cells with a baseline below {sc.MIN_BASE_YIELD} t/ha in any model are blank in % maps ({blank:.1f} % of cropland)")
absolute = {(e, s): sc.absolute_change(y, e, s) for e, s in SCENARIOS}
scale = sc.diverging(xr.concat([v.mean("model") for v in pct.values()], "k"))
fig, axes = sc.map_axes(2, 2, 3.9, 3.0)
for ax, (key, delta) in zip(axes.ravel(), pct.items()):
    mesh = sc.draw_map(ax, delta.mean("model").values, title=f"{key[0]}, {SPAN(key[1])}", hatch=~sc.agreement(delta).values, **scale)
fig.colorbar(mesh, ax=axes, shrink=0.6, label="Yield change (%)")
save(fig, "future_change_pct", FIGURES)
plt.show()
table = sc.regional_table({f"{e} {SPAN(s)}: {u}": (sc.regional_relative_change(y, e, s) if u == "%" else absolute[(e, s)]) for e, s in SCENARIOS for u in ("%", "t/ha")}, 2)
table.to_csv(TABLES / "future_change_regions.csv")
display(table.T)
"""),
MD("The same far-future change, model by model:"),
CODE("""
fig, axes = sc.map_axes(2, 4, 2.9, 2.3)
scale = sc.diverging(xr.concat([pct[(e, sc.FAR)] for e in sc.FUTURE], "k"))
for r, e in enumerate(sc.FUTURE):
    for c, m in enumerate(sc.MODELS):
        mesh = sc.draw_map(axes[r, c], pct[(e, sc.FAR)].sel(model=m).values, title=f"{m}, {e}", **scale)
fig.colorbar(mesh, ax=axes, shrink=0.6, label=f"Yield change, {SPAN(sc.FAR)} (%)")
save(fig, "future_change_models", FIGURES)
plt.show()
"""),
MD("## 5. Scenario difference: ssp370 − ssp126\n\nThe same models and periods, with different emissions. CO₂ reaches 867 ppm against 446 ppm in 2100, so a positive value can be CO₂ fertilisation winning over stronger warming. Notebook 05 separates the two."),
CODE("""
gap = {s: clim_gap for s, clim_gap in ((s, sc.clim(y.sel(experiment="ssp370"), s) - sc.clim(y.sel(experiment="ssp126"), s)) for s in (sc.NEAR, sc.FAR))}
scale = sc.diverging(gap[sc.FAR].mean("model"))
fig, axes = sc.map_axes(1, 2, 4.0, 4.0)
for ax, (s, delta) in zip(axes[0], gap.items()):
    mesh = sc.draw_map(ax, delta.mean("model").values, title=f"ssp370 − ssp126, {SPAN(s)}", hatch=~sc.agreement(delta).values, **scale)
fig.colorbar(mesh, ax=axes, shrink=0.7, label="t/ha")
save(fig, "ssp370_minus_ssp126", FIGURES)
plt.show()
display(sc.regional_table({SPAN(s): d for s, d in gap.items()}, 2).T)
"""),
MD("""
## 6. Forcing contrasts in the historical period

| Contrast | What differs | CO₂ on the crop |
|---|---|---|
| `hist-GHG − historical`, 1985–2014 | No aerosol or natural forcing in `hist-GHG` | Same |
| `historical − hist-aer`, 1985–2014 | GHG climate, natural forcing and CO₂ | **Differs by ~85 ppm on average** |
| Change 1985–2014 vs 1951–1980 within each experiment | The forcing's change over time | Rises except in `hist-aer` |

Only the first row isolates a climate effect. The residual `Δhistorical − (Δhist-GHG + Δhist-aer)` is what a two-forcing sum leaves out: natural forcing, interaction and internal variability.
"""),
CODE("""
contrasts = {
    "hist-GHG − historical": sc.clim(y.sel(experiment="hist-GHG"), sc.BASELINE) - sc.clim(y.sel(experiment="historical"), sc.BASELINE),
    "historical − hist-aer": sc.clim(y.sel(experiment="historical"), sc.BASELINE) - sc.clim(y.sel(experiment="hist-aer"), sc.BASELINE),
}
within = {e: sc.clim(y.sel(experiment=e), sc.BASELINE) - sc.clim(y.sel(experiment=e), sc.EARLY) for e in sc.HISTORICAL}
contrasts |= {f"Δ {e}": v for e, v in within.items()}
contrasts["Δ historical − (Δ hist-GHG + Δ hist-aer)"] = within["historical"] - within["hist-GHG"] - within["hist-aer"]
fig, axes = sc.map_axes(2, 3, 3.5, 2.75)
limit = float(np.nanquantile(np.abs(xr.concat([v.mean("model") for k, v in contrasts.items() if k != "historical − hist-aer"], "k").values), 0.98))
for ax, (name, delta) in zip(axes.ravel(), contrasts.items()):
    scale = sc.diverging(delta.mean("model")) if name == "historical − hist-aer" else {"cmap": DIVERGING_CMAP, "vmin": -limit, "vmax": limit}
    mesh = sc.draw_map(ax, delta.mean("model").values, title=name, hatch=~sc.agreement(delta).values, **scale)
    fig.colorbar(mesh, ax=ax, shrink=0.65, label="t/ha")
save(fig, "forcing_contrasts", FIGURES)
plt.show()
table = sc.regional_table(contrasts, 2)
table.to_csv(TABLES / "forcing_contrasts_regions.csv")
display(table.T)
"""),
MD("## 7. Change per country\n\nCyBench countries with at least 2,000 km² of cropland; bars show the multi-model mean and dots each model."),
CODE("""
fig, axes = plt.subplots(1, 2, figsize=(12.5, 6.0), sharey=True, layout="constrained")
store = {}
for ax, e in zip(axes, sc.FUTURE):
    by_country = sc.regional_relative_change(y, e, sc.FAR, aggregate=sc.country_series)
    order = by_country.mean("model").sortby(by_country.mean("model"))
    store[e] = order.to_series()
    ax.barh(order.country.values, order.values, color=EXP_COLOURS[e], alpha=0.65)
    for m in sc.MODELS:
        ax.plot(by_country.sel(model=m, country=order.country).values, order.country.values, "o", ms=3, color=MODEL_COLOURS[m], label=m)
    ax.axvline(0, color="k", lw=0.6)
    ax.set(title=f"{e}, {SPAN(sc.FAR)}", xlabel="Yield change vs 1985–2014 (%)")
axes[0].legend(frameon=False)
save(fig, "country_change", FIGURES)
plt.show()
pd.DataFrame(store).to_csv(TABLES / "country_change_far.csv")
"""),
MD("## 8. How robust is the spatial signal?\n\nThe share of cropland where the change is robust: at least 3 of 4 models agree on the sign, and the multi-model mean exceeds both 5 % and the between-model standard deviation."),
CODE("""
rows = []
for key, delta in pct.items():
    mean, spread = delta.mean("model"), delta.std("model")
    robust = sc.agreement(delta) & (np.abs(mean) > 5) & (np.abs(mean) > spread)
    w = cells.weight / cells.weight.sum()
    rows.append({"scenario": key[0], "period": SPAN(key[1]),
                 "cropland gaining, robust (%)": 100 * float((w * (robust & (mean > 0))).sum()),
                 "cropland losing, robust (%)": 100 * float((w * (robust & (mean < 0))).sum()),
                 "cropland not robust (%)": 100 * float((w * ~robust).sum()),
                 "median between-model SD (%)": float(spread.median())})
robustness = pd.DataFrame(rows)
robustness.to_csv(TABLES / "robustness.csv", index=False)
display(robustness.round(1))
"""),
MD("""
## 9. Calibration regions show in the change

The maps carry straight-edged bands in the east, in the same place in every model, so they do not come from a GCM's grid. Their edges are the boundaries of the calibration regions (28 clusters): each region has its own crop parameters, so neighbouring cells with almost the same weather can respond differently.

This section measures that effect, and how much cropland relies on parameters fitted to no local yield record. A CyBench cell belongs to a region fitted against its own unit's reported yields. A cell without a unit was only assigned to a region, so its parameters are extrapolated.
"""),
CODE("""
frame = cells.frame
far = sc.absolute_change(y, "ssp370", sc.FAR).mean("model").to_series()
base = sc.clim(y.sel(experiment="historical"), sc.BASELINE).mean("model").to_series()
south = frame.index + sc.N_LON
has = south.isin(frame.index)
north_ids, south_ids = frame.index[has], south[has]
across = frame.loc[north_ids, "region_id"].to_numpy() != frame.loc[south_ids, "region_id"].to_numpy()
def jump(values, mask):
    return float(np.nanmedian(np.abs(values.loc[north_ids].to_numpy() - values.loc[south_ids].to_numpy())[mask]))
print(f"North–south neighbours: {int(across.sum()):,} pairs across a region boundary, {int((~across).sum()):,} within")
display(pd.DataFrame({"same region": {"baseline yield (t/ha)": jump(base, ~across), "ssp370 far change (t/ha)": jump(far, ~across)},
                      "across a boundary": {"baseline yield (t/ha)": jump(base, across), "ssp370 far change (t/ha)": jump(far, across)}}).round(3))
coverage = (frame.assign(observed=frame["adm_id"].notna()).groupby("ar6")[["cropland_km2"]].sum().join(
    frame[frame["adm_id"].notna()].groupby("ar6")["cropland_km2"].sum().rename("with a CyBench unit")))
coverage["share (%)"] = 100 * coverage["with a CyBench unit"].fillna(0) / coverage["cropland_km2"]
print(f"Cropland with a CyBench unit: {100 * frame.loc[frame.adm_id.notna(), 'cropland_km2'].sum() / frame['cropland_km2'].sum():.1f} %")
coverage.to_csv(TABLES / "calibration_coverage.csv")
display(coverage.round(0))
calibrated = {int(p.stem.rsplit("_", 1)[1]) for p in (sc.CALIBRATION / "yield" / "crops").glob("crop_region_*.yaml")}
fallback = frame[~frame["region_id"].isin(calibrated)]
print(f"Regions with no calibrated crop file (run on the fallback crop): {sorted(fallback['region_id'].unique().tolist())}; {fallback['cropland_km2'].sum():,.0f} km² of cropland ({100 * fallback['cropland_km2'].sum() / frame['cropland_km2'].sum():.1f} %)")
codes = frame["region_id"].to_numpy()
fig, axes = sc.map_axes(1, 2, 4.4, 3.6)
fig.colorbar(sc.draw_map(axes[0, 0], far.to_numpy(), title="ssp370 2071–2100 change (t/ha)", **sc.diverging(far.to_numpy())), ax=axes[0, 0], shrink=0.7)
sc.draw_map(axes[0, 1], codes % 10, cmap="tab10", vmin=0, vmax=9, title="Calibration regions; hatched cells have no CyBench unit", hatch=frame["adm_id"].isna().to_numpy())
save(fig, "calibration_regions", FIGURES)
plt.show()
"""),
)

# ---------------------------------------------------------------------- time series
notebook("02_timeseries",
MD(f"""
# CMIP6 torchcrop runs — time series

Cropland-weighted yield over 1851–2100, regional and national. Smooth lines are 21-year centred means.
ACCESS-ESM1-5's forcing starts in 1950, so the 1851–2100 figure uses the three full-record models for
its mean and range, with ACCESS-ESM1-5 dashed. Mixing them would put a step in the mean at 1951. The
later figures start in 1951 and use all four models.

{CAVEATS}
"""),
MD("## 0. Setup"),
CODE(setup("timeseries", """
series = sc.regional_mean(y)
anomaly = 100.0 * (series / sc.clim(series.sel(experiment="historical"), sc.BASELINE) - 1.0)
smooth = lambda s: s.rolling(year=21, center=True, min_periods=15).mean()
LONG_RECORD = [m for m in sc.MODELS if m != "ACCESS-ESM1-5"]


def ensemble(ax, data, experiments, smoother=smooth):
    for e in experiments:
        block = smoother(data.sel(experiment=e))
        ax.fill_between(block.year, block.min("model"), block.max("model"), color=EXP_COLOURS[e], alpha=0.18, lw=0)
        ax.plot(block.year, block.mean("model"), color=EXP_COLOURS[e], lw=1.5, label=e)
    ax.axvspan(*sc.BASELINE, color="k", alpha=0.05, lw=0)
""")),
MD("## 1. Europe, 1851–2100\n\nThe grey band is the 1985–2014 baseline. Shading is the range of the three full-record models, and dashed lines are ACCESS-ESM1-5."),
CODE("""
co2 = sc.co2_series()
fig, axes = plt.subplots(3, 1, figsize=(11.0, 8.6), sharex=True, layout="constrained", gridspec_kw={"height_ratios": [3, 3, 1.3]})
ensemble(axes[0], series.sel(region="Europe", model=LONG_RECORD), sc.EXPERIMENTS)
axes[0].set_ylabel("Yield (t/ha)")
axes[0].legend(frameon=False, ncol=5)
ensemble(axes[1], anomaly.sel(region="Europe", model=LONG_RECORD), sc.EXPERIMENTS)
for ax, data in zip(axes, (series, anomaly)):
    for e in sc.EXPERIMENTS:
        s = smooth(data.sel(region="Europe", model="ACCESS-ESM1-5", experiment=e))
        ax.plot(s.year, s, color=EXP_COLOURS[e], lw=0.9, ls="--")
axes[1].axhline(0, color="k", lw=0.6)
axes[1].set_ylabel("Yield vs 1985–2014 (%)")
for e in sc.EXPERIMENTS:
    axes[2].plot(co2.index, co2[e], color=EXP_COLOURS[e], lw=1.2)
axes[2].set_ylabel("CO₂ (ppm)")
save(fig, "europe_full", FIGURES)
plt.show()
"""),
MD("Each model on its own, with the unsmoothed seasons behind the 21-year mean. The seasons show how large a single year is against the signal:"),
CODE("""
fig, axes = plt.subplots(2, 2, figsize=(12.5, 7.0), sharex=True, sharey=True, layout="constrained")
for ax, m in zip(axes.ravel(), sc.MODELS):
    for e in sc.EXPERIMENTS:
        s = series.sel(region="Europe", model=m, experiment=e)
        ax.plot(s.year, s, color=EXP_COLOURS[e], lw=0.35, alpha=0.5)
        ax.plot(s.year, smooth(s), color=EXP_COLOURS[e], lw=1.5, label=e)
    ax.set_title(m)
axes[0, 0].legend(frameon=False, ncol=2)
fig.supylabel("Yield (t/ha)")
save(fig, "europe_models", FIGURES)
plt.show()
"""),
MD("## 2. Against the MSWX run\n\nHistorical and ssp370 spliced across the missing 2015, around the MSWX torchcrop run. The model lines should straddle MSWX in level; year-to-year matches are not expected."),
CODE("""
mswx = sc.regional_mean(sc.mswx_cube()).sel(region="Europe")
fig, ax = plt.subplots(figsize=(10.0, 3.8), layout="constrained")
for m in sc.MODELS:
    for e in ("historical", "ssp370"):
        s = series.sel(region="Europe", model=m, experiment=e).sel(year=slice(1995, 2030))
        ax.plot(s.year, s, color=MODEL_COLOURS[m], lw=1.0, alpha=0.8, label=m if e == "historical" else None)
ax.plot(mswx.year, mswx, "k", lw=2.0, label="MSWX run")
ax.set(ylabel="Yield (t/ha)")
ax.legend(frameon=False, ncol=5)
save(fig, "vs_mswx", FIGURES)
plt.show()
window = lambda s, a, b: float(s.sel(year=slice(a, b)).mean())
seam = pd.DataFrame({m: {"historical 2005–2014": window(series.sel(region="Europe", model=m, experiment="historical"), 2005, 2014),
                         "ssp126 2016–2025": window(series.sel(region="Europe", model=m, experiment="ssp126"), 2016, 2025),
                         "ssp370 2016–2025": window(series.sel(region="Europe", model=m, experiment="ssp370"), 2016, 2025)} for m in sc.MODELS}).T
seam["MSWX 2005–2014"] = window(mswx, 2005, 2014)
seam["MSWX 2016–2024"] = window(mswx, 2016, 2024)
seam.to_csv(TABLES / "seam_levels.csv")
print("Level either side of the historical→SSP join (t/ha). A large step would mean the two are not one series:")
display(seam.round(2))
"""),
MD("## 3. Regions"),
CODE("""
regions = [r for r in series.region.values if r != "Europe"]
fig, axes = plt.subplots(2, 3, figsize=(13.0, 6.6), sharex=True, layout="constrained")
for ax, r in zip(axes.ravel(), regions):
    ensemble(ax, anomaly.sel(region=r).sel(year=slice(1951, None)), sc.EXPERIMENTS)
    ax.axhline(0, color="k", lw=0.5)
    ax.set_title(sc.REGION_NAMES[r])
axes[0, 0].legend(frameon=False, ncol=2)
axes[1, 2].set_visible(len(regions) > 5)
fig.supylabel("Yield vs 1985–2014 (%)")
save(fig, "regions", FIGURES)
plt.show()
"""),
MD("## 4. Forcing differences through time\n\nThe paired difference per model and year, which removes most of the shared crop response. The two historical climate runs still have different weather, so only the smoothed line is meaningful."),
CODE("""
pairs = {"hist-GHG − historical (no aerosol or natural forcing; same CO₂)": ("hist-GHG", "historical"),
         "historical − hist-aer (includes the CO₂ effect)": ("historical", "hist-aer"),
         "ssp370 − ssp126": ("ssp370", "ssp126")}
fig, axes = plt.subplots(1, 3, figsize=(13.5, 3.8), layout="constrained")
for ax, (name, (a, b)) in zip(axes, pairs.items()):
    d = smooth(series.sel(region="Europe", experiment=a) - series.sel(region="Europe", experiment=b)).dropna("year", how="all")
    for m in sc.MODELS:
        ax.plot(d.year, d.sel(model=m), color=MODEL_COLOURS[m], lw=1.0, label=m)
    ax.plot(d.year, d.mean("model"), "k", lw=2.0, label="Mean")
    ax.axhline(0, color="k", lw=0.5)
    ax.set_title(name, fontsize=9)
    ax.set_ylabel("t/ha")
axes[0].legend(frameon=False, fontsize=7)
save(fig, "forcing_differences", FIGURES)
plt.show()
"""),
MD("## 5. Interannual variability\n\nRolling 21-year standard deviation of each season's departure from the 21-year mean, as a share of that mean. A domain mean hides variability that offsets between regions, so the same is shown for Western & Central Europe and the Mediterranean."),
CODE("""
def rolling_cv(s):
    centre = smooth(s)
    return 100.0 * (s - centre).rolling(year=21, center=True, min_periods=15).std() / centre

fig, axes = plt.subplots(1, 3, figsize=(13.5, 3.8), sharey=True, layout="constrained")
for ax, r in zip(axes, ("Europe", "WCE", "MED")):
    ensemble(ax, rolling_cv(series.sel(region=r)).sel(year=slice(1951, None)), sc.EXPERIMENTS, smoother=lambda s: s)
    ax.set_title(sc.REGION_NAMES[r])
axes[0].set_ylabel("Interannual CV (%)")
axes[0].legend(frameon=False, fontsize=7)
save(fig, "variability", FIGURES)
plt.show()
"""),
MD("## 6. The crop processes behind the yield\n\nDomain means of the outputs torchcrop writes. Notebook 05 uses these for the yield decomposition."),
CODE("""
PROCESS = {"biomass_g_m2": "Biomass (g/m²)", "max_lai": "Max LAI", "days_to_anthesis": "Days to anthesis", "days_to_maturity": "Days to maturity",
           "tranrf_mean": "Transpiration stress factor (1 = none)", "nni_mean": "Nitrogen nutrition index", "heat_stress_factor": "Heat-stress factor (0 = none)"}
fig, axes = plt.subplots(2, 4, figsize=(14.0, 6.0), sharex=True, layout="constrained")
for ax, (v, label) in zip(axes.ravel(), PROCESS.items()):
    ensemble(ax, sc.regional_mean(sc.cube(v), regions=("Europe",)).sel(region="Europe").sel(year=slice(1951, None)), sc.EXPERIMENTS)
    ax.set_title(label, fontsize=9)
harvest_index = sc.regional_mean(y, regions=("Europe",)) * 100.0 / sc.regional_mean(sc.cube("biomass_g_m2"), regions=("Europe",))
ensemble(axes[1, 3], harvest_index.sel(region="Europe").sel(year=slice(1951, None)), sc.EXPERIMENTS)
axes[1, 3].set_title("Ratio of mean yield to mean biomass", fontsize=9)
axes[0, 0].legend(frameon=False, fontsize=7)
save(fig, "processes", FIGURES)
plt.show()
"""),
MD("## 7. Yield against CO₂\n\nEurope, per season. Separation between `ssp126` and `ssp370` at the same CO₂ shows that yield does not depend on CO₂ alone."),
CODE("""
fig, axes = plt.subplots(1, 4, figsize=(14.0, 3.6), sharex=True, sharey=True, layout="constrained")
for ax, m in zip(axes, sc.MODELS):
    for e in sc.EXPERIMENTS:
        s = series.sel(region="Europe", model=m, experiment=e).to_series().dropna()
        ax.scatter(co2[e].reindex(s.index), s, s=3, color=EXP_COLOURS[e], label=e, alpha=0.6)
    ax.set(title=m, xlabel="CO₂ (ppm)")
axes[0].set_ylabel("Yield (t/ha)")
axes[0].legend(frameon=False, fontsize=7, markerscale=3)
save(fig, "yield_vs_co2", FIGURES)
plt.show()
"""),
MD("## 8. Countries\n\nCyBench countries with at least 2,000 km² of cropland, 1951–2100."),
CODE("""
countries = sc.country_series(y)
base = sc.clim(countries.sel(experiment="historical"), sc.BASELINE)
by_country = (100.0 * (countries / base - 1.0)).sel(year=slice(1951, None))
codes = list(by_country.country.values)
n_rows = int(np.ceil(len(codes) / 5))
fig, axes = plt.subplots(n_rows, 5, figsize=(14.0, 2.2 * n_rows), sharex=True, sharey=True, layout="constrained")
for ax, code in zip(axes.ravel(), codes):
    ensemble(ax, by_country.sel(country=code), sc.EXPERIMENTS)
    ax.axhline(0, color="k", lw=0.4)
    ax.set_title(code, fontsize=9)
for ax in axes.ravel()[len(codes):]:
    ax.set_visible(False)
axes[0, 0].legend(frameon=False, fontsize=6)
fig.supylabel("Yield vs 1985–2014 (%)")
save(fig, "countries", FIGURES)
plt.show()
"""),
)

# --------------------------------------------------------------------------- trends
notebook("03_trends",
MD(f"""
# CMIP6 torchcrop runs — trends

Theil–Sen slopes with Mann–Kendall tests, per model, region and cell. Cell-level significance uses
Benjamini–Hochberg false-discovery control at α = 0.10, because a field of per-cell tests at p < 0.05
over-counts (Wilks 2016). Every trend here is climate and CO₂ alone, since the crop system is fixed;
a farm-scale trend also carries breeding and management.

{CAVEATS}
"""),
MD("## 0. Setup"),
CODE(setup("trends", """
WINDOWS = {"1951–2014": ((1951, 2014), sc.HISTORICAL), "1985–2014": ((1985, 2014), sc.HISTORICAL),
           "2016–2050": ((2016, 2050), sc.FUTURE), "2051–2100": ((2051, 2100), sc.FUTURE)}


def cell_trends(data):
    slopes, pvalues = {}, {}
    for name, (span, experiments) in WINDOWS.items():
        for e in experiments:
            block = sc.period(data.sel(experiment=e), span)
            years = block.year.values
            per_model = [sc.sen_mk(block.sel(model=m).values, years) for m in sc.MODELS]
            slopes[name, e] = xr.DataArray(np.stack([p[0] for p in per_model]), dims=("model", "cell"), coords={"model": list(sc.MODELS), "cell": data.cell})
            pvalues[name, e] = np.stack([p[1] for p in per_model])
    return slopes, pvalues
""")),
MD("## 1. Does a trend test apply?\n\nMann–Kendall assumes independent seasons. Lag-1 autocorrelation of the detrended seasons, per cell, shows whether that holds. Each season's crop starts from a re-initialised soil and has independent weather, so it should be near zero."),
CODE("""
block = sc.period(y.sel(experiment="historical"), (1951, 2014))
fit = block.polyfit("year", deg=1)["polyfit_coefficients"]
resid = block - xr.polyval(block["year"], fit)
lag1 = xr.corr(resid.isel(year=slice(1, None)).assign_coords(year=np.arange(63)), resid.isel(year=slice(None, -1)).assign_coords(year=np.arange(63)), dim="year")
print(lag1.quantile([0.05, 0.5, 0.95], dim="cell").to_pandas().round(3))
"""),
MD("## 2. Regional trends\n\nTheil–Sen per decade per model, on the region's mean series, with the Mann–Kendall result (`*` p < 0.05)."),
CODE("""
series = sc.regional_mean(y)
base = sc.clim(series.sel(experiment="historical"), sc.BASELINE)
rows = []
for name, (span, experiments) in WINDOWS.items():
    for e in experiments:
        for m in sc.MODELS:
            for r in series.region.values:
                s = sc.period(series.sel(model=m, experiment=e, region=r), span)
                slope, p = sc.sen_mk(s.values[:, None], s.year.values)
                rows.append({"window": name, "experiment": e, "model": m, "region": r, "slope": float(slope[0]), "p": float(p[0]),
                             "slope_pct": 100.0 * float(slope[0]) / float(base.sel(model=m, region=r))})
regional = pd.DataFrame(rows)
regional.to_csv(TABLES / "regional_trends.csv", index=False)
europe = regional[regional["region"] == "Europe"].assign(cell=lambda f: f["slope"].map("{:+.3f}".format) + np.where(f["p"] < 0.05, "*", ""))
print("Europe, t/ha per decade:")
display(europe.pivot(index=["window", "experiment"], columns="model", values="cell"))
"""),
MD("Trend per decade as % of each model's baseline, per region. Each dot is a model, and the bar is their mean:"),
CODE("""
order = [(w, e) for w, (_, es) in WINDOWS.items() for e in es]
regions = list(sc.REGION_NAMES)
fig, axes = plt.subplots(1, len(regions), figsize=(15.0, 5.0), sharey=True, layout="constrained")
for ax, r in zip(axes, regions):
    block = regional[regional["region"] == r]
    for i, (w, e) in enumerate(order):
        vals = block[(block["window"] == w) & (block["experiment"] == e)]["slope_pct"]
        ax.barh(i, vals.mean(), color=EXP_COLOURS[e], alpha=0.65)
        ax.plot(vals, [i] * len(vals), "o", ms=2.5, color="k")
    ax.axvline(0, color="k", lw=0.5)
    ax.set_title(sc.REGION_NAMES[r], fontsize=8)
    ax.set_xlabel("% per decade")
axes[0].set_yticks(range(len(order)), [f"{w}  {e}" for w, e in order], fontsize=8)
axes[0].invert_yaxis()
save(fig, "regional_trends", FIGURES)
plt.show()
"""),
MD("## 3. Trend maps\n\nThe multi-model mean of per-cell slopes. Hatching marks cells where fewer than 3 models share the sign."),
CODE("""
slopes, pvalues = cell_trends(y)
keys = list(slopes)
limit = float(np.nanquantile(np.abs(np.stack([slopes[k].mean("model").values for k in keys])), 0.98))
fig, axes = sc.map_axes(2, 5, 2.9, 2.3)
for ax, key in zip(axes.ravel(), keys):
    mesh = sc.draw_map(ax, slopes[key].mean("model").values, title=f"{key[1]}, {key[0]}", hatch=~sc.agreement(slopes[key]).values, cmap=DIVERGING_CMAP, vmin=-limit, vmax=limit)
fig.colorbar(mesh, ax=axes, shrink=0.6, label="Yield trend (t/ha per decade)")
save(fig, "trend_maps", FIGURES)
plt.show()
"""),
MD("## 4. How much cropland trends significantly?\n\nCropland share with a significant rise or fall after false-discovery control, per run, then averaged over models. The ranges give the lowest and highest model."),
CODE("""
w = cells.weight / cells.weight.sum()
rows = []
for key in keys:
    up, down = [], []
    for k, m in enumerate(sc.MODELS):
        mask = sc.fdr(pvalues[key][k])
        slope = slopes[key].sel(model=m).values
        up.append(100 * w[mask & (slope > 0)].sum())
        down.append(100 * w[mask & (slope < 0)].sum())
    rows.append({"window": key[0], "experiment": key[1], "rising (%)": f"{np.mean(up):.0f} [{min(up):.0f}, {max(up):.0f}]", "falling (%)": f"{np.mean(down):.0f} [{min(down):.0f}, {max(down):.0f}]"})
significant = pd.DataFrame(rows)
significant.to_csv(TABLES / "significant_area.csv", index=False)
display(significant)
"""),
MD("## 5. Do the forcing trends add up?\n\n1951–2014 in Europe. If forcings were additive, the `historical` trend would equal `hist-GHG` + `hist-aer`. The `hist-aer` trend has no CO₂ rise behind it, so it is the only climate-only trend in the table."),
CODE("""
pivot = regional[(regional["window"] == "1951–2014")].pivot_table(index=["region", "model"], columns="experiment", values="slope")[list(sc.HISTORICAL)]
pivot["hist-GHG + hist-aer"] = pivot["hist-GHG"] + pivot["hist-aer"]
pivot["residual"] = pivot["historical"] - pivot["hist-GHG + hist-aer"]
pivot.to_csv(TABLES / "forcing_trend_sum.csv")
display(pivot.loc["Europe"].round(3))
"""),
MD("## 6. When does a yield peak?\n\nThe year of each model's highest 21-year mean, per region and experiment. A year at the end of the run means yield is still rising in 2100, not that it peaks there."),
CODE("""
smooth = series.rolling(year=21, center=True, min_periods=21).mean()
peak = pd.DataFrame({(e, r): {m: int(smooth.sel(experiment=e, region=r, model=m).dropna("year").idxmax()) for m in sc.MODELS}
                     for e in sc.FUTURE for r in series.region.values}).T
peak.index.names = ["experiment", "region"]
peak.to_csv(TABLES / "peak_year.csv")
display(peak)
"""),
MD("## 7. Time of emergence\n\nThe first year after which a cell's 21-year mean stays more than one interannual standard deviation (of 1985–2014) from its baseline, until the run ends. Year per model, then the multi-model median; cells that never emerge are blank. The threshold is a convention, so read this as a ranking of where the signal shows first, not as a date."),
CODE(r"""
def emergence(data, experiment, k=1.0):
    base = sc.period(data.sel(experiment="historical"), sc.BASELINE)
    mean, sd = base.mean("year").values, base.std("year").values
    run = sc.period(data.sel(experiment=experiment), (2016, 2100))
    rolled = run.rolling(year=21, center=True, min_periods=21).mean().dropna("year", how="all")
    years = rolled.year.values
    departed = np.abs(rolled.transpose("year", "model", "cell").values - mean) > k * sd
    # The last year not departed, plus one, is when the departure becomes permanent.
    last_inside = departed.shape[0] - 1 - np.argmax(~departed[::-1], axis=0)
    first = np.where(departed.all(0), 0, last_inside + 1)
    out = np.where(first < len(years), years[np.minimum(first, len(years) - 1)], np.nan)
    return xr.DataArray(out, dims=("model", "cell"), coords={"model": list(sc.MODELS), "cell": data.cell})

toe = {e: emergence(y, e) for e in sc.FUTURE}
fig, axes = sc.map_axes(1, 2, 4.0, 4.0)
for ax, (e, field) in zip(axes[0], toe.items()):
    mesh = sc.draw_map(ax, field.median("model").values, cmap=SEQUENTIAL_CMAP, vmin=2026, vmax=2090, title=e)
fig.colorbar(mesh, ax=axes, shrink=0.7, label="Median year of emergence")
save(fig, "time_of_emergence", FIGURES)
plt.show()
w = cells.weight / cells.weight.sum()
print("Cropland emerged by year (%), multi-model median of models' share:")
display(pd.DataFrame({e: {year: float(np.median([100 * w[(f.sel(model=m).values <= year)].sum() for m in sc.MODELS])) for year in (2040, 2060, 2080, 2090)} for e, f in toe.items()}).round(0))
"""),
MD("## 8. Trends in the crop processes\n\nEurope, multi-model mean Theil–Sen per decade with the model range. Compared with the yield trend, this shows which process moves with it."),
CODE("""
PROCESS = ["yield_t_ha", "biomass_g_m2", "max_lai", "days_to_anthesis", "days_to_maturity", "tranrf_mean", "nni_mean", "heat_stress_factor"]
rows = []
for v in PROCESS:
    reg = sc.regional_mean(sc.cube(v), regions=("Europe",)).sel(region="Europe")
    for name, (span, experiments) in WINDOWS.items():
        for e in experiments:
            s = sc.period(reg.sel(experiment=e), span)
            vals = [float(sc.sen_mk(s.sel(model=m).values[:, None], s.year.values)[0][0]) for m in sc.MODELS]
            rows.append({"variable": v, "window": name, "experiment": e, "cell": f"{np.mean(vals):+.3g} [{min(vals):+.3g}, {max(vals):+.3g}]"})
process = pd.DataFrame(rows).pivot(index=["window", "experiment"], columns="variable", values="cell")[PROCESS]
process.to_csv(TABLES / "process_trends.csv")
display(process)
"""),
MD("## 9. Recent trend against the MSWX run\n\n2000–2024 in Europe: `historical` spliced to each SSP, against the MSWX torchcrop run. 25 seasons is short, and a free-running model's trend differs by internal variability alone, so the MSWX value only needs to fall inside a plausible range."),
CODE("""
mswx = sc.regional_mean(sc.mswx_cube()).sel(region="Europe")
slope, p = sc.sen_mk(mswx.values[:, None].astype(np.float32), mswx.year.values)
rows = [{"run": "MSWX", "slope": float(slope[0]), "p": float(p[0])}]
reg = series.sel(region="Europe")
for m in sc.MODELS:
    for e in sc.FUTURE:
        a, b = reg.sel(model=m, experiment="historical").sel(year=slice(2000, 2014)), reg.sel(model=m, experiment=e).sel(year=slice(2016, 2024))
        years = np.r_[a.year.values, b.year.values]
        s, q = sc.sen_mk(np.r_[a.values, b.values][:, None].astype(np.float32), years)
        rows.append({"run": f"{m} + {e}", "slope": float(s[0]), "p": float(q[0])})
recent = pd.DataFrame(rows)
recent.to_csv(TABLES / "recent_trend.csv", index=False)
display(recent.round(3))
"""),
)

# ------------------------------------------------------------------------- extremes
notebook("04_extreme_yield_probability",
MD(f"""
# CMIP6 torchcrop runs — changes in the probability of extreme yields

How much more or less often does a season cross a level that was rare in 1985–2014?

* **Events** are set per model and per cell on `historical` 1985–2014, so model bias does not read as
  risk: below the 10th and 5th percentiles, below half of the baseline mean, above the 90th percentile.
  With 30 seasons a 5th percentile sits between the two lowest values, so treat it as indicative.
* **Reference probability** is leave-one-year-out. An in-sample probability is exact by construction,
  while a future one has threshold sampling error, so comparing the two would overstate change.
* **Uncertainty**: 95 % intervals resample whole model-seasons, which keeps spatial correlation. They do
  not cover the small model sample. The model range is shown as well.
* **The events are absolute.** With CO₂ fertilisation in the runs, a rising mean lowers the chance of
  crossing a fixed level even if the bad years are as bad as before. §4 splits the change into mean and
  variability for this reason.

{CAVEATS}
"""),
MD("## 0. Setup"),
CODE(setup("extremes", """
import risk as rk

limits = {e.key: rk.threshold(y, e) for e in rk.EVENTS}
reference = {e.key: rk.loo_probability(y, e) for e in rk.EVENTS}
SCENARIOS = [("historical", sc.EARLY), ("hist-GHG", sc.BASELINE), ("hist-aer", sc.BASELINE), ("ssp126", sc.NEAR), ("ssp126", sc.FAR), ("ssp370", sc.NEAR), ("ssp370", sc.FAR)]
FUTURE = [s for s in SCENARIOS if s[0] in sc.FUTURE]
TAG = lambda e, s: f"{e} {SPAN(s)}"
""")),
MD("## 1. Probability in each region\n\nFor each event, scenario and region: the cropland-area probability, the risk ratio against the leave-one-year-out reference, the 95 % interval and the model range."),
CODE("""
def summarise(event, data=y, limit=None, base=None):
    limit = limits[event.key] if limit is None else limit
    base = reference[event.key] if base is None else base
    ref = sc.regional_mean(base)
    rows = []
    for e, span in SCENARIOS:
        yearly = sc.regional_mean(rk.hits(sc.period(data.sel(experiment=e), span), limit, event))
        for r in yearly.region.values:
            block = yearly.sel(region=r)
            p_models = block.mean("year")
            ratio = p_models / ref.sel(region=r)
            low, high = rk.bootstrap_ci(block.values.ravel())
            p_ref = float(ref.sel(region=r).mean())
            rows.append({"event": event.key, "scenario": TAG(e, span), "region": r, "reference (%)": 100 * p_ref, "probability (%)": 100 * float(p_models.mean()),
                         "CI low (%)": 100 * low, "CI high (%)": 100 * high, "risk ratio": float(p_models.mean()) / p_ref,
                         "RR model min": float(ratio.min()), "RR model max": float(ratio.max())})
    return pd.DataFrame(rows)

summary = pd.concat([summarise(e) for e in rk.EVENTS], ignore_index=True)
summary.to_csv(TABLES / "event_probability_regions.csv", index=False)
for event in rk.EVENTS:
    print(event.label)
    block = summary[(summary["event"] == event.key) & (summary["region"].isin(["Europe", "WCE", "MED", "EEU", "NEU"]))]
    display(block.pivot(index="scenario", columns="region", values="risk ratio").reindex([TAG(*s) for s in SCENARIOS]).round(2))
display(summary[(summary["region"] == "Europe") & (summary["event"] == "p10")].drop(columns=["event", "region"]).round(2))
"""),
MD("Risk ratios for the low-yield event, with intervals and each model, by region:"),
CODE("""
regions = ["Europe", "WCE", "MED", "EEU", "NEU"]
tags = [TAG(*s) for s in SCENARIOS if s[0] != "hist-aer"]
fig, axes = plt.subplots(1, 5, figsize=(15.0, 3.6), sharey=True, layout="constrained")
block = summary[summary["event"] == "p10"]
for ax, r in zip(axes, regions):
    sub = block[block["region"] == r].set_index("scenario").loc[tags]
    ref = sub["reference (%)"]
    ax.errorbar(sub["risk ratio"], range(len(tags)), xerr=[sub["risk ratio"] - sub["CI low (%)"] / ref, sub["CI high (%)"] / ref - sub["risk ratio"]], fmt="o", color=PALETTE["primary"], ms=4, lw=1.1)
    ax.hlines(range(len(tags)), sub["RR model min"], sub["RR model max"], color=PALETTE["series_2"], lw=3, alpha=0.35)
    ax.axvline(1, color="k", lw=0.5)
    ax.set_title(sc.REGION_NAMES[r], fontsize=9)
    ax.set_xscale("log")
    ax.set_xlabel("Risk ratio, 1-in-10 low yield")
axes[0].set_yticks(range(len(tags)), tags, fontsize=8)
axes[0].invert_yaxis()
save(fig, "risk_ratio_p10", FIGURES)
plt.show()
"""),
MD("## 2. Probability through time\n\nThe area-mean probability of the 1-in-10 low-yield event, in 21-year running means. The dotted line is the leave-one-year-out reference. Before 1951 three models are available."),
CODE("""
yearly = {key: sc.regional_mean(rk.hits(y, limits[key], rk.EVENT[key]), regions=("Europe", "WCE", "MED")) for key in ("p10", "half", "p90")}
fig, axes = plt.subplots(3, 3, figsize=(14.0, 8.0), sharex=True, layout="constrained")
for r_i, r in enumerate(("Europe", "WCE", "MED")):
    for c, (key, series) in enumerate(yearly.items()):
        ax = axes[r_i, c]
        for e in sc.EXPERIMENTS:
            s = series.sel(region=r, experiment=e).rolling(year=21, center=True, min_periods=15).mean() * 100
            ax.fill_between(s.year, s.min("model"), s.max("model"), color=EXP_COLOURS[e], alpha=0.18, lw=0)
            ax.plot(s.year, s.mean("model"), color=EXP_COLOURS[e], lw=1.3, label=e)
        ax.axhline(100 * float(sc.regional_mean(reference[key]).sel(region=r).mean()), color="k", ls=":", lw=0.8)
        ax.set_title(f"{sc.REGION_NAMES[r]}: {key}", fontsize=9)
        ax.set_ylabel("Probability (%)")
axes[0, 0].legend(frameon=False, fontsize=7)
save(fig, "probability_through_time", FIGURES)
plt.show()
"""),
MD("## 3. Where the probability changes\n\nChange in probability (percentage points) against the leave-one-year-out reference. Each model's probability rests on 30 seasons in a cell, so the map shows the mean of four models (120 seasons). Hatching marks cells where fewer than 3 models agree on the sign."),
CODE("""
for key in ("p10", "half", "p90"):
    event = rk.EVENT[key]
    delta = {(e, s): 100.0 * (rk.probability(y.sel(experiment=e), limits[key], event, s) - reference[key]) for e, s in FUTURE}
    limit = float(np.nanquantile(np.abs(np.stack([d.mean("model").values for d in delta.values()])), 0.99)) or 1.0
    fig, axes = sc.map_axes(1, 4, 3.0, 3.2)
    for ax, ((e, s), d) in zip(axes[0], delta.items()):
        mesh = sc.draw_map(ax, d.mean("model").values, title=TAG(e, s), hatch=~sc.agreement(d).values, cmap=DIVERGING_CMAP, vmin=-limit, vmax=limit)
    fig.colorbar(mesh, ax=axes, shrink=0.75, label="Δ probability (points)")
    fig.suptitle(event.label, fontsize=10)
    save(fig, f"probability_change_{key}", FIGURES)
    plt.show()
"""),
MD("## 4. Is it the mean or the variability?\n\nFor the far period, baseline seasons are rescaled by the cell's ratio of future to baseline mean. This keeps the coefficient of variation fixed. The probability of the rescaled seasons is the **mean-only** change; the remainder is the change in **variability and shape**."),
CODE("""
event = rk.EVENT["p10"]
frames, fields = [], {}
for e in sc.FUTURE:
    total = rk.probability(y.sel(experiment=e), limits["p10"], event, sc.FAR) - reference["p10"]
    mean_only = rk.scaled_baseline_probability(y, limits["p10"], event, e, sc.FAR) - reference["p10"]
    variability = total - mean_only
    fields[e] = {"total": total, "mean shift": mean_only, "variability": variability}
    reg = {n: sc.regional_mean(v) * 100 for n, v in fields[e].items()}
    for n, v in reg.items():
        frames.append(v.to_series().rename("points").reset_index().assign(part=n, experiment=e))
parts = pd.concat(frames, ignore_index=True)
parts.to_csv(TABLES / "mean_vs_variability.csv", index=False)
display(parts.groupby(["experiment", "region", "part"])["points"].mean().unstack("part")[["total", "mean shift", "variability"]].round(1))
limit = float(np.nanquantile(np.abs(np.stack([v.mean("model").values for f in fields.values() for v in f.values()])), 0.98)) * 100
fig, axes = sc.map_axes(2, 3, 3.5, 2.75)
for r, (e, f) in enumerate(fields.items()):
    for c, (n, v) in enumerate(f.items()):
        mesh = sc.draw_map(axes[r, c], 100 * v.mean("model").values, title=f"{e}: {n}", hatch=~sc.agreement(v).values, cmap=DIVERGING_CMAP, vmin=-limit, vmax=limit)
fig.colorbar(mesh, ax=axes, shrink=0.6, label=f"Δ probability of 1-in-10 low yield, {SPAN(sc.FAR)} (points)")
save(fig, "mean_vs_variability", FIGURES)
plt.show()
"""),
MD("Interannual coefficient of variation (linearly detrended within each window): baseline, then its change."),
CODE("""
cv_base = rk.coefficient_of_variation(y.sel(experiment="historical"), sc.BASELINE)
cv_change = {e: rk.coefficient_of_variation(y.sel(experiment=e), sc.FAR) - cv_base for e in sc.FUTURE}
fig, axes = sc.map_axes(1, 3, 3.5, 3.6)
fig.colorbar(sc.draw_map(axes[0, 0], cv_base.mean("model").values, cmap=SEQUENTIAL_CMAP, vmin=0, vmax=float(cv_base.mean("model").quantile(0.98)), title=f"CV, {SPAN(sc.BASELINE)}"), ax=axes[0, 0], shrink=0.65, label="%")
scale = sc.diverging(xr.concat([v.mean("model") for v in cv_change.values()], "k"))
for ax, (e, d) in zip(axes[0, 1:], cv_change.items()):
    mesh = sc.draw_map(ax, d.mean("model").values, title=f"Δ CV, {e}, {SPAN(sc.FAR)}", hatch=~sc.agreement(d).values, **scale)
fig.colorbar(mesh, ax=axes[0, 1:], shrink=0.65, label="points")
save(fig, "cv_change", FIGURES)
plt.show()
display(sc.regional_table({e: d for e, d in cv_change.items()}, 1).T)
"""),
MD("## 5. Synchronous shortfalls\n\nA 1-in-10 season in one cell can be offset elsewhere; many at once cannot. The share of Europe's cropland below its own 1-in-10 level per season, as the probability that at least x % of cropland is affected in one season. The baseline uses leave-one-year-out thresholds, like the reference in §1."),
CODE("""
area = sc.regional_mean(rk.hits(y, limits["p10"], rk.EVENT["p10"]), regions=("Europe",)).sel(region="Europe") * 100
levels = np.arange(0, 81, 2)
baseline_area = 100 * sc.regional_mean(rk.loo_hits(y, rk.EVENT["p10"]), regions=("Europe",)).sel(region="Europe").values.ravel()
fig, axes = plt.subplots(1, 2, figsize=(12.5, 4.0), sharey=True, layout="constrained")
rows = []
for ax, s in zip(axes, (sc.NEAR, sc.FAR)):
    base = baseline_area
    ax.plot(levels, [100 * (base >= x).mean() for x in levels], "k", lw=1.6, label=f"historical {SPAN(sc.BASELINE)}")
    for e in sc.FUTURE:
        vals = sc.period(area.sel(experiment=e), s).values.ravel()
        ax.plot(levels, [100 * (vals >= x).mean() for x in levels], color=EXP_COLOURS[e], lw=1.6, label=f"{e} {SPAN(s)}")
        rows.append({"scenario": TAG(e, s), **{f"P(≥{x} %) (%)": 100 * (vals >= x).mean() for x in (20, 30, 50)}})
    ax.set(yscale="log", xlabel="Cropland below its 1-in-10 level (%)", ylabel="Probability per season (%)", ylim=(0.4, 110))
    ax.legend(frameon=False)
rows.insert(0, {"scenario": f"historical {SPAN(sc.BASELINE)}", **{f"P(≥{x} %) (%)": 100 * (base >= x).mean() for x in (20, 30, 50)}})
synchronous = pd.DataFrame(rows)
synchronous.to_csv(TABLES / "synchronous_shortfall.csv", index=False)
save(fig, "synchronous", FIGURES)
plt.show()
display(synchronous.round(1))
"""),
MD("## 6. Continental return levels\n\nEurope's mean yield anomaly against each model's baseline mean. Curves pool all four models' seasons in each period (120 seasons, so about the 1-in-40 level is the limit of what the data support). Plotting position is Weibull, `rank / (n + 1)`."),
CODE("""
series = sc.regional_mean(y, regions=("Europe",)).sel(region="Europe")
anom = 100.0 * (series / sc.clim(series.sel(experiment="historical"), sc.BASELINE) - 1.0)
def return_curve(values):
    v = np.sort(values[np.isfinite(values)])
    return (len(v) + 1) / np.arange(1, len(v) + 1), v
fig, ax = plt.subplots(figsize=(8.0, 4.4), layout="constrained")
rt, lv = return_curve(sc.period(anom.sel(experiment="historical"), sc.BASELINE).values.ravel())
ax.plot(rt, lv, "k", lw=1.8, label=f"historical {SPAN(sc.BASELINE)}")
rows = [{"scenario": f"historical {SPAN(sc.BASELINE)}", **{f"1-in-{t}": float(np.interp(t, rt[::-1], lv[::-1])) for t in (5, 10, 20)}}]
for e, s in FUTURE:
    rt, lv = return_curve(sc.period(anom.sel(experiment=e), s).values.ravel())
    ax.plot(rt, lv, color=EXP_COLOURS[e], lw=1.6, ls="--" if s == sc.NEAR else "-", label=TAG(e, s))
    rows.append({"scenario": TAG(e, s), **{f"1-in-{t}": float(np.interp(t, rt[::-1], lv[::-1])) for t in (5, 10, 20)}})
ax.set(xscale="log", xlabel="Return period (seasons)", ylabel="Europe yield vs baseline mean (%)")
ax.axhline(0, color="k", lw=0.5)
ax.legend(frameon=False, fontsize=8)
save(fig, "return_levels", FIGURES)
plt.show()
levels_table = pd.DataFrame(rows).set_index("scenario")
levels_table.to_csv(TABLES / "return_levels.csv")
display(levels_table.round(1))
"""),
MD("## 7. Complete crop failure\n\nThe share of cropland area-seasons with zero yield, or not reaching maturity (`final_dvs` < 2). These are model outcomes with fixed sowing dates and a fixed cultivar, so they include a phenology mismatch that a real sowing or cultivar choice would avoid."),
CODE("""
dvs = sc.cube("final_dvs")
periods = [("historical", sc.EARLY), ("historical", sc.BASELINE), *FUTURE]
rows = []
for e, s in periods:
    z = sc.regional_mean((sc.period(y.sel(experiment=e), s) == 0).astype(np.float32).where(sc.period(y.sel(experiment=e), s).notnull())).mean("year") * 100
    d = sc.regional_mean((sc.period(dvs.sel(experiment=e), s) < 1.99).astype(np.float32).where(sc.period(dvs.sel(experiment=e), s).notnull())).mean("year") * 100
    rows.append({"scenario": TAG(e, s), **{f"zero yield: {r}": float(z.sel(region=r).mean()) for r in regions}, **{f"not mature: {r}": float(d.sel(region=r).mean()) for r in regions}})
failure = pd.DataFrame(rows).set_index("scenario")
failure.to_csv(TABLES / "complete_failure.csv")
display(failure.round(2).T)
"""),
MD("## 8. Per country"),
CODE("""
rows = {}
for e, s in FUTURE:
    p = sc.country_series(rk.probability(y.sel(experiment=e), limits["p10"], rk.EVENT["p10"], s))
    ref = sc.country_series(reference["p10"])
    rows[TAG(e, s)] = (100 * (p - ref)).mean("model").to_series()
country = pd.DataFrame(rows).sort_values(TAG("ssp370", sc.FAR))
country.to_csv(TABLES / "country_probability_change.csv")
fig, ax = plt.subplots(figsize=(7.0, 6.0), layout="constrained")
limit = float(np.abs(country.to_numpy()).max())
im = ax.imshow(country.to_numpy(), cmap=DIVERGING_CMAP, vmin=-limit, vmax=limit, aspect="auto")
ax.set_xticks(range(country.shape[1]), country.columns, rotation=30, ha="right", fontsize=8)
ax.set_yticks(range(len(country)), country.index, fontsize=8)
ax.grid(False)
for (i, j), v in np.ndenumerate(country.to_numpy()):
    ax.text(j, i, f"{v:+.1f}", ha="center", va="center", fontsize=6.5)
fig.colorbar(im, ax=ax, shrink=0.7, label="Δ probability of 1-in-10 low yield (points)")
save(fig, "country_probability", FIGURES)
plt.show()
"""),
MD("## 9. Sensitivity: heat-sterility-adjusted yield\n\n`yield_t_ha` leaves out heat sterility, so a change in risk from heat around flowering would not reach it. The tables repeat §1 on `adjusted_yield_t_ha`, using its own thresholds and reference."),
CODE("""
adjusted = sc.cube("adjusted_yield_t_ha")
event = rk.EVENT["p10"]
alt = summarise(event, adjusted, rk.threshold(adjusted, event), rk.loo_probability(adjusted, event))
alt.to_csv(TABLES / "event_probability_adjusted.csv", index=False)
compare = (summary[summary["event"] == "p10"].set_index(["scenario", "region"])[["risk ratio"]].rename(columns={"risk ratio": "yield_t_ha"})
           .join(alt.set_index(["scenario", "region"])["risk ratio"].rename("adjusted_yield_t_ha")))
compare["difference"] = compare["adjusted_yield_t_ha"] - compare["yield_t_ha"]
display(compare.loc[[TAG(*s) for s in FUTURE]].unstack("region")["difference"][regions].round(3))
print("Risk ratio, adjusted yield:")
display(compare.loc[[TAG(*s) for s in FUTURE]].unstack("region")["adjusted_yield_t_ha"][regions].round(2))
"""),
MD("""
## 10. Sensitivity: with the prescribed CO₂ effect on growth removed

Notebook 05 shows that crop parameter `COTB`, which multiplies radiation-use efficiency by a function of CO₂, accounts for nearly all of the mean yield gain. So how much of the falling risk comes from `COTB`? Each season's yield is divided by its `COTB` factor relative to the 1985–2014 mean, and §1 is repeated with its own thresholds.

This is an approximation, not a rerun. It assumes yield scales directly with RUE, and it leaves in CO₂'s reduction of transpiration. `hist-aer` tests the approximation: its climate has no greenhouse warming, so once its low CO₂ is divided out, its risk ratio should be close to or below 1.
"""),
CODE("""
import explain as xp

ppm, rue = xp.cotb_table()
co2 = sc.co2_series()
factor = xr.DataArray(np.stack([np.interp(co2[e].to_numpy(), ppm, rue) for e in sc.EXPERIMENTS]), dims=("experiment", "year"),
                      coords={"experiment": list(sc.EXPERIMENTS), "year": co2.index.to_numpy()})
factor = factor / float(factor.sel(experiment="historical").sel(year=slice(*sc.BASELINE)).mean())
rescaled = (y / factor.astype(np.float32)).rename("yield_without_cotb")
mean_change = sc.regional_table({TAG(e, s): sc.regional_relative_change(rescaled, e, s) for e, s in SCENARIOS}, 1)
mean_change.to_csv(TABLES / "mean_change_without_cotb.csv")
print("Mean yield change without COTB (%), multi-model mean [model range]:")
display(mean_change.T)
without = pd.concat([summarise(e, rescaled, rk.threshold(rescaled, e), rk.loo_probability(rescaled, e)) for e in (rk.EVENT["p10"], rk.EVENT["half"], rk.EVENT["p90"])], ignore_index=True)
without.to_csv(TABLES / "event_probability_without_cotb.csv", index=False)
for key in ("p10", "half", "p90"):
    print(rk.EVENT[key].label, "— risk ratio without COTB")
    display(without[(without["event"] == key) & without["region"].isin(regions)].pivot(index="scenario", columns="region", values="risk ratio").reindex([TAG(*s) for s in SCENARIOS])[regions].round(2))
display(without[(without["event"] == "p10") & (without["region"] == "Europe")].drop(columns=["event", "region"]).set_index("scenario").round(2))

limit_r, ref_r = rk.threshold(rescaled, rk.EVENT["p10"]), rk.loo_probability(rescaled, rk.EVENT["p10"])
delta = {(e, s): 100.0 * (rk.probability(rescaled.sel(experiment=e), limit_r, rk.EVENT["p10"], s) - ref_r) for e, s in FUTURE}
limit = float(np.nanquantile(np.abs(np.stack([d.mean("model").values for d in delta.values()])), 0.99)) or 1.0
fig, axes = sc.map_axes(1, 4, 3.0, 3.2)
for ax, ((e, s), d) in zip(axes[0], delta.items()):
    mesh = sc.draw_map(ax, d.mean("model").values, title=TAG(e, s), hatch=~sc.agreement(d).values, cmap=DIVERGING_CMAP, vmin=-limit, vmax=limit)
fig.colorbar(mesh, ax=axes, shrink=0.75, label="Δ probability, 1-in-10 low yield without COTB (points)")
save(fig, "probability_change_without_cotb", FIGURES)
plt.show()
w = cells.weight / cells.weight.sum()
display(pd.DataFrame([{"scenario": TAG(e, s), "cropland, risk rising (robust, %)": 100 * w[(sc.agreement(d).values) & (d.mean("model").values > 0)].sum(),
                       "cropland, risk falling (robust, %)": 100 * w[(sc.agreement(d).values) & (d.mean("model").values < 0)].sum()} for (e, s), d in delta.items()]).round(1))
"""),
)

# -------------------------------------------------------------------- explainability
notebook("05_explainability",
MD(f"""
# CMIP6 torchcrop runs — what explains the yield changes?

Two independent views:

1. **Inside the crop model** (§1, all cells): yield = biomass × harvest index, and the stress factors
   torchcrop writes. These are the simulated mechanisms, with no statistics in between.
2. **A statistical emulator** (§2–§6, a cropland-weighted sample of 5,000 cells from `features.py`):
   LightGBM predicts each season's yield anomaly from weather indices and CO₂, and TreeSHAP splits each
   prediction into additive parts.

**What this can and cannot say.** SHAP explains the emulator, and the emulator explains torchcrop. Neither
says what real wheat does. CO₂ and warming rise together, so the emulator can only separate them by
contrast: `hist-aer` (CO₂ fixed) and the two SSPs. §2 tests whether that works, by predicting an experiment left out of the fit,
and §4 compares SHAP with a second attribution rule. A clean separation would need torchcrop rerun with
CO₂ held fixed under SSP climate; that run does not exist yet, and §7 says what it would settle.

{CAVEATS}
"""),
MD("## 0. Setup"),
CODE(setup("explainability", """
import explain as xp
import features as ft

GROUP_COLOURS = dict(zip(xp.GROUP_LABELS, ("#4d4d4d", "#d95f02", "#1f78b4", "#66a61e", "#e31a1c", "#e6ab02", "#6a3d9a", "#a6cee3", "#bdbdbd")))
DRIVERS = [g for g in xp.GROUP_LABELS if g != "context"]
LABEL = lambda p: ft.FEATURES[p[2:]][2] if p.startswith("d_") else ("CO₂ (ppm)" if p == xp.CO2 else p)
SCEN = [("historical", "early"), ("hist-GHG", "baseline"), ("hist-aer", "baseline"), ("ssp126", "near"), ("ssp126", "far"), ("ssp370", "near"), ("ssp370", "far")]
""")),
MD("## 1. Inside the crop model\n\nThe change in log yield is exactly the change in log biomass plus the change in log harvest index, using Europe's cropland-weighted mean yield and biomass. The stress factors are then shown as simple differences."),
CODE("""
biomass, tran, nni, heat, lai, maturity, anthesis = (sc.cube(v) for v in ("biomass_g_m2", "tranrf_mean", "nni_mean", "heat_stress_factor", "max_lai", "days_to_maturity", "days_to_anthesis"))
SPANS = {"early": sc.EARLY, "baseline": sc.BASELINE, "near": sc.NEAR, "far": sc.FAR}
ybar, bbar = sc.regional_mean(y), sc.regional_mean(biomass)
rows = []
for e, name in SCEN:
    s = SPANS[name]
    for r in ybar.region.values:
        Y = np.log(sc.clim(ybar.sel(experiment=e, region=r), s) / sc.clim(ybar.sel(experiment="historical", region=r), sc.BASELINE)) * 100
        B = np.log(sc.clim(bbar.sel(experiment=e, region=r), s) / sc.clim(bbar.sel(experiment="historical", region=r), sc.BASELINE)) * 100
        rows.append({"scenario": f"{e} {SPAN(s)}", "region": r, "Δ ln yield (%)": float(Y.mean()), "Δ ln biomass (%)": float(B.mean()), "Δ ln harvest index (%)": float((Y - B).mean())})
decomposition = pd.DataFrame(rows)
decomposition.to_csv(TABLES / "yield_decomposition.csv", index=False)
display(decomposition[decomposition["region"].isin(["Europe", "WCE", "MED"])].set_index(["region", "scenario"]).round(1))

stress = {"Transpiration factor": tran, "Nitrogen nutrition index": nni, "Heat-stress factor": heat, "Max LAI": lai, "Days to anthesis": anthesis, "Days to maturity": maturity}
rows = {}
for name, cube_ in stress.items():
    rows[name] = {f"{e} {SPAN(s)}": f"{float(sc.regional_mean(sc.absolute_change(cube_, e, s), regions=('Europe',)).mean()):+.3g}" for e, s in [(e, SPANS[p]) for e, p in SCEN]}
    rows[name]["baseline value"] = f"{float(sc.regional_mean(sc.clim(cube_.sel(experiment='historical'), sc.BASELINE), regions=('Europe',)).mean()):.3g}"
process_change = pd.DataFrame(rows)
process_change.to_csv(TABLES / "process_change.csv")
display(process_change)
"""),
MD("Where each process moves in the high-emission far future (`ssp370`, 2071–2100 minus `historical` 1985–2014). Hatching marks cells where fewer than 3 of 4 models agree on the sign."),
CODE("""
panels = {"Biomass (%)": sc.relative_change(biomass, "ssp370", sc.FAR), "Harvest index (%)": sc.relative_change(y / biomass.where(biomass > 0), "ssp370", sc.FAR),
          "Transpiration factor": sc.absolute_change(tran, "ssp370", sc.FAR), "Nitrogen nutrition index": sc.absolute_change(nni, "ssp370", sc.FAR),
          "Heat-stress factor": sc.absolute_change(heat, "ssp370", sc.FAR), "Days to maturity": sc.absolute_change(maturity, "ssp370", sc.FAR)}
fig, axes = sc.map_axes(2, 3, 3.5, 2.75)
for ax, (name, delta) in zip(axes.ravel(), panels.items()):
    mean = delta.mean("model")
    fig.colorbar(sc.draw_map(ax, mean.values, title=name, hatch=~sc.agreement(delta).values, **sc.diverging(mean)), ax=ax, shrink=0.65)
save(fig, "process_maps", FIGURES)
plt.show()
"""),
MD("## 2. The emulator and whether to trust it\n\nHeld-out R² in rising order of difficulty. *Leave out model* asks whether the response transfers to a model the emulator has not seen. *Leave out experiment* is the extrapolation test that matters most for CO₂: `hist-aer` removes the only fixed-CO₂ data, and `ssp370` removes all CO₂ above about 475 ppm (`ssp126` peaks at 474)."),
CODE("""
panel = xp.panel()
panel["period"] = xp.label_periods(panel)
print(f"{len(panel):,} seasons, {panel.SimplaceID.nunique():,} cells, baseline ≥ {xp.MIN_BASE_YIELD} t/ha")
skill = xp.score_splits(panel)
skill.to_csv(TABLES / "emulator_skill.csv", index=False)
display(skill.round(3))
"""),
MD("## 3. Global importance\n\nFinal fit on all experiments. SHAP is computed on an equal draw from each model, experiment and period, so the long SSP runs do not dominate. Mean |SHAP| measures how much each part moves a prediction, whether up or down, in % of baseline yield."),
CODE("""
model = xp.fit(panel)
sample = panel.dropna(subset=["period"]).groupby(["model", "experiment", "period"]).sample(25_000, random_state=0).reset_index(drop=True)
shap = xp.contributions(model, sample)
groups = xp.by_group(shap)
print(f"Fit R² on the SHAP sample: {1 - ((sample.anomaly_pct - shap.sum(axis=1)) ** 2).sum() / ((sample.anomaly_pct - sample.anomaly_pct.mean()) ** 2).sum():.3f}")
fig, axes = plt.subplots(1, 2, figsize=(13.0, 5.0), layout="constrained")
imp_g = groups.abs().mean().sort_values()
axes[0].barh([xp.GROUP_LABELS[g] for g in imp_g.index], imp_g.values, color=[GROUP_COLOURS[g] for g in imp_g.index])
axes[0].set(xlabel="Mean |SHAP| (% of baseline yield)", title="By driver group")
imp_f = shap.drop(columns="base").abs().mean().sort_values().tail(16)
axes[1].barh([LABEL(p) for p in imp_f.index], imp_f.values, color=[GROUP_COLOURS[xp.GROUP_OF[p]] for p in imp_f.index])
axes[1].set(xlabel="Mean |SHAP| (% of baseline yield)", title="Top predictors")
axes[1].tick_params(axis="y", labelsize=7.5)
save(fig, "global_importance", FIGURES)
plt.show()
imp_g.rename("mean |SHAP|").to_csv(TABLES / "importance_groups.csv")
"""),
MD("### Response curves\n\nThe binned mean of each predictor's SHAP against its value. A flat line means the emulator ignores the predictor. For CO₂ the curve is shown per model, with a break where only `ssp370` supplies the data."),
CODE("""
CURVES = [xp.CO2, "d_tmean_grainfill", "d_tmean_winter", "d_prec_preanthesis", "d_prec_grainfill", "d_vpd_grainfill", "d_edd_grainfill", "d_rad_grainfill"]
fig, axes = plt.subplots(2, 4, figsize=(14.5, 6.2), layout="constrained")
for ax, p in zip(axes.ravel(), CURVES):
    edges = np.unique(np.nanquantile(sample[p], np.linspace(0.005, 0.995, 25)))
    for m in sc.MODELS:
        mask = (sample["model"] == m).to_numpy()
        bins = pd.cut(sample.loc[mask, p], edges)
        curve = shap.loc[mask, p].groupby(bins, observed=True).mean()
        ax.plot([b.mid for b in curve.index], curve.values, color=MODEL_COLOURS[m], lw=1.2, label=m)
    ax.axhline(0, color="k", lw=0.5)
    ax.set_title(LABEL(p), fontsize=8.5)
    ax.set_ylabel("SHAP (% of baseline)", fontsize=8)
axes[0, 0].legend(frameon=False, fontsize=7)
save(fig, "response_curves", FIGURES)
plt.show()
"""),
MD("## 4. What drives the change?\n\nThe change in mean SHAP from `historical` 1985–2014 to each scenario, per driver. SHAP is additive, so the bars sum to the emulator's change, and the black dot is the **simulated** change on the same rows. A gap between sum and dot is emulator error. Hollow markers are the second rule: what the prediction loses when that group alone is reset to normal."),
CODE("""
def period_means(frame, values):
    return values.groupby([frame["model"].to_numpy(), frame["experiment"].to_numpy(), frame["period"].to_numpy()]).mean()

mean_shap = period_means(sample, groups[DRIVERS])
sim = period_means(sample, sample["anomaly_pct"])
pred = period_means(sample, shap.sum(axis=1))
base_key = ("historical", "baseline")
rows, cf_rows = [], []
for e, name in SCEN:
    for m in sc.MODELS:
        delta = mean_shap.loc[(m, e, name)] - mean_shap.loc[(m, *base_key)]
        rows.append({"model": m, "scenario": f"{e} {name}", **delta.to_dict(), "sum": float(delta.sum()),
                     "emulated": float(pred.loc[(m, e, name)] - pred.loc[(m, *base_key)]), "simulated": float(sim.loc[(m, e, name)] - sim.loc[(m, *base_key)])})
        rows_now = sample[(sample["model"] == m) & (sample["experiment"] == e) & (sample["period"] == name)]
        rows_base = sample[(sample["model"] == m) & (sample["experiment"] == base_key[0]) & (sample["period"] == base_key[1])]
        cf_rows.append({"model": m, "scenario": f"{e} {name}", **{g: float(xp.counterfactual(model, rows_now, g).mean() - xp.counterfactual(model, rows_base, g).mean()) for g in DRIVERS}})
attribution, counter = pd.DataFrame(rows), pd.DataFrame(cf_rows)
attribution.to_csv(TABLES / "shap_attribution.csv", index=False)
counter.to_csv(TABLES / "counterfactual_attribution.csv", index=False)

means, cf_means = attribution.groupby("scenario", sort=False).mean(numeric_only=True), counter.groupby("scenario", sort=False).mean(numeric_only=True)
fig, ax = plt.subplots(figsize=(11.0, 5.0), layout="constrained")
pos, neg = np.zeros(len(means)), np.zeros(len(means))
yy = np.arange(len(means))
for g in DRIVERS:
    v = means[g].to_numpy()
    ax.barh(yy, v, left=np.where(v >= 0, pos, neg), color=GROUP_COLOURS[g], label=xp.GROUP_LABELS[g], height=0.75)
    pos, neg = pos + np.clip(v, 0, None), neg + np.clip(v, None, 0)
    ax.scatter(cf_means[g], yy, s=18, facecolors="none", edgecolors=GROUP_COLOURS[g], zorder=5, linewidths=1.0)
ax.scatter(means["simulated"], yy, color="k", s=26, zorder=6, label="Simulated change")
ax.axvline(0, color="k", lw=0.6)
ax.set_yticks(yy, means.index)
ax.invert_yaxis()
ax.set_xlabel("Change in yield anomaly vs historical 1985–2014 (% of baseline)")
ax.legend(frameon=False, ncol=3, fontsize=8)
save(fig, "attribution", FIGURES)
plt.show()
display(means.round(1).T)
print("Second rule, group reset to normal:")
display(cf_means.round(1).T)
"""),
MD("""
### Checking the CO₂ share against the crop file

Crop parameter `COTB` multiplies radiation-use efficiency by a prescribed function of CO₂, so its effect on biomass can be computed with no statistics. If growth scaled directly with RUE, the table's mean over each period would predict the CO₂ share. The last column divides that factor out of the simulated change and leaves an estimate of everything else: climate, plus CO₂'s other route through transpiration. The same `COTB` table is in all regional crops (the code raises an error otherwise).
"""),
CODE("""
ppm, rue = xp.cotb_table()
print("COTB:", dict(zip(ppm, rue)))
co2 = sc.co2_series()
factor = lambda e, s: float(np.interp(co2[e].loc[s[0]:s[1]].dropna().to_numpy(), ppm, rue).mean())
reference_factor = factor("historical", sc.BASELINE)
check = pd.DataFrame([{"scenario": f"{e} {name}", "mean CO₂ (ppm)": float(co2[e].loc[SPANS[name][0]:SPANS[name][1]].mean()),
                       "COTB alone (%)": 100 * (factor(e, SPANS[name]) / reference_factor - 1),
                       "SHAP CO₂ (%)": means.loc[f"{e} {name}", "co2"], "reset-to-normal CO₂ (%)": cf_means.loc[f"{e} {name}", "co2"],
                       "simulated (%)": means.loc[f"{e} {name}", "simulated"],
                       "simulated without COTB (%)": 100 * ((1 + means.loc[f"{e} {name}", "simulated"] / 100) * reference_factor / factor(e, SPANS[name]) - 1)} for e, name in SCEN]).set_index("scenario")
check.to_csv(TABLES / "co2_table_check.csv")
display(check.round(1))
"""),
MD("Per model, for `ssp370` far: does the same driver dominate in every model?"),
CODE("""
display(attribution[attribution["scenario"].isin(["ssp126 far", "ssp370 far"])].set_index(["scenario", "model"]).round(1).T)
"""),
MD("### By region\n\nThe same attribution by AR6 region, for the far-future scenarios."),
CODE("""
regions = {"WCE": "Western & Central Europe", "MED": "Mediterranean", "EEU": "Eastern Europe", "NEU": "Northern Europe"}
fig, axes = plt.subplots(2, 4, figsize=(15.0, 6.5), sharey=True, sharex=True, layout="constrained")
rows = []
for r_i, e in enumerate(sc.FUTURE):
    for ax, (r, label) in zip(axes[r_i], regions.items()):
        mask = (sample["ar6"] == r).to_numpy()
        now = groups.loc[mask & (sample["experiment"] == e).to_numpy() & (sample["period"] == "far").to_numpy(), DRIVERS].mean()
        ref = groups.loc[mask & (sample["experiment"] == "historical").to_numpy() & (sample["period"] == "baseline").to_numpy(), DRIVERS].mean()
        delta = now - ref
        rows.append({"experiment": e, "region": r, **delta.to_dict(), "cells in sample": int(sample.loc[mask, "SimplaceID"].nunique())})
        ax.barh([xp.GROUP_LABELS[g] for g in DRIVERS], delta.values, color=[GROUP_COLOURS[g] for g in DRIVERS])
        ax.axvline(0, color="k", lw=0.5)
        ax.set_title(f"{label}, {e}", fontsize=9)
axes[0, 0].invert_yaxis()
pd.DataFrame(rows).to_csv(TABLES / "shap_attribution_regions.csv", index=False)
save(fig, "attribution_regions", FIGURES)
plt.show()
display(pd.DataFrame(rows).set_index(["experiment", "region"]).round(1).T)
"""),
MD("## 5. Where each driver dominates\n\nThe strongest negative climate driver (CO₂ excluded) per 1° box in `ssp370` 2071–2100 against the baseline, using the sampled cells. Boxes with fewer than 3 sampled cells are blank. A box is shaded by the size of that driver's change, so a pale box has a weak signal."),
CODE("""
import cartopy.crs as ccrs
from matplotlib.lines import Line2D

climate = [g for g in DRIVERS if g != "co2"]
box = sample.assign(lat_box=np.floor(sample["lat"]) + 0.5, lon_box=np.floor(sample["lon"]) + 0.5)
fig, axes = sc.map_axes(1, 2, 5.2, 5.0)
for ax, e in zip(axes[0], sc.FUTURE):
    now = groups[climate][(box["experiment"] == e) & (box["period"] == "far")].join(box[["lat_box", "lon_box", "SimplaceID"]])
    ref = groups[climate][(box["experiment"] == "historical") & (box["period"] == "baseline")].join(box[["lat_box", "lon_box", "SimplaceID"]])
    enough = now.groupby(["lat_box", "lon_box"])["SimplaceID"].nunique() >= 3
    delta = (now.groupby(["lat_box", "lon_box"])[climate].mean() - ref.groupby(["lat_box", "lon_box"])[climate].mean())[enough].dropna()
    driver, size = delta.idxmin(axis=1), delta.min(axis=1)
    alpha = np.clip(-size / float(np.quantile(-size, 0.9)), 0.15, 1.0)
    ax.set_extent(sc.MAP_EXTENT, crs=ccrs.PlateCarree())
    ax.coastlines(lw=0.4)
    ax.scatter(delta.index.get_level_values("lon_box"), delta.index.get_level_values("lat_box"), c=driver.map(GROUP_COLOURS), alpha=alpha, s=26, marker="s", transform=ccrs.PlateCarree(), linewidths=0)
    ax.set_title(f"{e}, {SPAN(sc.FAR)}")
    display(driver.value_counts().rename(e).to_frame().T)
fig.legend(handles=[Line2D([], [], marker="s", ls="", color=GROUP_COLOURS[g], label=xp.GROUP_LABELS[g]) for g in climate], loc="outside lower center", ncol=len(climate), frameon=False, fontsize=8)
save(fig, "dominant_driver", FIGURES)
plt.show()
"""),
MD("## 6. What makes a season extreme?\n\nSeasons below the cell's 1-in-10 level (`historical` 1985–2014, per model and cell). Each bar is the mean SHAP in those seasons minus the mean SHAP over all seasons of the same scenario and period, so it isolates what separates a bad season from a normal one in that climate. CO₂ barely varies within a period, so it should barely appear."),
CODE("""
base_rows = panel[(panel["experiment"] == "historical") & panel["year"].between(*sc.BASELINE)]
limits = base_rows.groupby(["model", "SimplaceID"])["yield_t_ha"].quantile(0.10).rename("limit")
flagged = sample.join(limits, on=["model", "SimplaceID"])
low = (flagged["yield_t_ha"] < flagged["limit"]).to_numpy()
rows = []
for e, name in [("historical", "baseline"), ("ssp126", "far"), ("ssp370", "far")]:
    block = ((sample["experiment"] == e) & (sample["period"] == name)).to_numpy()
    for r in (None, "WCE", "MED", "EEU", "NEU"):
        where = block if r is None else block & (sample["ar6"] == r).to_numpy()
        delta = groups.loc[where & low, DRIVERS].mean() - groups.loc[where, DRIVERS].mean()
        rows.append({"scenario": f"{e} {name}", "region": r or "All", "seasons": int((where & low).sum()), "probability (%)": 100 * float(low[where].mean()), **delta.to_dict()})
anatomy = pd.DataFrame(rows)
anatomy.to_csv(TABLES / "extreme_anatomy.csv", index=False)
display(anatomy.set_index(["scenario", "region"]).round(1))

fig, axes = plt.subplots(1, 3, figsize=(13.5, 3.8), sharex=True, sharey=True, layout="constrained")
for ax, (name, block) in zip(axes, anatomy[anatomy["region"] == "All"].groupby("scenario", sort=False)):
    values = block[DRIVERS].iloc[0]
    ax.barh([xp.GROUP_LABELS[g] for g in DRIVERS], values.to_numpy(), color=[GROUP_COLOURS[g] for g in DRIVERS])
    ax.axvline(0, color="k", lw=0.5)
    ax.set(title=name, xlabel="SHAP in low-yield seasons − all seasons (points)")
axes[0].invert_yaxis()
save(fig, "extreme_anatomy", FIGURES)
plt.show()
"""),
MD("""
## 7. Limits of this attribution

* **Correlated drivers.** Mean temperature, VPD and heat extremes rise together, so their individual bars are less firm than their sum. Compare the group totals rather than ranking within the warm groups.
* **Fixed windows.** Indices use each cell's baseline phenology. In a warmer season the crop flowers earlier, so late-window heat may fall after real maturity. Windows that follow the crop would hide its response.
* **Emulated CO₂.** The CO₂ response above ~475 ppm is informed only by `ssp370`, and the leave-out row in §2 shows how far it can be trusted.
* **A decisive check.** Rerunning torchcrop on `ssp370` weather with CO₂ held at about 400 ppm, and the reverse, would give the CO₂ effect directly with no statistical split. It costs one run, about 2 h for each model and experiment.
"""),
)
