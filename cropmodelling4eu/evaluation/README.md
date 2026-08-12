# TorchCrop evaluation — country-level and gridded

Evaluates the 10 km [TorchCrop](../TORCHCROP.md) winter wheat run over
Europe against four references: **CyBench** at country level (EU-27 and
Schengen, Ukraine and Russia excluded), and **GDHY** / the **SAGE** crop
calendar per 0.5° grid cell.

```
evaluation/
├── yield_evaluation.ipynb          # yield vs CyBench: RMSE, MAE, Bias, MAPE, R2, Pearson r
├── phenology_evaluation.ipynb      # stage dates vs CyBench: RMSE, MAE, Bias, Pearson r
├── gdhy_yield_evaluation.ipynb     # yield vs GDHY, per 0.5° cell
├── sage_calendar_evaluation.ipynb  # sowing/maturity vs SAGE, per 0.5° cell
├── _build_notebooks.py             # regenerates all four notebooks from cell lists
└── outputs/                        # figures/, tables/, cache/ (git-ignored)
```

## Why four notebooks, not two

CyBench evaluates at the level a country runs its statistics service at:
sub-national regions, aggregated with real area weights, but collapsed to 23
national means. GDHY and SAGE evaluate at the level the model actually
predicts — a 0.5° grid cell — which is where a country-level bias can hide two
opposite regional errors, or a genuine spatial pattern the national mean
smooths away. The two pairs ask different questions and are not a
cross-check of each other so much as different resolutions of the same model.

## Running

The notebooks need `pandas`, `numpy`, `xarray`, `geopandas`, `cartopy`,
`matplotlib`, `scipy` and `pyarrow` — the project's `sdba` conda environment has
all of them.

```bash
conda activate sdba
cd cropmodelling4eu/evaluation
jupyter lab                       # then run any notebook top to bottom
```

Or headless:

```bash
python -m jupyter nbconvert --to notebook --execute --inplace yield_evaluation.ipynb
```

Each notebook takes a couple of minutes, dominated by the first read of the
1.7 M-row Parquet. The CyBench notebooks cache their cell-to-country join under
`outputs/cache/`.

Every input path in [`config.py`](../src/cropmodelling4eu/evaluation/config.py)
can be overridden by an
environment variable of the same name, so a different run needs no code change:

```bash
TORCHCROP_RUN_DIR=/data01/FDS/muduchuru/Data/SIMPLACE/torchcrop/potential jupyter lab
```

## Inputs

| What | Where |
| --- | --- |
| TorchCrop run | `<TORCHCROP_RUN_DIR>/torchcrop_europe.parquet` — one row per (cell, season) |
| CyBench yields | `<CYBENCH_ROOT>/cybench-data/wheat/<c>/yield_wheat_<c>.csv` |
| CyBench calendars | `.../crop_calendar_wheat_<c>.csv` — `sos`/`eos`, **static** |
| CyBench crop mask | `.../crop_mask_wheat_<c>.csv` — the calendar's area weight |
| Country geometry | `<CYBENCH_ROOT>/polygons/<c>/<c>.shp`, dissolved per country |
| GDHY yields | `<GDHY_ROOT>/<GDHY_CROP>/yield_<year>.nc4` — one NetCDF per year, 1981-2016, 0.5° |
| SAGE calendar | `<SAGE_ROOT>/<SAGE_CROP>.crop.calendar.fill.nc` — one static field, 0.5° |

**23 of the 31 target countries are evaluable** against CyBench. It publishes
no wheat for CY, LU, MT, SI (EU) or CH, IS, LI, NO (Schengen); the country-level
notebooks list them and drop them.

**GDHY overlaps the run in 2000-2016**, the intersection of its own 1981-2016
publication window and the run's 2000-2024. **SAGE has no year dimension at
all** — one climatological planting/harvest date per cell, so its comparison is
purely spatial, like CyBench's `sos`/`eos`.

## Two things changed when data4simplace gained its site stage

**Sowing is no longer a constant.** The export now carries a per-cell sowing
date (`site.csv`), the runner writes it into every result row as `sowing_doy`,
and `torchcrop.add_phenology_columns` uses that column when it is present. A
run made against an older export still works and still assumes DOY 270 — but it
says so in the log rather than leaving it to be inferred. Every date statistic
below therefore describes whichever of the two the run actually used.

**`sage_calendar_evaluation.ipynb`'s sowing comparison may now be circular.**
If the export was built with `site.calendar_source: sage` — the default — then
the simulated sowing date *came from* SAGE, and comparing the two is a round
trip rather than a test. Maturity stays a genuine comparison, because the model
predicts it. To keep sowing independent, build the export with
`site.calendar_source: ggcmi`.

## Design decisions worth knowing before reading a number

**Country footprints come from the CyBench polygons, not a world dataset.** The
reference yields are reported on exactly those administrative units, so
dissolving them gives a "France" whose simulated cells and observed regions
cover the same ground. A cell centre within 5 km of a border is snapped in — a
10 km cell whose centre lands just offshore still overlaps land, and at 0 km
Denmark and the Netherlands lose a fifth of their cells.

**Weighting is asymmetric, and that is deliberate.** The observed national yield
is `harvest_area`-weighted wherever at least half a country's regions report an
area (`MIN_WEIGHT_COVERAGE`), falling back to unweighted otherwise and recording
which was used. The simulated national mean is *unweighted* over cropland
cells: the SIMPLACE export carries no per-cell wheat area, so any weight would
have to be invented.

**No moisture conversion is applied.** TorchCrop reports grain dry matter,
CyBench reports market moisture (~13.5 %). Both are compared as published, so
part of the yield bias is that units difference rather than model error. Flip it
with `evaluation.aggregate.to_dry_matter`.

**Every date statistic is circular.** Means go through `circular_mean_doy`,
errors through `doy_difference`, correlations through both sides unwrapped about
the observed circular mean. Spain's regional `sos` runs from DOY 0.7 to 363; its
arithmetic mean is July, its circular mean is 30 December.

## What the phenology notebook can and cannot compare

| Stage | TorchCrop | CyBench | Verdict |
| --- | --- | --- | --- |
| Sowing | **constant input**, DOY 270 | `sos` | weak — see below |
| Flowering | **not in the run** | **not in the calendar** | not evaluated |
| Maturity | first day at `DVS >= 2` | `eos` | the one clean pairing |
| Harvest | not modelled; `= maturity + HARVEST_LAG_DAYS` (0) | `eos` | same as maturity |

* **Sowing is not a prediction.** The export carries no sowing calendar, so
  every cell from Crete to Lapland is sown on DOY 270.
* **CyBench `sos` changes meaning with climate**: ~DOY 330–360 in ES/EL/PT
  (autumn sowing) but ~DOY 40–75 in DE/PL/EE/LV (post-winter green-up). The
  notebook splits the two regimes rather than averaging across them.
* **Flowering is absent from both sides.** `cropmodelling4eu.torchcrop.run.run_batch` discards
  the DVS trajectory once it has dated the fertilizer schedule, so no anthesis
  date reaches the Parquet. Emitting the `DVS >= 1` crossing there and re-running
  the shards would supply the simulated side; the reference side would still
  need a source other than CyBench.
* **The reference is climatological.** With no year dimension on `sos`/`eos`,
  the per-country `pearson_r` is undefined (a constant has no variance) and the
  pooled one is a *spatial* correlation across countries. Nothing here tests
  whether the model tracks a warm year against a cold one.

## How the gridded notebooks work

**The run is binned onto the 0.5° reference grid; the reference is never
interpolated onto the 10 km grid.** `evaluation.grid.bin_cells` averages whichever
10 km cropland cells fall inside each 0.5° cell (`evaluation.grid.snap_to_grid`),
unweighted for the same reason as the country-level notebook — no per-cell
wheat area exists to weight by. A 0.5° cell is dropped unless it holds at least
`MIN_CELLS_PER_GRIDCELL` (default 3) simulated 10 km cells, so a cell that is
mostly sea does not carry the same weight as a fully covered inland one.

**GDHY is not an observation of the grid cell.** It downscales national and
sub-national yield statistics with a satellite vegetation-index proxy and a
crop mask, so a cell's *level* is largely its country's statistic and what
varies between neighbours is mostly the proxy. `gdhy_yield_evaluation.ipynb`
therefore splits the pooled correlation into a spatial term (cell means) and an
interannual term (`evaluation.metrics.to_anomalies`, each cell's deviation from its
own mean) — the pooled figure alone cannot tell "puts high-yielding cells in
the right place" apart from "tracks a good year from a bad one".

**A third of the SAGE domain is filled, not reported.** Where no reporting unit
covers a cell, the `.fill` product extrapolates from the nearest one; those
cells are flagged `filled` and every table in `sage_calendar_evaluation.ipynb`
can be read with or without them.

**A calendar date comes with a window**, 10-100 days wide for SAGE planting.
`evaluation.doy.doy_in_window` and `window_position` test whether the simulated date
falls inside it — a more forgiving and more informative question than a bias
in days, since a date a week from the midpoint of a wide window is not late.

**No p-value is reported for a pooled gridded fit.** Tens of thousands of
cell-years are not independent samples — neighbouring cells share most of
their signal, in both the run and a smoothed reference — so a p-value computed
as though they were would be fiction. Read the effect sizes only.

## Outputs

`outputs/figures/` — PNG (300 dpi) and PDF for every figure.
`outputs/tables/` — the metric tables and the paired data (country-year for
CyBench, cell-year or cell for GDHY/SAGE) as CSV.

## Regenerating the notebooks

All four notebooks are built from cell lists in `_build_notebooks.py`, so
prose and code changes are reviewable as a normal diff. **Rebuilding replaces
every cell's outputs with empty ones** — it does not merely append cells — so
re-execute afterwards or the notebooks on disk will carry no figures:

```bash
python _build_notebooks.py
python -m jupyter nbconvert --to notebook --execute --inplace *.ipynb
```

## Module map

The library lives in the installed package
([`src/cropmodelling4eu/evaluation/`](../src/cropmodelling4eu/evaluation/)), so
the notebooks import it like any other package rather than off `sys.path`:

```python
from cropmodelling4eu.evaluation import config, metrics, plots
```

| Module | Holds |
| --- | --- |
| `config.py` | Paths, country scope, constants, the phenology and calendar stage registries |
| `doy.py` | Circular day-of-year arithmetic, including window containment |
| `torchcrop.py` | Simulation loader, stage-date derivation |
| `cybench.py` | CyBench reference loaders (yield, calendar, crop mask) |
| `gdhy.py` | GDHY gridded-yield loader |
| `sage.py` | SAGE crop-calendar loader |
| `regions.py` | Country geometry, cell-to-country assignment |
| `grid.py` | Binning the 10 km run onto the 0.5° reference grid, gridded pairing |
| `aggregate.py` | Weighted and circular aggregation to country level, pairing |
| `metrics.py` | Linear and circular skill metrics, ranking, spatial/interannual anomalies |
| `plots.py` | Figures and styled tables |
| `style.py` | Palette, mark specs, colormaps |
