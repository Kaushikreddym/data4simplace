# torchcrop calibration

Calibrating the differentiable LINTUL-5 ([`torchcrop`](https://geonextgis.github.io/torchcrop))
against observed **phenology**, **yield** and **LAI** over Europe, by backpropagation
through the model's own day loop.

This plans work that does not exist yet. What *does* exist is
[`TORCHCROP.md`](TORCHCROP.md)'s uncalibrated production run, and its failure modes are
the specification for what follows.

## Why

The specification is the **full continental run**, not the 30-cell smoke test. Numbers
below are torchcrop over 2000–2024, scored in
[`evaluation/full_run_evaluation.ipynb`](evaluation/full_run_evaluation.ipynb) against
CyBench on its own reported units — 805 units / 14 923 unit-years for yield, 849 / 21 136
for maturity — with SIMPLACE on the **identical pairs** as a contrast, not as a truth.

| Symptom | torchcrop | SIMPLACE, same pairs | Stage that owns it |
| --- | --- | --- | --- |
| **Maturity is unbiased in the mean and wrong per unit** | bias **+1.0 d**, MAE 18.2 d, RMSE 23.3 d, r 0.73; per-unit bias spans **−61 … +108 d**, 31 % of units off by more than 20 d | −5.4 d, MAE 11.0, RMSE 14.1, r 0.75; −34 … +46 d, 10 % over 20 d | 1 · phenology |
| **The yield level is close, the skill is absent** | bias **+0.71 t/ha**, RMSE 2.42, r 0.34, **r² −0.32**; per-unit bias −3.2 … +7.0 t/ha | +0.16 t/ha, RMSE 2.38, r 0.26, r² −0.27 | 2 · yield |
| **The canopy is plausible, the partitioning is not** | peak LAI median **4.43**, 66 % inside the expected 4–6 — but harvest index median **0.538**, 26 % of cell-seasons above 0.6, on only 1 210 g m⁻² of biomass | LAI median 6.43, HI median **0.337**, biomass 1 721 g m⁻² | 3 · LAI |

Pooled `tranrf` 0.94 and `nni` 0.96 over the same cells: neither water nor nitrogen is the
limiter, so none of the three is a stress artefact.

> ⚠ **These numbers are torchcrop's on the *superseded* crop parameterisation** — the one
> described in *Resolved* below, where it ran `idsl = 0` with `tsum1 = 1623` while SIMPLACE
> ran `idsl = 2` with `tsum1 = 1125`. A re-run on the harmonised crop is in flight. Expect
> the phenology rows to move substantially and the §6 model-to-model comparison to change
> character entirely; the **stage ownership** below is what survives, because each stage is
> named by a *kind* of defect rather than by a number. Re-read §4–§6 of
> `full_run_evaluation.ipynb` before quoting any figure in this section.

What each stage has to fix:

- **Stage 1 — a spread, not a bias.** A pooled +1.0 d hides a −61 … +108 d per-unit range.
  A global `tsum` shift cannot touch it and any pooled metric will score it as an
  improvement, so stage 1 is per-region by necessity.
- **Stage 2 — skill, not level.** `r² −0.32` is worse than predicting the observed mean.
  The level/anomaly split in *Observations → Yield* is the stage, not a refinement of it.
- **Stage 3 — partitioning, not growth.** The canopy is built (median 4.43 m² m⁻²).
  `fotb` steps **0.0 → 1.0 between DVS 0.99 and 1.00** on the harmonised crop — the whole
  assimilate stream switches to the grain inside one hundredth of a development stage,
  sharper than the 0 → 0.95 the superseded file had. That is the direct cause of the
  harvest-index tail, and it is a partitioning shape, not a growth deficit.

### Resolved: the two models were running different crops

`idsl = 0` was never SIMPLACE's setting. It was torchcrop's, and it arrived there because
the harmonisation pointed at the wrong file:

| | Crop file | `idsl` | `tsum1` | `versat` | `tdwi` | `rgrl` |
| --- | --- | --- | --- | --- | --- | --- |
| SIMPLACE | SUSTAg `LINTUL5_crop.xml`, block `Crop=WW` | **2** | **1125** | **70** | 21.0 | 0.00817 |
| torchcrop, before | Brandenburg `crop.xml` | **0** | **1623** | absent | 15 | 0.018 |
| torchcrop, now | the same SUSTAg `WW` block | 2 | 1125 | 70 | 21.0 | 0.00817 |

So §6 of `full_run_evaluation.ipynb` was comparing two parameterisations, not two models,
and the `tsum1` gap is the mechanism this section described: at `idsl = 0` the crop needs
an inflated 1623 °C·d to delay anthesis with no vernalisation brake, while SIMPLACE's 1125
sits inside the JRC prior because vernalisation does that work.

Three things had to change for the harmonisation to be real, all now in place:

1. **`load_simplace_crop` keys on `CropName` *or* `Crop`.** `LINTUL5_crop.xml` holds eight
   blocks — `MAIZ MAIF WW BARL RAPE POTA SUGB CC` — and the old reader took the first,
   which is maize. A multi-crop file with no named block is now an error, not a default.
2. **Tables are read in both shapes.** Brandenburg stores a table as two parameters
   (`SLATableDVS` + `SLATableSLA`); SUSTAg stores one interleaved `<value>` list (`SLATB`).
   Reading only the first shape left every SUSTAg table silently on torchcrop's own preset.
3. **`versat` / `vbase` / `vernrt` are carried even though the bundled preset has no slot
   for them.** This is the one that mattered most: `idsl = 2` with `versat == vbase` makes
   `processes/phenology.py:194` return `vernfac = 1`, so the switch is inert — and it would
   have been inert *next to a `tsum1` calibrated with vernalisation*, which runs the crop
   far too fast with nothing in the file to say why.

`TC_CROP_XML` and `TC_SIMPLACE_CROP` select the file and the block; both submit paths set
them to the SIMPLACE run's own workspace copy automatically — the **smoke test carried the
same bug**, since `smoke.yaml` also runs the SUSTAg solution. `seeds.xml` keys its blocks
on long names (`winter_wheat`) where the crop file uses codes (`WW`), so `TC_SEEDS_CROP`
names it separately; its values are recorded and never mapped, so a miss there warns
rather than failing the build.

**What this does not reach** is in the next section. One item belongs here, though,
because it is specific to `idsl = 2`: the solution binds
`cMinimalVernalisationFactor = 0.4` on its `Phenology` component, while torchcrop's
`minimal_vernalisation_factor` defaults to 0. It is a *solution* constant, not a crop-file
parameter, so no crop mapping can carry it — and 0 versus 0.4 is the difference between a
crop whose development can halt completely without vernalisation and one that always keeps
40 % of its rate. It is a free parameter in [stage 1](#stage-1--phenology) anyway.

`idsl` itself stays out of the loss: it is in `Lintul5Model._NON_DIFFERENTIABLE_FIELDS`
(with `iopt`, `iairdu`), so no gradient reaches it and `materialize` warns if it is wrapped
as a free parameter.

### What harmonisation does not reach

Sharing the crop file closes the parameter gap. Two others stay open, and both matter to
calibration because a stage cannot fit what neither model reads from data.

**torchcrop inputs the export does not carry** (`torchcrop/run.py:ASSUMPTIONS`). These are
modelling assumptions with no SIMPLACE counterpart anywhere in the workspace — SIMPLACE's
water model is SLIM, whose soil parameters are `WMGENT / WRGENT / ALFA / DP / SMD / …`:

| Assumption | Value | Note |
| --- | --- | --- |
| `crairc` | 0.07, capped at `0.5·(wcst − wcfc)` | Critical air content. SLIM has **no** counterpart: a grep of the whole SIMPLACE workspace returns nothing. The cap exists because SoilGrids' `wv0010` is a drained upper limit, not porosity, so the uncapped default put `smair` below field capacity and read ordinary moisture as waterlogging |
| `ksub`, `runfr`, `cfev` | 100.0, 0.0, 2.0 | Subsoil drainage, runoff fraction, evaporation factor |
| `cn_ratio`, `mineralisable_frac`, `rtnmins` | 12.0, 0.003, 0.025 | Soil organic-matter turnover |
| `nmini_cap_g_m2`, `nminti_scale` | 15.0, 1.0 | Initial mineral N, which blocker 3 is about |

Freeing any of these in a stage would be fitting a torchcrop-only quantity against an
observation both models are supposed to share. Leave them fixed and record the values.

**SIMPLACE parameters torchcrop has no field for** — 17, logged by name on every workspace
build rather than dropped silently: `SeedWeight`, `vMaximumHeight`, `vZ_crop`,
`vSlopeTranCO2`, `vYInterceptTranCO2`, `vSpringFall`, `vRINPOP`, `vNSRPP`, `dfactor_max`,
`LateStageDVSStart`, `vrc`, `vHt_crop_REGION`, and the four heat-stress ones above. Most
drive SUSTAg modules torchcrop does not implement (canopy height, CO₂-transpiration
response, seed-to-sprout establishment). The heat four are the exception: they are a
mapping that has not been written, not a structural gap.

## Observations

### Phenology

| Source | Status | Role |
| --- | --- | --- |
| **PEP725** `/data01/FDS/muduchuru/Data/Agri/PEP725/` | on disk | **Primary.** 213 365 wheat rows, 3 018 stations, 1990–2024. BBCH **0** sowing, **10** emergence, 31, **51** heading, 85, **100** harvest. `cult_season == 2` isolates winter wheat (211 947 of 213 365 rows). |
| **CyBench** `crop_calendar_wheat_<c>.csv` | on disk | **Fallback.** Pan-European per `adm_id`, but **static** — one `sos`/`eos` per region, no year dimension. It anchors the climatological position where PEP725 has no stations; it can never reward interannual skill, only penalise it. Down-weight accordingly. |
| **JRC `EU_Phenology_Complete_ver2019`** `/data01/FDS/muduchuru/Data/Agri/Phenology/` | on disk | **Prior, not observation.** The MARS/BioMA per-cell calibration on 14 175 × 25 km cells: `TSUM1` (mean 860, 431–1240), `TSUM2` (794, 400–1350), `SOWING` (DOY 277), `VERNSAT` (54 d, 6–70). Joins `IDGRID` ↔ `grid_25km.shp:Grid_Code`, EPSG:3035. |
| **CLMS HRL Croplands** `CPMCE`/`CPMCH` | see [`CLMS_DOWNLOAD.md`](../src/data4simplace/CLMS_DOWNLOAD.md) | **The fix for PEP725's coverage hole.** 10 m, per-year, 2017–2024, pan-European. |

**PEP725 is not European — it is Central European.** That is the single fact that shapes
the regional design:

```
provider 101 (DE)   200 834 rows   6.0–15.0 °E, 47.6–54.9 °N
provider 301 (AT)     5 858
provider 601 (SK)     3 968
provider 2101 (SI)    1 343
providers 4 / 8 / 401 / 1401 / 2601 / 2701   ~1 400 total (CZ, BE-NL, ES-Cat, HR, BA, ME)
```

94 % of the phenology signal is German. Stage 1 can genuinely fit **CON, ATC, NEM** and
parts of **ALS/PAN**; every other EnS zone gets a shrinkage-to-prior fit against CyBench
`sos`/`eos` and the JRC map until CLMS lands. This is a limitation to publish in the
outputs, not to paper over.

**The JRC file is a prior with a caveat.** It is a WOFOST parameterisation and LINTUL-5's
`dtsmtb` caps the daily increment at 27 °Cd, so the numbers do not transfer 1:1. Use its
*spread across zones* to set per-region bounds, and its map as an **independent check** on
the calibrated `tsum1` field — it is never in the loss, so agreement is real evidence.

### Yield

**CyBench `yield_wheat_<c>.csv`** — subnational `(adm_id, harvest_year)`, t/ha, with
`harvest_area` weights and `crop_mask`. Nothing else on disk competes. Three mandatory
treatments:

1. **Dry matter.** CyBench is fresh weight at standard moisture; `wso` is g DM m⁻². Route
   through the existing `evaluation.aggregate.to_dry_matter` / `YIELD_MOISTURE_CONTENT`.
2. **Detrend.** CyBench carries a technology trend LINTUL-5 structurally cannot produce —
   no cultivar change, no management change. Fit raw levels and the parameters absorb the
   trend. Split the objective instead:

   ```
   L_yield = w_lvl · (mean_sim − mean_obs)²  +  w_ano · Huber(anom_sim, anom_obs)
   ```

   with observations detrended per `adm_id`. The level term holds the +0.71 t/ha bias; the
   anomaly term is the only part that tests climate response, and is what `r² −0.32` says
   is missing.
3. **Area weighting** — `aggregate.weighted_mean` already does this, including its
   documented fallback for the 75 % of German rows with no `harvest_area`.

GDHY (`evaluation/gdhy.py`) stays a **validation** set. It is itself modelled; training on
it would calibrate against another model's biases.

### LAI

**CyBench `fpar_wheat_<c>.csv`** — dekadal, 2001–2023, per `adm_id`, crop-mask weighted
(Germany: 401 regions, 332 028 rows, 10-day spacing confirmed).

> **Do not invert fPAR into LAI.** `DiagnosticState.frac_intercepted` *is* the model's own
> fAPAR, `1 − exp(−k·LAI)`. Compare simulation to observation in fAPAR space and the
> inversion never happens.

Two consequences:

- **Freeze `kdiftb` in stage 3.** Against fAPAR, `k` and `LAI` are exactly degenerate —
  any `k` is compensable by an `LAI`. Calibrating both fits the observation and gets the
  canopy wrong.
- **fAPAR saturates above LAI ≈ 3**, so it constrains canopy *timing* far better than
  peak magnitude. Weight the loss toward shape — greenup date, cumulative fAPAR,
  senescence half-fall date — not toward the plateau.

**One units check before first use.** `DE111` `fpar` ranges 7.9–30.7 while `DEB3K` opens
at 45.9. That is either percent with crop-mask dilution or an undocumented scale factor;
resolve it against a known dense-canopy region or it becomes a silent bias in every
stage-3 gradient.

`ndvi` is available and strictly worse — it needs an empirical NDVI→LAI function, i.e. a
nuisance parameter set fitted alongside the crop. Cross-check greenup timing with it, no
more. A true LAI product (CGLS 300 m, HR-VPP) would break the `k`/LAI degeneracy and
enable a stage 3b that frees `kdiftb`; neither is on disk.

## Regions

`/data01/FDS/muduchuru/Data/Agri/EnSv8/data/` gives **84 strata** (`ens_v8.shp`) in
**13 zones** (`enz_v8.shp`), EPSG:3035.

**Use EnZ (13) as the base, not EnS (84).** 84 strata against the observation density
above is hopelessly over-parameterised, and PEP725 reaches maybe 15 strata at all.

Per 10 km cell:

1. Reproject the cell centre to EPSG:3035; point-in-polygon → `EnZ` (13 classes), with
   `EnS` kept as diagnostic metadata.
2. Join to `adm_id` through `evaluation/regions.py`; attach CyBench `sos`, `eos` and the
   long-term mean yield.
3. **Within each EnZ**, cluster on `(sin/cos sos, sin/cos eos, mean yield)` — circular
   encoding for the dates, because the mean of DOY 300 and DOY 40 is not a date. `k` per
   zone by silhouette, subject to `min_cells_per_region` **and**
   `min_observation_years_per_region`.

Target **15–30 regions**. Each layer earns its place: EnZ carries agro-climate, the
calendar splits zones that straddle a season break (ATC spans Ireland and SW France), and
yield level proxies the management intensity the model does not represent — a region
mixing 3 t/ha and 9 t/ha districts will fit neither.

**Pooling defaults to hierarchical**: one global latent per parameter plus a per-region
deviation under L2 shrinkage, so data-poor regions inherit the global fit instead of
running to their bounds. `pooling: {none, hierarchical}` as a config switch.

## Three mechanics that shape the implementation

### A batch must be region-homogeneous

`torchcrop/calibration/paths.py:rebuild_table` assembles an `[N, 2]` table from **scalar**
ordinates (`torch.stack` over row values). Scalar *fields* can be `[B]`-shaped and
broadcast per cell — `test_batch_consistency_per_element_params` guarantees it — but
**table ordinates cannot**. `slatb`, `fltb`, `ruetb`, `rdrltb`, `dtsmtb` and `vernrt` are
all tables, and all of them need calibrating.

So: one `CalibrationManager` per region, one region per forward pass. This costs nothing —
a region-stratified sampler produces exactly that — but it has to be designed in, not
discovered.

### torchcrop has no gradient checkpointing

`ModelOutput` retains every per-day `ModelState`, rate dict and `DiagnosticState`. At
batch 2048 that was 1.7 GB *without* a graph; with autograd expect 10–20×. Mitigations in
order of cheapness:

1. **Cut the window** — sowing − 15 d to maturity + 30 d ≈ 330 days, against the
   production run's 600+ (`SIM_YEARS = 2`, `min_days_after_sowing: 365`).
2. **Batch 128–256**, not 2048.
3. **√T checkpointing** over the day loop, written in our own wrapper against the exposed
   `compute_rates` / `update_state` low-level API: roughly 1.3× compute for O(√T) memory.
   Budget this as real work, not a flag.

### `smooth=True` is optional for stage 1, and off by default

`Phenology(smooth=True)` replaces `emerged = tsump >= tsumem` and the `DVS < 1` / `DVS < 2`
branches with sigmoid blends of sharpness `k_sharp`. **Stage 1 does not need it**, and
torchcrop's own phenology-calibration example runs without it: the loss reads dates off
the accumulators through `crossing_day` (see *Losses*), and that path is differentiable
whether or not the branching is.

Where the gradient actually flows, at `smooth=False`:

| Target | Path to `tsumem` | Live? |
| --- | --- | --- |
| Emergence date | `crossing_day(tsump, tsumem)` — through the *threshold* | **yes**, exact |
| Anthesis, maturity dates | `dvs_rate × emerged_f`, and `emerged_f` is a bool cast | no |

So `tsumem` is identified by the emergence term alone — which is the reason the emergence
output had to exist before stage 1 could run at all, not a nice-to-have. `tsum1` and
`tsum2` are unaffected: they are thresholds on `dvs`, reached through the same
`crossing_day`.

Turn `smooth=True` on only if a stage-2 or stage-3 term needs a gradient *through* the
branching itself; it buys nothing for the dates and costs a sharpness parameter that
biases them. `HeatStressOnLeafSenescence` takes the same flag.

**The two-pass fertilizer schedule stays under `no_grad`.** `run._simulate` runs pass 1
unfertilised only to date the DVS-keyed schedule; keeping it detached means the gradient
ignores ∂dates/∂θ, which is defensible because those dates move discretely anyway.

## Losses

**Read a stage date off the accumulator by linear interpolation**, using
`cropmodelling4eu.torchcrop.run.crossing_day` — already written, already used to date
emergence in the production run:

```python
t0   = last step below the threshold
frac = (threshold − x[t0]) / (x[t0+1] − x[t0])
day  = t0 + frac
```

Not an argmax, which quantises to whole days and carries no gradient. And **not** a
sigmoid-sum `t₀ + Σ σ(k(target − x_t))`, which an earlier version of this section
prescribed: that is exact only in the sharp limit and biased at any finite `k`, and the
quantity is reported in days, so a smoothing bias is a bias in the answer. Linear
interpolation is exact, differentiable in both the trajectory and the threshold, and
sub-daily. This is what torchcrop's own phenology-calibration example uses.

The loss comes out **in days**, directly comparable to the SIMPLACE reference scores in
*Splits, diagnostics, acceptance* (maturity −5.4 d bias, MAE 11.0 d over 849 units).

**Which accumulator, per stage** — this is not a free choice:

| PEP725 | Accumulator | Threshold | Why not DVS |
| --- | --- | --- | --- |
| BBCH 10 emergence | `tsump` | `tsumem` | DVS is pinned at 0 until the crop emerges, so it carries no emergence signal at all; its first non-zero step is also one day late, because the rate on day *t* produces the state on day *t+1* |
| BBCH 51 heading | `dvs` | ≈ 0.85–0.90 | anthesis is BBCH 61 = DVS 1.0 |
| BBCH 100 harvest | `dvs` | 2.0, **hinged** | harvest ≥ maturity: penalise maturity *after* harvest hard, more than ~14 d *before* it mildly |

`crossing_day` returns **NaN where the threshold is never reached**, which is the right
behaviour for a loss too: a cell that never matured has no maturity date, and a clamped
last-day answer would be scored as a real one.

Work in **days since the window start**, not day-of-year. Upstream's example does
(`(date − SIM_START).dt.days`), and it removes circular arithmetic from the loss entirely —
no wrap, no shortest-path difference. The evaluation notebook still needs the circular
machinery because it pairs two products that share no origin; the loss does not.

A cheaper fallback — penalise `(DVS(t_obs) − DVS_target)²` at the observed date — loses the
day units, so it stays the fallback.

Yield and canopy losses as in *Observations*. The canopy loss adds a senescence-onset
term on fAPAR falling below half its peak — the one date that **does** need the sigmoid-sum
construction, because fAPAR is not monotone and `crossing_day` assumes it is.

## Staging — cumulative, not sequential

The requested order is **phenology → yield → LAI**. One structural caveat: LAI is causally
*upstream* of yield, so moving the canopy in stage 3 will de-tune what stage 2 fitted.

The fix keeps the order intact — **stages accumulate terms rather than replacing the
objective**:

| Stage | Freed | Loss |
| --- | --- | --- |
| 1 | phenology | `L_phen` |
| 2 | yield (phenology frozen) | `L_yield + λ₁·L_phen` |
| 3 | canopy (phenology frozen, yield params free but anchored) | `L_lai + λ₂·L_yield + λ₁·L_phen` |
| 4 (optional) | all | joint fine-tune at small lr |

The retention terms are free: one forward pass produces all three. `stage_order` stays a
config key so swapping 2 and 3 is a one-line change if the retention weights prove
insufficient.

## Parameters per stage

### Stage 1 — phenology

**Structural, decided before the stage (non-differentiable):** `idsl` 0 → **2**;
`site.idpl` sowing DOY, taken from PEP725 BBCH 0 where available, else `site.csv`.

**Current** is `<TC_OUT_DIR>/workspace/crop_wheat.yaml` today. A current value outside its
bound means the run starts from the wrong place.

**Tier 1 — free in every region.**

| Parameter | Unit | Current | Range | Constraint / source |
| --- | --- | --- | --- | --- |
| `tsum1` | °C·d | 1125 | **500 – 1300** | JRC `TSUM1` 431–1240. Now inside the prior; it was 1623 while the crop ran at `idsl = 0` |
| `tsum2` | °C·d | 1000 | **400 – 1350** | JRC `TSUM2` 400–1350 (mean 794) |
| `tsumem` | °C·d | 60 | **40 – 300** | torchcrop's own phenology example fits **160** on German ground observations. Unaffected by `idsl`: vernalisation gates DVS, not the `tsump` clock. The 33 d observed sowing→emergence lag (§5 of the evaluation) says 60 is far too low |
| `versat` | vernal d | **70** | **5 – 70** | JRC `VERNSAT` 6–70 (mean 54). At the top of the prior, so expect it to move down |
| `vbase` | vernal d | **14** | **0 – 25** | Hard constraint `vbase ≤ versat − 5` — it is the denominator in `vernf = (vern − vbase)/(versat − vbase)`, and `versat = vbase` makes the whole block a no-op |
| `vernrt` y-ordinates | d d⁻¹ | 0, 0, 1, 1, 0, 0 | **0 – 1** each | Fixed x = −10, −4, 3, 10, 17, 30 °C (SIMPLACE's own knots); unimodal. Free abscissae are not identifiable from three dates |
| `phottb` y-ordinates | – | 0 at 0 h → 1 at 17 h | **0 – 1** each | Two knots only, a straight ramp. **Non-decreasing.** Adding interior knots is a stage-1 decision, not a given |

**Tier 2 — freed only where three PEP725 stages are observed.**

| Parameter | Unit | Current | Range | Constraint / source |
| --- | --- | --- | --- | --- |
| `vernalisation_devstage` | DVS | 0.3 | **0.2 – 0.5** | Above this DVS `vernfac = 1`. Trades against `versat` |
| `minimal_vernalisation_factor` | – | 0.0 *(SIMPLACE's solution uses **0.4**)* | **0.0 – 0.5** | Floor on rate suppression. At 0 an unvernalised crop never develops — wrong for Mediterranean cells. The two models disagree here and no crop mapping can fix it (see *Resolved*), so freeing it is also what makes them comparable |
| `tbasem` | °C | 0.0 | **−2 – 4** | Pre-emergence base temperature |
| `dtsmtb` y-ordinates | °C·d | 30 at 30 °C | **20 – 32** at x = 30, 45 | The SUSTAg crop caps at 30, the same as WOFOST, so the JRC `TSUM` prior transfers directly. (Brandenburg's capped at 27, which is why an earlier version of this row said it did not.) |

**Identifiability rule: free parameters per region ≤ independent observed stages per
region.** Tier 1 is seven quantities against at most three PEP725 stages, so free `tsum1`,
`tsum2`, `versat`, `vbase` first and admit the two tables only where the
sowing→heading→harvest triple is observed across enough years to separate a photoperiod
response from a thermal one. A region with only CyBench `sos`/`eos` gets **two** (`tsum1`,
`tsum2`) under heavy shrinkage to the JRC prior, vernalisation held at the global latent.

### Stage 2 — yield

**Partitioning leads this stage.** Assimilation and partitioning compensate for each other
(1 210 g m⁻² biomass at HI 0.538), and only partitioning has an independent anchor:
HI 0.45–0.55 for winter wheat.

| Group | Parameter | Unit | Current | Range |
| --- | --- | --- | --- | --- |
| Partitioning | `fltb` y at DVS 0.646 / 0.95 | – | 0.30 / 0.00 | **0 – 0.7** each |
| | `fstb` y at DVS 0.646 / 0.95 | – | 0.70 / 1.00 | **0 – 1** each |
| | `fotb` y at DVS 0.99 / 1.0 | – | 0.00 / 1.00 | **0 – 1**, non-decreasing — a **step of the full fraction in one DVS step of 0.01** is the direct cause of the HI tail, and it is sharper than the 0 → 0.95 the previous crop file had |
| | *derived* `fo = 1 − fl − fs` | – | — | simplex, see the warning below |
| Assimilation | `ruetb` y at DVS 0 / 1.0 | g MJ⁻¹ | 2.6 / 2.5 | **2.0 – 4.0**, non-increasing |
| | `ruetb` y at DVS 1.3 | g MJ⁻¹ | 2.0 | **0.5 – 3.0** |
| | `scale_factor_rue` | – | 1.0 | **0.8 – 1.25** |
| N (`iopt = 3`) | `nmaxso` | g N g⁻¹ DM | 0.031 | **0.015 – 0.035** |
| | `frnx` | – | **0.5** | **0.3 – 1.0** — the optimum-N reduction is *active* on this crop; it was disabled (1.0) on the previous file |
| | `nlue` | – | 1.1 | **0.8 – 1.5** |
| | `tcnt` | d | 10 | **5 – 20** |
| | `dvsnt` | DVS | **1.0** | **0.6 – 1.0** — at the bound |
| | `nrf` | – | 0.7 | **0.4 – 0.9** — a *management* parameter in SIMPLACE, so freeing it decouples the two models |
| Water | `depnr` | – | 4.5 | **2.0 – 5.0** |
| | `cfet` | – | 1.0 | **0.8 – 1.2** |
| | `rdmcr` | m | 1.25 | **0.8 – 2.0**, capped by the soil profile's own rootable depth |
| | `rri` | m d⁻¹ | 0.012 | **0.008 – 0.030** |
| Heat | `grain_heat_temp_critical` | °C | 27 *(tc default; SIMPLACE 31)* | **24 – 32** |
| | `grain_heat_temp_limit` | °C | 40 *(tc default)* | **34 – 45**, with `limit ≥ critical + 4` |
| | `grain_heat_begin_devstage` | DVS | 0.8 *(tc default; SIMPLACE 0.75)* | **0.7 – 1.1** |
| | `grain_heat_end_devstage` | DVS | 1.3 *(tc default; SIMPLACE 1.25)* | **1.2 – 1.6** |

> ⚠ **The heat group is the one place the crop harmonisation does not reach.** SIMPLACE
> parameterises heat stress with `vTCritical` 31 °C, `vStartDVS` 0.75, `vEndDVS` 1.25 and
> `vReductionPerDHAboveTempCritical` 0.0025; three of those four have torchcrop
> counterparts and none is in `SCALARS`, so torchcrop runs the group on its own defaults
> (27 / 0.8 / 1.3). Add the three to the mapping before stage 2 frees them, or the stage
> starts from a difference nobody chose. The fourth has no counterpart: SIMPLACE reduces
> yield linearly per degree-hour above the critical temperature, torchcrop ramps between a
> critical and a limit temperature — a structural difference, not a parameter one.

> ⚠ **`fl + fs + fo = 1` is a simplex constraint the stock `CalibrationManager` cannot
> express.** `ConstraintGroup` does ordering only. Calibrating the three independently
> produces an invalid partitioning that still runs. We need a `SimplexGroup` in our own
> subclass — calibrate `fl` and `fs`, derive `fo = 1 − fl − fs` — applied in a
> post-`materialize` hook.

> ⚠ **The N parameters may not be identifiable at all.** Initial mineral N is a median
> ~243 kg/ha against a measured 30–90 (`soil.mineral_n_fraction = 0.01`, see
> `TORCHCROP.md`), so the crop is effectively N-unlimited and `nlue`/`frnx` have nothing
> to bite on — pooled `nni` is **0.96** over 760 k scored cell-seasons, so the N stress the
> group parameterises is essentially never active. Either fix the export first, or exclude
> the N group and say so in the outputs.

### Stage 3 — LAI

| Group | Parameter | Unit | Current | Range |
| --- | --- | --- | --- | --- |
| Specific leaf area | `slatb` y-ordinates | m² g⁻¹ | 0.015 – 0.020 | **0.008 – 0.025** each |
| | `scale_factor_sla` | – | 1.0 | **0.7 – 1.4** |
| Establishment | `rgrl` | (°C·d)⁻¹ | **0.00817** | **0.005 – 0.035** — the lower bound was 0.008, which the current value sits on top of |
| | `tdwi` | g DM m⁻² | 21 | **10 – 40** (100–400 kg/ha) |
| | `laii` | m² m⁻² | 0.012 | **0.005 – 0.05** |
| Closure | `laicr` | m² m⁻² | 4.0 | **3.0 – 6.0** |
| Senescence | `rdrl` | d⁻¹ | 0.05 | **0.02 – 0.10** |
| | `rdrshm` | d⁻¹ | 0.03 | **0.01 – 0.05** |
| | `rdrltb` y-ordinates | d⁻¹ | 0.0 – 0.09 | **0 – 0.12** each, non-decreasing in T — widened from 0.05, which the current 0.09 at 50 °C exceeded |
| | `scale_factor_rdr_leaves` | – | 1.0 | **0.6 – 1.6** |
| | `dvsdlt` | DVS | 1.0 | **0.8 – 1.4** |
| N feedback | `nlai` | – | 1.0 | **0.5 – 1.5** |
| | `nsla` | – | 0.5 | **0.2 – 1.0** |
| **Frozen** | `kdiftb`, `scale_factor_kdif` | – | 0.6 | — see the degeneracy note above |

**Target the tail, not the mean.** The canopy is built for most cells (median 4.43 m² m⁻²,
66 % inside 4–6), but 6.2 % of scored cell-seasons peak below 1.5 and 0.29 % never reach
DVS 2. That is a bimodal failure on specific cells: score it as a failure **rate** per
region, not as a mean LAI that hides it.

## Module layout

New package `src/cropmodelling4eu/calibration/`:

| File | Contents |
| --- | --- |
| `regions.py` | `CalibrationRegions` — EnS × calendar × yield clustering, cached |
| `observations.py` | `ObservationSet` — one long table `(unit, year, variable, value, weight, support)` unifying PEP725, CyBench yield, CyBench fpar and (later) CLMS |
| `dataset.py` | `CellYearDataset`, `StratifiedFractionSampler`, `collate_region_batch` |
| `losses.py` | `PhenologyLoss`, `YieldLoss`, `CanopyLoss`, the sigmoid-date helpers |
| `problem.py` | `CalibrationProblem` — per-region `CalibrationManager`, hierarchical pooling, `SimplexGroup` |
| `checkpointed.py` | segmented gradient-checkpointed forward over `compute_rates` / `update_state` |
| `trainer.py` | `Calibrator` — stage loop, Adam, early stop, logging, resumable state |
| `params_{phenology,yield,lai}.yaml` | parameter specs in torchcrop's **existing** `load_calibration_config` schema |

CLI: `cm4eu calibrate --stage phenology --config config.yaml`.

### The dataloader

**The sampling unit is `(observation_unit, harvest_year)`** — an `adm_id`-year for
CyBench, a station-year for PEP725 — not a bare cell. CyBench observations are reported on
`adm_id`, so the loss support and the sampling support should be the same object.

```python
StratifiedFractionSampler(fraction=0.15, strata=("region", "year"), seed=...)
```

- Draws **without replacement within an epoch**, stratified by region **and** by year. An
  unstratified draw can silently produce a "wet years only" epoch, and the parameters will
  chase it.
- Emits **region-homogeneous batches** (forced above), 128–256 cell-years each.
- Deterministic per-epoch seeding, so a killed run resumes the identical sequence.

**Spatial support of the loss:**

| Observation | Support |
| --- | --- |
| PEP725 | **point-matched** to the containing cell — `germany.py:MATCH_KM` already does this |
| CyBench yield, CyBench fpar | **aggregate-then-compare**: simulate the cells of an `adm_id`, area-weight to the adm mean, compare. The weighted mean is linear, so gradients pass cleanly |

That second row is why the sampler draws whole adm units rather than individual cells.

**Weather caching is the biggest speed decision.** The production runner reads each cell's
46-year gzip once; a calibration run revisits the same cells every epoch. Pre-extract the
sampled pool's seasons once into a memory-mapped `float32 [n_cell_years, T, C]` array — and
size it deliberately: 8 000 cells × 25 years × 330 d × 8 vars × 4 B ≈ 210 GB, so the pool
must be restricted to cell-years that carry an observation, and the cache built per pool,
not per domain.

## Splits, diagnostics, acceptance

- **Splits:** 70/15/15, split by **year blocks** (temporal extrapolation) *and* by held-out
  `adm_id` within each region (spatial transfer). Report both.
- **Regularisation:** L2 toward the physical prior — JRC per region for phenology, the
  SIMPLACE `crop.xml` values elsewhere — scaled by the prior's cross-zone spread.
- **Gradient sanity:** finite-difference check per free parameter on a handful of cells
  before any stage runs. Zero or non-finite gradients almost always mean `smooth=False`, or
  a non-differentiable target that slipped into the spec.
- **Identifiability:** Jacobian condition number and the parameter correlation matrix per
  region. Correlation above ~0.95 between two free parameters means one should be frozen.
- **Independent check:** calibrated `tsum1` against the JRC `TSUM1` field. It was never in
  the loss.

Targets are set against the **full run** (CyBench reported units, 2000–2024), SIMPLACE on
the identical pairs. Targets name spreads as well as biases: torchcrop already meets every
pooled bias, and the pooled bias is not what is wrong with it.

The ``torchcrop`` column is the **superseded** parameterisation (see the warning in
*Why*); SIMPLACE's is unaffected, since it was always reading its own crop file.

| Metric | Target | torchcrop (superseded crop) | SIMPLACE, same pairs |
| --- | --- | --- | --- |
| Maturity bias | \|bias\| < 7 d | **+1.0 d** — already met | −5.4 d |
| Maturity **MAE** | < 12 d | **18.2 d** | 11.0 d |
| Maturity per-unit spread | 90 % of units within ±20 d | **69 %** | 90 % |
| Yield bias | \|bias\| < 0.5 t/ha | **+0.71 t/ha** | +0.16 t/ha |
| Yield skill | **r² > 0** | **−0.32** — the binding constraint | −0.27 |
| Peak LAI | median 4–6, **and** < 2 % of cell-seasons below 1.5 | median 4.43 ✓, **6.2 %** below 1.5 ✗ | median 6.43 ✗ |
| Harvest index | median 0.45–0.55, < 5 % above 0.6 | 0.538 ✓, **26 %** above 0.6 ✗ | **0.337** ✗ |
| fAPAR seasonal RMSE | < 0.10 | — | — |

Neither model meets the LAI or harvest-index rows, in opposite directions — SIMPLACE builds
too much canopy and converts too little of it, torchcrop the reverse. Passing stage 3 means
landing between them, not matching SIMPLACE.

**No heading target.** The current `de_smoke` outputs carry no `anthesis_doy` on either
side, so `germany.PHASES[51]` has nothing to pair. Restore the heading date to the run
schema and add the row.

## Blockers to clear first

1. ~~**`idsl = 2` has no override path on the torchcrop side.**~~ **Cleared.** torchcrop is
   harmonised against the SUSTAg `WW` block and now runs `idsl = 2` with `versat = 70`,
   `vbase = 14` and SIMPLACE's own `vernrt` curve — see *Why*. **Every run before
   2026-09-04 used a different crop on each side**, so no earlier §6 comparison is a model
   comparison.
2. ~~**Neither run emits an emergence date.**~~ **Cleared.** SIMPLACE's `WLOutputs` now
   writes `Phenology.EmergenceDOY` / `AnthesisDOY` / `MaturityDOY`, and
   `torchcrop/run.py` writes `days_to_emergence` from the `tsump = tsumem` crossing.
   Stage 1 has an emergence target, and `tsumem` a gradient path, on both models.
   **Existing runs predate the change** — re-run before scoring emergence.
3. **The run this document is specified against is being re-made.** Blockers 1 and 2 both
   changed the inputs, so the numbers in *Why* and in *Splits, diagnostics, acceptance*
   describe a configuration that no longer exists. Stage 1 should not start until
   `full_run_evaluation.ipynb` has been re-run on the harmonised crop. Note that the
   re-run moves **two** things at once — the crop parameters and the emergence output — so
   `stresstest_evaluation.ipynb` is where a change is attributed to one or the other.
4. **The heat-stress group is unmapped.** `vTCritical` / `vStartDVS` / `vEndDVS` have
   torchcrop counterparts and are not in `SCALARS`, so the two models differ by 4 °C and
   0.05 DVS on a group stage 2 frees. Add them before stage 2 runs — see
   *What harmonisation does not reach*.
5. **Initial mineral N** at ~243 kg/ha median makes stage 2's N group unidentifiable
   (pooled `nni` 0.96). Fix `soil.mineral_n_fraction` in the export, or drop the group and
   record why.
6. **CLMS phenology** needs a CDSE account before the bulk route opens — see
   [`CLMS_DOWNLOAD.md`](../src/data4simplace/CLMS_DOWNLOAD.md). Not on the critical path: stages 1–3 can start
   on PEP725 + CyBench, and CLMS widens stage 1 beyond Central Europe when it lands.
