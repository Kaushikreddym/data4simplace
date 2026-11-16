# Task: CropFusionNet emulator of torchcrop winter wheat on the CMIP6 production runs

The generic spec this was adapted from is `cropfusionnet.orig.md`. This version
pins every choice that spec left open to what exists in this project on
2026-11-09. Where the original's guidance still applies unchanged (forward-pass
reading, masking semantics, code fixes, engineering), it is summarised here,
not repeated.

## Context

**Purpose: explainability.** We want to know *which forcing variables, in
which part of the season, drive torchcrop's winter-wheat yield and its CMIP6
changes*. We also want to separate CO₂ from warming and from the
water/VPD/heat signals, per region and in extreme seasons. The emulator is the
vehicle: CropFusionNet learns this from the raw daily forcing through its
static and temporal variable selection networks (VSNs), its attention
pooling, and per-day integrated gradients. Fast scenario sweeps are a
by-product, not the goal.

### What it replaces, and the bar it has to clear

`cropmodelling4eu/notebooks/production/cmip6/explain.py` already does this
with a **LightGBM emulator on hand-made season indices + TreeSHAP**. The
target is the yield anomaly against the cell's 1985–2014 historical mean.
Its tables are in `runs/.../cmip6/analysis/explainability/tables/`:

| Held out | R² | Bias |
|---|---|---|
| random 20% of years | 0.70 | +0.2% |
| one GCM | 0.65–0.71 | −1.4 … +3.6% |
| hist-GHG / historical / ssp126 | 0.62–0.64 | ≤ 1.3% |
| **hist-aer** | **0.35** | **+11.2%** |

By mean |SHAP|: CO₂ 14.0 > water 8.4 > VPD 7.0 > temperature 4.6 > context
4.3 > heat 4.0 > radiation 1.8 > heavy precipitation 1.8 > frost 0.4.

Its limits are what CropFusionNet should fix:
1. **Fixed windows.** The phases and indices are chosen before the fit, so
   the analysis cannot discover *when* a variable matters.
2. **CO₂ and warming confounded.** SHAP splits credit between correlated
   inputs arbitrarily, so the two separate only through hist-aer, and
   hist-aer is exactly where it fails (R² 0.35).
3. **Explains seasonal aggregates, not the daily process.**

### The model being explained is differentiable: use that as ground truth

torchcrop is a PyTorch model, so **exact attributions of torchcrop itself**
(integrated gradients of yield w.r.t. every daily input) are computable by
autograd on a sample of seasons. The emulator's VSN and attention weights are
*learned importances of the emulator*. TFT-style selection weights are known
not to be guaranteed faithful. Therefore:

- **torchcrop IG** is the reference attribution, on a small sample (~2 k
  seasons per GCM × experiment; the calibration code already backpropagates
  through full seasons).
- **Emulator explanations** (VSN weights, attention pooling, emulator IG)
  are trusted only where they agree with torchcrop IG on that sample. If they
  do, the emulator gives them cheaply over the full 10 M-season panel and
  across regions, extremes and experiments.
- **LightGBM + SHAP** stays as the baseline both are compared against.

Architecture: **CropFusionNet** (Srivastava et al. 2026,
https://doi.org/10.1016/j.aiia.2026.08.016; code
https://github.com/geonextgis/CropFusionNet). It is a TFT variant: per-variable
embeddings, then static GRN and VSN, then temporal VSN, multi-scale Conv1d,
LSTM, self-attention, static enrichment, attention pooling, and a quantile
head. This project adds a daily head, a causal option, and the fixes listed in
§8.

**Step 0 (unchanged):** clone the repo and read `model.py`, `dataset.py`,
`loss.py`, `train_CropFusionNet.py` and one config. Summarise the forward pass
in at most 15 lines before writing code.

## 1. What already exists (source of truth)

| Item | Where | Notes |
|---|---|---|
| CMIP6 torchcrop runs | `Data/SIMPLACE/cropmodelling4eu/runs/winter_wheat/calibrated/cmip6/<GCM>/<exp>/torchcrop/` | 4 GCMs (ACCESS-ESM1-5, IPSL-CM6A-LR, MIROC6, MRI-ESM2-0) × 5 experiments (historical, hist-GHG, hist-aer, ssp126, ssp370). All complete. GFDL is not exported. |
| Yearly targets | `torchcrop_europe.parquet` (one row per `SimplaceID × year`) | ~10.6 M rows for a 1851–2014 run (64,876 cells). ACCESS historical-type runs cover 1951–2014 only. |
| Daily targets, CMIP6 | **none** | CMIP6 runs were made with `TC_DAILY=0` (`submit_production.sh`, `CMIP6_TC_DAILY`). |
| Daily targets, MSWX | `runs/.../mswx/torchcrop/shards/torchcrop_daily_shard_*.parquet` | Long format `(SimplaceID, year, date, doy, das, variable, value)`, with LAI, AGB, NNI and TRANRF; 2000–2024; 20 shards, 4.2 GB. |
| Daily forcing | `cropmodelling4eu_inputs/exports/cmip6/<GCM>/<exp>/weather/` | One gzipped TSV per cell, `Date, Precipitation, TempMin, TempMean, TempMax, Radiation, Windspeed, RefET(-99.9), Gridcell, RelHumCalc`; ~65 GB per run. |
| CO₂ | `<export>/site/co2.csv` | Global annual. **hist-aer holds 284 ppm**; historical and hist-GHG reach 398 ppm by 2015; ssp126 reaches 446 and ssp370 867 by 2101. |
| Static | `<export>/soil/soil.csv` (6-layer wide SIMPLACE profile), `<export>/site/site.csv` (lat, lon, altitude, calendar), `management/fertilizer_winter_wheat.csv` (N/P/K by DVS), `management/irrigation_class_cereals.nc` | Identical across GCMs and experiments, because they come from the same base export. |
| Region crops | `winter_wheat_2000_2024_euval/calibration/regions.parquet` + `calibration/yield/crops/crop_region_<NN>.yaml` | 27 calibrated parameter sets; each cell runs one of them. |
| Sowing | `cropmodelling4eu_inputs/sowing_climatology_simplace_euval.csv` | **Fixed per cell, the same DOY every year** (median of MSWX SIMPLACE). Sowing does not respond to CMIP6 forcing. |

### Known defects the emulator inherits (do not "fix" them in the emulator)

- **Null cells:** about 4–6 k cells per GCM have all-−99.9 forcing and null
  torchcrop output. **Train and evaluate on the common 59,304-cell mask only.**
- **Inconsistent temperatures:** Tmin > Tmax on about 1.2% of days, from
  per-variable bias correction. torchcrop saw these values, so the emulator
  must see them too; add a QC flag channel rather than reordering.
- **Wind:** wind is the MSWX day-of-year climatology (`scenario.json` →
  `filled_from_mswx_climatology`), so it carries no interannual signal. Keep it
  as an input for fidelity, but expect near-zero attribution to it.
- **RefET:** always −99.9. Drop it, or derive ET0 in the dataset if wanted.

## 2. Scope decisions (pinned)

| Original option | Decision for CMIP6 | Why |
|---|---|---|
| Crop model | **torchcrop** only | It is the only crop model with CMIP6 output. SIMPLACE CMIP6 is being run now (MIROC6/hist-aer, array 872154) and can be added later as a second target with the same pipeline. |
| `mode` | **`season` only** | torchcrop starts every season fresh, so nothing carries over between years. Continuous mode becomes relevant only for SIMPLACE (soil water/N carry-over); keep the interface but defer the implementation (original Deliverable 4 → optional). |
| Causal | `causal=True` default; report True vs False | Unchanged. |
| Remote sensing | none | Unchanged: the inputs are exactly what torchcrop reads. |

## 3. Samples

One sample is one `(GCM, experiment, SimplaceID, harvest year)`:

- **Window:** from `sowing_doy − pre_sowing_days` (config, default 30) in year
  y−1 to a fixed end, 31 Aug of year y. The end is the latest torchcrop
  maturity in the euval run (Jun/Jul/Aug), plus margin. T ≈ 370 days, padded
  to `max_seq_len = 400`.
- **Phenology masks:** take emergence and maturity from torchcrop's own
  `days_to_emergence` and `days_to_maturity` in the yearly table, and use them
  only for `daily_target_mask`. The **inputs** are never cut at maturity,
  because the emulator must predict maturity itself.
- **Domain: all of Europe** (decided 2026-11-09). Every cell of the common
  59,304-cell mask is in the data, the daily targets and the inference maps;
  no fixed cell subset. About 59 k cells × (164 or 86 years) × 20 runs ≈ 150 M
  seasons. Sub-sampling is a property of the **sampler**, never of the data:
  1. **Per epoch:** draw a fresh season sample stratified over the
     27 calibration regions × soil texture class, so every cell is seen across
     epochs and no region dominates by cell count.
  2. **Extremes:** `WeightedRandomSampler` oversampling of each region's
     lowest-decile yield seasons (original §7).
  3. Held-out cells/regions for the spatial split (S4) are excluded from the
     sampler, not from the data.

## 4. Data pipeline (replaces original §8)

The CSVs cannot be read at training time: there are ~1.3 TB of gzipped text
across 20 runs. Add an offline **compaction step**, one SLURM array task per
run, within the 15-job cap.

1. `emulator4eu prepare --run cmip6/<GCM>/<exp>` reads
   each cell's weather once and writes a **Zarr** store
   `forcing/<GCM>/<exp>.zarr`. The store holds `dynamic[cell, day, var]` in
   float32, `qc[cell, day]`, and `co2[day]` broadcast from annual. The chunk
   is (64 cells, full time, all vars), so one read serves many seasons. The
   time axis is the run's own calendar: check MIROC6/IPSL for no-leap
   calendars before assuming 365.25.
2. `static.parquet`, once for the domain (static inputs are run-independent):
   - soil per layer: FC, WP, saturation, bulk density, carbon, N, texture;
     plus the depths;
   - site: lat, altitude;
   - management: N/P/K totals and the DVS of each event;
   - irrigation class (vIRR);
   - **calibration region id** as a categorical;
   - optionally, the region's crop parameters as real static inputs, so the
     emulator can later be queried for unseen parameter sets.
3. `targets_yearly/<GCM>/<exp>.parquet`: `torchcrop_europe.parquet` as is.
4. `targets_daily/<GCM>/<exp>.parquet`: only after the daily rerun (§5).
5. `Dataset` against the interface in the original spec: `dynamic [T,F_in]`,
   `static [F_s]`, `daily_targets [T,n_daily]`, `yearly_targets [n_yearly]`,
   `valid_mask`, `variable_mask`, `daily_target_mask`, `yearly_target_mask`,
   and metadata `(gcm, exp, SimplaceID, year)`. It slices the season window
   from Zarr using the cell's sowing DOY.
6. **Synthetic generator** (toy GDD phenology + water-limited growth + heat
   penalty) stays, for unit tests and CI only.

### Dynamic channels (F_in ≈ 12)

| Channel | Source |
|---|---|
| tmin, tmean, tmax, precip, radiation, rh, wind | weather TSV (units exactly as torchcrop reads them, after its own conversions) |
| co2 | `site/co2.csv`, daily broadcast |
| doy sin/cos | derived |
| sowing pulse | 1 on sowing day |
| qc_tminmax | 1 where Tmin > Tmax |

Fertilisation is DVS-triggered in torchcrop, so its dates are model output, not
input. It goes in as **static amounts + DVS thresholds**, not as daily pulses
(this deviates from original §5).

## 5. Targets

### Yearly head (`n_yearly = 8`, from `torchcrop_europe.parquet`)

`yield_t_ha`, `biomass_g_m2`, `max_lai`, `days_to_emergence`,
`days_to_anthesis`, `days_to_maturity`, `tranrf_mean`, `nni_mean`.

- `heat_stress_factor` is optional: check that it varies before adding it.
- `adjusted_yield_*` is a deterministic transform of yield; do not predict it
  separately.
- Yield and biomass can legitimately be 0 (crop failure). Use a hurdle option,
  P(failure) + conditional amount, if zeros exceed 1%.

### Daily head (`n_daily = 5`)

`LAI`, `AGB`, `DVS`, `NNI`, `TRANRF`; `run.py` already supports DVS.

**This needs new torchcrop runs** over the **full domain**, for all 20 CMIP6
runs and MSWX (whose existing daily shards lack DVS). The existing daily path
holds a whole shard in memory, ~0.7 G rows for one CMIP6 historical region
shard, so it must stream to disk before it can run at CMIP6 length. Until those
reruns exist:
- train the daily head on **MSWX daily** (2000–2024, same cells), and
- train the yearly head on CMIP6.

The daily head is masked off for CMIP6 samples until then (`w_daily = 0` per
sample).

### Consistency pairs (original §6)

- `yield_t_ha` ↔ storage-organ biomass at maturity is not available, because
  WSO is not emitted. Add `WSO` to the daily variables in the rerun, or use
  the pair `biomass_g_m2` ↔ `AGB` at maturity instead.
- `days_to_maturity` ↔ the first day `DVS ≥ 2`.

## 6. Splits (replaces original §9 split options)

Never random. All splits use the common 59,304-cell mask.

| Split | Train | Test | Question it answers |
|---|---|---|---|
| **S1 leave-GCM-out** (4 folds) | 3 GCMs, all experiments | the 4th GCM | Does it generalise to an unseen climate model's bias structure? |
| **S2 forcing extrapolation** | historical, hist-GHG, hist-aer, ssp126 | **ssp370** (CO₂ up to 867 ppm, the warmest climate) | Does it extrapolate beyond the training CO₂ and temperature range? This is the key test for sweeps. |
| **S3 attribution** | historical, ssp* | hist-GHG, hist-aer | Can single-forcing runs be reproduced? hist-aer is the only one at 284 ppm CO₂ throughout. |
| **S4 spatial blocks** | 22 regions | 5 held-out calibration regions | Does it generalise to an unseen soil, climate and parameter set? |
| **S5 extreme seasons** | — | each cell-run's lowest-decile yield seasons, evaluated separately inside S1/S2 | Fixed calendar years (2003, 2018) have no meaning in GCM time, so define extremes by decile. |

Report **S3 (hist-aer held out) and S1** as the headline numbers. For
explainability, hist-aer is the decisive split: it is the only experiment
where CO₂ (fixed at 284 ppm) and climate change are decoupled, and it is
where the LightGBM baseline fails (R² 0.35, +11% bias). An emulator that
cannot predict hist-aer has not separated CO₂ from warming, and its
attributions of the two cannot be trusted. Within any one run CO₂ hardly
varies across cells, so the CO₂ effect is learnt only *across* experiments
and years.

### Explainability outputs and acceptance (the deliverable that matters)

| Output | Method | Validated against |
|---|---|---|
| Global importance by variable group (same groups as `explain.py`: CO₂, temperature, water, VPD, heat, radiation, wet, frost, context) | static and temporal VSN weights; emulator IG summed per group | torchcrop IG, then LightGBM SHAP |
| **Seasonal importance profiles**: which variable matters on which day relative to sowing, or by DVS stage | per-day temporal VSN weights × IG, aligned on DVS from the daily head | torchcrop IG per day |
| Critical periods | attention-pooling weights over the season | torchcrop IG mass by phase |
| CO₂ vs warming decomposition of the CMIP6 yield change | counterfactuals: swap one input group to its historical-baseline value (CO₂ → 284, or climate → baseline) | the same counterfactual run in torchcrop on the sample |
| Regional and extreme-season anatomy | the above split by calibration region and by lowest-decile seasons | torchcrop IG on the same strata |

Acceptance:
1. **Fidelity:** beats LightGBM on every split, and clearly on hist-aer.
2. **Faithfulness:** rank correlation ≥ 0.8 between emulator and torchcrop
   IG at group level, and seasonal profiles that peak in the same phases.
3. **Counterfactuals:** the emulator's CO₂/warming split of the ssp370 yield
   change is within ±20% of torchcrop's.

## 7. Evaluation (original §9, specialised)

- **Per target:** R², RMSE, MAE, bias. Daily targets: NSE/KGE per trajectory
  and error by DVS stage.
- **Tail:** errors on the lowest-10% yields; interval coverage if quantiles
  are used.
- **Response curves against torchcrop itself:** pick about 200 cells, run
  torchcrop and the emulator on +1/+2/+4 K, ×0.8/×1.2 precip and CO₂ ∈
  {284, 400, 550, 700, 867}, then compare d(yield)/d(forcing). torchcrop is
  cheap and differentiable, so also compare **gradients**: torchcrop's
  autograd ∂yield/∂input against the emulator's integrated gradients.
- **Aggregates the CMIP6 notebooks use:** for each GCM × experiment, the
  regional-mean yield time series and trend, and the risk ratio of
  low-yield seasons, computed from the emulator and from torchcrop
  (notebooks under `cropmodelling4eu/notebooks/production/cmip6`). An
  emulator that matches per-season R² but distorts the trend or the risk
  ratio has failed for this purpose.
- **Baselines:**
  - LightGBM on seasonal aggregates (GDD, precipitation sums by phase, heat
    days > 30 °C, CO₂, static inputs) for yearly targets;
  - plain LSTM for daily targets.
- **Interpretability:** export static and temporal VSN weights, the
  attention-pooling weights, and integrated gradients per target.
  Sanity-check that wind gets near-zero weight (§1).

## 8. Model, loss, code fixes, engineering

Unchanged from the original spec §1, §2, §4, §6, §7 and §10:
- two heads; causal Conv1d padding and attention mask;
- the four masks;
- standardised targets with train-only scalers (JSON);
- `L = w_daily·L_daily + w_yearly·L_yearly`, optional Kendall weighting;
- Huber default; quantiles only on the yearly head, median + softplus offsets;
- the five code fixes, including the vectorised VSN and no `.to(device)` in
  `__init__`;
- AMP, `clip_grad_norm_(5)`, ReduceLROnPlateau, early stopping, seeding.

Project specifics:

- **Package:** `emulator4eu/` in this repo, PyPA src layout
  (`src/emulator4eu/{models,data,losses}/`, `train.py`, `evaluate.py`,
  `prepare.py`, `configs/`, `tests/`). Pydantic config (the same style as
  `cropmodelling4eu.config`). CLI: `emulator4eu prepare|train|evaluate`.
- **Hardware:** the `gpu` partition has 4 × V100 (48 cores, 95 GB RAM) and
  1 × H100 (128 cores, 768 GB RAM); no GRES is configured, so request the node.
  Develop on V100 (fp16 AMP, since V100 has no bf16). Use the H100 for the full
  S1/S2 trainings. Zarr reads go via `num_workers ≥ 8`. Full-domain forcing is
  ~59 k cells × 60 k days × 12 × 2 B (fp16 on disk) ≈ 85 GB per historical run,
  ≈ 1.4 TB for all 20 runs: too large to stage, so read from /data01 with
  cell-chunked Zarr and a per-epoch sampler that is chunk-aware (draw cells by
  chunk, seasons within).
- **Cluster:** all prepare and torchcrop-rerun arrays count towards the
  15-concurrent-job cap.
- **Tests:** shapes; mask exclusion; causality (perturbing day t leaves days < t
  unchanged); overfit one batch; synthetic end-to-end. Add one: the `prepare`
  round trip reproduces the exact torchcrop inputs for 3 cells (compare with
  torchcrop's own decoded weather cache).

## 9. Deliverables (stop and report after each)

1. Forward-pass summary of the original and the list of changes (Step 0).
2. Model + losses + unit tests on random tensors.
3. `prepare` for one run (MIROC6/historical, full domain) and its round-trip
   test.
4. Season-mode training on MIROC6 (all 5 experiments), yearly head only:
   losses falling, first S2-style result (train MIROC6 non-ssp370, test
   MIROC6 ssp370).
5. **Reference attributions:** integrated gradients of torchcrop's yield
   w.r.t. its daily inputs on ~2 k seasons per GCM × experiment (the same
   cells as the emulator), plus the CO₂/climate counterfactual torchcrop
   runs. This can run in parallel with 3–4 because it needs no emulator.
6. Daily-head support: MSWX daily first, then the full-domain torchcrop `TC_DAILY=1`
   reruns. Joint training. The DVS daily output aligns
   the seasonal importance profiles on phenology.
7. All 4 GCMs; S1–S5; the explainability outputs and acceptance tests of §6,
   against torchcrop IG and `explain.py`'s SHAP; LightGBM/LSTM baselines;
   README.
8. (Optional) SIMPLACE CMIP6 as a second target, plus continuous mode.

## 10. Open questions for the user

1. ~~Purpose~~: **answered 2026-11-09, explainability.** Acceptance is the
   faithfulness test against torchcrop IG plus hist-aer fidelity (§6).
   Should the target be absolute yield, or the yield anomaly against the 1985–2014
   historical mean as in `explain.py`? The anomaly is directly comparable with
   the existing SHAP results; absolute yield is what the daily head
   integrates to. The default proposed here: predict absolute yield and
   derive the anomaly.
2. ~~Daily rerun~~: **approved 2026-11-09, full domain, torchcrop and
   SIMPLACE** (preparation in progress; see §5).
3. Should region crop parameters be static inputs (the emulator generalises to
   new parameter sets, which calibration would need) or only a region id
   (simpler)?
4. ~~Cell subset~~: **answered 2026-11-09, full Europe.** The emulator is
   trained, evaluated and mapped over all cells; subsampling lives only in the
   sampler.
