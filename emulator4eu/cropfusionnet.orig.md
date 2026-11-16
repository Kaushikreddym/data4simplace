# Task: Implement a multi-output CropFusionNet-based crop model emulator (daily + yearly outputs)

## Context

I want to train a deep-learning **emulator of a process-based crop model**. Inputs are daily forcing time series plus static site/management descriptors. Outputs are (a) **daily simulated state/flux variables** of the crop model and (b) **yearly (seasonal) summary outputs**.

The architecture should be based on **CropFusionNet**, an interpretable Temporal Fusion Transformer variant for district-level crop yield forecasting:

- Paper: Srivastava et al. (2026), *CropFusionNet: an interpretable deep learning framework for uncertainty-aware crop yield forecasting across Germany*, Artificial Intelligence in Agriculture. https://doi.org/10.1016/j.aiia.2026.08.016 (ScienceDirect: https://www.sciencedirect.com/science/article/pii/S2589721726001029)
- Code: https://github.com/geonextgis/CropFusionNet (model in `src/models/CropFusionNet/model.py`, dataset in `src/dataset/dataset.py`, loss in `src/loss/loss.py`, training in `src/train/train_CropFusionNet.py`, configs in `src/config/`)

**Step 0: read both before writing code.** Clone the repo, read `model.py`, `dataset.py`, `loss.py`, `train_CropFusionNet.py` and one config, and summarise the forward pass back to me in ≤15 lines before implementing.

### What CropFusionNet does (for orientation; verify against the code)

Per-variable embeddings (`Linear(1, emb_dim)` per real variable, `nn.Embedding` per categorical) → static GRN context + static Variable Selection Network (VSN) → temporal VSN conditioned on static context, with per-variable and per-timestep masks → multi-scale depthwise Conv1d (k=3,5,7) + residual → LSTM with h0/c0 projected from static context → GLU/LayerNorm skip → sinusoidal positional encoding + multi-head self-attention with key padding mask → static enrichment GRN → multi-head attention pooling ("DynamicPyramidalPooling") → projection → quantile head (pinball loss, q = 0.1/0.5/0.9). It predicts **one** end-of-season target from **one** season of daily inputs (~270–400 days), with phenology-based masking.

## What to change

### 1. Two output heads
- **Daily head (sequence-to-sequence):** from the per-timestep representation (after static enrichment), predict `n_daily` crop-model variables at every day, e.g. LAI, above-ground biomass, storage-organ biomass, development stage, root-zone soil water, actual ET, N uptake (final list via config). Use a shared GRN + `Linear(hidden, n_daily)` applied per timestep.
- **Yearly head:** keep attention pooling → projection, but output `n_yearly` targets, e.g. yield, total biomass, seasonal ET, N leaching, anthesis/maturity DOY (list via config).
- Both heads are trained jointly. Daily supervision is the main point: it gives dense physical supervision and should improve generalisation and extreme-year behaviour of the yearly targets.

### 2. Causality option
Crop state at day *t* depends only on forcing up to *t*. Add `causal: bool` (default **True**):
- Conv1d: left-padding only (no centred padding as in the original).
- Self-attention: causal attention mask in addition to the padding mask.
- LSTM is already unidirectional; keep it.
- The yearly pooling head may still see the whole season.
Report validation results for `causal=True` vs `False`.

### 3. Season mode vs continuous multi-year mode (config switch)
- `mode="season"`: one sample = one season window (as in CropFusionNet: from N days before sowing to harvest, configurable), one set of yearly targets. This is the default and the first thing to get working.
- `mode="continuous"`: one sample = a multi-year daily series of a continuous crop-model run (soil water/N/SOC carry over between years). Yearly outputs come from **per-year segmented attention pooling**: give each timestep a year index, compute masked softmax pooling within each year segment (vectorised, e.g. one-hot `[B,T,Y]` masks + einsum), and output `[B, Y, n_yearly]`. Full self-attention over ~10k+ days is too expensive. In this mode, use attention only within windows (e.g. per season or a sliding window, configurable) or turn it off, and rely on the LSTM for cross-year memory. Use truncated BPTT or chunking if memory requires it.

### 4. Masking
- `valid_mask [B,T]`: padding/out-of-window days.
- `variable_mask [B,T,F_in]`: missing inputs (keep the original semantics).
- `daily_target_mask [B,T,n_daily]`: daily targets can be undefined (e.g. LAI before emergence or after harvest, NaN in crop-model output). The loss must ignore these.
- `yearly_target_mask [B,(Y,)n_yearly]`: missing yearly targets.

### 5. Inputs
- Dynamic daily: weather forcings (Tmin, Tmax, Tavg, precip, radiation, VPD/RH, wind, ET0, CO₂). Management events (sowing, fertilisation amount, irrigation, tillage) go in as **daily pulse channels**, not static scalars. Optional time-varying categoricals (e.g. crop type in rotations).
- Static: soil (texture, AWC, SOC, depth, bulk density), cultivar parameters, latitude/elevation, management levels; real and categorical.
- **No remote-sensing inputs**: this is an emulator, so inputs are exactly what the crop model sees.

### 6. Loss
- Standardise every target with training-set statistics (scalers fit on train only, saved to JSON as in the original).
- `L = w_daily * masked_mean(L_daily) + w_yearly * masked_mean(L_yearly)`, with per-variable weights in config. Implement learned homoscedastic uncertainty weighting (Kendall et al., 2018) as an option.
- Point loss: MSE or Huber (configurable). Keep quantile/pinball loss as an option for the yearly head. If quantiles are used, prevent quantile crossing (e.g. predict median + softplus-positive offsets).
- Optional **consistency penalty**: yearly yield vs. the daily storage-organ biomass at harvest (or other configured pairs), so the two heads agree.

### 7. Fixes/cleanups relative to the original code
- `x.get("variable_mask", None).to(...)` crashes when the mask is absent; handle `None`.
- Vectorise the per-variable GRNs in the VSN (grouped linear / batched einsum) instead of the Python loop; emulators may have 20–40 inputs.
- Remove `.to(self.device)` calls inside `__init__`; move the model once.
- `PositionalEncoder` `max_seq_len` must cover the longest sequence or window.
- Wire in optional extreme-year oversampling (`WeightedRandomSampler` based on the yearly target tails). The original defines `get_sample_weights()` but does not appear to use it.

### 8. Data pipeline
- Do not assume a file format. Write a `Dataset` against a clear interface: per sample, return `dynamic [T,F_in]`, `static [F_s]`, `daily_targets [T,n_daily]`, `yearly_targets`, the masks, `year_index [T]` (continuous mode), and metadata (site id, years).
- Provide an adapter for **NetCDF/xarray** and one for **Parquet**, plus a **synthetic data generator** (toy crop model: GDD-driven phenology, water-limited biomass growth, heat-stress penalty around anthesis, soil-water bucket with carryover) so everything runs end-to-end without my data.
- Phenology masks come from the crop model's own simulated sowing/harvest dates when available, otherwise from config.

### 9. Splits and evaluation
- Split options: leave-sites-out (spatial blocks), leave-years-out, and a dedicated **extreme-years holdout** (lowest-decile yield years or user-listed years, e.g. 2003, 2018, 2022). Never random sample splits.
- Metrics per target: R², RMSE, MAE, bias. For daily outputs also NSE/KGE per trajectory and error by development stage.
- Tail metrics: errors for the lowest 10% of yearly yields. If quantiles are used, interval coverage overall and in the tail.
- Physical sanity checks: response curves of yearly yield to +ΔT, ×precip, CO₂ steps, with the emulator compared against the crop model when reference runs are provided.
- Baseline: gradient boosting (LightGBM/XGBoost) on seasonal aggregate features for yearly targets, plus a plain LSTM for daily targets. Report the emulator vs. the baselines.
- Interpretability: export static/temporal VSN weights, attention-pooling weights, and per-target integrated gradients.

### 10. Engineering
- PyTorch ≥ 2.x. Lightning is optional; a plain loop is fine. Config via YAML or dataclasses (one config per crop/model).
- Mixed precision, gradient clipping (as original, max_norm=5), ReduceLROnPlateau, early stopping, checkpointing, seeding.
- Repo layout: `models/`, `data/`, `losses/`, `train.py`, `evaluate.py`, `configs/`, `tests/`.
- Tests (pytest):
  - shape tests for both modes;
  - masks really exclude padded steps and NaN targets from the loss;
  - causality test: perturbing inputs at day t must not change daily predictions at days < t when `causal=True`;
  - overfit-one-batch test;
  - synthetic end-to-end run.

## Deliverables (in order, stop and report after each)
1. Summary of the original forward pass and a list of planned changes.
2. Model + losses + unit tests passing on random tensors.
3. Synthetic-data pipeline and a short training run showing loss decreasing for both heads, in season mode.
4. Continuous multi-year mode with per-year segmented pooling, tested on synthetic carryover data.
5. Evaluation script with metrics, tail metrics, response curves and baselines; README with usage.

## Things to ask me rather than guess
- Which crop model (e.g. MONICA/APSIM/DSSAT/LPJmL), and whether it resets each season or runs continuously.
- Final lists of daily and yearly output variables, and which can legitimately be zero (crop failure).
- Data format, sizes (sites × years × parameter sets), and available GPU.