# Learned sowing dates for torchcrop

!!! info "Status — 2026-11-13"
    All six arms (A0, A1, B1, B0, C, D) are complete and evaluated in
    `notebooks/sowing_comparison/sowing_comparison.ipynb`.

## Question

Can a model that sees only **past** weather choose a better sowing day than
SIMPLACE's sowing rule? Here "better" is judged against corrected satellite
emergence (CLMS) and ground-observed sowing (PEP725). And does the choice
change torchcrop's simulated maturity and yield?

The experiment covers **torchcrop only**, forced by MSWX for the 2000–2024
harvest years over the EU_val domain (70,705 cells).

## Why the rule needs replacing

Both models currently sow on the date table that SIMPLACE's `SowingRule`
produces (`sowing_from_simplace_euval.csv`). The rule sows on the first day of
the SAGE window on which **any one** of three conditions holds:

| Condition | Threshold |
| --- | --- |
| 7-day mean Tmin | ≤ 12 °C |
| 3-day rain sum | ≥ 20 mm |
| Top 5 cm soil moisture | ≥ WP + 0.5 · (FC − WP) |

If none holds, sowing is forced on the last day of the window.

The temperature condition is already true when most European windows open, so
the rule **saturates**: 72 % of cell-years sow on the window's opening day
(+1 d, because of the rule's one-day latch). Evaluation against PEP725 in the
production notebook (§7) gives:

- **Bias:** −7.3 d overall, and about −30 d in AT, SK, PL, HU and BE.
- **Year-to-year correlation:** 0.00. The dates carry no weather signal; they
  reproduce the SAGE national blocks.
- **Dependence on simulated soil water:** the moisture condition reads
  simulated soil water, so changing the crop file changes the date.
  Production and euval disagree on 8 % of cell-years.

CMIP6 runs sow on the per-cell median of the same table, so they inherit the
national blocks.

## Method

Code: `cropmodelling4eu.sowing` (CLI `cm4eu sowing …`). The method is
described in full in `cropmodelling4eu/SOWING.md`.

### Model

A **discrete-time hazard network**. For each candidate day *s* in the window
(DOY 193 of the year before harvest, up to DOY 365), it predicts the
probability *h<sub>s</sub>* of sowing that day, given that sowing has not
happened yet:

```text
P(sow = s) = h_s · Π_{t<s} (1 − h_t),     h_365 = 1   (sowing forced on the last day)
```

Its inputs come from three branches, each covering a different time scale:

| Branch | Inputs | Role |
| --- | --- | --- |
| Short term | Causal TCN over daily Tmin, Tmax, Tmean, rain, radiation, vapour pressure and wind; receptive field 21 d | The weather-triggered part of the decision |
| Medium term | Topsoil bucket model, Tmin7, Rain3, Rain30, dry-spell length, degree days | State a farmer would judge (field trafficable, seedbed moist) |
| Long term | Autumn climate normals over Y−31 … Y−2; location, SAGE window, topsoil FC/WP and texture | The local baseline date; modulates the short-term branch (FiLM) |

!!! note "Strictly causal"
    The network never sees weather after a candidate sowing day. The weather
    between sowing and emergence enters only the loss, through a fixed
    emergence clock, just as it does inside torchcrop.

### Training target

There are no gridded sowing observations, so the network is trained on
**emergence**. Each candidate sowing day *s* is mapped to an emergence day
*e(s)* by torchcrop's own thermal clock. Emergence happens when the cumulated
`clamp(Tmean − tbasem, 0, teffmx − tbasem)` from the sowing day reaches
`tsumem`. This step reproduces torchcrop to < 0.1 d in 99.96 % of production
cell-seasons.

The loss is the negative log-likelihood of observed emergence under the
mixture Σ<sub>s</sub> P(sow = s) · Laplace(e(s), b). Its three terms are:

- **CLMS emergence:** CPMCE corrected by −6 d, gridded, 2017–2024.
- **PEP725 emergence:** BBCH 10, stations.
- **PEP725 sowing:** BBCH 0, stations, scored directly against the sowing
  distribution with a fixed 3 d Laplace scale.

### `tsumem`

`tsumem` is fitted **once**, globally, on PEP725 sowing→emergence pairs from
the training blocks only. The fit is **128 °C·d** with `tbasem` 0: 22,518
station-seasons, MAE 5.2 d, simulated lag 11.4 d against 13.0 d observed. The
calibrated crops carry 185–200 °C·d per region, a value that partly
compensated for the rule's early sowing.

### Hold-out

| Data | Split |
| --- | --- |
| PEP725 | 1° station blocks, 70 / 15 / 15 train / validation / test, drawn once (`pep725_split.parquet`) |
| CLMS / CyBench | The calibration's own `phenology_split.parquet`, so the comparison shares its test units |

## Comparison arms

Every arm is a full torchcrop MSWX run into
`runs/winter_wheat/sowing_comparison/<arm>/`. Each arm records its inputs in a
`provenance.json` (sowing table with sha256, crops, `tsumem` source, overrides).

| Arm | Sowing | `tsumem` | Purpose |
| --- | --- | --- | --- |
| **A0** | SIMPLACE rule (euval table) | calibrated, per region | The old method |
| **A1** | SIMPLACE rule | PEP725 fit (128) | `tsumem` effect alone |
| **B1** | Network | calibrated | Sowing effect alone |
| **B0** | Network | PEP725 fit | **The new method** |
| **C** | Per-cell median of the network table | PEP725 fit | What the network's weather response adds; the CMIP6 analogue |
| **D** | Per cell-season the sowing that maximises simulated yield | PEP725 fit | Upper bound, not a method |

The 2 × 2 of A0, A1, B1 and B0 separates three things: the sowing effect
(B1 − A0 and B0 − A1), the `tsumem` effect (A1 − A0) and their interaction.
B0 − C isolates the year-to-year weather response.

**Arm D.**

- **What it runs:** B0's sowing table shifted by −42, −28, −14, +14, +28 and
  +42 d, clipped to DOY 200–365. Each shift is a full torchcrop run under
  `D_grid/`, and B0 itself serves as the zero shift.
- **What it keeps:** per cell-season, the run with the highest `yield_t_ha`;
  ties go to the smaller shift.
- **Why it is only a bound:** it uses the whole season's weather and the
  model's own yield, so no real sowing rule could make it. It measures how
  much yield sowing alone can move in torchcrop, and how far torchcrop's
  optimum lies from when farmers actually sow.

## Results

### Network sowing, PEP725 test blocks

These station blocks were never used in training or for `tsumem`.

| Sowing table | Bias | RMSE | MAE | Year-to-year r | Year-to-year SD ratio | Spatial r |
| --- | --- | --- | --- | --- | --- | --- |
| SIMPLACE rule | −7.2 d | 15.1 | 11.9 | 0.02 | 0.04 | 0.01 |
| **Network (B0)** | **+0.4 d** | **11.4** | **8.4** | 0.13 | 0.38 | **0.57** |
| Network climatology (C) | +0.1 d | 11.3 | 8.4 | — | — | 0.55 |
| SAGE window midpoint | +9.1 d | 15.7 | — | — | — | 0.09 |

- **Level and spatial pattern:** the network removes the rule's bias and most
  of the national-block pattern.
- **Weather response:** weak. The year-to-year correlation is 0.13, the
  year-to-year spread is 38 % of the observed one, and the fixed per-cell
  climatology scores just as well. Almost all the gain comes from where and on
  what average date sowing happens, not from reacting to the season.
- **Distribution:** network DOY percentiles (1 / 25 / 50 / 75 / 99) are
  254 / 274 / 286 / 301 / 342. Under 1 % of seasons are forced to the
  deadline.

### torchcrop arms against PEP725 (test blocks)

| Arm | Sowing bias | Sowing RMSE | Emergence bias | Emergence RMSE | Emergence year-to-year r | Emergence spatial r |
| --- | --- | --- | --- | --- | --- | --- |
| A0 (old) | −6.1 d | 15.2 | −0.9 d | 16.9 | 0.05 | 0.35 |
| A1 | −6.1 d | 15.2 | −9.4 d | 19.6 | 0.01 | 0.23 |
| B1 | +1.0 d | 11.7 | +12.4 d | 27.5 | 0.38 | 0.79 |
| **B0 (new)** | **+1.0 d** | **11.7** | **−0.8 d** | **14.9** | **0.41** | **0.83** |
| C | +1.3 d | 11.6 | −0.1 d | 15.2 | 0.44 | 0.83 |

A0's small emergence bias is two errors cancelling: sowing is 6 d early and
the calibrated `tsumem` (185–200 °C·d) is too slow. A1 shows this, because
fixing only the clock leaves A1 9 d early. B1 shows it from the other side: the
right sowing with the old slow clock is 12 d late. Only B0 gets both right.

### torchcrop arms against CLMS and CyBench

These use the per-unit references from the production notebook: CLMS
emergence corrected by −6 d, CLMS harvest by +18.7 d, and dry-matter CyBench
yield.

| Target | Split | A0 | A1 | B1 | **B0** | C |
| --- | --- | --- | --- | --- | --- | --- |
| Emergence bias (d) | test | −14.1 | −19.9 | +8.4 | **−1.7** | −3.0 |
| Emergence RMSE (d) | test | 27.2 | 30.2 | 20.8 | **16.1** | 16.7 |
| Season-length bias (d) | test | +12.5 | +18.2 | −8.5 | **+1.4** | +2.4 |
| Harvest bias (d) | test | −1.5 | −1.6 | +0.0 | **−0.2** | −0.4 |
| Harvest RMSE (d) | test | 7.1 | 7.2 | 7.2 | 7.1 | 7.1 |
| Yield bias (t/ha) | test | +0.04 | +0.05 | −0.04 | −0.02 | −0.01 |
| Yield RMSE (t/ha) | test | 0.85 | 0.84 | 0.85 | 0.84 | 0.84 |

Paired per unit against A0 (mean Δ|error| over units, 95 % bootstrap interval):

| Target | B0 | C | Units improved, B0 |
| --- | --- | --- | --- |
| Emergence | **−10.9 d** (−11.8 … −10.1) | −9.1 d | 91 % |
| Season length | −9.1 d (−9.9 … −8.3) | −7.5 d | 87 % |
| Harvest | −0.1 d (−0.2 … 0.0) | −0.1 d | 60 % |
| Yield | 0.00 t/ha | 0.00 t/ha | 53 % |

**National blocks.** Within-country spread of unit emergence, simulated over
observed (1 = gradient reproduced). The rule left large countries flat. A0 gives
DE 0.32, FR 0.26, PL 0.25, IT 0.17 and SK 0.18; B0 gives 0.56, 0.81, 0.95, 1.05
and 1.07. A0's high ratios in CZ (7.0), AT (2.7) and DK (3.0) are noise, not
gradient; B0 brings them to 1.1, 0.7 and 0.8.

### Interannual skill

This is the per-unit correlation between simulated and observed values across
seasons, taken as the median over units. "Test" means the calibration's test
split; at least 8 seasons per unit are required. Detrended values remove a
linear trend per unit. Table: `evaluation/tables/interannual_unit_r.csv`.

| Target | Split | A0 | A1 | B1 | **B0** | C |
| --- | --- | --- | --- | --- | --- | --- |
| Yield (CyBench), detrended | all (755 units) | 0.43 | 0.44 | 0.43 | 0.44 | 0.43 |
| Yield (CyBench), detrended | test (98) | 0.34 | 0.34 | 0.35 | 0.35 | 0.34 |
| Emergence (CLMS) | all (488) | −0.06 | −0.01 | 0.50 | **0.64** | −0.05 |
| Emergence (CLMS) | test (82) | −0.09 | −0.07 | 0.54 | **0.64** | −0.10 |
| Harvest (CLMS) | test (82) | 0.73 | 0.73 | 0.72 | 0.72 | 0.73 |

- **Yield: no change.** The median r moves by +0.003 at most, in every split,
  with or without detrending.
- **Emergence: transformed.** It goes from no year-to-year signal (A0
  r ≈ 0) to r 0.64 on the test units. The gain needs the network's per-season
  dates: the per-cell climatology (C) stays at r ≈ 0. This is where the
  weather response pays off, more clearly than in the PEP725 sowing scores.
- **Why it doesn't reach yield.** Year-to-year yield variation in torchcrop
  comes from spring and summer weather. Where autumn emergence falls in a
  given year does not reach grain fill.

### Why sowing does not reach yield

The D grid runs every cell-season at seven sowing dates (B0 ± 14/28/42 d) with
everything else fixed, so it measures the model's own response. Values are
medians over 1.7 M cell-seasons, relative to B0. Table:
`evaluation/tables/sowing_shift_propagation.csv`.

| Sowing shift | −42 d | −14 d | +14 d | +42 d |
| --- | --- | --- | --- | --- |
| Emergence | −47 d | −16 d | +17 d | +66 d |
| Anthesis | −3.8 d | −1.6 d | +2.1 d | +7.3 d |
| Maturity | −3 d | −1 d | +2 d | +6 d |
| Grain-fill length | +0.5 d | +0.2 d | −0.3 d | −1.0 d |
| Max LAI | +0.24 | +0.08 | −0.06 | −0.17 |
| Yield | +3.4 % | +1.5 % | −2.0 % | −7.2 % |

1. **Phenology re-converges in spring.** Emergence follows sowing one-for-one,
   or more than that for late sowings into cold soil. Anthesis then moves only
   0.14 d per day of sowing shift (median; 5–95 %: 0.05–0.31). Vernalisation
   and photoperiod gate development to anthesis, so a crop sown late catches
   up during spring. The network's +30 d becomes about +4 d at anthesis and
   +2 d at maturity, and grain-fill length barely changes.
2. **Yield is set after the point of convergence.** The median response is
   −1.4 % per 10 d of later sowing. That is a real effect: more pre-winter
   biomass and LAI from earlier sowing. But it is small, so B0's 30 d later
   sowing costs about −2 % (−0.08 t/ha).
3. **Year-to-year sowing variation is too small to show up.** The network's
   per-season dates differ from the per-cell climatology (C) by 10.6 d SD. The
   yield they cause differs by 0.10 t/ha SD, against a year-to-year yield SD
   of 0.67 t/ha. The direction is consistent (r −0.58), but it explains 2 % of
   the yield variance. In B0, a cell's yield anomaly correlates with its
   sowing anomaly at −0.09, and with its grain-fill-length anomaly at +0.32.
   Interannual yield is decided by spring and summer weather.
4. **Little that sowing could change is limiting.** The median water-stress
   factor is 0.98, the N index 1.00, and heat stress zero. torchcrop has no
   frost kill, winter hardening, tillering or plant-density response, and no
   penalty for too much growth before winter. Those are the processes through
   which sowing date affects yield in the field, so the model cannot produce
   them.

The sowing network fixes the **emergence** output, which has no route to
yield in this crop model. A better yield response needs those missing
processes in the model, not a better sowing date.

### Arm D: the yield-maximising sowing

For each cell-season, D picks the shift with the highest simulated yield.

- **Where the optimum lies.** 65 % of cell-seasons pick the earliest shift
  (−42 d) and 68 % pick one of the two grid ends. The median best shift is
  −42 d in every country, and 80–100 % of each country's cell-seasons prefer
  an earlier date. torchcrop's optimum lies **beyond** the grid, so the gain
  is a lower bound.
- **Gain.** Median +0.14 t/ha over B0 (mean +0.20), about 4 %.

Against observations (test split):

| | A0 | B0 | D |
| --- | --- | --- | --- |
| Sowing bias vs PEP725 | −6.1 d | +1.0 d | **−32.5 d** |
| Emergence bias vs CLMS | −14.1 d | −1.7 d | **−35.9 d** |
| Harvest bias vs CLMS | −1.5 d | −0.2 d | −2.3 d |
| Yield bias vs CyBench (t/ha) | +0.04 | −0.02 | +0.15 |
| Yield RMSE (t/ha) | 0.85 | 0.84 | 0.84 |
| Yield year-to-year r (median per unit) | 0.36 | 0.39 | 0.37 |

- **The model's optimum contradicts what farmers do.** torchcrop always
  rewards earlier sowing, because it has no frost kill, no penalty for too much
  growth before winter, and no disease or pest pressure. Farmers sow about
  a month later than that optimum.
- **The yield gain is not skill.** Even knowing the whole season, the best
  sowing raises the yield level but leaves yield RMSE (0.84) and year-to-year
  correlation unchanged. Year-to-year yield variation is out of sowing's
  reach in this model.
- **Consequence for calibration.** Any calibration that let sowing move freely
  would drive it to the early edge to gain yield. Sowing has to stay
  constrained by observations, as the network does.

### Reading

- **Emergence is fixed; maturity and yield are not moved.** The new sowing and
  clock (B0) remove the 14 d emergence bias against CLMS and halve its error in
  91 % of units. Maturity, harvest and yield skill are unchanged.
- **Why maturity barely moves:** the sowing is a median 30 d later, but
  maturity is set by spring thermal time and photoperiod. The median shift is
  +2 d and the harvest bias against CLMS goes from −1.5 to −0.2 d.
- **The weather response adds little.** B0 beats C on CLMS emergence by
  1.8 d per unit, but on PEP725 the two are within 0.3 d. For CMIP6, the
  per-cell climatology (C) is an adequate sowing table.
- **The calibrated `tsumem` existed to absorb the rule's early sowing.** With
  realistic sowing it makes emergence 12 d late (B1), so it must not be reused
  with the network table.

## Recalibration decision

`tsumem` moves emergence and nothing else. Sowing moves maturity and yield,
through the season's thermal time and the length of the autumn growth phase.
The decision rule:

- **Maturity moves by more than ~2 d** (the paired B1 − A0 shift in notebook
  §4): refit the phenology and yield stages on the network's sowing, with
  `tsumem` held at the PEP725 value:

    ```yaml
    fixed_parameters: {crop.tsumem: 128.0, crop.tbasem: 0.0}
    loss:
      emergence_level: 0
    ```

- **Otherwise:** the calibrated crops stand.

!!! success "Outcome"
    The paired maturity shift (B1 − A0 and B0 − A0) is **+2 d**, exactly at the
    threshold. Harvest bias against CLMS *improves* (−1.5 → −0.2 d) and yield
    skill is unchanged (Δ|error| 0.00 t/ha). A refit is therefore **not
    required**. Adopt B0 as: the network sowing table plus the calibrated crops
    with `tsumem` 128 / `tbasem` 0 (`sowing_network/crops_pep725`). A refit
    with `fixed_parameters` stays optional, worthwhile only if a future change
    moves maturity further.

`fixed_parameters` removes the named parameters from the search, so `tsumem`
is no longer calibrated against emergence once the network supplies the
sowing.

## Running it

```bash
cd cropmodelling4eu
./submit/sowing/submit_sowing_network.sh                 # split, cache, tsumem, train, predict (one node, ~20 min)
./submit/sowing/submit_sowing_comparison.sh --dry-run    # check inputs, print the plan
TC_SLOTS=3 ./submit/sowing/submit_sowing_comparison.sh   # A0 A1 B1 B0 C, chained
TC_SLOTS=3 ./submit/sowing/submit_sowing_comparison.sh --after <job> D
python notebooks/sowing_comparison/build_notebooks.py    # then execute sowing_comparison.ipynb
```

| Output | Location (`Data/SIMPLACE/cropmodelling4eu/`) |
| --- | --- |
| Season cache, model, `tsumem.json`, split | `sowing_network/` |
| Sowing tables (B0/B1, C) | `sowing_network/sowing_network.csv`, `sowing_network_climatology.csv` |
| Per-day distribution summary (p10, p90, deadline probability) | `sowing_network/sowing_network_detail.parquet` |
| Crops with the PEP725 `tsumem` (a copy) | `sowing_network/crops_pep725/` |
| Shifted tables for D | `sowing_network/oracle/` |
| torchcrop arms | `runs/winter_wheat/sowing_comparison/{A0,A1,B1,B0,C,D,D_grid}/` |
| Evaluation tables and figures | `runs/winter_wheat/sowing_comparison/evaluation/` |

## Falling back to the old method

Nothing existing was edited in place; the old path is the default everywhere.

| To keep the old… | Do |
| --- | --- |
| MSWX sowing | Leave `MSWX_SOWING` unset in `submit_production.sh` |
| CMIP6 sowing | Leave `CMIP6_SOWING` unset |
| Crops / `tsumem` | Point `CAL_DIR` at `calibration_dm`; its crops are never modified |
| Fallback crop | Leave `TC_CROP_OVERRIDES` unset in `submit_torchcrop.sh` |
| Calibration | `fixed_parameters: {}` (the default) and `loss.emergence_level: 1.0` |
| Production runs | `runs/winter_wheat/calibrated/` is never written by the sowing scripts |

## Known issues and next steps

- **Early stopping mostly tracks CLMS.** Validation batches are ordered CLMS
  first, so the early-stopping metric reflects the CLMS term. Fix: score each
  source separately and stop on a weighted sum.
- **Weak weather response.** Options: weight PEP725 more, or restrict the
  long-term branch so the network cannot rely on the climatological baseline
  alone.
- **The rule arms (A0, A1) carry a hidden fallback crop value.** Regions
  without a calibrated crop use a fallback crop with `tsumem` 60 °C·d. The
  PEP725 arms override it to 128 through `TC_CROP_OVERRIDES`; A0 keeps 60, as
  production does.
- **CMIP6.** Run `cm4eu sowing predict` on a cache built from each model's
  export, so the long-term branch reads that model's own prior autumns.
  Alternatively, use the climatology table (arm C) if C matches B0 in the full
  evaluation.
- **Circularity.** CLMS emergence was also a calibration target. Judge the
  sowing skill on the PEP725 test blocks and the CLMS test split only.
