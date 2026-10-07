# torchcrop calibration

Calibrating the differentiable LINTUL-5 ([`torchcrop`](https://geonextgis.github.io/torchcrop))
against observed **phenology**, **yield** and **LAI** over Europe, by backpropagation
through the model's own day loop.

This plans work that does not exist yet. What *does* exist is the uncalibrated
`winter_wheat_2000_2024_euval` production run — both models on the harmonised SUSTAg `WW`
crop, sown on the same day in every cell — and its failure modes are the specification for
what follows. Every number below is from
[`evaluation/full_run_evaluation.ipynb`](evaluation/full_run_evaluation.ipynb) on that run,
scored against **CyBench yield** and **CLMS HRL Croplands phenology**; nothing here is
carried over from the earlier `winter_wheat_2000_2024` run, whose crop file and export
were both different. See [`TORCHCROP.md`](TORCHCROP.md) for how the run itself is built.

## Why

The specification is the **full continental run**, not the 30-cell smoke test. Numbers
below are the `winter_wheat_2000_2024_euval` run — the EU_val export, both models on the
harmonised SUSTAg `WW` crop, 68 685 cells × 25 seasons = 1 711 524 cell-seasons each —
scored in [`evaluation/full_run_evaluation.ipynb`](evaluation/full_run_evaluation.ipynb).
Yield is CyBench on its own reported units (805 units / 14 923 unit-years); **phenology is
CLMS HRL Croplands**, per unit *and per season* (815 units / 5 811 unit-years, 2017–2024).
SIMPLACE runs on the **identical pairs**, as a contrast rather than as a truth.

**Sowing is identical on both sides for every one of the 1 711 524 pairs** (§6.1,
`sowing_doy_delta` exactly 0). This is the first run in which §6 is a comparison of two
models rather than of two seasons.

| Symptom | torchcrop | SIMPLACE, same pairs | Stage that owns it |
| --- | --- | --- | --- |
| **Emergence is a month early and carries no interannual signal at all** | bias **−31.5 d** against CLMS, MAE 31.9, RMSE 38.0; sowing→emergence **3.3 d** against a ground-observed **12 d**; anomaly r **0.00**, simulated anomaly sd **2.1 d** | −31.1 d, MAE 31.5, RMSE 37.7; lag 5.0 d; anomaly r −0.00, sd 2.1 | 1 · phenology |
| **Harvest timing is the one thing that works** | bias +24.9 d against CLMS — but CLMS is itself **18.7 d early against PEP725** (below), so ≈ **+6 d** on the ground scale; spatial r 0.89, **anomaly r 0.67**, sd 4.8 against 6.1 | +19.1 d → ≈ **+0.4 d**; spatial r 0.91, anomaly r 0.65 | 1 · phenology |
| **The yield skill is now positive, and the level over-corrected** | bias **−0.68 t/ha**, MAE 1.65, RMSE 2.08, r **0.52**, **r² +0.03**; per-country bias −1.8 … +3.0 | +1.19 t/ha, MAE 1.93, RMSE 2.56, r 0.35, r² **−0.47** | 2 · yield |
| **The canopy is too big, and the partitioning is no longer the problem** | peak LAI median **7.31**, only **5.5 %** inside 4–6; HI median **0.408**, 0.7 % above 0.6, on 1 067 g m⁻² of biomass | LAI median 6.50, 18.9 % inside 4–6; HI median **0.343**, biomass 1 807 g m⁻² | 3 · LAI |

Pooled `tranrf` 0.895 and `nni` 0.985 over the same cells: water is now mildly limiting,
nitrogen is not limiting at all.

**Read the harvest row through the reference offset, which is now measured against the
ground.** CLMS `CPMCH` sits **−28.9 d** from CyBench `eos` (median −28.8, IQR −33.5 …
−24.5, r 0.87 across 835 units — the two products agree on *where*, not on *when*; §5.2)
and **−18.7 d from PEP725**, identically on both supports
(`data4simplace/notebooks/clms_pep725_validation.ipynb`). PEP725 is the one that counts: it
is a person in a field, and the offset is constant year by year (−15.3 … −23.0 d, sd 2.7,
with the 2022 drought year at −16.5 — so it is a **product bias, not a hot summer**) and
unchanged when CLMS is smoothed over a 3×3 cell block (−18.6), so it is not a support
artefact either. On the ground scale torchcrop harvests **+6.2 d late** and SIMPLACE
**+0.4 d** — both far better than the raw CLMS bias suggests, and the reason the target
below is written as a residual after the offset. What the offset does **not** touch is the
anomaly correlation, taken within a unit and blind to any constant level difference, which
is why the interannual rows above are the ones to trust.

What each stage has to fix:

- **Stage 1 — emergence, not maturity.** The old "+1.0 d pooled, −61 … +108 d per unit"
  maturity spread was an artefact of the unharmonised crop and is gone; harvest now has
  spatial r 0.89 and anomaly r 0.67, which is a working thermal-time response. What does
  not work is emergence: `tsumem = 60 °C·d` is reached in **3.3 days**, so no weather can
  move it, and the model produces a 2.1 d anomaly against CLMS's 13.6 d — though most of
  that 13.6 is CLMS noise and the ground figure is ~8 d (see *Observations*). That is a
  structural non-response, not a bias — a `tsumem` shift alone fixes the level but the
  anomaly amplitude is the test.
- **Stage 2 — the level, and the sign flip.** `r² +0.03` clears the old binding
  constraint, barely; `r 0.52` is genuine skill. The bias flipped from +0.71 to
  **−0.68 t/ha** and is strongly regional — Mediterranean units are near zero
  (EL −0.00, ES −0.09, IT +0.16), the north-west is uniformly low (BE −1.75, IE −1.78,
  DK −1.72, NL −1.60, DE −1.27), the Baltics uniformly high (LV +2.92, LT +2.66,
  EE +2.55). A pooled bias term will average those three regimes into nothing.
- **Stage 3 — the canopy, not the harvest index.** The HI tail this section used to name
  is **gone**: 0.7 % of cell-seasons above 0.6 against 26 % before, and the median has
  fallen from 0.538 to **0.408** — now just *below* the 0.45–0.55 anchor rather than well
  above it. So is the bimodal canopy failure — 0.13 % of cell-seasons
  peak below 1.5, against 6.2 % before. What replaced both is a canopy that is simply too
  large: median peak LAI **7.31**, with only 5.5 % of cell-seasons inside the expected
  4–6, on 1 067 g m⁻² of biomass. torchcrop now builds *more* canopy than SIMPLACE (6.50)
  out of *less* biomass, which is an SLA and senescence statement, and it is the reverse
  of what the previous run said.

**With sowing forced equal, §6 finally isolates the models** (1 711 524 pairs,
`torchcrop − simplace`): yield **−1.90 t/ha** (r 0.84), biomass **−699 g m⁻²** (r 0.84),
peak LAI **+1.04** (r 0.48), maturity **+8.4 d** (r 0.94). Same crop file, same weather,
same soil, same sowing day: torchcrop grows a bigger canopy, accumulates 40 % less
biomass, and runs eight days longer.

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

**`idsl = 2` is the setting, fixed, for every stage and every region.** It is SIMPLACE's
own value in the SUSTAg `WW` block, it is what the harmonised run uses, and it is not a
tuning knob: it is in `Lintul5Model._NON_DIFFERENTIABLE_FIELDS` (with `iopt`, `iairdu`), so
no gradient reaches it and `materialize` warns if it is wrapped as a free parameter.
Nothing below proposes changing it. Where `idsl = 0` appears in this document it is either
**history** (what torchcrop ran before the harmonisation) or a **control** (upstream's
example runs at 0; so does the identifiability check in *Stage 1*, purely to prove the
measurement is real) — never a configuration on the table.

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

**CLMS is the primary observation, not the coverage patch it was planned as.** It landed,
the evaluation now runs on it, and the numbers in *Why* are its.

| Source | Status | Role |
| --- | --- | --- |
| **CLMS HRL Croplands** `CPMCE`/`CPMCH`, reduced to `phenology_adm.parquet` / `phenology_grid.parquet` | **in use** — `EU_val/validation/phenology`, see [`CLMS_DOWNLOAD.md`](../src/data4simplace/CLMS_DOWNLOAD.md) | **Primary.** 10 m, per year, pan-European. After the quality gates: **835 units × 8 seasons (2017–2024), 5 895 unit-years**, 815 of those units reached by the run. Emergence *and* harvest, with a per-unit p10/p90 spread. |
| **PEP725** `/data01/FDS/muduchuru/Data/Agri/PEP725/` | on disk, **not yet wired in** | **The station-level detail CLMS cannot give.** 213 365 wheat rows, 3 018 stations, 1990–2024. BBCH **0** sowing, **10** emergence, 31, **51** heading, 85, **100** harvest. `cult_season == 2` isolates winter wheat (211 947 rows). Its unique contribution is now **heading** (BBCH 51) and the 1990–2016 seasons CLMS does not cover. |
| **JRC `EU_Phenology_Complete_ver2019`** `/data01/FDS/muduchuru/Data/Agri/Phenology/` | on disk | **Prior, not observation.** The MARS/BioMA per-cell calibration on 14 175 × 25 km cells: `TSUM1` (mean 860, 431–1240), `TSUM2` (794, 400–1350), `SOWING` (DOY 277), `VERNSAT` (54 d, 6–70). Joins `IDGRID` ↔ `grid_25km.shp:Grid_Code`, EPSG:3035. |
| **CyBench** `crop_calendar_wheat_<c>.csv` | on disk | **Not an observation any more — a cross-check on CLMS's level.** Static `sos`/`eos`, no year dimension, so it can only penalise interannual skill. Its one remaining job is §5.2's offset measurement. |

**Three properties of CLMS decide how the loss must read it**, all measured rather than
assumed:

1. **The level is offset and the shape is not — and the offset is now measured.**
   `CPMCH` is −28.9 d from CyBench `eos` (median −28.8, IQR −33.5 … −24.5, r 0.87 across
   835 units) and **−18.7 d from PEP725**, the ground observation. The PEP725 number is the
   usable one: it is identical on both supports, unchanged under 3×3 spatial smoothing, and
   constant year to year (−15.3 … −23.0 d, with the 2022 drought year at −16.5), so it is a
   product bias rather than a support artefact or a hot summer. **Apply it as a fixed
   correction rather than letting `tsum2` absorb it**, and keep the level term — the
   earlier advice to weight the level to zero was written when the offset was an unresolved
   product difference, and it no longer is.
2. **The anomaly term is where the signal is — for harvest.** CLMS harvest anomaly sd is
   6.1 d and both models already reach anomaly r ≈ 0.66 against it, and PEP725 confirms
   that signal is real (CLMS-vs-ground anomaly r 0.54–0.58). CLMS *emergence* anomaly sd is
   13.6 d against the models' 2.1 at r 0.00 — but that 13.6 does **not** survive contact
   with the ground, as the next paragraph shows, so emergence has the most headroom in the
   plan and the least trustworthy yardstick.

   **Most of the CLMS emergence anomaly is CLMS noise, and the target must be set against
   ground truth instead.** `data4simplace`'s
   `notebooks/clms_pep725_validation.ipynb` scores CLMS against PEP725 on both supports:

   | | emergence | harvest |
   | --- | --- | --- |
   | bias (CLMS − PEP725), 10 km / NUTS | **+6.0 / +5.6 d** | **−18.7 / −18.7 d** |
   | RMSE | 19.7 / 16.7 d | 20.7 / 20.4 d |
   | spatial r | 0.44 / 0.59 | 0.58 / 0.66 |
   | **anomaly r** | **0.11 / 0.13** | **0.54 / 0.58** |
   | anomaly sd, PEP725 | 8.3 / 7.6 d | 7.8 / 7.5 d |
   | anomaly sd, CLMS | **14.2 / 12.4 d** | 5.8 / 5.6 d |

   CLMS emergence therefore has an interannual scatter **~1.7x larger than the ground
   observation's and essentially uncorrelated with it** (r 0.11). The 13.6 d "observed"
   anomaly is mostly product noise, and a `tsumem` fitted to reproduce it would be fitting
   noise. **Set the emergence anomaly target against PEP725's ~8 d**, weight the CLMS
   emergence anomaly term down to near zero, and keep CLMS emergence for its *level* and
   its spatial gradient. CLMS **harvest** needs no such discount: anomaly r 0.54–0.58
   against ground truth, with a slightly conservative amplitude.

   **Even against 8 d, stage 1 cannot close the gap alone.** Simulated emergence is
   `sowing + days_to_emergence`, and the variance budget on this run is: per-cell sowing sd
   **3.24 d** (SIMPLACE's rule-based date, handed to torchcrop — and 21 % of cells get a
   single constant date across all 25 seasons), per-cell `days_to_emergence` sd **0.67 d**.
   Raising `tsumem` from 60 to ~150 scales the second roughly with its mean, so ~1.7 d —
   *estimated, not measured* — which with the sowing term gives ~3.7 d against PEP725's 8.
   **The rest is the sowing window**, an input to both models and outside every stage of
   this plan; it is the same saturation `TORCHCROP.md` records, where the rule fires on the
   window's first day in the large majority of cell-years. Report the emergence anomaly
   target as "as far as `tsumem` can take it", and raise the sowing window as a separate
   finding against the export.
3. **Eight seasons, not twenty-five.** 2017–2024 only. A year-block split (see *Splits*)
   over 8 years gives at most one held-out block, so the temporal split must be
   leave-one-year-out on phenology, not the 70/15/15 the yield term gets.

**PEP725 is not European — it is Central European.** That is the fact that shaped the
regional design, and CLMS is what relaxes it:

```
provider 101 (DE)   200 834 rows   6.0–15.0 °E, 47.6–54.9 °N
provider 301 (AT)     5 858
provider 601 (SK)     3 968
provider 2101 (SI)    1 343
providers 4 / 8 / 401 / 1401 / 2601 / 2701   ~1 400 total (CZ, BE-NL, ES-Cat, HR, BA, ME)
```

94 % of the PEP725 signal is German. With CLMS primary, **stage 1 can now fit every EnZ
zone on emergence and harvest**; PEP725 adds heading only where it has stations, so the
identifiability rule below binds at *three* stages in CON/ATC/NEM and at *two* everywhere
else — a much weaker restriction than the "shrinkage to prior outside Central Europe" this
section previously prescribed.

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

   with observations detrended per `adm_id`. The level term holds the **−0.68 t/ha** bias;
   the anomaly term is the part that tests climate response. On the harmonised run the
   anomaly term is no longer starting from nothing — `r 0.52`, `r² +0.03` — so its job has
   changed from *creating* skill to *not destroying* it while the level is corrected.
3. **The level term must be per region, not pooled.** The bias is three regimes, not one:
   Mediterranean ≈ 0 (EL −0.00, ES −0.09, IT +0.16), north-west uniformly low
   (IE −1.78, BE −1.75, DK −1.72, NL −1.60, DE −1.27), Baltic uniformly high
   (LV +2.92, LT +2.66, EE +2.55). A pooled `(mean_sim − mean_obs)²` averages them to
   −0.68 and moves nothing where it matters — the same argument stage 1 makes about
   pooled maturity, one section up.
4. **Area weighting** — `aggregate.weighted_mean` already does this, including its
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
above is hopelessly over-parameterised, and PEP725 reaches maybe 15 strata at all. CLMS
weakens that argument without overturning it — 835 units across 22 countries would support
more than 13 groups on phenology — but yield is still 805 CyBench units over 25 years and
LAI is still fPAR, so the region set has to serve the sparsest term, and the target below
stands.

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

On the EU_val export the silhouette takes four sub-clusters in most zones and lands at
**33**, just above the band, so `regions.max_clusters_per_zone` defaults to **3**: every
extra region is another parameter set fitted on fewer observations. The band is a warning
rather than an error, since a domain smaller than Europe legitimately falls below it. Two
consequences the run has to state rather than absorb:

- **The region cache keys on the settings as well as the cell set.** Clustering is what
  these settings *do*, so a cache built under different ones is rebuilt rather than reused.
- **A region no observation reaches is not calibrated, and is named.** `ANA` (1 569 cells,
  0 CyBench units) gets no manager and no crop file. Its cells still run — on the
  *uncalibrated* crop — so a map of the result would show no discontinuity, which is why
  the region is logged at build time and listed in each stage's `summary.json`.

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

**One region, but not one year.** The constraint is on the *parameters*, not on the
calendar: every window in a region opens on the same day-of-year, so seasons from
different harvest years share one relative time axis and the single scalar `start_doy` the
engine takes. Batching by region *and* year as well would leave a batch about 40 cells
against a 192 limit, and LINTUL-5's day loop costs almost the same at either — see
*The dataloader*.

### torchcrop has no gradient checkpointing

`ModelOutput` retains every per-day `ModelState`, rate dict and `DiagnosticState`. At
batch 2048 that was 1.7 GB *without* a graph; with autograd expect 10–20×. Mitigations in
order of cheapness:

1. **Cut the window** — sowing − 15 d to maturity + 30 d ≈ 330 days, against the
   production run's 600+ (`SIM_YEARS = 2`, `min_days_after_sowing: 365`). **Measured at
   450**, not 330: see *Two findings from building it*.
2. **Batch 128–256**, not 2048.
3. **√T checkpointing** over the day loop, written in our own wrapper against the exposed
   `compute_rates` / `update_state` low-level API: roughly 1.3× compute for O(√T) memory.
   Budget this as real work, not a flag. Written; it reproduces the stock forward
   bit-for-bit, and the gradients through it are identical to the unchunked ones.

**One node, used properly — not ten.** The parallel axis is *regions*: a batch
carries one region's parameters, and once the penalties are scoped to that region
(finding 3) nothing in a batch touches another region's latents, so batches of
different regions have disjoint gradients and run concurrently for the same
numbers. `calibration/parallel.py` forks one worker per region, groups an epoch's
batches into waves that never repeat a region, and returns **gradients rather than
graphs** — a few dozen floats per batch instead of a 465-day autograd tape.

The ceiling is therefore the **region count**, ~29 here, since at batch 768 each
region contributes about one batch an epoch. A `compute` node has 80 cores, so one
node covers the whole available parallelism and a second would idle — this is
deliberately not a multi-node job. Threads are no substitute: the day loop is a
Python loop holding the GIL, which is why 1/4/10/20 threads measured identical.
The single approximation is that the pooling penalty's *detached* cross-region
mean is one wave stale rather than updated between the steps inside a wave — the
ordinary synchronous-versus-sequential distinction, under a penalty weighted 0.1.
`--workers 1` is the exactly sequential path.

**An epoch is not a step.** With one parameter set per region, a batch updates *one*
region, so a region takes `batches_per_epoch / n_regions` Adam steps an epoch — on this
run 51/28 ≈ **1.8**, or about **146 steps per region** over the 80 epochs. Against
upstream's 50 steps for a single global set that is the right order and not a generous
one, and it is why the first epochs move the loss so little (val 608 → 598 over epoch 1).
Raising `data.fraction` buys more steps per region as well as more data per step.

**What it costs, measured.** One stage-1 epoch on the EU_val export — 29 regions, a 20 %
draw of 3 661 training unit-years, 51 batches, plus the full 678-unit-year validation pass
— is **19 minutes** on 4 CPU threads, so `max_epochs: 80` is about **25 hours**. Two thirds
of that is the day loop itself, which is Python and per-day rather than per-cell: a batch
of 40 cells costs almost what 192 does, which is why the batches are packed across years
rather than being one region-year each. The checkpointed backward is ~3× a forward pass,
and the unfertilised pass-1 that dates the DVS-keyed schedule is a fourth. Cutting
`max_epochs` is not the lever it looks like — upstream's example converges by 50 — but the
first epoch already moves the emergence bias, so a short run is a real smoke test.

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
*Splits, diagnostics, acceptance* (harvest +19.1 d bias against CLMS, MAE 18.8 d over
815 units — or +0.4 d once CLMS's own −18.7 d offset is removed).

**Which accumulator, per stage** — this is not a free choice:

| Observation | Accumulator | Threshold | Why not DVS |
| --- | --- | --- | --- |
| **CLMS `CPMCE`** (PEP725 BBCH 10) emergence | `tsump` | `tsumem` | DVS is pinned at 0 until the crop emerges, so it carries no emergence signal at all; its first non-zero step is also one day late, because the rate on day *t* produces the state on day *t+1* |
| PEP725 BBCH 51 heading | `dvs` | ≈ 0.85–0.90 | anthesis is BBCH 61 = DVS 1.0. Needs the anthesis output blocker 7 asks for |
| **CLMS `CPMCH`** (PEP725 BBCH 100) harvest | `dvs` | 2.0, **hinged** | harvest ≥ maturity: penalise maturity *after* harvest hard, more than ~14 d *before* it mildly |

The two CLMS rows are per `(adm_id, year)` and aggregate-then-compare, like the yield
term; the PEP725 row is point-matched to a cell. Both flow through the same
`crossing_day`, so the loss is one construction over two supports rather than two losses.

**Weight the three terms by what the reference can actually measure**, which
*Observations → Phenology* establishes and which is not uniform:

| Term | Weight | Why |
| --- | --- | --- |
| Harvest level | full, **on the offset-corrected date** (`CPMCH + 18.7 d`) | the offset is constant, measured against ground truth, and identical on both supports — correcting it is better than letting `tsum2` absorb it |
| Harvest anomaly | full | CLMS-vs-ground anomaly r 0.54–0.58; this is real interannual signal |
| Emergence level | full | CLMS emergence is only +6 d from ground truth and its spatial gradient is sound |
| **Emergence anomaly** | **near zero against CLMS**; full against PEP725 where stations exist | CLMS-vs-ground anomaly r is **0.11**, with CLMS scattering 1.7× more than the ground observation. Fitting it fits noise |
| Season length | dropped, or derived | it is the difference of the two rows above and adds no independent information |

`crossing_day` returns **NaN where the threshold is never reached**, which is the right
behaviour for a loss too: a cell that never matured has no maturity date, and a clamped
last-day answer would be scored as a real one.

Work in **days since the window start**, not day-of-year. Upstream's example does
(`(date − SIM_START).dt.days`), and it removes circular arithmetic from the loss entirely —
no wrap, no shortest-path difference. The evaluation notebook still needs the circular
machinery because it pairs two products that share no origin; the loss does not.

A cheaper fallback — penalise `(DVS(t_obs) − DVS_target)²` at the observed date — loses the
day units, so it stays the fallback.

**Upstream's phenology example is the reference implementation for all of this**, and it
calibrates exactly the three thermal-time parameters stage 1 leads with —
[`04_calibration/phenology_calibration_(winter_wheat)`](https://geonextgis.github.io/torchcrop/examples/04_calibration/phenology_calibration_%28winter_wheat%29/).
It confirms every mechanical choice above rather than merely being compatible with them:

| | Upstream | Here |
| --- | --- | --- |
| Free | `crop.tsumem`, `crop.tsum1`, `crop.tsum2` | the same three, plus the vernalisation group |
| Dates | `crossing_day` on `tsump` @ `tsumem`, `dvs` @ 1.0, `dvs` @ 2.0 | identical |
| Loss | `Σ_events ((pred − obs)² over train_idx).mean() / n_events`, in days | the same, plus the hinge on maturity ≤ harvest |
| `smooth` | not used | not used — see the next section |
| Manager | `CalibrationManager(model, specs)`, latent↔bounded bijection | the same, plus `SimplexGroup` for stage 2 |
| Optimiser | Adam, `lr=0.12`, `betas=(0.8, 0.95)`, 80 steps (converged by 50) | start here |
| Split | 12 of 18 sites train, 6 held out | 70/15/15 by year block *and* held-out `adm_id` |
| Support | 18 Brandenburg NUTS-3 regions, **one** harvest year (2021) | 815 units × 8 seasons (CLMS) + PEP725 stations |

**The damped `betas=(0.8, 0.95)` is not a stylistic choice** — upstream documents it as
what stops `tsum1` and `tsum2` making compensating excursions, since both move maturity.
That is the same conditioning problem the identifiability rule below is about, met with a
different instrument. Use both.

**Two differences that matter.** Upstream runs `idsl = 0` and bounds `tsum1` at
(900, 2000) and `tsum2` at (700, 1800); we run `idsl = 2` with the JRC-derived bounds in
*Stage 1*, which are much lower because vernalisation does the delaying work an
`idsl = 0` crop has to buy with an inflated `tsum1` — the mechanism *Resolved* describes.
Do not carry upstream's bounds over. And upstream fits **one season**, so it never
separates a level error from a climate-response error; the year dimension CLMS brings is
exactly what makes the anomaly terms in *Observations → Phenology* possible.

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
| 4 | all | joint fine-tune at small lr |

The retention terms are free: one forward pass produces all three. `stage_order` stays a
config key so swapping 2 and 3 is a one-line change if the retention weights prove
insufficient.

**Stage 4 is no longer optional.** The harmonised run puts the canopy 20–60 % *above*
target rather than below it, so stage 3 removes leaf area that stage 2 was fitting yield
through. That is a strictly larger perturbation than the one `λ₂` was sized for, and it
runs the wrong way — a stage 3 that raised a deficient canopy would have pushed yield up,
toward the −0.68 t/ha bias stage 2 is correcting, not away from it.

**Stage 1 also moves more than the order suggests.** `tsumem` going from 60 to ~200 °C·d
delays emergence by about a month, which removes a month of early growth from every
season before stage 2 ever runs. So stage 1's headline fix is also the single largest
input change stage 2 will see — do not size stage 2's starting point on the numbers in
*Why*, which are pre-stage-1.

## Parameters per stage

### Stage 1 — phenology

**Structural, decided before the stage (non-differentiable):** `idsl` = **2**, fixed —
already the run's setting, never calibrated, see *Resolved*; `site.idpl` sowing DOY, taken
from PEP725 BBCH 0 where available, else `site.csv`.

**Current** is `<TC_OUT_DIR>/workspace/crop_wheat.yaml` today. A current value outside its
bound means the run starts from the wrong place.

**Tier 1 — free in every region.**

| Parameter | Unit | Current | Range | Constraint / source |
| --- | --- | --- | --- | --- |
| `tsum1` | °C·d | 1125 | **500 – 1300** | JRC `TSUM1` 431–1240. Now inside the prior; it was 1623 while the crop ran at `idsl = 0` |
| `tsum2` | °C·d | 1000 | **400 – 1350** | JRC `TSUM2` 400–1350 (mean 794) |
| `tsumem` | °C·d | 60 | **100 – 300** | **The single most identifiable parameter in the plan.** At 60 the crop emerges in **3.3 days** against a ground-observed **12** (PEP725 median, IQR 10–16, 4 433 station-seasons — `notebooks/clms_pep725_validation.ipynb` §2.1). `tsump` accumulates `clamp(T − tbasem, 0)` capped at `teffmx − tbasem`, so at an autumn mean of 10–13 °C over `tbasem = 0` a 12-day lag implies **120–160 °C·d** — and upstream's example independently fits **160** on German ground observations, bounding it (40, 300). The lower bound is 100 because the range below it reproduces the 3-day failure. Unaffected by `idsl`: vernalisation gates DVS, not the `tsump` clock |
| `versat` | vernal d | **70** | **5 – 70** | JRC `VERNSAT` 6–70 (mean 54). At the top of the prior, so expect it to move down |
| `vbase` | vernal d | **14** | **0 – 25** | Hard constraint `vbase ≤ versat − 5` — it is the denominator in `vernf = (vern − vbase)/(versat − vbase)`, and `versat = vbase` makes the whole block a no-op |
| `vernrt` y-ordinates | d d⁻¹ | 0, 0, 1, 1, 0, 0 | **0 – 1** each | Fixed x = −10, −4, 3, 10, 17, 30 °C (SIMPLACE's own knots); unimodal. Free abscissae are not identifiable from three dates |
| `phottb` y-ordinates | – | 0 at 0 h → 1 at 17 h | **0 – 1** each | Two knots only, a straight ramp. **Non-decreasing.** Adding interior knots is a stage-1 decision, not a given |

**Tier 2 — freed only where a third dated stage is observed** (PEP725 heading, or torchcrop anthesis once blocker 7 lands; see the coverage table below).

| Parameter | Unit | Current | Range | Constraint / source |
| --- | --- | --- | --- | --- |
| `vernalisation_devstage` | DVS | 0.3 | **0.2 – 0.5** | Above this DVS `vernfac = 1`. Trades against `versat` |
| `minimal_vernalisation_factor` | – | 0.0 *(SIMPLACE's solution uses **0.4**)* | **0.0 – 0.5** | Floor on rate suppression. At 0 an unvernalised crop never develops — wrong for Mediterranean cells. The two models disagree here and no crop mapping can fix it (see *Resolved*), so freeing it is also what makes them comparable |
| `tbasem` | °C | 0.0 | **−2 – 4** | Pre-emergence base temperature |
| `dtsmtb` y-ordinates | °C·d | 30 at 30 °C | **20 – 32** at x = 30, 45 | The SUSTAg crop caps at 30, the same as WOFOST, so the JRC `TSUM` prior transfers directly. (Brandenburg's capped at 27, which is why an earlier version of this row said it did not.) |

**Identifiability rule: free parameters per region <= independent observed stages per
region.** With CLMS primary, **every region has two dated stages** (emergence and harvest)
over 8 seasons, and CON/ATC/NEM additionally have PEP725 heading.

**torchcrop calibrates `tsum1` and `tsum2` as ordinary free parameters** — upstream's
phenology example fits exactly `tsumem`, `tsum1`, `tsum2` together, and nothing in the
library treats them specially. The question is not whether they *can* be freed but whether
the observations here **separate** them, and that was measured rather than argued:

| | `dvr_veg` | `dvr_gen` |
| --- | --- | --- |
| `processes/phenology.py:215` | `dtsu · photofac · vernfac / tsum1` | `dtsu / tsum2` |

At `idsl = 0` both factors are pinned to 1, so a season needs `tsum1 + tsum2` degree-days
and **only the sum is identified** — which is why upstream, running `idsl = 0`, needs its
observed *flowering* date to fit the two. At our `idsl = 2` the vegetative phase is
discounted and the generative phase is not, so the split does move maturity. Holding
`tsum1 + tsum2 = 2125` and sliding the split over 24 cells × 2 seasons:

| Split | `idsl = 2` | `idsl = 0` (control) |
| --- | --- | --- |
| 900 / 1225 | **−0.75 d** (sd 0.81, max 3) | −0.12 d (sd 0.33, max 1) |
| 1125 / 1000 | reference | reference |
| 1350 / 775 | **+0.62 d** (sd 0.67, max 3) | +0.23 d (sd 0.42, max 1) |

The control is the integer-day quantisation floor — `days_to_maturity` is a whole-day
count — so the `idsl = 2` response is real, symmetric and near-linear at about
**0.003 d per °C·d** reallocated. It is also **hopeless as a lever**: a 21 % shift of the
thermal budget moves maturity by two thirds of a day, against a CLMS harvest RMSE of 21 d
and an observed anomaly sd of 6 d. Pinning the split to ±100 °C·d would need maturity
resolved to ±0.3 d.

The reason the discount is so weak is worth recording, because it is not the naive
photoperiod argument: `phottb` ramps 0 → 1 over 0 → 17 h, so mid-winter `photofac` is
about 0.47 — but `dtsmtb` is 0 below 0 °C and only ~2 °C·d/day at 2 °C, so almost no DVS
accumulates while the discount is deep. The bulk of the vegetative thermal time is spent
in spring at `photofac` 0.75–0.95, and it is the **marginal** rate either side of anthesis
that sets the lever, not the phase average.

So:

| Region has | Free in stage 1 | `params_phenology.yaml` |
| --- | --- | --- |
| CLMS emergence + harvest (all 22 countries) | `tsumem`; `tsum1 + tsum2` as a **sum**, with the ratio shrunk hard to the JRC prior; `versat` under shrinkage | tier 1 |
| \+ PEP725 heading (CON, ATC, NEM, parts of ALS/PAN), scored against the anthesis date `run.py` now writes | the above with `tsum1` and `tsum2` genuinely **separated**, plus `vbase`, `phottb`, `vernrt`, `vernalisation_devstage` | tier 2 |

**The tiering is per region, and so is the ratio prior's release.** One shared
spec set would freeze the vernalisation block everywhere because 17 of 28 regions
lack a third dated stage — discarding what the other 11 know. `pooling_penalty`
already means over the managers *carrying* a given latent, so a parameter free in
eleven regions pools across those eleven and is simply absent from the rest;
nothing ever required a common spec set. On the EU_val export tier 2 frees in
**11 of 28 regions** — 266 latents against 112 — and those regions come out
**better conditioned than tier 1**: a Jacobian condition number of **4.0 over 18
free parameters**, against 8.3 over 4 where only CLMS's two stages exist. That is
the plan's own argument, measured: heading separates the thermal sums, and freeing
more parameters *with* it conditions the problem better than freeing fewer
without.

**This table is now measured, not argued.** A first implementation put `vbase`, `phottb`
and `vernrt` at tier 1, and `Calibrator.identifiability` refused it: on the EU_val export,
with only CLMS's two dated stages, the parameter correlation matrix comes back at
**0.997–0.999** across `vbase`/`versat`/`phottb@0`/`phottb@17`/`vernrt@10` — far past the
0.95 at which one of a pair should be frozen. Tier 1 is therefore exactly four scalars:
`tsumem`, `tsum1`, `tsum2`, `versat`, and on those the Jacobian condition number is
**8.7** — an ordinary, well-posed problem. The diagnostic runs on one batch and so on one
region, which its output names; it is a statement about what *that region's* observations
separate.

What survives the trim is the structure this section predicted, now measured. Tier 1's own
correlations split cleanly in two:

| Pair | r | |
| --- | --- | --- |
| `tsumem` ↔ `versat` | **−0.951** | the three pre-anthesis delays, trading |
| `tsumem` ↔ `tsum1` | **−0.921** | against each other |
| `tsum1` ↔ `versat` | **+0.901** | |
| `tsumem` ↔ `tsum2` | −0.488 | `tsum2` moves maturity and nothing else does, |
| `tsum1` ↔ `tsum2` | +0.438 | so it is the best-identified of the four |
| `tsum2` ↔ `versat` | +0.327 | |

Everything that delays development *before* anthesis is confounded with everything else
that does — a slower emergence, a longer vegetative sum and a longer vernalisation
requirement all produce the same harvest date, which is precisely why anthesis is the
observation that resolves them. The three are not frozen, because each carries an
*independent* prior the others do not touch: `tsumem` from PEP725's ground-observed
12-day sowing-to-emergence lag (150 ± 50 °C·d), `versat` from the JRC `VERNSAT` field
(54 ± 16 d), and the `tsum1`/`tsum2` split from the ratio prior. Those priors are what make
the directions identifiable; without them one of the three would have to go.

The **gradient check** on the same batch passes cleanly: all four parameters live, with
analytic and finite-difference gradients agreeing to 0.3 % on `tsumem`, `tsum1` and
`tsum2`. `versat` disagrees by 11 %, which is the finite difference's own floor rather
than a fault — its gradient is ~1 against the others' 60–165, so a ±1e-3 latent step
barely moves a loss quoted in days.

Parameterise the pair as **(sum, ratio)** rather than as two independent thermal sums.
Both stay free everywhere — that is what makes the manager's bounds and upstream's damped
`betas=(0.8, 0.95)` sufficient — but the ratio carries a strong prior wherever anthesis is
unobserved, so the optimiser cannot wander the near-singular direction the table above
measures. Blocker 7 is what turns that prior back into data, and it is three lines.

### Stage 2 — yield

**Assimilation leads this stage now, not partitioning.** On the harmonised crop HI is
**0.408** — just below the 0.45–0.55 anchor, a fifth of the distance the old 0.538 sat
above it — while
biomass is **1 067 g m⁻²** against SIMPLACE's 1 807 on the identical cells, seasons and
sowing dates. The yield gap is a growth gap, not a partitioning gap: torchcrop reaches a
*larger* peak LAI (7.31 vs 6.50) out of 40 % less biomass, which points at `ruetb` /
`scale_factor_rue` and at the SLA that turns assimilate into leaf area, not at `fltb` /
`fstb` / `fotb`. Free the partitioning group anyway — it is what keeps HI from drifting
while RUE moves — but do not expect it to carry the stage.

| Group | Parameter | Unit | Current | Range |
| --- | --- | --- | --- | --- |
| Partitioning | `fltb` y at DVS 0.646 / 0.95 | – | 0.30 / 0.00 | **0 – 0.7** each |
| | `fstb` y at DVS 0.646 / 0.95 | – | 0.70 / 1.00 | **0 – 1** each |
| | `fotb` y at DVS 0.99 / 1.0 | – | 0.00 / 1.00 | **0 – 1**, non-decreasing. The step of the full fraction in one hundredth of a DVS was named here as the cause of the harvest-index tail; the harmonised run **falsifies that** — the step is still there and only 0.7 % of cell-seasons exceed HI 0.6. Keep it free as a shape parameter, not as a fix |
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

> ⚠ **The N parameters are not identifiable.** Initial mineral N is a median ~243 kg/ha
> against a measured 30–90 (`soil.mineral_n_fraction = 0.01`, see `TORCHCROP.md`), so the
> crop is effectively N-unlimited and `nlue`/`frnx` have nothing to bite on. Pooled `nni`
> on the harmonised run is **0.985** over 1 711 524 cell-seasons — *worse* than the 0.96
> this warning was written against. This is no longer a "may not": either fix the export
> first, or exclude the N group and say so in the outputs. Water is closer to being
> informative (`tranrf` 0.895) but is still not a limiter over most of the domain.

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

**Target the mean, not the tail — this reversed.** The bimodal failure this section was
written around is gone: **0.13 %** of cell-seasons peak below 1.5 (against 6.2 %) and
0.02 % never reach DVS 2 (against 0.29 %). What replaced it is a uniform over-build —
median peak LAI **7.31**, only **5.5 %** of cell-seasons inside the expected 4–6 — on
1 067 g m⁻² of biomass, i.e. a canopy larger than SIMPLACE's (6.50) out of 40 % less dry
matter. That ratio is `slatb` and `scale_factor_sla` almost by definition, with
`scale_factor_rdr_leaves` and `rdrltb` as the other side of the same balance: leaf area
per gram is too high, or leaves are kept too long, or both. Keep the failure-rate metric
per region — it is what would catch a regression — but the stage is now scored on the
median and on the 4–6 share.

**One consequence for staging.** Stage 3 will *lower* LAI, and lower LAI intercepts less
light, so the stage-2 yield fit will move down with it. The retention term `λ₂·L_yield`
was sized for a stage 3 that raised a deficient canopy; sized for one that cuts an
excessive one it has to be larger, or stage 2 has to be re-fitted after stage 3 (the
optional stage 4 joint fine-tune, which is now closer to mandatory than optional).

## Module layout

`src/cropmodelling4eu/calibration/` — **written**. Stage 1 runs end to end on the
`EU_val` export: 33 regions over 68 685 cells, 835 CLMS units × 8 seasons = 5 895
unit-years, 50 170 cell-years in the memory-mapped season cache. Two things the plan got
wrong turned up in the building and are recorded in *What building it turned up* below;
everything else is as specified.

| File | Contents |
| --- | --- |
| `config.py` | `CalibrationConfig` and the constants this document fixes — the CLMS offsets, the autumn cut, the per-stage spec files |
| `regions.py` | `CalibrationRegions` — EnZ × calendar × yield clustering, cached against **both** the cell set and the settings |
| `observations.py` | `ObservationSet` — one long table `(unit, year, variable, value, weight, support)` unifying **CLMS phenology** (primary), CyBench yield, CyBench fpar and PEP725. **Wraps `data4simplace.phenology.pep725`; does not reimplement it** — that module already does the calendar-year → harvest-year alignment, the containment match and the circular statistics, and re-deriving them is how the alignment bug comes back |
| `dataset.py` | `CellYearDataset`, `StratifiedFractionSampler`, `collate_region_batch` |
| `losses.py` | `PhenologyLoss`, `YieldLoss`, `CanopyLoss`. Dates come from `crossing_day`; the sigmoid-sum helper is for the fAPAR senescence term only, which is the one non-monotone target |
| `problem.py` | `CalibrationProblem` — per-region `CalibrationManager`, hierarchical pooling, `SimplexGroup` |
| `checkpointed.py` | segmented gradient-checkpointed forward over `compute_rates` / `update_state` |
| `splits.py` | `make_split` — the year-block and held-out-unit splits, and the leave-one-year-out the phenology term needs instead |
| `trainer.py` | `Calibrator` — stage loop, Adam, early stop, logging, resumable state, and the two pre-stage diagnostics |
| `params_{phenology,yield,lai}.yaml` | parameter specs in torchcrop's **existing** `load_calibration_config` schema, plus three keys this package reads itself: `tier`, `blocked` and `simplex` |

CLI: `cm4eu calibrate --stage phenology --config config.yaml --crop-file <workspace>/crop_wheat.yaml`.
`--prepare-only` builds the regions, observations and weather cache and reports them, which
is what to run first on a new export. `--pep725 <dir>` adds the station data, without which
there is no heading stage and no ground-truth emergence anomaly.

**A parameter is not freed unless the observations identify it, and the reason is carried
into the output.** `tier: 2` marks a parameter needing a third dated stage in the region;
`blocked: "<reason>"` marks one the data cannot identify at all. Both are filtered at load
and both reasons are written into the stage's `summary.json`, so a run that could not fit
the N group says *why* rather than silently omitting it. `--free-blocked` overrides,
deliberately awkwardly: every current blocker is a data problem, not a code one. On this
export that is 6 of 24 yield parameters (the N group, `nrf`), 4 more (the heat group) and
2 of 20 canopy parameters.

**The output of a stage is a runnable crop file, not a table of numbers.** Each stage
writes `crops/crop_region_NN.yaml` in torchcrop's own preset layout, on the layout of the
workspace file the run started from, so a calibrated region is used by pointing
`CropParameters(config_file=...)` at it and diffs cleanly against its own starting point.

### What building it turned up

Seven of these, and every one is a silent failure — nothing raises, and the numbers stay
plausible. Four were found only by reading a *running* job's log rather than
trusting that it was fine.

**1. `crossing_day(dvs, 2.0)` has no gradient, because `euler_update` clamps `dvs` to
`[0, 2]`.** On the day the crop matures the stored value is *exactly* 2.0, so the
interpolation fraction `(2 − x0)/(x1 − x0)` is exactly 1.0 — sitting on its own
`clamp(0, 1)` boundary, where the derivative is zero. The maturity date then comes back as
a whole number and `d(maturity)/d(tsum1)` is **0 analytically while a finite difference
measures 0.27 d per 100 °C·d**. A stage 1 built on the stored trajectory trains `tsumem`
and nothing else. The fix is to read the *uncapped* trajectory `dvs_prev + dvs_rate`,
identical to the stored one on every other day since the rate is gated on `dvs < 2`;
`checkpointed.py` exposes it as `dvs_uncapped` and every DVS crossing uses it. Emergence
was never affected — `tsump` has no upper clamp — which is why the failure would have
looked like "only `tsumem` moves", i.e. exactly the result stage 1 expects.

**2. The 330-day window is far too short — the crop never matures in it.** *Three
mechanics* sized it as "sowing − 15 d to maturity + 30 d ≈ 330 days". Measured on the
harmonised SUSTAg `WW` crop at `idsl = 2`: anthesis lands ~295 d after sowing and **DVS 2
is not reached inside 345 days at all**. The window also has to span the gap between a
region's anchor sowing day and its latest sower. `data.window_days` is therefore **450**,
not 330. This is the same long season `simplace.collect`'s duration wrap is about — a
number of these seasons genuinely exceed 365 days — so it is a property of the crop, not
of the window. At 330 days the maturity term would simply have been dropped as
"never reached", by the same NaN masking that is correct when a crop really does not
mature.

**3. The prior and pooling penalties were applied once per *batch*, not once
per *step*.** Both summed over **every** region and were added to every batch's
loss, so with 51 batches over 29 regions each region was pulled toward its
prior ~28 times for every time its own data was seen — the prior winning by
arithmetic rather than by evidence. Two further consequences followed from the
same line: `optimizer.step()` ran on every batch, so a region took ~50 Adam
steps an epoch on prior gradient alone, carrying momentum with it; and
`clip_grad_norm_` over the whole parameter set scaled the one real gradient by
a norm dominated by the 28 regions the batch never touched. Both penalties now
take a `region_id` and the clip is scoped to it, so a batch touches exactly one
region's latents. **This is also what forced the whole calibration onto one
core** — with every batch touching every region there were no independent
batches to run concurrently.

**4. PEP725 heading was never loaded, so tier 2 could never unlock.**
`pep725.load_observations` filters to `PHASES` — codes 0, 10 and 100, the stages
with a **CLMS counterpart** — and that module drops BBCH 51 on purpose, saying
so. A wrapper asking for heading *afterwards* therefore finds nothing, silently:
the first full run logged `0 of 28 regions have a third dated stage` and froze
the vernalisation block and the photoperiod ramp for the whole calibration. The
fix is an additive `extra_phases` argument, so `PHASES` keeps meaning "pairable
against CLMS" while a caller that needs no counterpart can ask for more. The
same call also defaulted to CLMS's 2017–2024 window, discarding the 1990–2016
seasons this document names as PEP725's other unique contribution. Loading both
takes the station pool from 5 455 emergence rows over 979 stations to
**19 587 emergence, 21 199 harvest and 19 309 heading** over 1 991.

**5. A missing date entered the loss as a finite sentinel.**
`pd.to_numeric` on a `datetime64` column returns nanoseconds and maps `NaT` to
the int64 minimum, `-9223372036854775808`. That is *finite*, so neither the
`notna` filter that built the table nor the `isfinite` mask in `_masked_mse`
removed it, and ~940 station-seasons were scored against a date 292 years before
the Big Bang. It never bit on CLMS, whose reduced parquet has no gaps — it needed
PEP725, where 3 800 seasons carry no emergence and 4 566 no heading, which is the
common case rather than the edge one. Missingness is now tested on the datetime
column *before* it is converted.

**6. A cell can sit in the pool under two regions, and needs a window per region.**
PEP725 makes a cell its own observation unit; CyBench puts that same cell inside
an `adm_id`. The two units can fall in different regions, and each region anchors
its window on its *own* earliest sowing day. The season cache was keyed on
`(cell, year)` and kept whichever anchor it met first, so in the other region's
batch that cell disagreed with its peers about which day the window opens —
`collate_region_batch` refused it and the first full run died there. The key is
now `(cell, year, anchor)`. The same fix carries a second, subtler one: the
window start is snapped to an exact day-of-year, because the same calendar date
has a different `dayofyear` in a leap year and the engine derives everything from
one scalar `start_doy`. **Neither appears without PEP725 loaded**, which is why
nothing before it hit them.

**7. A latent seeded exactly on its bound is frozen, not free.** The
manager's bijection is `lo + (hi − lo)·σ(z)`, and inverting it at a bound clamps `z` to
±13.8 where `σ'(z)` is ~1e-6. The SUSTAg crop starts *on* several bounds by construction —
`versat` at 70, the `vernrt` plateau at 0 and 1, both `phottb` knots — so `RegionManager`
seeds every init 2 % of its range inside the box, and a test asserts no latent sits on one.

### First run: stage 1 on EU_val

Three epochs — a smoke test, not a fit: at 1.8 Adam steps per region per epoch that is
about five steps each, against the ~146 the full 80 epochs buy. It is here because it is
the first end-to-end evidence the machinery works, and because two of its numbers are
worth acting on. Scored on the **test split** — harvest year 2024 plus 111 held-out
`adm_id` — against the *same* split for the uncalibrated crop, so the two columns differ
only by the parameters.

| | Uncalibrated (`tsumem = 60`) | After 3 epochs | Target |
| --- | --- | --- | --- |
| Loss (days², all terms) | 1123.6 | **780.5** | — |
| Emergence bias | −4.5 d | +4.6 d | \|bias\| < 7 ✓ both |
| Emergence RMSE | 22.7 d | 21.7 d | |
| Harvest bias, offset-adjusted | +7.6 d | **−2.7 d** | \|bias\| < 7 ✓ calibrated only |
| Harvest RMSE | 9.8 d | **7.8 d** | |
| Sowing → emergence lag | 4.4 d | **13.7 d** | within 5 d of 12 ✓ calibrated only |
| Maturity after harvest | 8.2 d | **2.0 d** | the hinge, working |
| Emergence anomaly RMSE | 13.5 d | 19.4 d | ✗ worse |
| Harvest anomaly RMSE | 3.9 d | 5.3 d | ✗ worse |
| Never matured | 0.0 % | 0.0 % | |

**The levels move the right way and the anomalies move the wrong way.** That is what five
steps of a level-dominated objective should do — `emergence_anomaly_clms` is weighted 0.05
precisely because CLMS emergence anomaly is mostly product noise, so nothing is holding
that column. Whether it recovers over 80 epochs is the first thing a real run has to show.

Where the four parameters went, over 28 regions:

| | Start | Median | Range | Regional spread |
| --- | --- | --- | --- | --- |
| `tsumem` | 150 (init) | 147 | 113 – 208 | 95 |
| `tsum1` | 1125 | **952** | 890 – 1115 | 225 |
| `tsum2` | 1000 | **869** | 741 – 1006 | 266 |
| `versat` | 68.7 | 67.9 | 65 – 69 | **3.9** |

One thing to read off it. **`tsumem`'s median barely moved, because the spec's
`init: 150` already did that work** — the uncalibrated 60 is outside the bound on purpose,
so most of the sowing-to-emergence correction (4.4 → 13.7 d) is the initialisation and not
the fit; what the fit adds is the 113–208 regional spread.

> ⚠ **This table was produced before finding 3, and `versat`'s row is not
> evidence.** Its spread of 3.9 against `tsum2`'s 266 was first read as "weakly
> identified, prior-dominated" — a plausible story, and consistent with its
> gradient of ~1. But the run that produced it applied the prior ~28 times per
> epoch against the data's ~1.8, so *every* parameter was over-pulled toward
> its prior and `versat`, having the smallest data gradient, was pulled hardest.
> Whether it is genuinely unidentifiable without a third dated stage is now an
> open question, not a finding. It is also on the **site calendar**, not
> SIMPLACE's sowing dates. Both are fixed in the run this table will be
> replaced from.

**One number does not reconcile, and it should be resolved before either is quoted.**
`full_run_evaluation.ipynb` reports the uncalibrated emergence bias as **−31.5 d**; this
pipeline measures **−4.5 d** on its own pool (−10.8 d before the CLMS emergence offset).
The alignment here is verified rather than assumed — converting the window-relative dates
back to day-of-year gives simulated emergence at DOY 294 against an observed 296, and
simulated maturity at DOY 212 against an observed harvest of 212, on the correct calendar
years. And the **sowing-to-emergence lag agrees closely** (4.4 d here, 3.3 d in the
notebook), which is the diagnostic that localises the disagreement: both are read off the
same simulated emergence, so the difference is on the **observation** side, not the model's.
Candidates are the −6.0 d offset applied here and not there, and how the notebook pairs
CLMS emergence against the run's `emergence_doy`. Until it is settled, the *relative*
column above is the trustworthy one and the absolute emergence bias is not.

### The dataloader

**The sampling unit is `(observation_unit, harvest_year)`** — an `adm_id`-year for
CyBench, a station-year for PEP725 — not a bare cell. CyBench observations are reported on
`adm_id`, so the loss support and the sampling support should be the same object.

```python
StratifiedFractionSampler(fraction=0.20, strata=("region", "year"), seed=...)
```

- Draws **without replacement within an epoch**, stratified by region **and** by year. An
  unstratified draw can silently produce a "wet years only" epoch, and the parameters will
  chase it.
- Emits **region-homogeneous batches** (forced above), 128–256 cell-years each.
- Deterministic per-epoch seeding, so a killed run resumes the identical sequence.

**The batches are region-homogeneous but not year-homogeneous, and that is a measured
choice.** Every window in a region opens on the same *day-of-year* — the region's anchor
sowing day minus `pre_days` — so seasons from different harvest years share one relative
time axis and one scalar `start_doy`, which is all the engine needs. Forcing one year per
batch as well would have made them tiny: 835 units over 33 regions and 8 seasons gives a
region-year about five units at a 20 % draw, some 40 cells against a 192 limit, and
LINTUL-5's day loop costs almost the same at 40 cells as at 192. So the *draw* is
stratified by region **and** year and the *packing* is by region alone, which fills the
batch and lets the running climatology see several seasons of a unit at once. CO2 became
per-cell in `build_site_params` for the same reason: it rose 90 ppm across the export's
window, so one value per batch would be a real error.

**`fraction: 0.20` — the per-iteration draw.** With ~6 700 CLMS unit-years over 15–30
regions, one epoch draws ~1 340 unit-years, i.e. **~45–90 per region**, which is 1 batch at
the low end and 1–2 at the high end — enough that every region's gradient is estimated from
a spread of years rather than from one. Yield's pool (805 CyBench units × 25 years) is
three times larger, so 20 % there is ~4 000 unit-years and the constraint is memory, not
support: cap it with the batch size, not by dropping the fraction. The floor is
`min_observation_years_per_region` — a region whose 20 % draw cannot reach two distinct
years should be merged at region-construction time, not patched in the sampler.

**Spatial support of the loss:**

| Observation | Support |
| --- | --- |
| PEP725 | **point-matched to the cell that contains it** — `data4simplace.phenology.pep725.match_to_grid`. Not `germany.py:MATCH_KM`, which takes the nearest centre within 25 km: on a 0.1° grid the nearest centre to an interior point *is* its own cell, so the radius only ever admits stations from cells they are not in |
| CLMS phenology, CyBench yield, CyBench fpar | **aggregate-then-compare**: simulate the cells of an `adm_id`, area-weight to the adm mean, compare. The weighted mean is linear, so gradients pass cleanly. CLMS also publishes a per-cell table, so a phenology term can run on either support — prefer `adm` where the yield term already does |

That second row is why the sampler draws whole adm units rather than individual cells.

**Weather caching is the biggest speed decision.** The production runner reads each cell's
46-year gzip once; a calibration run revisits the same cells every epoch. Pre-extract the
sampled pool's seasons once into a memory-mapped `float32 [n_cell_years, T, C]` array — and
size it deliberately: 8 000 cells × 25 years × 330 d × 8 vars × 4 B ≈ 210 GB, so the pool
must be restricted to cell-years that carry an observation, and the cache built per pool,
not per domain.

## Splits, diagnostics, acceptance

- **Splits:** 70/15/15, split by **year blocks** (temporal extrapolation) *and* by held-out
  `adm_id` within each region (spatial transfer). Report both. **The phenology term cannot
  have this split**: CLMS covers 8 seasons, so a 15 % year block is one year. Use
  leave-one-year-out there and keep the 70/15/15 for the yield term, which has 25.
- **Regularisation:** L2 toward the physical prior — JRC per region for phenology, the
  SIMPLACE `crop.xml` values elsewhere — scaled by the prior's cross-zone spread.
- **Gradient sanity:** finite-difference check per free parameter on a handful of cells
  before any stage runs. Zero or non-finite gradients almost always mean `smooth=False`, or
  a non-differentiable target that slipped into the spec.
- **Identifiability:** Jacobian condition number and the parameter correlation matrix per
  region. Correlation above ~0.95 between two free parameters means one should be frozen.
- **Independent check:** calibrated `tsum1` against the JRC `TSUM1` field. It was never in
  the loss.

Targets are set against the **full run** (`winter_wheat_2000_2024_euval`, 2000–2024;
phenology on CLMS's 2017–2024 window), SIMPLACE on the identical pairs. Targets name
spreads and anomaly amplitudes as well as biases: a pooled bias is the thing torchcrop was
always closest to and the thing that says least.

**Phenology targets are stated against CLMS**, so they are not comparable with any earlier
version of this table. The harvest level target is written as a *residual after the
offset* rather than as a raw bias, and the offset used is the **ground-truth −18.7 d
against PEP725**, not the −28.9 d against CyBench `eos` — the two references disagree with
each other as well as with CLMS, and only one of them is a person in a field.

| Metric | Target | torchcrop | SIMPLACE, same pairs |
| --- | --- | --- | --- |
| Emergence bias | \|bias\| < 7 d | **−31.5 d** — the binding phenology constraint | −31.1 d |
| Emergence anomaly amplitude | against **PEP725's ~8 d**, not CLMS's 13.6 (mostly product noise); as far as `tsumem` reaches, est. ~3.7 d — the rest is the sowing window | **2.1 d** | 2.1 d |
| Emergence anomaly r | > 0.3 **against PEP725** — scored against CLMS emergence it is scoring noise, since CLMS-vs-PEP725 is itself only 0.11 | **0.00** | −0.00 |
| Sowing → emergence lag | within 5 d of the ground-observed **12 d** (PEP725) | **3.3 d** | 5.0 d |
| Harvest bias, offset-adjusted | \|bias + 18.7\| < 7 d — the offset is now **measured against ground truth** (PEP725), not inferred from CyBench | +24.9 → **+6.2 d** ✓ | +19.1 → **+0.4 d** ✓ |
| Harvest spatial r | > 0.85 | **0.89** ✓ | 0.91 ✓ |
| Harvest anomaly r | > 0.6 | **0.67** ✓ | 0.65 ✓ |
| Harvest anomaly amplitude | within 50 % of the observed 6.1 d | **4.8 d** (79 %) ✓ | 4.9 d ✓ |
| Yield bias | \|bias\| < 0.5 t/ha | **−0.68 t/ha** | +1.19 t/ha |
| Yield per-country bias spread | 90 % of countries within ±1.5 t/ha | **61 %** (14/23), range −1.78 … +2.96 | 43 % (10/23), −0.27 … +6.11 |
| Yield skill | **r² > 0.2** | **+0.03** — the binding yield constraint | −0.47 |
| Peak LAI | median 4–6, **and** < 2 % of cell-seasons below 1.5 | median **7.31** ✗, 0.13 % ✓ | median 6.50 ✗, 2.23 % ✗ |
| Harvest index | median 0.45–0.55, < 5 % above 0.6 | **0.408** ✗ (low), 0.7 % ✓ | **0.343** ✗, 0.0 % ✓ |
| fAPAR seasonal RMSE | < 0.10 | — | — |

**Three targets moved because the run moved, and they are worth naming.** Yield skill was
`r² > 0` and torchcrop met it (+0.03), so the target is raised to 0.2 — passing a
constraint by 0.03 is not a calibrated model. The old maturity targets (bias, MAE, ±20 d
per-unit spread) are **retired**: they were measured against CyBench `eos` on the
unharmonised crop, and harvest now passes every skill test that survives the reference
change. The harvest-index target survives unchanged but is now missed from **below** on
both models rather than from above on torchcrop.

**No heading target, and now for one reason rather than two.** SIMPLACE's collected run
*does* write `anthesis_doy`; torchcrop's does not — `run_batch` discards the DVS
trajectory once it has placed the fertilizer schedule. So the missing half is torchcrop's,
and it is one `crossing_day(dvs, 1.0)` call in the same place `days_to_emergence` was
added. Do that before stage 1 frees `phottb` or `vernalisation_devstage`: anthesis is the
only observation that separates the pre- from the post-anthesis thermal sum, and without
it `tsum1` and `tsum2` are identified only by their sum.

**A data note that outlives this run.** `simplace.collect.to_run_schema` derives
`days_to_maturity` and `days_to_emergence` as `stage_doy − sowing_doy` wrapped forward
into `(0, 365]`, so a season running past its own sowing anniversary is recorded as the
remainder — a 380-day season is written as 15 days. On this run that is **1.97 %** of the
2000–2024 rows (3.61 % over SIMPLACE's full 1979–2024 window). The populations separate
cleanly: genuine seasons start at 160 days, wrapped ones stop at 79, and 12 of 1.7 M rows
lie between. `fullrun._load_simplace` therefore drops the **durations** of rows below
`config.MIN_PLAUSIBLE_SEASON_DAYS` (150) and keeps their dates, which is why the harvest
row above is scored on 5 811 unit-years and the season-length row on 5 809. Fixing it
properly means carrying the sowing *date* rather than its DOY through the collector.

## Blockers to clear first

Five of the seven are cleared, and **stage 1 has been written and run**. The two that
remain are both stage 2's, both are data problems rather than code ones, and both are now
enforced in the parameter specs rather than only recorded here: the parameters they block
are marked `blocked:` with their reason, dropped at load, and the reason is written into
the stage's `summary.json`.

1. ~~**`idsl = 2` has no override path on the torchcrop side.**~~ **Cleared.** torchcrop is
   harmonised against the SUSTAg `WW` block and runs `idsl = 2` with `versat = 70`,
   `vbase = 14` and SIMPLACE's own `vernrt` curve — see *Why*. **Every run before
   2026-09-04 used a different crop on each side**, so no earlier §6 comparison is a model
   comparison.
2. ~~**Neither run emits an emergence date.**~~ **Cleared and exercised.** SIMPLACE's
   `WLOutputs` writes `Phenology.EmergenceDOY` / `AnthesisDOY` / `MaturityDOY`;
   `torchcrop/run.py` writes `days_to_emergence` from the `tsump = tsumem` crossing. Both
   are in the `_euval` Parquets, `config.CLMS_STAGES` scores emergence as a real bias
   rather than as a sowing proxy, and §5.8's season-length decomposition closes to
   **−0.2 d** on both models — which is the check that the two dates and the duration
   describe the same season.
3. ~~**The run this document is specified against is being re-made.**~~ **Cleared.**
   `winter_wheat_2000_2024_euval` is finished on both sides and
   `full_run_evaluation.ipynb` has been re-run on it; every number in *Why* and in
   *Splits, diagnostics, acceptance* is from that run. Two things moved at once — the crop
   parameters *and* the export (EU → EU_val) — so a change is not attributable to either
   from this notebook alone; `stresstest_evaluation.ipynb` is where that separation is
   done, and it has **not** been re-run.
4. ~~**CLMS phenology needs a CDSE account.**~~ **Cleared.** The reduction is on disk at
   `EU_val/validation/phenology`, 835 units × 8 seasons survive the quality gates, and it
   is the phenology reference the evaluation now uses. Its cost is the 2017–2024 window
   and the −28.9 d level offset against CyBench, both handled in *Observations →
   Phenology*.
5. **The heat-stress group is unmapped.** `vTCritical` / `vStartDVS` / `vEndDVS` have
   torchcrop counterparts and are not in `SCALARS`, so the two models differ by 4 °C and
   0.05 DVS on a group stage 2 frees. Add them before stage 2 runs — see
   *What harmonisation does not reach*. **Still open.**
6. **Initial mineral N** makes stage 2's N group unidentifiable. Pooled `nni` is now
   **0.985** over 1.7 M cell-seasons — worse than the 0.96 this blocker was written
   against, so nitrogen is essentially never limiting anywhere in the domain. Fix
   `soil.mineral_n_fraction` in the export, or drop `nmaxso` / `frnx` / `nlue` / `tcnt` /
   `dvsnt` / `nrf` from stage 2 and record why. **Still open, and now unambiguous.**
7. ~~**torchcrop emits no anthesis date.**~~ **Cleared.** `run.py`'s summary frame now
   writes `days_to_anthesis` from `crossing_day(dvs, 1.0)`, beside the emergence column
   and through the same function. It is the only observation that separates the pre- from
   the post-anthesis thermal sum, so until a run carries it the ratio prior in
   `params_phenology.yaml` is what pins that direction — see the identifiability rule,
   which is about how *weakly* the split is identified without it.
