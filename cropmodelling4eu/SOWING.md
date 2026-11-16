# Learned sowing dates (`cropmodelling4eu.sowing`)

A causal hazard network that replaces SIMPLACE's rule-based sowing **date**,
trained through torchcrop's own emergence clock on corrected CLMS emergence
and PEP725 sowing. It produces the `(SimplaceID, year, sowing_doy)` table both
models already read; no crop model code changes.

## Why

The SUSTAg rule (`SowingRule`) sows on the first window day on which
`Avg7Tmin <= 12 C` **or** `Rain3Sum >= 20 mm` **or** topsoil moisture is above
half the plant-available water. The temperature branch is true when most
European windows open, so 72 % of cell-years sow on the opening day.
Production notebook §7, against PEP725: bias −7.3 d, year-to-year correlation
0.00, and about −30 d in AT/SK/PL/HU/BE. The moisture branch reads simulated
soil water, so a different crop file moves the date: production and euval
disagree on 8 % of cell-years.

## Method

| Part | What | Where |
| --- | --- | --- |
| Cache | per (cell, harvest year): daily weather from **DOY 193 of Y-1** for 240 d (SAGE `plant.start` min 213 − 20) | `cache.py`, `features.py` |
| Short-term branch | causal TCN, receptive field **21 d**, weather only: no DOY, no statics | `model.py` |
| Medium-term branch | causal state: topsoil bucket (continuous over the record), Tmin7, Rain3, Rain30, dry spell, degree days | `features.py` |
| Long-term branch | normals over autumns **Y−31 … Y−2** plus statics (location, SAGE window, topsoil FC/WP/texture). Sets the baseline `mu` and FiLM-modulates the short branch | `features.py`, `model.py` |
| Hazard | `P(sow=s) = h_s ∏(1−h)`, last candidate day (DOY 365) forced | `model.py` |
| Emergence step | torchcrop's thermal clock, **fixed**: `clamp((tmin+tmax)/2 − tbasem, 0, teffmx − tbasem)` summed from the sowing day, `crossing_day` at `tsumem`. Matches torchcrop to < 0.1 d in 99.96 % of production cell-seasons | `emergence.py` |
| Loss | Laplace-mixture NLL of CLMS emergence (CPMCE − 6 d) and PEP725 emergence over `P(sow)`, plus PEP725 sowing | `model.py`, `train.py` |
| `tsumem` | one global value fitted on PEP725 sowing→emergence lags, **train blocks only**, before the network | `tsumem.py` |
| Hold-out | PEP725 station blocks (1°) 70/15/15, drawn once (`pep725_split.parquet`); CLMS by the calibration's `phenology_split.parquet` | `targets.py` |

The network never sees weather after a candidate sowing day. That weather
reaches only the fixed emergence table inside the loss, as it does in
torchcrop itself.

## Running

```bash
./submit/sowing/submit_sowing_network.sh              # split, build-cache, fit-tsumem, train, predict (one node)
cm4eu sowing <step> [--sowing-config my.yaml]         # any step by hand
./submit/sowing/submit_sowing_comparison.sh --dry-run # then the five torchcrop arms
python notebooks/sowing_comparison/build_notebooks.py # and execute sowing_comparison.ipynb
```

Outputs go under `Data/SIMPLACE/cropmodelling4eu/sowing_network/`: `cache/`,
`tsumem.json`, `pep725_split.parquet`, `model/`, `sowing_network.csv` (B0/B1),
`sowing_network_climatology.csv` (C), `sowing_network_detail.parquet`
(p10/p90, deadline probability), and `crops_pep725/` (a copy of the calibrated
crops with `tsumem`/`tbasem` replaced; `override.json` records the old values).

## Comparison arms (torchcrop, MSWX) → `runs/winter_wheat/sowing_comparison/<arm>`

| Arm | Sowing | Crops |
| --- | --- | --- |
| A0 | `sowing_from_simplace_euval.csv` | `calibration_dm/yield/crops` = the old method |
| A1 | same | `crops_pep725` |
| B1 | `sowing_network.csv` | `calibration_dm/yield/crops` |
| B0 | `sowing_network.csv` | `crops_pep725` = the new method |
| C | `sowing_network_climatology.csv` | `crops_pep725` |
| D | per cell-season argmax of simulated yield over `sowing_network.csv` + offsets (B0 = offset 0) | `crops_pep725` |

**D is an upper bound, not a method.** `cm4eu sowing shift` writes the network
table moved by each of `ORACLE_OFFSETS` (default ±14/28/42 d, clipped to DOY
200–365) into `sowing_network/oracle/`; each runs as a plain arm under
`sowing_comparison/D_grid/<m42..p42>`; `cm4eu sowing oracle` keeps per (cell,
season) the offset with the highest `yield_t_ha` (ties to the smaller shift) and
writes `D/torchcrop/torchcrop_europe.parquet`, `D/sowing_yield_max.csv` and
`D/oracle_summary.json`. It sees the whole season, so it bounds what sowing
alone can do in torchcrop. Comparison notebook §6.

`provenance.json` per arm records the sowing table and its sha256, the crops,
the `tsumem` source and the fallback-crop overrides.

## Falling back to the old method

Nothing that existed before was changed in place. Each old path is the default:

| To use the old… | Do |
| --- | --- |
| sowing (MSWX torchcrop) | `MSWX_SOWING` unset in `submit_production.sh` → `sowing_from_simplace_euval.csv` |
| sowing (CMIP6) | `CMIP6_SOWING` unset → `sowing_climatology_simplace_euval.csv` |
| crops / `tsumem` | `CAL_DIR=…/calibration_dm` (its `yield/crops` are never edited; arms read a *copy*) |
| fallback crop | `TC_CROP_OVERRIDES` unset (the default) in `submit_torchcrop.sh` |
| calibration | `fixed_parameters: {}` (the default) and `loss.emergence_level: 1.0` |
| production runs | `runs/winter_wheat/calibrated/` is never written by the sowing scripts |

To adopt the new method instead:

- **Production MSWX:** point `MSWX_SOWING` at `sowing_network.csv` and
  `CAL_DIR` at a calibration whose crops carry the PEP725 `tsumem`.
- **Calibration:** set `fixed_parameters: {crop.tsumem: <tsumem.json>,
  crop.tbasem: 0.0}` and `loss.emergence_level: 0`.
- **CMIP6:** run `predict` on a cache built from that model's export. The
  long-term branch then reads that model's own prior-year normals.

## Recalibrate or not

`tsumem` moves emergence only. The comparison notebook's §4 reports the
**sowing effect on maturity and yield (B1 − A0)**. If maturity moves by more
than about 2 d, refit the phenology and yield stages on the network's sowing
with `tsumem` fixed. Otherwise the calibrated crops stand.
