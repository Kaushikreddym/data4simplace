# submit/

SLURM entry points for cropmodelling4eu, grouped by what a run is for. Run every
script from the package root (`cropmodelling4eu/`), e.g.
`./submit/production/submit_production.sh --dry-run mswx`. Every entry point
takes `--dry-run` (or `--help`) and submits nothing with it.

| Folder | Purpose | Start with |
| --- | --- | --- |
| [production/](production/README.md) | SIMPLACE + torchcrop on the **calibrated** crops, for MSWX and every CMIP6 model × experiment | `submit_production.sh` |
| [calibration/](calibration/README.md) | Fit torchcrop's per-region crop parameters against CLMS phenology and yields | `submit_calibration.sh` |
| [baseline/](baseline/README.md) | Both models on the **default** (uncalibrated) crop; smoke tests, status, retries | `submit_cropmodelling.sh` |
| [benchmark/](benchmark/README.md) | CPU/GPU throughput and the SIMPLACE ↔ torchcrop stress test | `submit_benchmark.sh` |
| [utils/](utils/README.md) | Environment files and job bodies the entry points call; not run by hand | — |
| `logs/` | SLURM logs of the calibration and torchcrop jobs (not tracked) | — |

Typical order: `calibration/` once → `production/` for each forcing, with
`baseline/` as the uncalibrated reference to compare against.

Cluster rule: keep all concurrent jobs at or below 15 nodes. The production
driver chains forcings so it stays within 10 SIMPLACE + 4 torchcrop slots.

Data locations:

| What | Where |
| --- | --- |
| Inputs (data4simplace exports) | `Data/SIMPLACE/cropmodelling4eu_inputs/exports/{mswx,cmip6/<model>/<exp>}` |
| Calibration | `Data/SIMPLACE/cropmodelling4eu/winter_wheat_2000_2024_euval/calibration` |
| Production runs | `Data/SIMPLACE/cropmodelling4eu/runs/winter_wheat/calibrated/{mswx,cmip6/<model>/<exp>}` |
| SIMPLACE templates | `Data/SIMPLACE/TEMPLATES/{Rotation_monocrop,ERA5_SIMPLACE}` |
