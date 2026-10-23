# calibration/

Fits torchcrop's crop parameters per region against observed data.
Phenology is fitted against the CLMS harvest and emergence dates, and yield
against the regional yield statistics. The yield stage starts from the
phenology fit.

| File | Purpose |
| --- | --- |
| `submit_calibration.sh` | One staged calibration (phenology → yield) as one SLURM job. Builds the SUSTAg-derived crop file first if it is missing |
| `submit_calibration_ensemble.sh` | N independent calibrations side by side on one node, one per seed (`CAL_SEEDS`); the spread is the fit's uncertainty |

```bash
./submit/calibration/submit_calibration.sh --dry-run
./submit/calibration/submit_calibration.sh
CAL_STAGE=phenology CAL_EPOCHS=20 ./submit/calibration/submit_calibration.sh
./submit/calibration/submit_calibration_ensemble.sh --dry-run
```

All settings are `CAL_*` variables in
[../utils/calibration_env.sh](../utils/calibration_env.sh). The job body is
[../utils/calibrate.sh](../utils/calibrate.sh). Logs go to `submit/logs/`.

## Output

Written to `Data/SIMPLACE/cropmodelling4eu/winter_wheat_2000_2024_euval/calibration/`:

- `regions.parquet`: the region definitions.
- `<stage>/crops/crop_region_NN.yaml`: the fitted crop per region.
- `<stage>_split.parquet`, `*_identifiability.csv`, `*_gradient_check.csv`: diagnostics.

[../production/](../production/README.md) reads `yield/crops/`.

LAI is disabled (no observations). There is no joint stage.
