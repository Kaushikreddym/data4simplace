# production/

Both models on the per-region **calibrated** crops, for any forcing.

| File | Purpose |
| --- | --- |
| `submit_production.sh` | The driver. Takes forcings: `mswx`, `cmip6`, `cmip6/<model>`, `cmip6/<model>/<exp>` |
| `config_production.yaml` | Base config of every run. Export `EU_val_nfix` (corrected initial mineral N); the driver writes a per-forcing copy with `run_name`, `export_dir` and the window replaced |

```bash
./submit/production/submit_production.sh --dry-run mswx cmip6   # checks + plan
./submit/production/submit_production.sh mswx                   # EU validation run
./submit/production/submit_production.sh cmip6/MIROC6           # one model, all experiments
./submit/production/submit_production.sh --after <jobid> cmip6  # hold behind a job
```

Options: `--simplace-only`, `--torchcrop-only`, `--after JOB`. Settings
(environment): `CAL_DIR`, `CAL_STAGE` (default `yield`), `SP_SLOTS` (10),
`TC_SLOTS` (4), `CMIP6_TC_DAILY` (0).

## Output

```
runs/winter_wheat/calibrated/<forcing>/
  config.yaml          the run's own config
  provenance.json      calibration, export, window, submit time
  simplace_regions/    one SIMPLACE build per calibration region
  simplace/            simplace_europe.parquet, sowing_from_simplace.csv
  torchcrop/           torchcrop_europe.parquet, maps/
  logs/
```

## What happens per forcing

1. **SIMPLACE.** One build per calibration region, each with that region's
   calibrated `crop.xml`, then a single array over every region and line chunk.
   A merge job collects the regions into `simplace/`.
2. **torchcrop.** A shard array in which each cell takes its region's
   calibrated crop. Cells in an uncalibrated region fall back to the SUSTAg WW
   crop, never Brandenburg.
3. **Sowing.** CMIP6 torchcrop sows on its own run's SIMPLACE dates, so it
   waits for the merge. MSWX uses the table the calibration was fitted on, so
   it starts at once.
4. **CMIP6 window.** Taken from the export's `scenario.json`. Harvest years run
   from the first year + 1 to the last year. CO₂ is set per year.

Forcings are chained: each SIMPLACE array waits for the previous one, and each
torchcrop array waits for the previous torchcrop array. Region builds run on
the login node, about 28 per forcing.

The uncalibrated counterpart is [../baseline/](../baseline/README.md).
