# utils/

Shared pieces the entry points source or `sbatch`. Nothing here needs to be
run by hand. Job bodies locate this folder through `<PROJECT_DIR>/submit/utils`,
because SLURM runs a spooled copy of each script.

## Environment files (sourced)

| File | Variables | Used by |
| --- | --- | --- |
| `simplace_env.sh` | `SP_*`: config, run dir, partition, memory, lines per task; `sp_activate` | baseline, production |
| `torchcrop_env.sh` | `TC_*`: export, run dir, shards, crop source, region crops, sowing file, daily output | baseline, production |
| `calibration_env.sh` | `CAL_*`: stage, batch sizes, workers, device, crop/sowing inputs; `cal_build_crop`, `cal_check_inputs` | calibration, benchmark |

## SLURM job bodies

| File | Job |
| --- | --- |
| `calibrate.sh` | The whole staged calibration in one process |
| `torchcrop_array.sh` | One torchcrop shard per array task |
| `torchcrop_maps.sh` | Combines the shards into `torchcrop_europe.parquet`/`_grid.nc` and renders maps |
| `cropmodelling_handoff.sh` | Collects a SIMPLACE run and writes `sowing_from_simplace.csv` for torchcrop (baseline chain) |
| `simplace_task_table.sh` | One task of a production SIMPLACE array: one line range of one region build |
| `simplace_regions_collect.sh` | Collects every region build and merges them into `simplace_europe.parquet` and `sowing_from_simplace.csv` |

## Helper script

| File | Purpose |
| --- | --- |
| `submit_simplace_regions.sh` | Builds one SIMPLACE run per calibration region, each with its calibrated `crop.xml`. Production calls it with `--build-only`; it still works standalone with `--export-dir` |
