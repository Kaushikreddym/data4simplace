# baseline/

Both models on the **default**, uncalibrated crop. This is the reference the
calibrated production runs are compared against, and the home of the smoke
test, status and retry tools.

| File | Purpose |
| --- | --- |
| `submit_cropmodelling.sh` | SIMPLACE → sowing handoff → torchcrop, chained. Also `--smoke` (30 German cells, run locally), `--status`, `--retry` |
| `submit_simplace.sh` | SIMPLACE alone. Builds a run directory from a config, then delegates to the run's own generated `submit.sh` |
| `submit_torchcrop.sh` | torchcrop alone: shard array plus a combine/maps job. `--prepare` writes the workspace only |
| `torchcrop_status.sh` | Which torchcrop shards are finished or failed for a run |

```bash
./submit/baseline/submit_cropmodelling.sh --dry-run
./submit/baseline/submit_cropmodelling.sh --smoke
./submit/baseline/submit_simplace.sh --status
TC_SOWING_FILE=<run>/simplace/sowing_from_simplace.csv ./submit/baseline/submit_torchcrop.sh
```

Configuration comes from `cropmodelling4eu/config.yaml` (`SP_CONFIG` /
`TC_CONFIG` override it). Settings are the `SP_*` variables in
[../utils/simplace_env.sh](../utils/simplace_env.sh) and the `TC_*` variables
in [../utils/torchcrop_env.sh](../utils/torchcrop_env.sh). Output goes to
`<output_dir>/<run_name>/{simplace,torchcrop}`.

`config.yaml` points at the `EU_val` export, whose `soil_long.csv` has 10× too
much initial mineral N. For a like-for-like comparison with production, set
`SP_CONFIG=submit/production/config_production.yaml`.
