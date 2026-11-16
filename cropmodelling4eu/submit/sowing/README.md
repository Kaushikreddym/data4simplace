# sowing/

The learned-sowing pipeline and the torchcrop comparison against the old
rule-based sowing. See `SOWING.md` at the package root.

| File | Purpose |
| --- | --- |
| `submit_sowing_network.sh [steps…]` | One SLURM job (64 CPUs, 90G): `split`, `build-cache`, `fit-tsumem`, `train`, `predict` |
| `submit_sowing_comparison.sh [--dry-run] [--after JOB] [A0 A1 B1 B0 C D]` | The torchcrop MSWX arms (D = yield-max sowing bound over a 6-offset grid + B0), chained, each into `runs/winter_wheat/sowing_comparison/<arm>` |

Settings (environment): `SOWING_CONFIG` (a YAML of `SowingConfig` overrides),
`SOWING_CPUS`, `SOWING_MEM`, `CAL_DIR` (default `calibration_dm`), `TC_SLOTS` (4), `ORACLE_OFFSETS` (`-42 -28 -14 14 28 42`).

Neither script writes under `runs/winter_wheat/calibrated/` or edits a
calibrated crop: the arms read a copy (`sowing_network/crops_pep725`).
