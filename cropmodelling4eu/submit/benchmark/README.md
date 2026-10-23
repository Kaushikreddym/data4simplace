# benchmark/

Performance and model-agreement tests. These don't produce results for
analysis.

| File | Purpose |
| --- | --- |
| `submit_benchmark.sh` | Throughput of torchcrop calibration batches: a CPU job, then a GPU job, both appending to one `benchmark.parquet`. SIMPLACE timings are read from its task logs. Plots: `notebooks/benchmark/benchmark.ipynb` |
| `submit_stresstest.py` | Runs SIMPLACE and torchcrop on the same crop and sowing date with no spin-up, removing each non-model difference in turn. Answers "do the two models agree?", not "how close to observations?". Analysis: `evaluation/stresstest_evaluation.ipynb` |

```bash
./submit/benchmark/submit_benchmark.sh --dry-run
./submit/benchmark/submit_stresstest.py --dry-run
./submit/benchmark/submit_stresstest.py --scenarios limited --years 2003
```

`submit_benchmark.sh` reads an existing calibration weather cache
(`BENCH_CACHE_FROM`). Its default, `calibration_mixcheck`, was deleted in the
cleanup, so point it at `winter_wheat_2000_2024_euval/calibration`.
