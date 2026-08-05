# torchcrop Europe run (SLURM)

Runs the differentiable LINTUL-5 model
([`torchcrop`](https://geonextgis.github.io/torchcrop)) over the **finished**
SIMPLACE Europe export as a SLURM **array job — one task per shard**, then
combines the shards and renders spatial maps.

This consumes what [`submit_europe.sh`](README.md) produces; it does not run the
`data4simplace` pipeline. The modelling assumptions, unit conversions and the
list of inputs the export cannot supply are derived in
[`notebooks/torchcrop_europe_run.ipynb`](../notebooks/torchcrop_europe_run.ipynb)
— read that first.

## Quick start

```bash
./submit/submit_torchcrop.sh --dry-run    # show the plan, submit nothing
./submit/submit_torchcrop.sh              # shard array + combine + maps
./submit/torchcrop_status.sh              # progress, queue, failed shards
./submit/submit_torchcrop.sh --retry      # resubmit only the missing shards
./submit/submit_torchcrop.sh --maps-only  # re-render maps from existing shards
```

Defaults (all overridable, see [`torchcrop_env.sh`](torchcrop_env.sh)):

| | |
| --- | --- |
| input | `/data01/FDS/muduchuru/Data/SIMPLACE/europe_torchcrop` |
| cells | weather ∩ soil ∩ management = **68 685** of 70 705 |
| seasons | `TC_START_YEAR=2000` … `TC_END_YEAR=2024` (winter wheat, sown 1 Sep of Y−1) |
| run mode | `TC_IOPT=3` — water + N limited |
| shards | `TC_N_SHARDS=10` → ~6 900 cells each |
| batching | `TC_BATCH_SIZE=2048` cells per model call, `TC_IO_WORKERS=16` |
| resources | `compute`, 16 cpus, 16 G, 6 h per task, 10 concurrent |
| output | `/data01/FDS/muduchuru/Data/SIMPLACE/torchcrop/<TC_RUN_NAME>` |
| env | conda `sdba` (holds `torch` + the editable `torchcrop`) |

```bash
TC_START_YEAR=1980 TC_TIME=12:00:00 ./submit/submit_torchcrop.sh   # all 45 seasons
TC_RUN_NAME=potential TC_IOPT=1 ./submit/submit_torchcrop.sh       # potential yield
TC_RUN_NAME=test TC_END_YEAR=2001 ./submit/submit_torchcrop.sh     # 2 seasons
```

## Sharding and batching

Two independent axes, and they do different jobs.

**Shards** are the SLURM parallelism — one array task each. Cells are dealt
**round-robin** (`ids[shard::n_shards]`), not in contiguous blocks: `SimplaceID`
runs north-to-south, so a contiguous split would hand one task all of
Scandinavia and another all of Iberia, with very different weather-file sizes
and very different wall times. Round-robin makes every shard a domain-wide
sample, so the tasks finish together.

**Batches** are the within-task parallelism. LINTUL-5 is a Python loop over
days, so per-cell cost collapses as the batch grows — the loop runs once for
2 000 cells instead of 2 000 times:

| batch | ms per cell-year | peak RSS |
| --- | --- | --- |
| 128 | 40 | — |
| 1 024 | 9.4 | — |
| 2 048 | **5.2** | 1.7 GB |

The ceiling is memory, not cells: `ModelOutput` retains every per-day state,
rate and diagnostic for the whole batch. 2 048 sits well inside a 16 G task.

At that rate a 45-season run over 68 685 cells is ~30 min per shard, of which
~2 min is I/O.

## Weather I/O

The 46-year weather file is read **once per cell** and every requested season is
sliced out of it. A gzip stream has no random access, so the whole file is
decoded whichever season is wanted — decoding it once instead of once per year
is the difference between one pass and 45. Reads are gzip-bound and release the
GIL, so `TC_IO_WORKERS` threads scale; `TC_TORCH_THREADS` stays low because the
model works on `[B]`-shaped tensors that are too small to repay many intra-op
threads.

## Outputs

```
$TC_OUT_DIR/
├── shards/torchcrop_shard_000..009.parquet   one per array task
├── torchcrop_europe.parquet                  every (cell, season) row
├── torchcrop_europe_grid.nc                  (year, lat, lon) cubes + climatology
└── maps/
    ├── map_overview.png                      8-panel climatology
    ├── map_yield_anomaly_by_year.png         per-season anomaly panels
    └── map_<variable>.png                    one per variable
```

Per-(cell, season) columns: `yield_g_m2`, `adjusted_yield_g_m2` (heat-stress
adjusted), `biomass_g_m2`, `max_lai`, `final_dvs`, `days_to_maturity`,
`n/p/k_applied_g_m2`, `tranrf_mean` and `nni_mean` (water and N stress averaged
over the days `0 < DVS < 2`), `heat_stress_factor`, `irri`.

In the NetCDF, the across-year climatology carries a **`_clim`** suffix, not
`_mean` — `tranrf_mean` and `nni_mean` are already growing-season means, so
`_mean` would be ambiguous between the 3-D per-year cube and its 2-D
climatology.

## Failure handling

Each shard writes only its own Parquet, so a failed task is re-runnable alone
and `--retry` submits exactly the missing indices (a comma list, not a range —
failures are rarely contiguous).

`torchcrop_maps.py` **refuses** to map an incomplete shard set. Because shards
are dealt round-robin, a missing shard does not leave an empty corner; it
speckles ~1/N of the cells across all of Europe, which would look like sparse
data rather than a broken run.

## Files

| File | Role |
| --- | --- |
| `submit_torchcrop.sh` | Driver: sizes the array, submits the array + map jobs with the right dependency |
| `torchcrop_env.sh` | Shared settings + conda activation, sourced by every script |
| `torchcrop_array.sh` | The array job — `SLURM_ARRAY_TASK_ID` **is** the shard index |
| `torchcrop_run.py` | One shard: load, batch, two-pass simulate, write Parquet |
| `torchcrop_maps.sh` | Dependent combine + map job |
| `torchcrop_maps.py` | Concatenate shards, grid to NetCDF, render the maps |
| `torchcrop_status.sh` | Progress, queue state, failed shards |

## Caveats

Carried over from the notebook's gap analysis — these bound what the maps mean:

- **Sowing date** is a constant DOY 270 across 34–72 °N; the export carries no
  sowing calendar. This is the largest single error in the run.
- **Wind speed** depends on the export. `weather_export.py` now maps MSWX
  `sfcwind → Windspeed`, converted from the 10 m product to the 2 m equivalent
  SIMPLACE expects (FAO-56 eq. 47, factor 0.748). The current
  `europe_torchcrop` weather files predate that fix and still carry `-99.9`, so
  `torchcrop_run.py` falls back to a constant 2 m s⁻¹ and Penman ET₀ is biased.
  Re-export the weather stage and the runner picks the real column up
  automatically — the fallback triggers only when the column is entirely
  sentinel, so there is no flag to change.
- **Initial mineral N** comes straight from the export, where
  `soil.mineral_n_fraction = 0.01` puts a median 243 kg N ha⁻¹ in the profile
  against a measured 30–90. At that level the crop is effectively N-unlimited,
  so `TC_IOPT=3` behaves much like `TC_IOPT=2` and the NNI map sits near 1.
- **`TC_IOPT=4` is not data-driven** — no European soil P or K layer exists, so
  it would run on torchcrop's built-in default pools.
