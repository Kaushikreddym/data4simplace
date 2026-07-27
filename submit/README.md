# Tiled Europe run (SLURM)

Runs the `data4simplace` pipeline over the continental grid as a SLURM **array
job — one task per tile**, then mosaics the results. A monolithic run OOMs the
node and exceeds what the SoilGrids WCS will serve in one request; the tiling
machinery in [`tiling.py`](../src/data4simplace/tiling.py) already provides the
unit of work (`--tile-index`), so these scripts only schedule and combine it.

## Quick start

```bash
./submit/submit_europe.sh --dry-run     # show the plan, submit nothing
./submit/submit_europe.sh               # tile array + combine + management
./submit/status.sh                      # progress, queue, failed tasks
./submit/submit_europe.sh --retry       # resubmit only unfinished tiles
```

Defaults (all overridable, see [`env.sh`](env.sh)):

| | |
| --- | --- |
| grid | 17°W–52°E, 34°N–72°N at 0.1° → 690 × 380 = 262,200 cells |
| tile size | `D4S_TILE_DEG=5.0` → 50 × 50 cells → **112 tiles** |
| resources | `compute`, 8 cpus, 48 G, 2 days per task, 8 concurrent |
| output | `/beegfs/muduchuru/data4simplace/europe` |
| env | conda `sdba` (holds the editable install) |

```bash
D4S_TILE_DEG=2.5 D4S_MAX_CONCURRENT=6 ./submit/submit_europe.sh   # 448 smaller tiles
D4S_RUN_NAME=europe_test D4S_OUT_DIR=/beegfs/.../test ./submit/submit_europe.sh
```

## Files

| File | Role |
| --- | --- |
| `submit_europe.sh` | Driver: freezes the run config, sizes the array, submits all three jobs with the right dependencies |
| `env.sh` | Shared settings + conda activation, sourced by every script |
| `tile_array.sh` | The array job — `SLURM_ARRAY_TASK_ID` **is** the tile index |
| `combine.sh` | Mosaics soil shards into `soil/soil.csv` (`--combine-only`) |
| `management.sh` | NPK/fertilizer export, which the tile array does **not** cover (see caveats) |
| `status.sh` | Progress, queue state, failed tasks |
| `tile_config.py` | Writes the per-run / per-tile config overrides |
| `tile_status.py` | Maps `.done` markers back to array ids for `--retry` |

## How it works

1. **Freeze the config.** `config.yaml` is copied to
   `<out>/_work/config_run.yaml` with `paths.output_dir` pointed at the run
   directory. Every job reads that copy, so editing the tracked `config.yaml`
   later cannot retarget tiles that are already queued.
2. **Size the array.** `data4simplace --count-tiles` gives the array bound; the
   driver checks it against SLURM's `MaxArraySize` (1001 here).
3. **Run tiles.** Each task processes one tile and writes its own weather CSVs
   (named by *global* `gcol`/`grow`, so tiles never collide), its own soil shard
   `soil/_shards/tile_<r0>_<c0>.csv`, and a `.tiles/<tile>.done` marker. No two
   tasks touch the same file, and no task depends on another.
4. **Combine.** A dependent job (`afterany`, so one bad tile cannot block the
   mosaic of the other 111) concatenates the shards and reports any tile whose
   marker is missing — an incomplete `soil.csv` is never silent.

**Restartability.** A tile with a `.done` marker is skipped. Resubmitting is
always safe; `--retry` narrows the array to just the unfinished ids
(`--array=3,9-11,111`).

## Why each tile gets its own WCS cache

`SoilGridsHandler.fetch_wcs` caches a download as
`<wcs_cache_dir>/<coverage_id>.tif` — **the key carries no bounding box**, while
the request is subset to the tile's bbox. Tiles sharing one cache directory
would read back whichever tile downloaded `clay_0-5cm_mean.tif` first, silently
exporting one tile's soil across the whole domain. `tile_config.py` therefore
gives each task `soil.wcs_cache_dir = <cache_root>/tile_<NNNN>`. Verified: four
adjacent tiles produced four different files under that one coverage name.

Re-running the *same* tile still reuses its cache, so a retry costs no downloads.
The proper fix is to include the bbox in the cache key (as
`corine.py::_cached_png` already does) — worth doing in the package, after which
this workaround can go.

The CORINE cache is already bbox-keyed and is safely shared across tiles.

## Runtime expectations

Measured on this cluster, extrapolated from a real 0.5° tile and a 182-day MSWX load:

- MSWX is **one netCDF per day per variable** (16,237 files × 6 vars), and
  `MSWXHandler._open_variable` opens each file, concatenates, *then* subsets the
  bbox. At ~354 ms/day for all six variables that is **~100 min of file opening
  per tile**, before any computation.
- Soil (WCS fetch, 250 m `usda_profile` classification, aggregation) ran ~25 s
  for a 0.5° tile; expect a few minutes at 5°.
- Weather export writes one gzipped CSV per cropland cell — up to 2,500 files ×
  16,802 rows per tile.

Budget **3–8 h per tile**; the 2-day walltime is deliberately generous. At 8
concurrent tasks the full run is on the order of a couple of days.

> **Worth doing before a production run:** pre-subset MSWX to the Europe bbox
> once into per-year files. Every tile currently re-reads the entire global daily
> archive (112 tiles × ~97k file opens ≈ 10.9 M opens against `/data01`), and
> that single change removes almost all of it. Keep `D4S_MAX_CONCURRENT` modest
> until then — the bottleneck is NFS metadata, not CPU.

## Caveats

- **`write_soil_statistics` produces nothing under tiled execution.**
  `tiling._run_tile` covers only the climate and soil *export* stages; it drops
  `SoilGridsHandler.class_statistics`, so `soil_class_statistics.nc`,
  `soil_class_shares.nc` and `soil_class_shares.csv` are never written. The flag
  is silently ignored — it is not wired through the tile path at all. Fixing it
  means having `_run_tile` write a per-tile statistics shard and merging the
  shards in `combine_tiles`.
- **NPK/management is not tiled** — `_run_tile` skips it. `management.sh` runs it
  once over the whole grid instead (it needs no tiling: coarse global rasters,
  one output CSV). It self-skips while `paths.npk_root` is `null`, which is the
  current config, so no management file is produced yet.
- **Tile size does not change the science.** The dominant class is chosen per
  target cell from pixels inside that cell, so there are no cross-tile edge
  effects; tile size only trades memory against task count.
