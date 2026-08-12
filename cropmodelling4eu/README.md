# cropmodelling4eu

Runs and evaluates crop models over the European inputs
[`data4simplace`](../README.md) produces: build a run from a finished export,
execute it with **SIMPLACE** (via its Singularity container) or **torchcrop**
(differentiable LINTUL-5), collect the results into one schema, and compare them
against CyBench, GDHY and the SAGE crop calendar.

The two models share one reader layer, so they differ only in *how a cell is
simulated* — not in which cells exist, where they are, what soil they have or
when the crop goes in. That is what makes their results comparable rather than
merely adjacent.

```bash
pip install -e '.[torchcrop,dev]'          # torch is optional; SIMPLACE needs none of it
cm4eu inspect --export /data01/FDS/muduchuru/Data/SIMPLACE/europe_torchcrop \
              --composition /beegfs/.../fertilizer_composition.xml
```

```
Export   : /data01/FDS/muduchuru/Data/SIMPLACE/europe_torchcrop
Grid     : 690 x 380 cells at 0.1 deg (-17.0..52.0 E, 34.0..72.0 N)
Runnable : 68685 cells (SimplaceID 60427..262196)
Seasons  : 2000-2024 (25 harvest years)
Soil     : 6 layers to 2.00 m, 20 properties
Fertiliser: 68685 cells with a plan
site: FALLBACK -- sowing DOY 270 and altitude 0 m assumed for every cell
```

## Why `inspect` leads

Because the most consequential thing about a run is **which of its inputs are
data and which are assumptions**, and that is invisible once the results exist.
An export written before `data4simplace`'s site stage carries no sowing
calendar, so every cell from Crete to Lapland is sown on the same day — the
single largest error in the published European run. `inspect` says so, in the
run log and on the console, rather than leaving it to be rediscovered.

## The reader layer

Everything in [`export/`](src/cropmodelling4eu/export/) is model-agnostic.

| Module | Reads |
| --- | --- |
| `cells.py` | `SimplaceID` ↔ (row, col) ↔ (lon, lat) ↔ weather filename; round-robin sharding |
| `weather.py` | The gzipped per-cell record; season slicing and unit conversions |
| `soil.py` | `soil.csv` **or** `soil_long.csv` into one tidy profile table |
| `management.py` | The fertilizer schedule and `fertilizer_composition.xml` |
| `site.py` | `site.csv` and `co2.csv`, with a labelled fallback |

Three decisions worth knowing before reading a number:

**The grid is configuration, not a constant.** `SimplaceID` is row-major over
the export's own grid, so decoding one needs the bounds the export was written
with. `RunConfig.from_export()` reads them back out of `_work/config_run.yaml`;
the previous runner hard-coded one grid, and a mismatch places every cell at the
wrong coordinates and reads the wrong weather file, silently.

**The soil layer geometry comes from the file.** The previous runner carried
`LAYER_BOTTOMS_M = [0.1 … 2.0]` as a module constant. A long export carrying
SoilGrids' native horizons (0.05/0.15/0.30/0.60/1.00/2.00 m) would have been
integrated over the wrong depths with no error anywhere. Here the depths are
read from the file, and falling back to the default logs a warning.

**Runnable means weather ∧ soil ∧ a fertilizer plan.** The same intersection
for both models, so a cell appears in every result or in none — a cell with
weather but no plan would otherwise run on some default schedule and report a
yield the export does not describe. Site is deliberately *not* in the
intersection: a cell missing from `site.csv` takes the documented fallback
rather than dropping out.

## Running torchcrop

```bash
./submit/submit_torchcrop.sh --dry-run    # show the plan, submit nothing
./submit/submit_torchcrop.sh              # shard array + combine + maps
./submit/torchcrop_status.sh              # progress, queue, failed shards
./submit/submit_torchcrop.sh --retry      # resubmit only the missing shards
```

See [TORCHCROP.md](TORCHCROP.md) for the sharding, batching and I/O design, and
for the modelling caveats that bound what the maps mean.

**Sowing dates are now per cell**, read from the export's `site.csv`. That
changes the batching: a batch shares one time axis and one `idpl` latch, so
cells are grouped by sowing day-of-year *before* being cut into batches
(`group_by_sowing`). Mixing dates inside a batch is not a rounding question but
a correctness one — it would either skip the batch or sow a Spanish cell on a
German date. A 0.5° calendar takes a handful of distinct values over one shard,
so the groups stay large enough to keep the batches full.

## Running SIMPLACE

```bash
# Build the run directory: XML, project CSV, symlinked inputs, and its own
# submit/retry scripts. Nothing is copied — every data file is a symlink.
cm4eu simplace build --config config.yaml --out-dir /path/to/run [--cells 4]

# From then on the run drives itself, with no need for this package:
/path/to/run/submit.sh                # the whole run, as a SLURM array
/path/to/run/submit.sh --status       # what is finished
/path/to/run/submit.sh --retry        # only the tasks that did not exit zero
/path/to/run/run_task.sh 1-4          # one range, right here

cm4eu simplace collect --config config.yaml --out-dir /path/to/run
```

See [SIMPLACE.md](SIMPLACE.md) — and read its unit warning before trusting
a yield: the export's `Radiation` is in W m⁻² where the solution expects
kJ m⁻² d⁻¹, which costs a factor of ~77 and is not yet fixed.

## Validation

A 30-cell Germany smoke test runs both models over the same cells and
compares them with CyBench yields and PEP725 phenology — see
[VALIDATION.md](VALIDATION.md). Headline: SIMPLACE is unbiased against the
national statistics (−0.12 t/ha) and predicts heading and harvest to within
a day; torchcrop's published configuration yields 3.6x low, which the port
reproduces bit-for-bit and therefore did not cause.

## Layout

```
cropmodelling4eu/
├── config.yaml                     # one run config for both models
├── TORCHCROP.md                    # the torchcrop run's design and caveats
├── SIMPLACE.md                     # the SIMPLACE run's, and the unit warning
├── VALIDATION.md                   # the Germany smoke test vs CyBench + PEP725
├── scripts/                        # cell selection, per-cell runs, validation
├── src/cropmodelling4eu/
│   ├── cli.py                      # cm4eu
│   ├── config.py                   # pydantic RunConfig
│   ├── export/                     # the shared reader layer
│   ├── torchcrop/                  # run.py (one shard) + maps.py (combine, grid, plot)
│   ├── simplace/                   # workspace, project, solution, run, collect
│   └── evaluation/                 # the library the notebooks call
├── evaluation/                     # the four evaluation notebooks + outputs/
├── submit/                         # SLURM drivers: torchcrop's array, and the
│                                   # config -> build step SIMPLACE needs first
└── tests/
```

A SIMPLACE run has no driver under `submit/` beyond that build step, on
purpose: `cm4eu simplace build` writes `submit.sh` and `run_task.sh` **into the
run directory**, where they carry that run's own resolved paths and need
neither this package nor a config to re-run a failed range.


## Testing

```bash
cd cropmodelling4eu && pytest
```

The fixtures build a miniature but complete export on disk — weather files,
both soil layouts, a schedule, a composition file and a site table — so the
readers are tested against real files rather than mocks.
