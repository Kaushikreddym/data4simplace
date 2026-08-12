# SIMPLACE Europe run

Runs the Brandenburg Lintul5+SLIM solution over the `data4simplace` export in
its Singularity container, as a SLURM **array job — one task per range of
project-file lines** — then collects the per-cell outputs into the same schema
the torchcrop run writes.

```bash
# Build the run directory, then drive it from there.
cm4eu simplace build --config config.yaml --out-dir <dir> [--cells 4]
<dir>/submit.sh                          # the array
<dir>/submit.sh --status                 # what is finished
<dir>/submit.sh --retry                  # only what did not exit zero
cm4eu simplace collect --config config.yaml --out-dir <dir>

./submit/submit_simplace.sh --dry-run    # config -> build -> the above
./submit/submit_simplace.sh --build-only # build and validate, submit nothing
```

| | |
| --- | --- |
| template | `/data01/FDS/muduchuru/codes/SIMPLACE/Brandenburg_1KM_winter_wheat` |
| image | `/beegfs/common/singularity/simplace/simplace_5.0-3897.sif` |
| sharding | `SP_LINES_PER_TASK=500` → 138 tasks over 68 685 cells (~10 h each) |
| output | `--out-dir`, or `<output_dir>/<run_name>/simplace/` |

## The export's radiation unit, and where it is fixed

**An unreconciled run against `europe_torchcrop` produces ~0.04 t/ha.** The
mechanism is sound — the same solution gives 5.5 t/ha on the reference DWD
weather — but the export's radiation is in a different unit from the reference
file the exporter claims to conform to:

| Column | Brandenburg reference | data4simplace export | Factor |
| --- | --- | --- | --- |
| `Radiation` | `21967` (kJ m⁻² d⁻¹) | `286.69` (W m⁻²) | 86.4× |

The ~77× effective shortfall matches the observed biomass shortfall (0.22 vs
17 t/ha) almost exactly: SIMPLACE reads the file as written, so it grows a crop
in about 1 % of the real light. This stayed hidden because the torchcrop runner
converts in the *consumer* (`Radiation × 0.0864` to MJ), so only a model reading
the file as-is exposes it.

**Radiation is the only affected column here.** The weather resource declares
six, and `RelHumCalc` is not among them — this solution derives vapour pressure
from temperature inside the transform below and never reads humidity. (It
matters for `sustag_v2`, whose contract does read it.) `Windspeed` and `RefET`
are `-99.9` in the same files but likewise unread; see
[TORCHCROP.md](TORCHCROP.md).

### The fix is one token of SQL, not a converted file

The solution already transforms its weather before use:

```xml
<transform id="weather_transform" frequence="DAILY"
           class="net.simplace.sim.transformers.DefaultSQLStatementTransformer">
  <input id="statement">SELECT CURRENTDATE, ...
    Irradiation/1000 as SradiationMJ, ... FROM weather</input>
</transform>
```

So `simplace.weather_conversion: transform` (the default) rewrites that one
expression to `(Irradiation*86.4)/1000` at build time and **leaves the weather
symlinked**. Verified against the file-conversion path on 4 German cells ×
13 seasons: **52 of 52 (cell, year) yields identical to 6 decimal places.**

Two details make it safe:

- **The SQL names are the resource's declared ids, not the file's.** SIMPLACE
  binds CSV columns by position, so the export's `Radiation` is the solution's
  `Irradiation`. `declared_factors` zips the contract's column order against
  the declared order, which translates the key *and* asserts the alignment;
  a contract wider than the resource is an error, not a silent truncation.
- **The one other reference to raw `weather.Irradiation` is a sentinel test** —
  `if(${weather.Irradiation} lt -99)`, choosing the temperature-based radiation
  fallback — which is unit-independent.

| | `transform` (default) | `files` |
| --- | --- | --- |
| extra disk | none | ~15 GB for Europe |
| build cost | none | ~1.5 h first time |
| weather in the workspace | symlink | link into a converted cache |

`files` remains for a solution with no transform to patch, or a contract that
changes the file's *shape* rather than its units — `sustag_v2` reorders columns
and adds `vprsd`/`dewp`, which SQL over the declared six cannot do.

Both are reconciliations at the consumer. The export still claims to conform to
a reference whose units it does not use, so
[`weather_export.py`](../src/data4simplace/exporters/weather_export.py) writing
kJ m⁻² d⁻¹ directly is still the honest fix — it is just no longer urgent, since
the transform costs nothing.

## How a run is put together

**`build`** materialises everything a run needs in one directory:

```bash
cm4eu simplace build --config config.yaml --out-dir /path/to/run
```

```
/path/to/run/
├── build.json                        what this tree was built from
├── run.env                           resolved paths, binds and defaults
├── submit.sh                         submit / status / retry
├── run_task.sh                       one line range, in the container
├── workspace/
│   ├── solution/solution.sol.xml     from the template, variables overridden
│   ├── project/project.proj.xml      from the template, filename repointed
│   ├── project/project_<crop>.csv    generated — one line per cell
│   └── data/                         symlinks: template XML + the export
├── weather/<row>/…                   the row-nested tree the solution reads
├── state/task_<first>-<last>.done    one stamp per range that exited zero
├── log/
└── out/                              SIMPLACE's own output
```

`--out-dir` is optional; without it the tree goes to
`<paths.output_dir>/<run_name>/simplace`.

The template is parsed, adjusted in memory and written into the workspace —
never edited in place, so a run cannot corrupt the project it came from.

**Nothing is copied.** Every data file in the tree is a symlink: the export's
`soil.csv` and fertilizer schedule, the weather, and the template's own crop,
SLIM and soil C/N/P parameters. A built workspace is a few hundred kilobytes,
so it can be inspected, thrown away and rebuilt without a thought. The cost is
that editing the template changes every workspace built from it — which is what
`build.json` is for, since the tree itself no longer records where its inputs
came from.

**Four files are generated rather than linked**, and each for a reason:

- `project_<crop>.csv` — one line per cell, and the line range is the unit of
  work. Its `vColumn`/`vRow` are what the weather filename is built from.
- `location.csv` — per-cell latitude, which drives day length. Copying the
  template's would give every European cell Brandenburg's 52.6 °N.
- The row-nested weather tree — the solution reads
  `${_DATADIR_}/${vRow}/daily_mean_RES1_C${vColumn}R${vRow}.csv.gz`, while the
  export writes every file flat in one directory.
- `submit.sh` / `run_task.sh` — see below.

`vNUTSID`, `vSTATE_ID` and `vSTATE_NAME` are written as `NA`. The solution
echoes the first two straight into its daily output, so a fabricated code would
travel into the results looking like a real region.

## Running it, and running it again

The build writes the run's own scripts into the output directory:

```bash
/path/to/run/submit.sh                  # the whole run, as a SLURM array
/path/to/run/submit.sh --status         # what is finished
/path/to/run/submit.sh --retry          # only the tasks that did not exit zero
/path/to/run/submit.sh --local          # here, in the foreground, no SLURM
/path/to/run/run_task.sh 1-2000         # one range, by hand
```

Three properties make them worth generating rather than shipping:

**They need nothing from this package.** `run_task.sh` builds the `singularity`
command from `run.env` itself. A retry works from a bare shell — no conda
environment, no `cm4eu`, no config — which is why they live with the run
instead of in `submit/`.

**They take the run directory as input.** Both default to their own location,
so `./submit.sh` from inside the folder is enough, and both accept `--root` to
drive a run from anywhere. The bind list is written relative to that root, so
the whole directory can be copied elsewhere and still run.

**A retry is derived, not remembered.** `--retry` recomputes what is
outstanding on every call, so it stays right after any mix of array runs,
manual re-runs and cancellations.

What it derives it from is the one subtle decision. A task counts as finished
when **it exited zero**, stamped under `state/`, not when every one of its cells
has an output file — because a cell can legitimately produce none. Winter wheat
at 63 °N never reaches maturity, so SIMPLACE writes no yearly file for it, and a
coverage-based check would resubmit that task for ever. `--status` still reports
the per-cell counts, where they are information rather than a trigger:

```
task   range           cells    exited 0
0      1-6             5/6      yes
```

The stamp is named after the *line range*, not the task index, so re-splitting
the work with a different `lines_per_task` invalidates the stamps instead of
silently crediting them to a different set of lines.

## Why the export is bound into the container

The workspace **symlinks** the export's `soil.csv`, fertilizer schedule and
weather rather than copying ~70 000 files. A symlink resolves *inside* the
container, so the export is also bound read-only at its own absolute path — as
are the template and, when the weather is converted, its cache. Without that
every link dangles and SIMPLACE fails with a `NullPointerException` on a file
whose path it has just printed — which is a long way from "the bind list is
short one entry".

Under `weather_conversion: files` the numbers themselves change, so a link to
the export would be wrong: the converted files go to a **shared cache** outside
the run directory (`<output_dir>/weather_cache/<contract>`, or
`--weather-cache`) and the workspace links into that. The default
`weather_conversion: transform` avoids the question entirely.

## Validation

`build` checks every CSV resource the solution declares against the file it
points at, on the login node, before anything is queued. Problems are split by
how certain they are to be fatal:

- **A missing file is an error.** The build stops.
- **A missing column is a warning.** SIMPLACE tolerates some: this solution
  declares `Type` where its own working schedule has `vType`, and
  `LowerBoundaryPConcentration` where its `soil.csv` has
  `LowerBoundaryConcentration` — and that pair is a run that has produced
  54 years of sensible output. `--strict` makes them fatal.

`DOUBLEARRAY` says a value is an array, **not** how the file encodes it:
SIMPLACE fills one from `<id>_1`…`<id>_N` columns (wide) or from repeated rows
under the key (long). Both are accepted.

## Sharding

SIMPLACE selects work with `-l=START-END` over the project file, so ranges are
**contiguous** — unlike the torchcrop shards, which are dealt round-robin.
`SimplaceID` runs north-to-south, so one task gets Scandinavia and another
Iberia, with different wall times; the array's concurrency throttle absorbs it.

`submit.sh` checks the task count against the cluster's `MaxArraySize` (1001
here) before submitting, because `sbatch` rejects an oversized array with a
message that does not say which number was too big.

### Sizing a task, and why it matters more than it looks

**A cell costs ~72 s over the 1979–2024 window** — five German cells in
6 m 26 s, measured on the built Europe workspace. So:

| | |
| --- | --- |
| one cell, 46 seasons | ~72 s |
| 68 685 cells | ~1 370 core-hours |
| `lines_per_task: 500` | ~10 h/task, 138 tasks |
| `lines_per_task: 2000` | **~40 h/task** — dies on a 12 h wall clock |

500 is therefore the default, and it is measured rather than guessed. Scale it
with the simulated window: a 15-year run costs a third as much per cell, so
1500 lines fits the same wall clock.

**Getting this wrong is the one failure the run scripts handle badly.** A task
the scheduler kills has written output for the cells it reached but has no
completion stamp, so `--retry` restarts that range **from its first line** — it
redoes the finished cells, runs out of wall clock at the same place, and never
finishes. `--status` makes it visible (the task's cell count climbs while
`exited 0` stays `NO`), but the fix is to size the task so it can finish, not to
retry it harder.

## Collection

`collect` reads every `<id>_yearly.csv` (semicolon-separated, and carrying no
location column — the identity is the filename) and maps it into the **shared
run schema**, so one evaluation code path serves both models:

| SIMPLACE | Shared schema | |
| --- | --- | --- |
| `Yield_t_ha` | `yield_g_m2` | × 100 |
| `AGBiomass_t_ha` | `biomass_g_m2` | × 100 |
| `inputChemN_kg_ha` | `n_applied_g_m2` | ÷ 10 |
| `MaturityDOY` − `PlantingDOY` | `days_to_maturity` | wrapped over New Year |

Rows carry `model = "simplace"`.

## Three solutions, two ways of sowing

| | Brandenburg | **Brandenburg + rules** | EU SUSTAg v2 |
| --- | --- | --- | --- |
| solution | `.../codes/SIMPLACE/Brandenburg_1KM_winter_wheat` | [`templates/brandenburg/`](templates/brandenburg/) | [`templates/sustag/solution/`](templates/sustag/solution/) |
| soil layout | wide `soil.csv` | wide `soil.csv` | **long** `soil_long.csv` (`sustag_v2` dialect) |
| sowing | `vIDPL`, one constant | **rule-based, per-cell window** | **rule-based, per-cell window** |
| weather | the export's own format | the export's own format | a different schema — see below |
| runnable today | yes | **yes — the default** | no, see the gap list below |

**The stock Brandenburg solution cannot take a per-cell sowing date.** Its
planting day is the `vIDPL` variable (250 for every cell), its project resource
declares no sowing column, and its `PlantingDOY` output is `rule="vIDPL"` — so
that column is the input echoed back, not a result. A run against it reports one
sowing date for every cell and every year, whatever `site.csv` says.

**`templates/brandenburg/` is that solution with the v2 sowing rules dropped
in**, and it is what `config.yaml` points at. The three sowing components are
copied verbatim from `EU_SUSTAg_data4simplace.sol.xml` with only their input
names remapped (`weather.AirTemperatureMin`, `weather.Rain`,
`DefaultManagement.WithCrop`); `DefaultManagement` sows on `${SowingRule.DoSow}`,
and `Phenology.cIDPL` and `PlantingDOY` read the realized
`SowingDate.SowingDOY`. Nothing else differs from the template — `diff` shows
five removed lines — so a run against it differs from a stock run by the sowing
date and nothing else. The project template adds `vSowWindowStartDOY` and
`vSowWindowLengthDays`, which `build_project_frame` fills from `site.csv`.

A new `ForcedSow` column flags the seasons where the deadline, not a rule, set
the date. **Check it before reading spread as skill**: with
`vSowForceAtWindowEnd = 1` a cell where no rule fires sows on the window's last
day, which is a constant in disguise.

### Where the window comes from

The rules need two day-of-year thresholds, and the solution takes the first of
these that exists:

| Source | Read as | Covers |
| --- | --- | --- |
| **The fertilizer schedule** | `sowwindow.vSowWindowStartDOY` / `vSowWindowEndDOY`, SAGE's own `plant.start`…`plant.end`, written by data4simplace's management exporter | cells with an NPKGRIDS rate |
| The project file | `vSowWindowStartDOY` + `vSowWindowLengthDays`, from `SiteTable.sowing_window` | every simulated cell |

The schedule wins because it is the published window rather than a window
derived from the date. It cannot be the only source: a cell with no NPKGRIDS
rate is absent from the schedule entirely, so the project file stays the
fallback. `SowingWindowDates` resolves the two into `WindowStart` / `WindowEnd`
once a day, and both are written to the yearly output — with two possible
sources, a sowing date cannot be read without knowing which applied.

`sowwindow` is a second resource on a second interface over the *same* CSV:
`managementfile2` is keyed on `(location, EventNr)` and refreshed only while a
crop is in the ground, which is exactly when the sowing rule is not running.

**Without either, the window is constant** — `SiteTable.sowing_window` falls
back to the sowing date ± 7 days, so every cell gets DOY 263–277 and the rules
can only move the date inside those 14 days. And SAGE's own window is constant
*within* Germany: its reporting unit covers the whole country, so the 30-cell
smoke test gets one window (DOY 272–308) for every cell. Cell-to-cell spread in
the sowing date therefore needs a domain wider than one SAGE reporting unit;
what the window buys inside Germany is 36 days of room for the weather rules
instead of 14.

**The v2 SUSTAg solution does, and better.** It sows on the first day inside a
planting window on which a weather rule holds — a 7-day mean minimum
temperature, a 3-day rainfall sum, or top-5 cm soil moisture — forcing a sowing
on the last window day if none fires. And it reads the window's start per grid
cell:

```xml
<var id="vSowWindowStartDOY" description="... set per grid cell from the project file">200</var>
```

That is a better use of the SAGE calendar than a single date, because SAGE
publishes a *window* (`plant.start` … `plant.end`) and collapsing it to a
midpoint throws away what it actually measured. `SiteTable.sowing_window`
turns the exported window into `(start, length)`, clamped so it neither
degenerates to a fixed date nor wraps the year end — which the solution's
`DOY <= start + length` test cannot express.

### What the v2 solution still needs

Run `cm4eu simplace build` against it and the validator names these. Three are
work, one is configuration:

| | Status |
| --- | --- |
| Soil | **Done** — `export.layout: long` with the `sustag_v2` dialect writes all 19 derivable columns it declares, and nothing it does not |
| Sowing window | **Done** — from `site.csv` via the project file |
| Profile constants | **Config** — `alfa`, `n`, `ksat`, `macroporevolume`, `dampingdepth`, `drainage_rate`, `deltatheta`, `maxRootingDepth`, `Soiltype` have no SoilGrids source; set `soil.long_constants` in data4simplace |
| Fertilizer | **Not done** — it keys on `gcm_rcp`, `period`, `CO2_level` and `Rate`, where the export's long schedule keys on `Year`. A new management dialect |
| Weather | **Not done** — it reads `period, gcm_rcp, DATE (yyyyMMdd), tmax, tmin, vprsd, wind, rain, srad, rhumd_tn, rhumd_tx, dewp`, comma-separated, with `srad` in MJ m⁻² d⁻¹ and `wind` in **km/day**. The export writes a different schema entirely, so this needs a weather converter |
| Scenario keys | **Config** — `vCrop`, `vTrt`, `vClimPerCO2_ID`, `vENZ` select parameter sets; set `simplace.project_constants` |

The weather gap is the substantial one, and it is worth noting that its
`srad` unit (MJ m⁻² d⁻¹) is a *third* convention, after the export's W m⁻² and
Brandenburg's kJ m⁻² d⁻¹ — which is the same defect described above seen from
another angle.

### The v2 weather file, for an observational (MSWX) run

**Do not trust the `unit=` attributes.** SIMPLACE does not enforce them and
several are wrong — `rhumd_tn`/`rhumd_tx` are annotated `metre_per_second-time`
and `vprsd` `degree_Celsius`. The units below are read off the working file's
values and off the transforms the solution applies to them.

| Column | Unit | Example (1 Jan) | Used? |
| --- | --- | --- | --- |
| `period` | scenario key | `0` | join only |
| `gcm_rcp` | scenario key | `0_0` | join only |
| `DATE` | `yyyyMMdd`, no separators | `19800101` | ✓ |
| `tmax`, `tmin` | °C | `1.48`, `-1.02` | ✓ |
| `vprsd` | **kPa** (actual vapour pressure) | `0.52` | ✓ ×4 |
| `wind` | **km/day** — `ToDiurnal` applies ×0.011574 to m/s | `304` | ✓ |
| `rain` | mm/day | `1.5` | ✓ |
| `srad` | **MJ m⁻² d⁻¹** — `TransformUnit` applies ×10⁶ to J | `2.1` | ✓ |
| `rhumd_tn`, `rhumd_tx`, `dewp` | — | `91.8`, `76.5`, `-2.2` | **never referenced** |

The last three are declared in the weather resource and used nowhere in the
solution; fill them with anything, or delete the three `<res>` lines. Humidity
still matters — it enters as `vprsd`, which drives net radiation, FAO-56 ET₀,
the hourly canopy temperature and the diurnal transform. Build it as
`es(Tmean) × RH/100`, which is what
[`export/weather.py`](src/cropmodelling4eu/export/weather.py) already does.

From the MSWX export the conversions are `Radiation × 0.0864` (W m⁻² →
MJ m⁻² d⁻¹) and `Windspeed × 86.4` (m s⁻¹ → km/day).

### `period` / `gcm_rcp` are a row selector, and CO₂ rides on them

```
vClimPerCO2_ID ("C1")
   └─► ClimPerCO2.csv → vPeriod, vgcm_rcp, vCO2, vCO2_level
          ├─► keys the weather resource    (period, gcm_rcp)
          ├─► keys the fertilizer resource (period, gcm_rcp, CO2_level)
          └─► feeds vCO2 to RadiationUseEfficiency, CO2Tran, rcCO2
```

The weather file may hold several scenarios stacked; the scenario row picks
which rows are read. An observational run needs one consistent pair
(`period=0`, `gcm_rcp=0_0`) on every weather **and** fertilizer row, plus one
matching `ClimPerCO2.csv` row.

Two consequences worth knowing before a run:

- **CO₂ is physics and comes from the scenario file, not the weather.** With
  `frequence="ONCE"` it is a single value for the whole simulation — so a
  1979–2024 run collapses 336→422 ppm to a constant, and the annual series the
  site stage exports cannot be consumed as the solution stands.
- **A missed lookup skips the run silently**:
  `<simmodel skip="check:isNull(${gcm_rcp_scenario.vCO2}) || ...">`. A typo in
  `gcm_rcp` or a mismatched `period` produces no error and no output — check
  this first if a run comes back empty.
