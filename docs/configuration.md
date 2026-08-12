# Configuration

Every run is driven by a single YAML file (default: `config.yaml`). It is parsed
and validated by [`load_config`][data4simplace.config.load_config] into an
immutable, type-checked [`PipelineConfig`][data4simplace.config.PipelineConfig]
(Pydantic v2). Unknown keys are rejected (`extra="forbid"`), so typos fail fast.

## Full example

```yaml
--8<-- "config.yaml"
```

## Sections

### `flags` — execution switches

Each stage runs only when its flag is `true`. See
[`ExecutionFlags`][data4simplace.config.ExecutionFlags].

| Flag | Controls |
| --- | --- |
| `run_climate_processing` | MSWX climate processing |
| `run_soil_processing` | SoilGrids soil processing |
| `compute_ptf` | Pedotransfer functions (hydraulics); fallback for `soilwater_*` columns the SoilGrids `wv*` layers do not cover |
| `run_npk_processing` | NPK / fertilizer processing (NPKGRIDS crop rates) |
| `run_site_processing` | Per-cell sowing calendar, altitude and the CO₂ series |
| `apply_agricultural_mask` | Restrict outputs to PROBA-V cropland cells |
| `export_simplace_weather` | SIMPLACE weather file export |
| `export_simplace_soil` | SIMPLACE soil file export |
| `export_simplace_site` | `site/site.csv` + `site/co2.csv`; needs `run_site_processing` |
| `export_simplace_management` | SIMPLACE management / fertilizer export |
| `export_top3_soil_csvs` | One SIMPLACE soil file per primary class (`soil_1..n.csv`); needs `soil.aggregation_method: top3` |

!!! info "The exported cell set"
    Weather, soil and management always cover the same cells. The set is the
    intersection of the PROBA-V cropland mask (when
    `apply_agricultural_mask` is on) and the cells the soil stage produced
    values for. With `soil.fill_missing: false` (the default) that means cells
    whose profile was aggregated from their own 250 m pixels — no cell gets a
    weather file unless it also gets a soil profile.

!!! warning "Export needs its input stage"
    An export flag only produces files if the matching processing stage also
    ran. For example, `export_simplace_weather: true` with
    `run_climate_processing: false` logs a warning and writes nothing.

### `grid` — the 10 km target grid

Defines the regular lon/lat grid every dataset is aligned onto. See
[`GridConfig`][data4simplace.config.GridConfig]. `resolution_deg: 0.1` is
approximately 10 km. Bounds are validated (`min < max`, within WGS84 range).

### `time` — climate window

ISO `YYYY-MM-DD` `start` / `end` used by the climate handler. See
[`TimeConfig`][data4simplace.config.TimeConfig].

### `paths` — input sources and output

See [`PathsConfig`][data4simplace.config.PathsConfig]. `mswx_root` is required;
`soilgrids_root`, `npk_root` and `cropland_weights_path` may be `null` to fetch
remotely or skip. `cropland_weights_path` is the PROBA-V LC100
`Crops-CoverFraction` GeoTIFF and is the pipeline's only cropland definition:
unset it and no cropland filtering happens at either resolution. Outputs are
written under `output_dir`.

`calendar_root` and `dem_path` feed the site stage and are **required** when
`run_site_processing` is on; `co2_file` stays optional.

!!! danger "`dem_path` must be a terrain DEM"
    `Land/Elevation/EGM96_30arcsec.nc4` holds `geoid_altitude` — a
    WGS84-to-EGM96 vertical *offset* of roughly ±100 m, not a height above sea
    level. It sits in a directory called `Elevation` and its units are metres,
    so it is the easy wrong choice; the stage rejects a geoid variable by name.
    Use `Land/GMTED/GMTED2010_15n015_00625deg.nc`.

### `site` — sowing calendar, altitude & CO₂

See [`SiteConfig`][data4simplace.config.SiteConfig]. Writes `site/site.csv`
(one row per exported cell) and `site/co2.csv` (one global annual series).
SIMPLACE reads these as `vWGS84_lat`/`vWGS84_lon`, `vAltitude` and
`vSowingDOY`; torchcrop as `site.latitude`, `site.altitude` and `site.idpl`.

`calendar_source` selects the product:

| Setting | Data |
| --- | --- |
| `sage` (default) | Sacks et al. (2010), 0.5° climatology of planting/harvest dates with a start/end window |
| `ggcmi` | GGCMI phase 3 planting + maturity day, built for gridded crop models |

Sampling is **nearest-neighbour, never interpolated** — these are dates from
discrete reporting units, and the mean of DOY 300 and DOY 40 is not a date.

Every exported cell carries a `calendar_source` label saying where its date
came from, so an assumption is never indistinguishable from data:

| Label | Meaning |
| --- | --- |
| `product` | The calendar covers this cell |
| `nearest` | Copied from the nearest covered cell (`fill_calendar_gaps`), bounded by the exported cell mask |
| `fallback` | No date could be reached; `site.fallback_sowing_doy` was used |

### The planting window in the fertilizer schedule

A rule-based solution reads a *window*, not a date. With
`site.write_management_window` (default `true`) SAGE's `plant.start` /
`plant.end` pair is appended to every event row of
`management/fertilizer_<crop>.csv` as `site.window_start_column` /
`site.window_end_column` (default `vSowWindowStartDOY` / `vSowWindowEndDOY`).

| Setting | Default | Effect |
| --- | --- | --- |
| `write_management_window` | `true` | Write the pair. Needs `run_site_processing`; without it the schedule carries no window and the solution's own default sows every cell in the same window |
| `window_min_days` | `7` | Shortest window, and the half-width used where the product publishes a date but no window |
| `window_max_days` | `120` | Longest window |

Lengths are clamped to that range, a window crossing New Year is re-wrapped and
moved off the year end, and the **long** layout writes no window at all — no
long dialect declares a column for it.

The separate `calendar_filled` flag is the **product's own**: SAGE extrapolates
cells no reporting unit covers, and flags them. Roughly a third of the European
domain is filled that way — half of the Alps, none of the North German Plain.

!!! warning "SAGE as an input makes one evaluation circular"
    `sage_calendar_evaluation.ipynb` compares the simulated sowing date against
    SAGE. Once sowing *comes from* SAGE, that comparison is a round trip, not a
    test. Maturity stays a genuine comparison; switch `calendar_source: ggcmi`
    to keep sowing independent too.

### `reference` — SIMPLACE source-of-truth files

See [`ReferenceConfig`][data4simplace.config.ReferenceConfig]. The exporters
inspect these files to recover the exact delimiter, headers, missing sentinel
and depth horizons. Any that are `null` fall back to the documented default
schema.

### `export` — wide or long layout

See [`ExportConfig`][data4simplace.config.ExportConfig]. `layout: wide | long |
both` selects the shape of the soil and management files
(`soil_layout` / `management_layout` override per file).

| Layout | Shape | Files |
| --- | --- | --- |
| `wide` (default) | One row per `location`, depth in the column names (`clay_1`..`clay_6`) | `soil/soil.csv`, `management/fertilizer_<crop>.csv` |
| `long` | One row per `(location, depth)` — SIMPLACE assembles the arrays from the rows sharing a key | `soil/soil_long.csv`, `management/fertilizer_<crop>_long.csv` |
| `both` | Both files | all four |

The long form is what a solution declaring `datatype="DOUBLEARRAY"` expects,
which is how the EU SUSTAg and ERA5 projects read their soil. Column names,
units and delimiter come from the dialect selected from
`reference.soil_file_long`; unset falls back to the built-in SUSTAg spelling.

`both` is nearly free — the two files are two serialisations of one
computation — and lets a wide-driven and a long-driven SIMPLACE run be compared
on identical inputs.

!!! warning "`npk.long_amount_basis` is not cosmetic"
    The SUSTAg long dialect has no `vType` column, so there is no carrier to
    divide by and `Amount` is the **nutrient** in g/m². Writing product grams
    into a nutrient field is a silent factor-of-3.7 error for KAS. `product` is
    refused on a dialect without a fertilizer-type column.

### `climate` — MSWX variable mapping & chunks

See [`ClimateConfig`][data4simplace.config.ClimateConfig]. Maps MSWX variable
folders to canonical names and sets the dask chunk sizes.

### `npk` — fertilizer source & schedule

See [`NPKConfig`][data4simplace.config.NPKConfig]. `source: npkgrids` (the
default) reads one crop-specific **NPKGRIDS v1.08** netCDF from
`paths.npk_root` — `NPKGRIDSv1.08_<crop>.nc`, holding N, P2O5 and K2O
application rates in kg/ha at 0.05° — and aligns it to the target grid.
`source: rasters` keeps the generic per-nutrient GeoTIFF/netCDF discovery for
other gridded products.

| Key | Default | Meaning |
| --- | --- | --- |
| `crop` | `wheat` | NPKGRIDS filename suffix |
| `simplace_crop` | `winter_wheat` | Value of the schedule's `crop` column |
| `include_zero_rate` | `false` | Whether zero-rate land pixels count toward a cell's mean |
| `min_quality` | `null` | Minimum NPKGRIDS quality score (1 best … 0 worst) |
| `n_fertilizer` | `KAS` | Carrier the cell's N rate is applied as |
| `p_fertilizer` / `k_fertilizer` | `P` / `K` | Straight carriers replacing the reference's compound PK |
| `n_split` | `null` | Relative split of N across the dressings; `null` derives it from the reference |
| `composition_file` | `null` | `fertilizer_composition.xml`; `null` uses the copy next to the reference CSV |

!!! info "Sentinels and dropped cells"
    NPKGRIDS marks ocean with `-1` (always dropped) and land the crop is not
    grown on with `0`. Averaging those zeros into a 10 km cell would dilute the
    rate of the pixels that *do* grow the crop, so by default only positive
    pixels contribute and a cell left without one is omitted from the schedule
    rather than given the reference's flat amounts. The management cell set is
    therefore a **subset** of the weather/soil cell set.

### `soil` — SoilGrids layers, depths & CRS

See [`SoilConfig`][data4simplace.config.SoilConfig]. Lists the SoilGrids layers
and depth horizons, and the Homolosine → target CRS pair.

The default layer list includes the volumetric water contents `wv0010`, `wv0033`
and `wv1500`, which fill the SIMPLACE `soilwater_sat`/`_fc`(`_init`)/`_wp`
columns directly. Two further keys turn the total-N layer into initial mineral N:

| Key | Default | Meaning |
| --- | --- | --- |
| `mineral_n_fraction` | `0.01` | Share of the per-layer total-N stock written as mineral N; `0` keeps the reference constants |
| `ammonium_share` | `0.3` | Share of that mineral N written as ammonium, the rest as nitrate |

### `soil.aggregation_method` — one profile per cell, or the top N

| Setting | Behaviour |
| --- | --- |
| `dominant` (default) | **Method A**: one profile per cell, aggregated from the dominant class' pixels. |
| `top3` | **Method B**: the `soil.n_primary_classes` most frequent classes, each aggregated over its own pixels, with per-class surface areas (km²) and the cell's dominance ratio and normalised Shannon entropy. |

Rank 1 is the dominant class either way, so `soil.csv` is identical; Method B
adds the per-class products on top:

- `flags.write_soil_statistics` writes `intermediate_soil_properties.nc`,
  `intermediate_top3_classes.nc` and `intermediate_soil_uncertainty.nc` to
  `<output_dir>/soil/netcdf_tiles/`, before any CSV is written (a tiled run
  writes one set per tile, suffixed with the tile name).
- `flags.export_top3_soil_csvs` writes `soil_1.csv` … `soil_n.csv`, each the full
  SIMPLACE schema followed by `SimplaceID`, `latitude`, `longitude`,
  `soil_class_id`, `class_name`, `area_km2`, `area_fraction`,
  `cell_shannon_entropy` and `cell_dominance_ratio`.

Cells with fewer classes than the rank are absent from that rank's file, so
`soil_3.csv` is shorter than `soil_1.csv`.

### `missing_value`

Sentinel for absent values (default `-99`), overridden by reference inspection
when a reference file is present.
