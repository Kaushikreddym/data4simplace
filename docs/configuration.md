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
| `compute_ptf` | Optional pedotransfer functions (hydraulics) |
| `run_npk_processing` | NPK / fertilizer processing |
| `apply_agricultural_mask` | Cropland mask filtering |
| `export_simplace_weather` | SIMPLACE weather file export |
| `export_simplace_soil` | SIMPLACE soil file export |
| `export_simplace_management` | SIMPLACE management / fertilizer export |

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
`soilgrids_root`, `npk_root` and `cropland_mask` may be `null` to fetch remotely
or skip. Outputs are written under `output_dir`.

### `reference` — SIMPLACE source-of-truth files

See [`ReferenceConfig`][data4simplace.config.ReferenceConfig]. The exporters
inspect these files to recover the exact delimiter, headers, missing sentinel
and depth horizons. Any that are `null` fall back to the documented default
schema.

### `climate` — MSWX variable mapping & chunks

See [`ClimateConfig`][data4simplace.config.ClimateConfig]. Maps MSWX variable
folders to canonical names and sets the dask chunk sizes.

### `soil` — SoilGrids layers, depths & CRS

See [`SoilConfig`][data4simplace.config.SoilConfig]. Lists the SoilGrids layers
and depth horizons, and the Homolosine → target CRS pair.

### `missing_value`

Sentinel for absent values (default `-99`), overridden by reference inspection
when a reference file is present.
