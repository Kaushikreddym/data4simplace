# CLAUDE.md — Project Guidelines & Developer Instructions

## Role

Act as a **senior Python data engineer and geospatial developer**.

## Project Overview

`data4simplace` is a modular, production-ready, **PyPI-publishable**
Python package that ingests, processes, aggregates, and formats **climate
(MSWX)**, **soil (SoilGrids)**, and **nutrient (NPK/fertilizer)** datasets into a
unified **10 km spatial grid**.

All point-based CSV outputs must **strictly** conform to the exact file
structures, schemas, variable headers, units, depth layers, and delimiter
conventions required by the **SIMPLACE** crop-model simulation framework.

## Reference Data Paths

| Purpose | Path |
| --- | --- |
| Climate input (MSWX) | `/data01/FDS/muduchuru/Atmos/MSWX` |
| Cropland cover (Copernicus PROBA-V LC100) | `/data01/FDS/muduchuru/Land/LULC/CopernicusLandCover` |
| SIMPLACE reference — weather | `/beegfs/muduchuru/simplace/Brandenburg_1KM_winter_wheat/data/weather` |
| SIMPLACE reference — soil | `/beegfs/muduchuru/simplace/Brandenburg_1KM_winter_wheat/data/soil` |
| SIMPLACE reference — management | `/beegfs/muduchuru/simplace/Brandenburg_1KM_winter_wheat/data/management/fertilizer_winter_wheat.csv` |

The reference files above are the source of truth for output structure. Inspect
them dynamically rather than hard-coding headers.

## Repository & Package Layout

Follow the standard PyPA `src` layout:

```text
data4simplace/
├── pyproject.toml                  # Build config, dependencies, CLI registration
├── config.yaml                     # Execution config & module flags
├── README.md
├── LICENSE
├── CLAUDE.md
├── src/
│   └── data4simplace/
│       ├── __init__.py
│       ├── cli.py                  # CLI entrypoint (`data4simplace`)
│       ├── config.py               # Config parser & Pydantic validator
│       ├── climate/
│       │   ├── __init__.py
│       │   └── mswx_handler.py     # MSWX netCDF loader & spatial extractor
│       ├── soil/
│       │   ├── __init__.py
│       │   ├── soilgrids.py        # SoilGrids fetcher, scale factor & CRS transformer
│       │   └── ptf.py              # Optional Pedotransfer Functions
│       ├── npk/
│       │   ├── __init__.py
│       │   └── npk_handler.py      # NPK dataset aligner & loader
│       ├── spatial/
│       │   ├── __init__.py
│       │   └── masking.py          # Agricultural cropland mask application
│       └── exporters/
│           ├── __init__.py
│           ├── base_exporter.py    # Reference CSV parser engine
│           ├── weather_export.py   # SIMPLACE weather file generator
│           ├── soil_export.py      # SIMPLACE soil profile file generator
│           └── mgmt_export.py      # SIMPLACE fertilizer schedule exporter
└── tests/
    └── __init__.py
```

## Packaging & CLI

- **Build backend:** `setuptools.build_meta`, configured via `pyproject.toml`.
- **CLI command:** register `data4simplace` → `data4simplace.cli:main`
  under `project.scripts`.
- **Core dependencies:** `xarray`, `rioxarray`, `rasterio`, `geopandas`,
  `pandas`, `dask[complete]`, `pyyaml`, `pydantic`.

## Execution Flags (`config.yaml`)

Every pipeline step is controlled by an explicit boolean flag:

| Flag | Controls |
| --- | --- |
| `run_climate_processing` | MSWX climate processing |
| `run_soil_processing` | SoilGrids soil processing |
| `compute_ptf` | Optional Pedotransfer Functions (hydraulic parameters) |
| `run_npk_processing` | NPK/fertilizer processing |
| `apply_agricultural_mask` | Cropland mask filtering |
| `write_soil_statistics` | Per-class NetCDF statistics + class-share table |
| `export_simplace_weather` | SIMPLACE weather file export |
| `export_simplace_soil` | SIMPLACE soil file export |
| `export_simplace_management` | SIMPLACE management/fertilizer export |

## Functional Requirements

### Climate Handler (MSWX)

- Load daily MSWX climate metrics from `/data01/FDS/muduchuru/Atmos/MSWX`.
- Parse and align spatial bounds at **10 km resolution** using
  `xarray`/`rioxarray` with **dask chunking**.

### Soil Handler (SoilGrids)

- Programmatically fetch native **250 m** SoilGrids layers: clay, silt, sand,
  bulk density, organic carbon, pH, and nitrogen.
- **Scale factors:** automatically apply (un-scale) the official SoilGrids
  scaling factors before any calculation.
- **CRS reprojection:** reproject from Homolosine (`EPSG:152160`) to `EPSG:4326`
  **before** 10 km spatial aggregation.
- **Optional PTF:** if `compute_ptf=true`, derive hydraulic parameters
  (Saxton-Rawls or Wösten); otherwise skip hydraulic derivations. PTFs are
  non-linear, so they must be computed at **250 m before aggregation** — see
  [Agricultural Mask & Soil Aggregation Workflow](#agricultural-mask--soil-aggregation-workflow).
- **Cropland masking & aggregation:** the soil pipeline masks to the dominant
  soil type per cell using cropland weights and aggregates at 250 m; see the
  dedicated workflow section below for the full procedure.
- **Per-class statistics:** with `write_soil_statistics=true` the stage also
  describes the `soil.n_primary_classes` most frequent classes per cell; see
  [Primary-Class Statistics](#primary-class-statistics-intermediate-outputs).

### NPK Handler

- Align global NPK application/soil datasets to the 10 km target grid.

### SIMPLACE Export Engine

- Assign uniform `SimplaceID` cell identifiers across **all** outputs.
- Dynamically inspect the reference CSV files for exact headers, column order,
  missing-value sentinels (`-99`), and depth horizons.
- Generate weather, soil, and management/fertilizer files matching SIMPLACE
  input standards.

## Agricultural Mask & Soil Aggregation Workflow

Cropland masking and soil aggregation are performed at the **native 250 m
resolution first**, then aggregated to the 10 km target grid. Never aggregate
raw inputs before masking or before running non-linear equations.

**Cropland weights** come from the Copernicus PROBA-V LC100 100 m
`Crops-CoverFraction` layer under
`/data01/FDS/muduchuru/Land/LULC/CopernicusLandCover`:

```text
PROBAV_LC100_global_v3.0.1_2019-nrt_Crops-CoverFraction-layer_EPSG-4326.tif
```

### 1. Core workflow

1. Load SoilGrids 250 m rasters and un-scale them via the standard module logic.
2. Align and apply the PROBA-V 100 m cropland `Crops-CoverFraction` weights to
   the 250 m soil grid.
3. Classify/select the **dominant soil type** per target cell according to the
   `config.yaml` setting (`usda`, `usda_profile` or `wrb`; see §2).
4. Mask 250 m pixels to keep **only** the selected dominant soil type within each
   target grid cell.
5. Compute non-linear Pedotransfer Functions (PTFs) at **250 m first**, on the
   masked pixels.
6. Aggregate properties spatially over the dominant soil type using cropland
   weights and the variable-specific rules (see §3).
7. Post-normalize texture fractions so `clay + silt + sand = 100 %`.

### 2. Dominant soil-type selection (`config.yaml`)

Selected via a config setting, `soil.dominant_mode: usda | usda_profile | wrb`.

**Mode A — `usda` (USDA texture-class dominance)**

1. Derive the 250 m USDA soil texture class (12 classes) from unscaled sand,
   silt and clay **of the topsoil layer** (`soil.depths[0]`).
2. Sum cropland weights per USDA class within each target grid cell.
3. Select the USDA texture class with the highest total cropland weight as the
   dominant class.
4. Mask 250 m pixels: keep **only** pixels matching the dominant USDA class in
   that cell.

**Mode C — `usda_profile` (composite topsoil × rooting-zone dominance)**

A surface class alone cannot separate a uniform sand from a sandy cover over
loamy till — the layered profiles that are widespread in NE Germany and that
decide plant-available water. This mode classifies on **two keys**:

1. **Topsoil key** — the USDA class of `soil.depths[0]` (0–5 cm), as in Mode A.
2. **Rooting-zone key** — the USDA class of the **thickness-weighted mean**
   texture from the bottom of the topsoil layer (5 cm) down to
   `soil.rootzone_bottom_cm` (default 100 cm). Layers that only partly overlap
   the window contribute only their overlapping thickness.
3. Pack the pair into one code, `(topsoil − 1) × 12 + rootzone` (1–144, `0` if
   either key is unclassified), then run the **same majority vote and mask** as
   Mode A on the composite code.

The mask stays a *single* class, so every depth is still aggregated over the
same pixel set (no chimera profiles assembled from different locations at
different depths).

**Mode B — `wrb` (WRB Reference Soil Group dominance)**

1. Load the 250 m SoilGrids WRB `MostProbable` layer.
2. Sum cropland weights per WRB soil code within each target grid cell.
3. Select the WRB class with the highest total cropland weight as the dominant
   class.
4. Mask 250 m pixels: keep **only** pixels matching the dominant WRB class in
   that cell.

### 3. Variable-specific aggregation rules

Run on the masked dominant pixels, weighted by cropland cover fraction `W`.

| Variable group | Variables | Aggregation |
| --- | --- | --- |
| Linear | `clay`, `silt`, `sand`, `bdod`, `cfvo` | Cropland-weighted arithmetic mean |
| Skewed / log-normal | `soc`, `nitrogen` | Cropland-weighted geometric mean: `exp(Σ(ln(X)·W) / ΣW)` |
| Logarithmic | `phh2o` | Convert to `[H⁺] = 10^(−pH)`, weighted mean, back-convert `−log₁₀(mean[H⁺])` |
| Non-linear PTFs | `AWC`, `Ksat` | Compute the PTF at 250 m **first**, then aggregate the output — never aggregate the inputs before a non-linear equation |

### 4. Post-processing & gap filling

- **Texture normalization:** `out_fraction = (fraction / (clay + silt + sand)) × 100`.
- **Missing target cells** (coastal / islands): fill via nearest-neighbour
  distance search from the nearest valid land cell.

### 5. Exported statistic (`soil.export_statistic`)

The CSVs carry one value per property, depth and cell, taken from the **dominant
class** (rank 1):

| Setting | Behaviour |
| --- | --- |
| `mean` (default) | The variable-specific mean rules of §3 |
| `median` | The plain per-cell median for every variable — the median commutes with the log/H⁺ transforms, so no per-variable rule is needed |

## Primary-Class Statistics (intermediate outputs)

Set `flags.write_soil_statistics: true` to also describe the
`soil.n_primary_classes` (default 3) most frequent classes per target cell.
Rank 1 is the dominant class the CSVs carry; the lower ranks quantify the
inter-class spread a single exported profile leaves out. Files land in
`<output_dir>/soil/`:

| File | Contents |
| --- | --- |
| `soil_class_statistics.nc` | `<layer>_<statistic>` on `(rank, depth, lat, lon)` — `mean`, `median`, `std` (sample), `kurt` (excess kurtosis, normal = 0) and `count` for **every property, depth and class** |
| `soil_class_shares.nc` | `class_code`, `pixels` and `share_percent` on `(rank, lat, lon)`; the code → name map is the `class_code_names` attribute |
| `soil_class_shares.csv` | The same per cell and rank, with `SimplaceID`, lat/lon, **class code and class name** (`usda`: texture class; `usda_profile`: `topsoil/rooting-zone`) and the **percent of the cell's classified pixels** |

Rank-1 `share_percent` is the weight of the exported profile; `1 − share` is the
fraction of the cell's cropland the export does not represent, and the rank-2/3
statistics say how different that remainder is.

## Code Standards

- Complete, functional **Python 3.10+** code.
- Strict type annotations, docstrings, and error handling throughout.
- Memory-efficient processing using **dask** and **xarray**.

## Tool Permissions
- Allow file edits in `src/` and `tests/`
- Allow running `pytest`
- Allow running `simplace-pipeline`

## Development Commands

```bash
# Install package locally in editable mode (with dev extras)
pip install -e .[dev]

# Run the pipeline CLI
data4simplace --config config.yaml

# Build source distribution and wheel for PyPI
python -m build

# Run unit tests
pytest
```
