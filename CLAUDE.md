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
| SIMPLACE reference — weather | `/beegfs/halder/SIMPLACE_WDIR/Brandenburg_1KM_winter_wheat/data/weather` |
| SIMPLACE reference — soil | `/beegfs/halder/SIMPLACE_WDIR/Brandenburg_1KM_winter_wheat/data/soil` |
| SIMPLACE reference — management | `/beegfs/halder/SIMPLACE_WDIR/Brandenburg_1KM_winter_wheat/data/management/fertilizer_winter_wheat.csv` |

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
  (Saxton-Rawls or Wösten); otherwise skip hydraulic derivations.

### NPK Handler

- Align global NPK application/soil datasets to the 10 km target grid.

### SIMPLACE Export Engine

- Assign uniform `SimplaceID` cell identifiers across **all** outputs.
- Dynamically inspect the reference CSV files for exact headers, column order,
  missing-value sentinels (`-99`), and depth horizons.
- Generate weather, soil, and management/fertilizer files matching SIMPLACE
  input standards.

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
