# data4simplace

Ingest, process, aggregate and format **climate (MSWX)**, **soil (SoilGrids)**
and **nutrient (NPK/fertilizer)** datasets into a unified **10 km grid**, and
export point-based CSV files that conform to the
[SIMPLACE](https://www.simplace.net/) crop-model input conventions.

## Features

- **Climate (MSWX)** — loads daily `YYYYDDD.nc` files with dask chunking,
  subsets to the target bounding box, and aggregates to 10 km.
- **Soil (SoilGrids)** — loads native 250 m layers, applies the official scale
  factors, reprojects from Homolosine (`EPSG:152160`) to `EPSG:4326`, and
  aggregates to 10 km. Water retention from the SoilGrids `wv*` layers, with
  Saxton–Rawls pedotransfer functions as a fallback.
- **NPK (NPKGRIDS)** — reads the crop's NPKGRIDS v1.08 netCDF (N, P2O5 and K2O
  application rates at 0.05°), aligns it to the target grid, and rescales the
  SIMPLACE fertilizer schedule's product amounts to each cell's own rates.
- **Cropland filtering** — one PROBA-V LC100 `Crops-CoverFraction` source
  drives both filters: the 250 m soil pixels feeding the aggregation, and the
  10 km cells that get exported at all.
- **Unified cell set** — weather, soil and management cover exactly the same
  cells: cropland cells that also have a soil profile aggregated from their own
  250 m pixels.
- **SIMPLACE export engine** — inspects the reference files *dynamically* to
  recover delimiter, column order, missing sentinel (`-99`) and depth horizons,
  and assigns a uniform `SimplaceID` across weather, soil and management outputs.

## Installation

```bash
pip install -e .[dev]
```

## Usage

```bash
# Validate config and list enabled stages
data4simplace --config config.yaml --dry-run

# Run the full pipeline
data4simplace --config config.yaml --verbose
```

Every stage is toggled by an explicit flag in `config.yaml`:

| Flag | Controls |
| --- | --- |
| `run_climate_processing` | MSWX climate processing |
| `run_soil_processing` | SoilGrids soil processing |
| `compute_ptf` | Pedotransfer functions (fallback for the SoilGrids `wv*` water contents) |
| `run_npk_processing` | NPK/fertilizer processing |
| `apply_agricultural_mask` | Restrict outputs to PROBA-V cropland cells |
| `export_simplace_weather` | SIMPLACE weather export |
| `export_simplace_soil` | SIMPLACE soil export |
| `export_simplace_management` | SIMPLACE management/fertilizer export |
| `export_top3_soil_csvs` | One SIMPLACE soil file per primary class (`soil_1..n.csv`) |

## Package layout

```text
src/data4simplace/
├── cli.py            # CLI entry point
├── config.py         # Pydantic config parser & validator
├── grid.py           # Target 10 km grid + SimplaceID assignment
├── pipeline.py       # Stage orchestration
├── climate/          # MSWX handler
├── soil/             # SoilGrids handler + PTF
├── npk/              # NPKGRIDS reader + fertilizer compositions
├── spatial/          # PROBA-V cropland masks (250 m pixels, 10 km cells)
└── exporters/        # Reference parser + weather/soil/management writers
```

The SIMPLACE reference files are the **source of truth** for output structure;
the exporters inspect them at runtime rather than hard-coding schemas. When a
reference is unavailable, a documented fallback schema keeps the pipeline usable
offline.

## Documentation

Full documentation (guides, example notebooks and an auto-generated API
reference) is built with MkDocs Material under [`docs/`](docs/):

```bash
pip install -e .[docs]   # or: pip install -r docs/requirements.txt
mkdocs serve             # live preview at http://127.0.0.1:8000
mkdocs build             # static site into ./site
```

A GitHub Actions workflow ([.github/workflows/docs.yml](.github/workflows/docs.yml))
publishes the site to GitHub Pages on every push to `main`.

## Development

```bash
pytest          # run tests
python -m build # build sdist + wheel
```
