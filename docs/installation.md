# Installation

## Requirements

- **Python 3.10+**
- The geospatial stack (`rasterio`, `rioxarray`, `geopandas`, `pyproj`) builds
  on GDAL. On most systems `pip` wheels bundle GDAL; if you hit build errors,
  install the toolchain with conda/mamba first.

## From a source checkout

```bash
git clone <repo-url> data4simplace
cd data4simplace

# Editable install with the development extras (pytest, ruff, mypy, black, build)
pip install -e .[dev]
```

Verify the install:

```bash
data4simplace --version
python -c "import data4simplace; print(data4simplace.__version__)"
```

## With conda / mamba (recommended for GDAL)

```bash
mamba create -n data4simplace python=3.10 \
    xarray rioxarray rasterio geopandas pandas numpy \
    dask netCDF4 pyproj pyyaml pydantic -c conda-forge
mamba activate data4simplace
pip install -e .[dev]
```

## Building the documentation

The docs use [MkDocs Material](https://squidfunk.github.io/mkdocs-material/)
with `mkdocs-jupyter` (to render the example notebooks) and `mkdocstrings`
(to auto-generate the API reference from docstrings).

```bash
# Install the documentation toolchain
pip install -e .[docs]        # or: pip install -r docs/requirements.txt

# Live-reload preview at http://127.0.0.1:8000
mkdocs serve

# Build the static site into ./site
mkdocs build
```

!!! tip "Executing notebooks at build time"
    By default `mkdocs.yml` sets `execute: false` so the site builds without the
    large input datasets. Once the MSWX / SoilGrids / NPK inputs are reachable
    from your build machine, flip it to `execute: true` to render live outputs.

## Building distributions for PyPI

```bash
python -m build          # creates dist/*.whl and dist/*.tar.gz
```
