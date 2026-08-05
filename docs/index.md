# data4simplace

**Ingest, process, aggregate and format climate (MSWX), soil (SoilGrids) and
nutrient (NPK/fertilizer) datasets into a unified 10 km grid**, and export
point-based CSV files that conform to the
[SIMPLACE](https://www.simplace.net/) crop-model input conventions.

---

<div class="d4s-grid" markdown>

<div class="d4s-card" markdown>
### 🌦️ Climate (MSWX)
Loads daily `YYYYDDD.nc` files with dask chunking, subsets to the target
bounding box and aggregates to 10 km.
</div>

<div class="d4s-card" markdown>
### 🪨 Soil (SoilGrids)
Loads native 250 m layers, applies official scale factors, reprojects from
Homolosine (`EPSG:152160`) to `EPSG:4326`, aggregates to 10 km. Water
retention comes from the SoilGrids `wv0010`/`wv0033`/`wv1500` layers, with
Saxton–Rawls pedotransfer functions as a fallback.
</div>

<div class="d4s-card" markdown>
### 🧪 NPK
Aligns global fertilizer / nutrient rasters onto the target grid.
</div>

<div class="d4s-card" markdown>
### 🌱 Cropland filtering
One PROBA-V `Crops-CoverFraction` source filters the 250 m soil pixels and
selects the 10 km cells that are exported.
</div>

<div class="d4s-card" markdown>
### 📤 SIMPLACE exporters
Inspects the reference files *at runtime* to recover the delimiter, column
order, missing sentinel (`-99`) and depth horizons.
</div>

<div class="d4s-card" markdown>
### 🔖 Stable `SimplaceID`
One deterministic cell identifier shared across weather, soil and management
outputs.
</div>

</div>

## Quick look

```bash
pip install -e .[dev]

# Validate the config and list enabled stages
data4simplace --config config.yaml --dry-run

# Run the full pipeline
data4simplace --config config.yaml --verbose
```

```python
from data4simplace import load_config
from data4simplace.pipeline import Pipeline

config = load_config("config.yaml")
result = Pipeline(config).run()
print(f"Wrote {len(result.written)} SIMPLACE input file(s)")
```

## Where to go next

- **[Installation](installation.md)** — set up the package and the docs toolchain.
- **[Configuration](configuration.md)** — every flag and section of `config.yaml`.
- **[Command line](cli.md)** — the `data4simplace` CLI.
- **[Examples](examples/index.md)** — runnable Jupyter notebooks.
- **[API reference](api/index.md)** — auto-generated from the source docstrings.

!!! note "Source of truth"
    The SIMPLACE reference files are the **source of truth** for output
    structure; the exporters inspect them at runtime rather than hard-coding
    schemas. When a reference is unavailable, a documented fallback schema keeps
    the pipeline usable offline.
