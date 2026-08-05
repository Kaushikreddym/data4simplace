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
| Harvested area — MIRCA-OS (irrigated + rainfed) | `/data01/FDS/muduchuru/Land/MIRCA-OS/data/contents/Monthly Growing Area Grids/Monthly Growing Area Grids` |
| Harvested area — ECIRA (irrigated + growing) | `/data01/FDS/muduchuru/Land/ECIRA` |
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
│       ├── management/
│       │   ├── __init__.py
│       │   └── irrigation.py       # Harvested area -> vIRR cell classification
│       ├── spatial/
│       │   ├── __init__.py
│       │   └── cropland_weights.py # PROBA-V cropland masks (pixels + cells)
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
| `run_irrigation_classification` | Irrigated/rainfed cell classification (`vIRR`) |
| `apply_agricultural_mask` | Restrict every output to PROBA-V cropland cells |
| `write_soil_statistics` | Per-class NetCDF statistics + class-share table |
| `export_simplace_weather` | SIMPLACE weather file export |
| `export_simplace_soil` | SIMPLACE soil file export |
| `export_simplace_management` | SIMPLACE management/fertilizer export |
| `export_top3_soil_csvs` | One SIMPLACE soil file per primary class (`soil_1..n.csv`) |

## Functional Requirements

### Climate Handler (MSWX)

- Load daily MSWX climate metrics from `/data01/FDS/muduchuru/Atmos/MSWX`.
- Parse and align spatial bounds at **10 km resolution** using
  `xarray`/`rioxarray` with **dask chunking**.

### Soil Handler (SoilGrids)

- Programmatically fetch native **250 m** SoilGrids layers: clay, silt, sand,
  bulk density, organic carbon, pH, nitrogen, and the volumetric water contents
  at 10 / 33 / 1500 kPa (`wv0010`, `wv0033`, `wv1500`).
- **Scale factors:** automatically apply (un-scale) the official SoilGrids
  scaling factors before any calculation.
- **CRS reprojection:** reproject from Homolosine (`EPSG:152160`) to `EPSG:4326`
  **before** 10 km spatial aggregation.
- **Water retention:** the `wv*` layers fill the SIMPLACE `soilwater_*` block
  directly — `wv0033` → field capacity (and the initial content), `wv1500` →
  wilting point, `wv0010` → saturation (SoilGrids publishes no 0 kPa layer, so
  saturation is a drained upper limit). They are linear, so they aggregate with
  the plain weighted mean. Because the three are predicted by independent
  models, `harmonise_water_retention` clips each drier suction to the wetter one.
- **Optional PTF:** `compute_ptf=true` derives hydraulic parameters
  (Saxton-Rawls or Wösten) as a **fallback** for whichever `soilwater_*` columns
  the `wv*` layers do not cover. PTFs are non-linear, so they must be computed
  at **250 m before aggregation** — see
  [Agricultural Mask & Soil Aggregation Workflow](#agricultural-mask--soil-aggregation-workflow).
- **Initial mineral N:** total N and bulk density give a per-layer N stock
  (kg N/ha); `soil.mineral_n_fraction` of it is written as mineral N and split by
  `soil.ammonium_share` into the `ammonium_*` / `nitrate_*` columns. Both are
  initialisation assumptions and must stay explicit in `config.yaml`.
- **Cropland masking & aggregation:** the soil pipeline masks to the dominant
  soil type per cell using cropland weights and aggregates at 250 m; see the
  dedicated workflow section below for the full procedure.
- **Aggregation method:** `soil.aggregation_method` selects between one profile
  per cell (`dominant`) and the top-N per-class profiles with area and
  uncertainty metrics (`top3`); see
  [Multi-Class Aggregation](#multi-class-aggregation-method-b).
- **Per-class statistics:** with `write_soil_statistics=true` the stage also
  describes the `soil.n_primary_classes` most frequent classes per cell; see
  [Primary-Class Statistics](#primary-class-statistics-intermediate-outputs).

### NPK Handler (NPKGRIDS)

- **Source:** NPKGRIDS v1.08 (`/data01/FDS/muduchuru/Land/NPKGRIDS/NC_FILES`),
  one netCDF per crop on a global 0.05° grid holding `Nrate` [kg-N/ha],
  `P2O5rate` [kg-P2O5/ha] and `K2Orate` [kg-K2O/ha] plus a per-nutrient quality
  score. `npk.crop` picks the file (`wheat` → `NPKGRIDSv1.08_wheat.nc`);
  `npk.source: rasters` falls back to the generic per-nutrient raster discovery.
- **Sentinels:** `-1` is ocean and is always dropped. A `0` marks land the crop
  is not grown on; averaging those into a 10 km cell would dilute the rate of
  the pixels that *do* grow it, so only positive pixels contribute
  (`npk.include_zero_rate` overrides) and a cell left without one is **omitted
  from the schedule** rather than given the reference's flat amounts. The
  management cell set is therefore a subset of the
  [exported cell set](#exported-cell-set).
- **Alignment:** cropped to the grid bbox, then aggregated to 10 km with the
  standard binned mean.

### Fertilizer Schedule (SIMPLACE management)

The reference `fertilizer_<crop>.csv` is a long table — one row per location,
event and fertilizer type — carrying *product* amounts in g/m² at a development
stage (`DVS`), with the same flat scenario for every location:

```text
location,FertilizerScenario,crop,Event,vType,DVS,Amount
49612,2,winter_wheat,1,PK,0.001,40
49612,2,winter_wheat,2,KAS,0.25,32
```

The exporter keeps the reference's **structure** — event count, DVS timing and
the relative N split — and replaces the flat amounts with each cell's NPKGRIDS
rates. Two conversions apply:

1. **Oxide → element.** NPKGRIDS reports P2O5/K2O; `fertilizer_composition.xml`
   declares elemental contents (`P2O5_TO_P = 0.4364`, `K2O_TO_K = 0.8302`).
2. **Nutrient → product.** `Amount` is grams of *product*, so a nutrient demand
   is divided by the carrier's content, read from `fertilizer_composition.xml`
   next to the reference CSV — 1 g N is 3.70 g of KAS (27 % mineral N) but only
   2.17 g of Urea (46 %).

The N rate is split across the reference's dressings by the **nutrient** each
delivers (`Amount × mineral N content`), not by the raw amounts, so a schedule
mixing carriers of different strengths still splits the rate the way the
reference does.

The reference's compound `PK` cannot honour two independent rates — its P:K is
fixed at 0.0792:0.083 g/g — so it is replaced by the straight carriers
`npk.p_fertilizer` / `npk.k_fertilizer` (`P` = P2O5, `K` = K2O in the reference
composition file) at the same DVS. Events are then ordered by `DVS` and
renumbered 1..n per cell.

### Irrigation Classification (`vIRR`)

`management/irrigation.py` labels every target cell irrigated or rainfed from the
irrigated share of the crop's harvested area, and `mgmt_export.py` appends it to
the schedule as the `vIRR` column (one label per location, repeated across that
cell's event rows):

```text
vIRR = 1  where  A_irrigated / (A_irrigated + A_rainfed) > irrigation.threshold
vIRR = 0  otherwise
```

A cell holding less than `irrigation.min_crop_area_ha` of the crop is
**unclassified** and written as `0` — SIMPLACE takes a flag, not a three-state
code — but stays distinguishable in the `irrigated_fraction` (NaN) and
`source_id` fields of `management/irrigation_class_<crop>.nc`.

**Sources** (`irrigation.source`):

| Setting | Data |
| --- | --- |
| `mirca` | MIRCA-OS v0.1 Monthly Growing Area Grids, 5 arcmin, global. Annual harvested area = the sum over sub-crops of each sub-crop's peak month, for `_ir` and `_rf` |
| `ecira` | ECIRA v2.0, 1 km, EU/EEA only. `Crop_IR` over `Crop_A` (its README guarantees `Crop_A = Crop_IR + Crop_RF`); falls back to `Crop_IR + Crop_RF` |
| `merged` (default) | ECIRA where it classifies the cell, MIRCA-OS elsewhere |

ECIRA leads in `merged` because MIRCA-OS inherits its irrigated/rainfed split
from national statistics that report **zero** irrigated cereals for Germany,
Romania, the UK, Sweden, Hungary and Poland and zero irrigated maize for
Portugal, and separately loses Italy's 118 kha of irrigated wheat between its own
crop calendar and its grids. Outside ECIRA's EU/EEA footprint MIRCA-OS is the
only source, so the merged layer still covers the whole domain. The evidence is
in `notebooks/irrigation_mirca_ecira_comparison.ipynb`.

**Crop groups.** ECIRA publishes no wheat class — its `CERE` is *cereals
excluding maize and rice* — so `npk.simplace_crop: winter_wheat` classifies
against `cereals` on both sides (MIRCA-OS wheat + barley + rye + millet +
sorghum). `irrigation.crop_group` overrides the mapping.

Both products carry hectares per cell, an **extensive** quantity, so regridding
conserves the sum: MIRCA-OS by exact 1-D overlap weights (its NetCDF coordinates
are cell *corners*, so a half-cell correction is applied first), ECIRA by binning
its 1 km pixel centres, which never splits a pixel across cells.

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

This layer is the pipeline's **only** cropland definition. One threshold,
`soil.cropland_min_fraction`, drives both filters derived from it:

| Filter | Resolution | Decides |
| --- | --- | --- |
| `CroplandWeights.keep_mask` | 250 m pixels | which pixels vote on the dominant class and feed the aggregation — i.e. the *value* a cell carries |
| `CroplandWeights.cell_mask` | 10 km cells | which cells are *exported at all* — a cell needs `soil.min_cropland_pixels` qualifying native pixels |

The cell filter runs under `flags.apply_agricultural_mask` and is applied to
every product (climate, soil, hydraulics, NPK) **and** to the cell table, so the
exported cell set is identical across weather, soil and management. It also
bounds the nearest-neighbour gap fill of §4 — an unbounded search would carry
land profiles out over the sea.

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
| Linear | `clay`, `silt`, `sand`, `bdod`, `cfvo`, `wv0010`, `wv0033`, `wv1500` | Cropland-weighted arithmetic mean |
| Skewed / log-normal | `soc`, `nitrogen` | Cropland-weighted geometric mean: `exp(Σ(ln(X)·W) / ΣW)` |
| Logarithmic | `phh2o` | Convert to `[H⁺] = 10^(−pH)`, weighted mean, back-convert `−log₁₀(mean[H⁺])` |
| Non-linear PTFs | `AWC`, `Ksat` | Compute the PTF at 250 m **first**, then aggregate the output — never aggregate the inputs before a non-linear equation |

The water-retention layers are measured-data predictions, so they need no PTF
and no 250 m pre-computation: they aggregate like any other linear property, and
the physical ordering is restored after aggregation.

### 4. Post-processing & gap filling

- **Texture normalization:** `out_fraction = (fraction / (clay + silt + sand)) × 100`.
- **Missing target cells** (coastal / islands): `soil.fill_missing` fills them
  via nearest-neighbour distance search from the nearest valid land cell. It is
  **off by default**, so every exported profile is aggregated from that cell's
  own 250 m pixels and none is borrowed from a neighbour. When enabled the
  search is bounded to the cropland cell mask, and the filled cells join the
  export (see [Exported Cell Set](#exported-cell-set)).

## Exported Cell Set

Weather, soil and management must cover **exactly the same cells**: a weather
file for a cell SIMPLACE has no soil profile for is unusable. The cell set is
resolved once, in `spatial.export_cell_mask`, as the intersection of two
conditions:

| Condition | Applies when | Keeps a cell if |
| --- | --- | --- |
| Cropland | `flags.apply_agricultural_mask` | it holds `soil.min_cropland_pixels` PROBA-V pixels at ≥ `soil.cropland_min_fraction` cover |
| Valid soil | the soil stage ran | any soil property at any depth is non-NaN |

The resulting mask is applied to climate, soil, hydraulics and NPK **and** to
the cell table, so every exporter iterates the same cells. With
`soil.fill_missing: false` the valid-soil condition means *aggregated from the
cell's own 250 m pixels*; enabling the fill widens it to the filled cells.

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

## Multi-Class Aggregation (Method B)

`soil.aggregation_method` selects how many soil classes a cell contributes:

| Method | Setting | Produces |
| --- | --- | --- |
| **A — single dominant** | `dominant` (default) | The legacy single profile per cell, from the dominant class. `soil.csv`. |
| **B — top-N multi-class** | `top3` | The `soil.n_primary_classes` most frequent classes, **each aggregated over its own 250 m pixels** with the §3 rules, plus per-class areas and cell-level uncertainty. |

Rank 1 of Method B *is* Method A: the ranking and the majority vote both break
ties on the lower class code, so the dominant profile is bit-identical and
`soil.csv` never depends on the method. Method B costs one extra aggregation
pass per additional rank.

### Area & uncertainty metrics

Areas are true surface areas, not pixel counts: a 0.1° cell covers ~103 km² on
Crete and ~39 km² at North Cape, so every pixel is weighted by
`R² · Δlon · (sin φ_north − sin φ_south)` (`spatial/area.py`). Per cell:

| Metric | Definition |
| --- | --- |
| `area_km2` (per rank) | Cropland area of that class in the cell (× 100 for ha) |
| `area_fraction` (`p₁, p₂, p₃`) | `area_km2` over the cell's classified area |
| `total_area_km2` | The cell's classified cropland area |
| Dominance ratio `U_d` | `1 − p₁` — the share of the cell the exported profile does **not** represent |
| Normalized Shannon entropy `H′` | `−Σ pᵢ ln pᵢ / ln N` over **all** N classes in the cell: 0 = uniform, 1 = all classes equally frequent, comparable across cells with different N |

### Intermediate NetCDF rasters

With `flags.write_soil_statistics: true` the gridded state is written to
`<output_dir>/soil/netcdf_tiles/` **before** any CSV, so it survives an exporter
failure. A tiled run writes one set per tile, suffixed with the tile name.

| File | Contents |
| --- | --- |
| `intermediate_soil_properties.nc` | The aggregated 10 km property stack `soil.csv` is written from (`(depth, lat, lon)`, plus PTF outputs when present) |
| `intermediate_top3_classes.nc` | Per-rank properties on `(rank, depth, lat, lon)` with `class_code`, `pixels`, `area_km2`, `area_fraction`, and the code → name lookup as `class_code_value`/`class_code_name` |
| `intermediate_soil_uncertainty.nc` | `total_area_km2`, `n_classes`, the per-rank shares, `dominance_ratio` and `shannon_entropy` |

### Per-class CSV exports

`flags.export_top3_soil_csvs: true` writes `soil_1.csv`, `soil_2.csv` and
`soil_3.csv` — the full SIMPLACE soil schema, so any one of them can drive a run
— each followed by a metadata block: `SimplaceID`, `latitude`, `longitude`,
`soil_class_id`, `class_name`, `area_km2`, `area_fraction`,
`cell_shannon_entropy`, `cell_dominance_ratio`.

A cell with fewer classes than the rank is simply absent from that file, so
`soil_3.csv` is shorter than `soil_1.csv`. PTF hydraulics are derived from the
dominant class' pixels, so they are written to `soil_1.csv` only.

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
