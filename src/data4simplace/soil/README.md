# How the SIMPLACE soil columns are produced

Where every column of `soil.csv` comes from, and which ones are **not** derived
from data. The writer is [`../exporters/soil_export.py`](../exporters/soil_export.py);
the values it consumes are produced by [`soilgrids.py`](soilgrids.py),
[`classify.py`](classify.py) and [`ptf.py`](ptf.py).

The reference schema is read at runtime from
`/beegfs/muduchuru/simplace/Brandenburg_1KM_winter_wheat/data/soil/soil.csv`
(142 columns). Every column falls into exactly one of four buckets:

| Bucket | Columns | Source |
| --- | --- | --- |
| Computed from SoilGrids | 66 | 250 m rasters, aggregated per cell |
| Computed by the PTF | 0 | Saxton–Rawls, **fallback only** (see §3) |
| Cell identifier | 1 | our `SimplaceID` |
| Constant | 75 | copied from the reference's first row |

## 1. Computed from SoilGrids — 66 columns

`clay_1..6`, `sand_1..6`, `bulkdensity_1..6`, `carbon_1..6`, `PH_1..6`,
`soilwater_sat_1..6`, `soilwater_fc_1..6`, `soilwater_init_1..6`,
`soilwater_wp_1..6`, `ammonium_1..6`, `nitrate_1..6`

Mapped by `_STEM_TO_SOILGRIDS` in `soil_export.py`:

| Column stem | SoilGrids layer | Unit written |
| --- | --- | --- |
| `clay` | `clay` | % |
| `sand` | `sand` | % |
| `bulkdensity` | `bdod` | kg dm⁻³ |
| `carbon` | `soc` | g kg⁻¹ |
| `PH` | `phh2o` | pH |
| `soilwater_sat` | `wv0010` | m³ m⁻³ |
| `soilwater_fc` | `wv0033` | m³ m⁻³ |
| `soilwater_init` | `wv0033` | m³ m⁻³ |
| `soilwater_wp` | `wv1500` | m³ m⁻³ |

`ammonium`/`nitrate` are derived rather than mapped — see §3b.

### Water retention

`wv0010`, `wv0033` and `wv1500` are SoilGrids' own volumetric water contents at
10, 33 and 1500 kPa, predicted from measured retention data instead of a texture
regression. They are **linear** quantities, so they take the plain cropland-
weighted mean of the masked 250 m pixels like `clay` — no `flags.compute_ptf`,
no non-linear-before-aggregation constraint. Un-scaled to vol % by the standard
factor of 10, then × 0.01 at export for SIMPLACE's m³ m⁻³ (`_STEM_UNIT_FACTOR`).

Two things to know about this mapping:

- **`soilwater_sat` is the 10 kPa content, not total porosity.** SoilGrids
  publishes no saturation layer; 10 kPa is the wettest suction available, so the
  saturated water content is a drained upper limit (≈ 0.45 where porosity from
  `1 − bdod/2.65` would be ≈ 0.62). The ordering the water balance needs still
  holds; the ceiling on it is lower than a true saturation.
- **The three layers come from independent models**, so a cell can be predicted
  wetter at 1500 kPa than at 33 kPa, which would make plant-available water
  negative. `SoilGridsHandler.harmonise_water_retention` clips each drier
  suction to the wetter one (`wv0010 ≥ wv0033 ≥ wv1500`) after aggregation,
  keeping the wettest layer authoritative. A cell missing any layer stays
  missing rather than inheriting a neighbouring suction's value.

Chain per target cell:

1. Fetch the 250 m coverage (local tile, else SoilGrids WCS), apply the official
   scale factors, mask no-data, reproject Homolosine → `EPSG:4326`.
2. Keep only 250 m pixels whose PROBA-V cropland cover ≥ `soil.cropland_min_fraction`.
3. Keep only pixels of the cell's dominant class (`soil.dominant_mode`).
4. Aggregate those pixels with the variable's rule — arithmetic mean for
   `clay`/`sand`/`bdod`, **geometric** mean for `soc`, **H⁺** mean for `phh2o`
   (`aggregate.RULES`). With `soil.export_statistic: median` the plain median
   replaces all three.
5. Normalise texture so `clay + silt + sand = 100 %`.
6. Remap the SoilGrids depth intervals onto the SIMPLACE layers by
   overlap-weighted averaging (`remap_depth_weighted`), then round to 5 dp.

### Depth remapping

SIMPLACE layer bottoms are read from the reference (`SoilLayerDepth_1..6`,
falling back to `depth_1..6`): 0.1 / 0.3 / 0.5 / 0.7 / 1 / 2 m. Against the
SoilGrids intervals that gives:

| Layer | Depth | Built from |
| --- | --- | --- |
| `_1` | 0–10 cm | 0–5 cm (50 %) + 5–15 cm (50 %) |
| `_2` | 10–30 cm | 5–15 cm (25 %) + 15–30 cm (75 %) |
| `_3` | 30–50 cm | 30–60 cm (100 %) |
| `_4` | 50–70 cm | 30–60 cm (50 %) + 60–100 cm (50 %) |
| `_5` | 70–100 cm | 60–100 cm (100 %) |
| `_6` | 100–200 cm | 100–200 cm (100 %) |

A destination layer with no source overlap is `NaN`.

## 2. `location`

Our own `SimplaceID` (row-major over the 10 km grid), **not** the reference's
location ID: the project's `location.csv` carries latitude only, while sampling
SoilGrids needs longitude too.

## 3a. Computed by the Saxton–Rawls PTF — fallback only

`_STEM_TO_PTF` maps the same four `soilwater_*` stems, but the exporter consults
it **only for stems the SoilGrids water-content layers did not fill**. With
`wv0010`/`wv0033`/`wv1500` in `soil.layers` the PTF writes nothing; drop them
(or run against tiles that lack them) and `flags.compute_ptf: true` restores the
old behaviour, deriving the block from sand, clay and organic matter
(`soc × 1.724`) at **250 m before aggregation** because the equations are
non-linear. The PTF carries its own source depth intervals, so it may cover
fewer depths than the soil dataset.

## 3b. Initial mineral N — 12 columns

`ammonium_1..6`, `nitrate_1..6`, from `SoilExporter._mineral_nitrogen`.

SoilGrids `nitrogen` is **total** (largely organic) N in g kg⁻¹; SIMPLACE
initialises with **mineral** N per layer. Bulk density turns the concentration
into an N density, which is remapped onto the SIMPLACE layers and integrated
over their thickness:

```text
density [g N dm⁻³] = nitrogen [g kg⁻¹] × bdod [kg dm⁻³]
stock   [kg N ha⁻¹] = density × 100 × thickness [cm]
```

The density is formed **before** the depth remap — the stock is additive over
depth, so remapping the product is exact where remapping the two factors
separately would not be. `soil.mineral_n_fraction` (default 0.01) of the stock
is taken as mineral N and split by `soil.ammonium_share` (default 0.3), the rest
becoming nitrate. Both are **initialisation assumptions, not measurements**:
1 % is the usual order of magnitude for arable topsoils, but the split and the
fraction are yours to calibrate. Set `mineral_n_fraction: 0` — or drop
`nitrogen`/`bdod` from `soil.layers` — and both columns fall back to bucket 4.

## 4. Constant, copied from the reference — 75 columns

Not derived from any dataset. `soil_export.py` reads the **first row** of the
reference `soil.csv` and writes that value into every cell
(`template.get(col, sentinel)`); with no reference file they become the missing
sentinel (`-99`).

`soiltype`, `dampingdepth`, `soilwater_fc_global`, `soilwater_sat_global`,
`drainage_rate`, `deltatheta`, `DZF`, `depth_1..6`, `SoilLayerDepth_1..6`,
`soilwater_red_*`, `soilwater_res_*`, `macroporevolume_*`, `CaCO3_*`, `BSA_*`,
`LowerBoundaryConcentration`, `RootMaxUptakeRate`,
`InitialDissolvedPConcentration_*`, `InitialAdsorbedPConcentration_*`,
`InitialFixedPConcentration_*`, `slimalfa_*`.

## Caveats

**`theta_paw` and `ksat` are computed and discarded** when the PTF runs at all.
`saxton_rawls` returns
five variables; only four are mapped in `_STEM_TO_PTF`. `ksat` (mm h⁻¹) is the
natural source for `drainage_rate` and plausibly `macroporevolume_*`, both of
which currently stay at the reference constant.

**Rows without soil are dropped.** A cell whose derived columns are all `NaN` is
removed from the frame; the exported cell set is resolved upstream by
`spatial.export_cell_mask`, so weather, soil and management cover identical cells.

## 5. Method B — one file per soil class

With `soil.aggregation_method: top3` the same chain runs once per primary class:
each rank is masked to **its own** pixels before aggregation, so a rank-2 profile
is built from rank-2 pixels only and never blended with rank 1. The code lives in
[`multiclass.py`](multiclass.py) (`TopClassAggregation`),
[`classify.py`](classify.py) (`class_composition`) and
[`../exporters/soil_export.py`](../exporters/soil_export.py)
(`TopSoilExporter`).

`flags.export_top3_soil_csvs` then writes `soil_1.csv` … `soil_n.csv`, each with
the 142 reference columns above plus this metadata block:

| Column | Meaning |
| --- | --- |
| `SimplaceID` | Cell identifier — the same one `location` carries |
| `latitude`, `longitude` | Cell centre |
| `soil_class_id`, `class_name` | The class this row's profile was aggregated from |
| `area_km2`, `area_fraction` | Its cropland area in the cell, and that over the cell's classified area |
| `cell_shannon_entropy` | Normalised entropy over **all** classes in the cell |
| `cell_dominance_ratio` | `1 − p₁`: what `soil.csv` leaves out |

Buckets 1–4 above are unchanged for every rank, with two exceptions:

- **Rank 1 equals `soil.csv`.** The ranking and the majority vote break ties the
  same way, so `soil_1.csv` carries the same profiles as the single-file export.
- **PTF hydraulics are rank 1 only.** `saxton_rawls` runs on the dominant class'
  masked pixels; rather than lend those values to another class, ranks 2..n leave
  the `soilwater_*` fallback columns at the reference constant. This only bites
  when the `wv*` layers are absent — with them, every rank gets its own water
  retention from bucket 1.
