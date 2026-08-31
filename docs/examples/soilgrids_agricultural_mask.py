"""Download SoilGrids for a lat/lon box and walk through the agricultural mask.

Follows soil/CLAUDE.md's "Agricultural Mask & Soil Aggregation Workflow" one
step at a time, on a single layer (topsoil clay), then runs the full
production path (dominant-class selection, every configured layer/depth) for
comparison:

  1. Fetch one 250 m SoilGrids layer (native resolution, un-scaled).
  2. Build the 250 m PROBA-V cropland keep-mask, aligned to that layer's grid.
  3. Apply the mask at 250 m -- *before* any aggregation.
  4. Aggregate the masked 250 m pixels onto the 10 km target grid.
  5. Compare against ``load_processed()``, the actual pipeline call (adds
     dominant-soil-class selection on top of the same two steps, run for
     every configured layer and depth).

Run: ``python docs/examples/soilgrids_agricultural_mask.py``
"""

from __future__ import annotations

from pathlib import Path

from data4simplace.config import PipelineConfig
from data4simplace.grid import TargetGrid
from data4simplace.soil.classify import bin_reduce, reducer_for
from data4simplace.soil.soilgrids import SoilGridsHandler
from data4simplace.spatial import CroplandWeights

# A small bbox around Magdeburg, Brandenburg (~10x10 target cells at 0.1 deg).
# SoilGrids and PROBA-V are both fetched/clipped to this box plus a 1-cell halo.
BBOX = dict(min_lon=11.0, max_lon=12.0, min_lat=52.0, max_lat=53.0)

# Copernicus PROBA-V LC100 Crops-CoverFraction GeoTIFF -- the pipeline's single
# cropland definition (see the project CLAUDE.md "Reference Data Paths" table).
# Point this at your own copy: https://lcviewer.vito.be/download
CROPLAND_WEIGHTS_PATH = (
    "/data01/FDS/muduchuru/Land/LULC/CopernicusLandCover/"
    "PROBAV_LC100_global_v3.0.1_2019-nrt_Crops-CoverFraction-layer_EPSG-4326.tif"
)

OUTPUT_DIR = Path("./soilgrids_mask_demo")


def build_config() -> PipelineConfig:
    """A minimal config: soil-only, a few layers, no reference/export stages."""
    return PipelineConfig.model_validate(
        {
            "flags": {
                "run_soil_processing": True,
            },
            "grid": {"resolution_deg": 0.1, **BBOX},
            "time": {"start": "1979-01-01", "end": "1979-01-01"},
            "paths": {
                "soilgrids_root": None,  # None -> fetch from the WCS
                "cropland_weights_path": CROPLAND_WEIGHTS_PATH,
                "output_dir": str(OUTPUT_DIR),
            },
            "soil": {
                # A short layer/depth list keeps the WCS download small; the
                # project's config.yaml lists the full production set.
                "layers": ["clay", "silt", "sand"],
                "depths": ["0-5cm"],
                "dominant_mode": "usda",
                "aggregation_method": "dominant",
                "cropland_min_fraction": 0.8,
                "min_cropland_pixels": 1,
                "use_wcs": True,
            },
        }
    )


def main() -> None:
    config = build_config()
    grid = TargetGrid.from_config(config.grid)
    handler = SoilGridsHandler(config)

    # --- 1. Fetch one 250 m layer, native resolution, un-scaled ------------
    raw = handler.fetch_wcs("clay", "0-5cm")
    clay_250m = SoilGridsHandler.mask_nodata(SoilGridsHandler.unscale(raw, "clay"), "clay")
    print(f"1. Fetched clay 0-5cm: {dict(clay_250m.sizes)} pixels, "
          f"mean={float(clay_250m.mean()):.1f}%")

    # --- 2. The 250 m cropland keep-mask, aligned to the SoilGrids grid ----
    weights = CroplandWeights(config)
    keep_250m = weights.keep_mask(clay_250m)  # None if no cropland raster configured
    kept_share = float(keep_250m.mean()) if keep_250m is not None else 1.0
    print(f"2. Cropland keep-mask (>= {config.soil.cropland_min_fraction:.0%} cover): "
          f"{kept_share:.1%} of pixels kept")

    # --- 3. Apply the mask at 250 m -- before any aggregation --------------
    clay_masked_250m = clay_250m.where(keep_250m) if keep_250m is not None else clay_250m
    print(f"3. Clay mean at 250 m -- unmasked: {float(clay_250m.mean()):.2f}%, "
          f"cropland-masked: {float(clay_masked_250m.mean()):.2f}%")

    # --- 4. Aggregate the masked 250 m pixels onto the 10 km target grid ---
    # The same primitive soilgrids.py's _aggregate_masked uses per layer/depth.
    clay_10km = bin_reduce(clay_masked_250m, grid, reducer_for("clay"))
    print(f"4. Aggregated to the {grid.shape[0]}x{grid.shape[1]} 10 km grid: "
          f"{int(clay_10km.notnull().sum())} cells carry a value")

    # --- 5. The full production path, for comparison -----------------------
    # load_processed() runs the same keep_mask -> mask -> aggregate sequence
    # for every configured layer and depth, plus dominant-soil-class selection
    # (soil/CLAUDE.md): pixels are additionally restricted to the majority
    # USDA/WRB class of each cell before aggregation, so its clay mean differs
    # from the single-layer walkthrough above (which used every cropland pixel).
    soil, _hydraulic = handler.load_processed()
    print("\n5. Production soil dataset (dominant-class aggregation, all layers):")
    print(soil)

    OUTPUT_DIR.mkdir(exist_ok=True)
    clay_10km.rename("clay_cropland_mean").to_netcdf(OUTPUT_DIR / "clay_10km_walkthrough.nc")
    soil.to_netcdf(OUTPUT_DIR / "soil_masked.nc")
    print(f"\nWrote outputs to {OUTPUT_DIR}/")


if __name__ == "__main__":
    main()
