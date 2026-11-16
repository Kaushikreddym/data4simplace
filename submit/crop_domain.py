"""Cells where a crop holds at least a minimum share of the cell's cropland.

    python submit/crop_domain.py --config <export>/_work/config_run.yaml \\
        --min-share 0.05 --out <export>/domain_maize_5pct.csv

Share = harvested area of the crop group (``IrrigationClassifier.harvested_area``:
ECIRA where it classifies the cell, MIRCA-OS elsewhere) over the cell's cropland
area (mean PROBA-V cover fraction x cell area). Only cells the export already
writes are considered, so the domain can narrow the run but never widen it.
"""

from __future__ import annotations

import argparse
import logging

import numpy as np
import pandas as pd

from data4simplace.config import load_config
from data4simplace.grid import TargetGrid
from data4simplace.management.irrigation import IrrigationClassifier
from data4simplace.soil.classify import cell_assignment
from data4simplace.spatial.area import latitude_band_area_km2
from data4simplace.spatial.cropland_weights import CroplandWeights

logger = logging.getLogger("crop_domain")


def cropland_area_ha(config, grid: TargetGrid) -> np.ndarray:
    """Mean PROBA-V cover fraction x cell area, ha, on ``(lat, lon)``."""
    fraction = CroplandWeights(config).load_fraction()
    n_lat, n_lon = grid.shape
    cell, values = cell_assignment(fraction.astype("float32"), grid)
    pixels = np.bincount(cell, minlength=n_lat * n_lon)
    cover = np.bincount(cell, weights=values, minlength=n_lat * n_lon)
    mean_cover = np.divide(cover, pixels, out=np.zeros_like(cover), where=pixels > 0)
    area_km2 = latitude_band_area_km2(grid.lat_centers, grid.resolution_deg, grid.resolution_deg)
    return mean_cover.reshape(n_lat, n_lon) * np.asarray(area_km2)[:, None] * 100.0


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--config", required=True)
    parser.add_argument("--min-share", type=float, default=0.05)
    parser.add_argument("--out", required=True)
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")

    config = load_config(args.config)
    grid = TargetGrid(**{k: getattr(config.grid, k) for k in
                         ("resolution_deg", "min_lon", "max_lon", "min_lat", "max_lat")})
    crop_ha = IrrigationClassifier(config).harvested_area()
    cropland_ha = cropland_area_ha(config, grid)
    share = np.divide(crop_ha, cropland_ha, out=np.full_like(crop_ha, np.nan), where=cropland_ha > 0)

    cells = grid.cell_table()
    flat = cells["row"].to_numpy() * grid.shape[1] + cells["col"].to_numpy()
    cells = cells.assign(crop_ha=crop_ha.ravel()[flat], cropland_ha=cropland_ha.ravel()[flat],
                         share=share.ravel()[flat])
    exported = pd.read_csv(f"{config.paths.output_dir}/site/site.csv", usecols=["location"])["location"]
    cells = cells[cells["SimplaceID"].isin(exported)]
    keep = cells[cells["share"] >= args.min_share]
    keep[["SimplaceID", "lat", "lon", "crop_ha", "cropland_ha", "share"]].to_csv(args.out, index=False)
    logger.info("%d of %d exported cells hold >= %.0f %% of their cropland in the crop "
                "(%.0f of %.0f kha of the crop kept)", len(keep), len(cells), 100 * args.min_share,
                keep["crop_ha"].sum() / 1e3, cells["crop_ha"].sum() / 1e3)


if __name__ == "__main__":
    main()
