"""Pick a spread of German cropland cells both models can run.

A smoke test wants few cells but not a clustered few: yield and phenology both
vary strongly north-to-south across Germany, so a handful taken from one corner
would agree with a reference for the wrong reason. Cells are therefore drawn
**evenly across the latitude range**, and each is tagged with the CyBench NUTS-3
region whose polygon contains it, so the yield comparison joins without a
nearest-neighbour guess.
"""

from __future__ import annotations

import argparse
import logging
from pathlib import Path

import geopandas as gpd
import numpy as np
import pandas as pd

from cropmodelling4eu.config import RunConfig
from cropmodelling4eu.export import resolve_export

logger = logging.getLogger("select_german_cells")

CYBENCH_POLYGONS = Path("/data01/FDS/muduchuru/Data/Agri/cybench/polygons/DE/DE.shp")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--n-cells", type=int, default=40)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--composition", type=Path)
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(levelname)-7s %(message)s")

    config = RunConfig.model_validate(
        {
            **_load(args.config),
            **({"paths": {**_load(args.config)["paths"],
                          "composition_file": str(args.composition)}}
               if args.composition else {}),
        }
    )
    bundle = resolve_export(config, require_management=True)
    cells = bundle.cells()

    regions = gpd.read_file(CYBENCH_POLYGONS).to_crs("EPSG:4326")
    id_column = next(
        c for c in ("adm_id", "ADM_ID", "NUTS_ID", "nuts_id") if c in regions.columns
    )
    points = gpd.GeoDataFrame(
        cells,
        geometry=gpd.points_from_xy(cells["lon"], cells["lat"]),
        crs="EPSG:4326",
    )
    joined = gpd.sjoin(points, regions[[id_column, "geometry"]], how="inner", predicate="within")
    joined = joined.rename(columns={id_column: "adm_id"}).drop(columns=["index_right"])
    logger.info(
        "%d of %d runnable cells fall inside a German NUTS-3 polygon (%d regions)",
        len(joined), len(cells), joined["adm_id"].nunique(),
    )
    if joined.empty:
        raise SystemExit("No runnable cell falls inside Germany")

    # Even coverage in **both** directions. Banding on latitude alone is not
    # enough: SimplaceID runs west-to-east within a row, so taking the first
    # cell of each latitude band walks down Germany's western border and never
    # sees Brandenburg, Saxony or eastern Bavaria. Binning on both axes and
    # taking the cell nearest each bin's centre gives a transect across the
    # country instead of along one edge.
    n_lat = max(1, int(round(np.sqrt(args.n_cells * 1.3))))
    n_lon = max(1, int(np.ceil(args.n_cells / n_lat)))
    joined["_lat_bin"] = pd.qcut(joined["lat"], q=n_lat, duplicates="drop")
    joined["_lon_bin"] = pd.qcut(joined["lon"], q=n_lon, duplicates="drop")

    def _central(group: pd.DataFrame) -> pd.DataFrame:
        centre_lon = group["lon"].median()
        centre_lat = group["lat"].median()
        distance = (group["lon"] - centre_lon) ** 2 + (group["lat"] - centre_lat) ** 2
        return group.loc[[distance.idxmin()]]

    picked = (
        joined.groupby(["_lat_bin", "_lon_bin"], observed=True, sort=False)
        .apply(_central, include_groups=False)
        .reset_index(drop=True)
        .sort_values("SimplaceID")
        .head(args.n_cells)
        .reset_index(drop=True)
    )

    out = picked[["SimplaceID", "row", "col", "lon", "lat", "adm_id"]]
    args.out.parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(args.out, index=False)
    logger.info(
        "Selected %d cells, %.2f-%.2f N, %.2f-%.2f E, %d regions -> %s",
        len(out), out["lat"].min(), out["lat"].max(),
        out["lon"].min(), out["lon"].max(), out["adm_id"].nunique(), args.out,
    )
    print(out.to_string(index=False))
    return 0


def _load(path: Path) -> dict:
    import yaml

    with Path(path).open("r", encoding="utf-8") as handle:
        return yaml.safe_load(handle)


if __name__ == "__main__":
    raise SystemExit(main())
