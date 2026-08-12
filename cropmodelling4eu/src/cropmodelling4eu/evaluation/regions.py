"""Country geometry and the 10 km cell to country assignment.

The national footprints come from the CyBench administrative polygons, not
from a separate world dataset. That is deliberate: the reference yields are
reported on exactly those units, so dissolving them gives a "France" whose
simulated cells and observed regions cover the same ground. A Natural Earth
outline would include French territory CyBench never reports on, and the
difference would land silently in the bias.
"""

from __future__ import annotations

import logging
from pathlib import Path

import geopandas as gpd
import pandas as pd

from .config import CACHE_DIR, CYBENCH_POLYGON_DIR, SNAP_KM

logger = logging.getLogger(__name__)

__all__ = [
    "EQUAL_AREA_CRS",
    "assign_cells_to_countries",
    "default_cache_path",
    "load_admin_polygons",
    "load_country_polygons",
]

#: ETRS89 / LAEA Europe. Equal-area and the Eurostat standard for the domain,
#: so the metric snap distance below is a true distance rather than a degree
#: approximation that would shrink by a factor of two between Crete and Lapland.
EQUAL_AREA_CRS: str = "EPSG:3035"


def load_admin_polygons(
    countries: list[str], polygon_dir: Path | None = None
) -> gpd.GeoDataFrame:
    """Administrative polygons for a list of countries, concatenated.

    Args:
        countries: Two-letter CyBench codes.
        polygon_dir: Root holding ``<country>/<country>.shp``.

    Returns:
        Columns ``country``, ``adm_id``, ``geometry`` in EPSG:4326.

    Raises:
        ValueError: If no shapefile was found.
    """
    polygon_dir = polygon_dir or CYBENCH_POLYGON_DIR
    frames = []
    for code in countries:
        path = polygon_dir / code / f"{code}.shp"
        if not path.is_file():
            logger.info("%s: no polygons at %s", code, path)
            continue
        frame = gpd.read_file(path)
        frame["country"] = code
        frames.append(frame[["country", "adm_id", "geometry"]])

    if not frames:
        raise ValueError(f"no CyBench polygons found under {polygon_dir}")

    admin = gpd.GeoDataFrame(
        pd.concat(frames, ignore_index=True), crs=frames[0].crs
    ).to_crs("EPSG:4326")
    logger.info("%d admin polygons across %d countries", len(admin),
                admin["country"].nunique())
    return admin


def load_country_polygons(
    countries: list[str], polygon_dir: Path | None = None
) -> gpd.GeoDataFrame:
    """National footprints, dissolved from the administrative polygons.

    Args:
        countries: Two-letter CyBench codes.
        polygon_dir: Root holding ``<country>/<country>.shp``.

    Returns:
        One row per country: ``country``, ``geometry`` in EPSG:4326, sorted by
        code so map legends and table rows agree.
    """
    admin = load_admin_polygons(countries, polygon_dir)
    dissolved = (
        admin.dissolve(by="country")
        .reset_index()[["country", "geometry"]]
        .sort_values("country")
        .reset_index(drop=True)
    )
    logger.info("dissolved to %d national footprints", len(dissolved))
    return gpd.GeoDataFrame(dissolved, crs=admin.crs)


def assign_cells_to_countries(
    cells: pd.DataFrame,
    countries: list[str],
    polygon_dir: Path | None = None,
    snap_km: float = SNAP_KM,
    cache: Path | None = None,
    id_col: str = "SimplaceID",
) -> pd.DataFrame:
    """Label each 10 km cell with the country it falls in.

    Two passes. First a strict point-in-polygon test on the cell centre. Cells
    that fall in no polygon are then matched to the nearest country within
    ``snap_km``: a 10 km cell whose centre lands just offshore still overlaps
    land, and dropping it would cost Denmark and the Netherlands a fifth of
    their cells. Cells further out than the snap — the UK, Norway, the western
    Balkans, North Africa, everything the CyBench footprint excludes — are
    returned with ``country = NaN`` for the caller to drop.

    Args:
        cells: Unique cells with ``SimplaceID``, ``lon``, ``lat`` (see
            :func:`utils.torchcrop.simulation_cells`).
        countries: Candidate country codes.
        polygon_dir: Root holding the CyBench polygons.
        snap_km: Nearest-country tolerance for unmatched cells. Half a grid
            cell by default; ``0`` disables the second pass.
        cache: Parquet to read the result from and write it to. The join costs
            a couple of seconds, but caching keeps a notebook re-run free.
            ``None`` disables caching.
        id_col: Column holding the cell identifier. The 10 km run keys on
            ``SimplaceID``; the gridded evaluations pass ``grid_id`` to run the
            same join over 0.5° cell centres. A cache written for one is not
            valid for the other, so give them different ``cache`` paths.

    Returns:
        ``cells`` plus ``country`` (nullable string) and ``snapped`` (bool,
        True where the cell was matched by distance rather than containment),
        one row per input cell and in the input order.

    Raises:
        KeyError: If ``cells`` lacks a required column.
    """
    required = {id_col, "lon", "lat"}
    missing = required - set(cells.columns)
    if missing:
        raise KeyError(f"cells lacks {sorted(missing)}")

    if cache is not None and Path(cache).is_file():
        cached = pd.read_parquet(cache)
        if id_col in cached.columns and set(cached[id_col]) == set(cells[id_col]):
            logger.info("cell-to-country map read from %s", cache)
            return cells.merge(cached, on=id_col, how="left")
        logger.info("%s covers a different cell set; rebuilding", cache)

    polygons = load_country_polygons(countries, polygon_dir)
    points = gpd.GeoDataFrame(
        cells[[id_col]].copy(),
        geometry=gpd.points_from_xy(cells["lon"], cells["lat"]),
        crs="EPSG:4326",
    )

    inside = (
        gpd.sjoin(points, polygons, how="left", predicate="within")
        .drop(columns="index_right")
        # A cell centre on a shared border can match both neighbours. Keeping
        # the first is arbitrary but stable, and it affects a handful of cells.
        .drop_duplicates(subset=id_col)
        .set_index(id_col)["country"]
    )
    result = pd.DataFrame(
        {id_col: cells[id_col].to_numpy()}
    ).assign(country=lambda d: d[id_col].map(inside), snapped=False)

    unmatched = result["country"].isna()
    logger.info(
        "%d of %d cells inside a CyBench country", int((~unmatched).sum()), len(result)
    )

    if snap_km > 0 and unmatched.any():
        near = gpd.sjoin_nearest(
            points[points[id_col].isin(result.loc[unmatched, id_col])]
            .to_crs(EQUAL_AREA_CRS),
            polygons.to_crs(EQUAL_AREA_CRS),
            how="inner",
            max_distance=snap_km * 1000.0,
        ).drop_duplicates(subset=id_col).set_index(id_col)["country"]

        snapped = result[id_col].map(near)
        take = result["country"].isna() & snapped.notna()
        result.loc[take, "country"] = snapped[take]
        result.loc[take, "snapped"] = True
        logger.info(
            "snapped %d further cells within %.1f km of a border",
            int(take.sum()), snap_km,
        )

    result["country"] = result["country"].astype("string")
    logger.info(
        "%d cells assigned, %d outside the CyBench footprint",
        int(result["country"].notna().sum()), int(result["country"].isna().sum()),
    )

    if cache is not None:
        Path(cache).parent.mkdir(parents=True, exist_ok=True)
        result.to_parquet(cache, index=False)
        logger.info("cached cell-to-country map to %s", cache)

    return cells.merge(result, on=id_col, how="left")


def default_cache_path(
    n_countries: int, snap_km: float = SNAP_KM, prefix: str = "cell"
) -> Path:
    """Cache filename that changes when the join's inputs change.

    The country list and the snap distance both alter the result, so both are
    in the name — otherwise a re-run with a wider snap would silently reuse the
    narrow one. ``prefix`` separates the 10 km join from the 0.5° one, whose
    identifiers live in a different namespace entirely.
    """
    return CACHE_DIR / f"{prefix}_country_n{n_countries}_snap{snap_km:g}km.parquet"
