"""Administrative geometry and the 10 km cell to region/country assignments.

The national footprints come from the CyBench administrative polygons, not
from a separate world dataset. That is deliberate: the reference yields are
reported on exactly those units, so dissolving them gives a "France" whose
simulated cells and observed regions cover the same ground. A Natural Earth
outline would include French territory CyBench never reports on, and the
difference would land silently in the bias.

The same polygons serve both resolutions a comparison can run at:
:func:`assign_cells_to_countries` labels a cell with the dissolved country,
:func:`assign_cells_to_regions` with the undissolved unit the statistic is
actually published on — NUTS-2 for eight of the European countries, NUTS-3
for the other fifteen (:func:`nuts_level`).
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
    "assign_cells_to_regions",
    "default_cache_path",
    "load_admin_polygons",
    "load_country_polygons",
    "nuts_level",
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


def _cell_points(cells: pd.DataFrame, id_col: str) -> gpd.GeoDataFrame:
    """Cell centres as points, carrying only the identifier."""
    return gpd.GeoDataFrame(
        cells[[id_col]].copy(),
        geometry=gpd.points_from_xy(cells["lon"], cells["lat"]),
        crs="EPSG:4326",
    )


def _label_points(
    points: gpd.GeoDataFrame,
    polygons: gpd.GeoDataFrame,
    label_cols: list[str],
    id_col: str,
    snap_km: float,
) -> pd.DataFrame:
    """Label each point with the polygon it falls in, snapping the strays.

    Shared by the country and the administrative-unit joins, which differ only
    in which polygons they are given: both label a 10 km cell centre by
    containment first, then match whatever is left to the nearest polygon
    within ``snap_km``, because a cell whose centre lands just offshore still
    overlaps land and dropping it would cost Denmark and the Netherlands a
    fifth of their cells.

    Args:
        points: Cell centres from :func:`_cell_points`.
        polygons: Any polygon layer carrying ``label_cols``.
        label_cols: Columns to copy off the matched polygon. The first is the
            one containment is judged on.
        id_col: Identifier column of ``points``.
        snap_km: Nearest-polygon tolerance; ``0`` disables the second pass.

    Returns:
        One row per point, in the input order: ``id_col``, the ``label_cols``,
        and ``snapped`` (True where the match came from the distance pass).
    """
    inside = (
        gpd.sjoin(points, polygons, how="left", predicate="within")
        .drop(columns="index_right")
        # A cell centre on a shared border can match both neighbours. Keeping
        # the first is arbitrary but stable, and it affects a handful of cells.
        .drop_duplicates(subset=id_col)
        .set_index(id_col)[label_cols]
    )
    result = pd.DataFrame({id_col: points[id_col].to_numpy()})
    for col in label_cols:
        result[col] = result[id_col].map(inside[col])
    result["snapped"] = False

    unmatched = result[label_cols[0]].isna()
    logger.info(
        "%d of %d cells inside a polygon", int((~unmatched).sum()), len(result)
    )

    if snap_km > 0 and unmatched.any():
        near = (
            gpd.sjoin_nearest(
                points[points[id_col].isin(result.loc[unmatched, id_col])]
                .to_crs(EQUAL_AREA_CRS),
                polygons.to_crs(EQUAL_AREA_CRS),
                how="inner",
                max_distance=snap_km * 1000.0,
            )
            .drop_duplicates(subset=id_col)
            .set_index(id_col)[label_cols]
        )
        snapped = result[id_col].map(near[label_cols[0]])
        take = unmatched & snapped.notna()
        for col in label_cols:
            result.loc[take, col] = result.loc[take, id_col].map(near[col])
        result.loc[take, "snapped"] = True
        logger.info(
            "snapped %d further cells within %.1f km of a border",
            int(take.sum()), snap_km,
        )

    return result


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
    result = _label_points(
        _cell_points(cells, id_col), polygons, ["country"], id_col, snap_km
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


def assign_cells_to_regions(
    cells: pd.DataFrame,
    countries: list[str],
    polygon_dir: Path | None = None,
    snap_km: float = SNAP_KM,
    cache: Path | None = None,
    id_col: str = "SimplaceID",
) -> pd.DataFrame:
    """Label each 10 km cell with the administrative unit CyBench reports on.

    The same two passes as :func:`assign_cells_to_countries` — containment on
    the cell centre, then a nearest-polygon snap within ``snap_km`` — run
    against the undissolved polygons, so a cell is labelled with the very unit
    (NUTS-2 or NUTS-3, whichever that country's statistic is published on) its
    yield will be compared against.

    **A cell belongs to exactly one unit, chosen by its centre.** A 10 km cell
    straddling a boundary contributes all of itself to one side and nothing to
    the other; splitting it by overlap area would be defensible but would also
    make each unit's simulated mean depend on cells its statistic does not
    cover. The bias this leaves is second-order at NUTS-2 and real at NUTS-3,
    where a unit can be smaller than a single cell — which is why the paired
    table carries ``n_cells`` and the notebook reports its distribution rather
    than presenting every unit as equally well resolved.

    Args:
        cells: Unique cells with ``SimplaceID``, ``lon``, ``lat``.
        countries: Candidate country codes.
        polygon_dir: Root holding the CyBench polygons.
        snap_km: Nearest-unit tolerance for unmatched cells; ``0`` disables it.
        cache: Parquet to read the result from and write it to. Give it a
            different path from the country join's — the two carry different
            columns for the same cells.
        id_col: Column holding the cell identifier.

    Returns:
        ``cells`` plus ``adm_id`` and ``adm_country`` (both nullable string)
        and ``adm_snapped`` (bool), one row per input cell and in the input
        order. ``adm_country`` is the unit's own country, which a caller can
        check against the country join: the two are the same polygons, so they
        agree by containment and can differ only for a snapped border cell.

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
            logger.info("cell-to-region map read from %s", cache)
            return cells.merge(cached, on=id_col, how="left")
        logger.info("%s covers a different cell set; rebuilding", cache)

    admin = load_admin_polygons(countries, polygon_dir)
    result = _label_points(
        _cell_points(cells, id_col), admin, ["adm_id", "country"], id_col, snap_km
    ).rename(columns={"country": "adm_country", "snapped": "adm_snapped"})
    for col in ("adm_id", "adm_country"):
        result[col] = result[col].astype("string")
    logger.info(
        "%d cells assigned to %d of %d administrative units",
        int(result["adm_id"].notna().sum()), result["adm_id"].nunique(), len(admin),
    )

    if cache is not None:
        Path(cache).parent.mkdir(parents=True, exist_ok=True)
        result.to_parquet(cache, index=False)
        logger.info("cached cell-to-region map to %s", cache)

    return cells.merge(result, on=id_col, how="left")


def nuts_level(adm_id: pd.Series) -> pd.Series:
    """NUTS level of a CyBench ``adm_id``, read off the code length.

    A NUTS code is the two-letter country code plus one character per level, so
    ``PL21`` is NUTS-2 and ``DE111`` is NUTS-3. CyBench publishes each European
    country at exactly one level, and which one is a property of the national
    statistic rather than a choice this code gets to make — so the level is
    derived, never assumed.

    Identifiers that are not NUTS at all (CyBench's Argentinian, Brazilian,
    Indian and US units) come back as ``<NA>`` rather than a made-up level.

    Args:
        adm_id: Administrative unit codes.

    Returns:
        Nullable integer level in 1-3, aligned to ``adm_id``.
    """
    code = adm_id.astype("string").str.strip()
    level = code.str.len().astype("Int64") - 2
    is_nuts = code.str.fullmatch(r"[A-Z]{2}[A-Z0-9]{1,3}").fillna(False)
    return level.where(is_nuts & level.between(1, 3))


def default_cache_path(
    n_countries: int, snap_km: float = SNAP_KM, prefix: str = "cell_country"
) -> Path:
    """Cache filename that changes when the join's inputs change.

    The country list and the snap distance both alter the result, so both are
    in the name — otherwise a re-run with a wider snap would silently reuse the
    narrow one. ``prefix`` names the join itself: ``cell_country`` for
    :func:`assign_cells_to_countries`, ``cell_region`` for
    :func:`assign_cells_to_regions`, and something else again for the 0.5°
    gridded joins, whose identifiers live in a different namespace entirely. A
    cache written for one join is not valid for another, so they must not share
    a prefix.
    """
    return CACHE_DIR / f"{prefix}_n{n_countries}_snap{snap_km:g}km.parquet"
