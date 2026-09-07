"""Which CLMS tiles exist upstream, and which of them this run needs.

CDSE publishes a catalogue CSV per product listing every tile with its UUID, S3
path, byte length, MD5 and bounding box. Reading those three CSVs is the whole
of discovery -- no authentication, no STAC walk.

**Selection is by the exported cell set, not by a bounding box.** The domain box
contains a great deal of sea and non-cropland; intersecting the tiles with the
cells the export actually writes takes the fetch from 897 tiles to 664, and from
109 GB to 80.7 GB.

Note the two products live under different themes: ``CPMCE``/``CPMCH`` sit under
``cropping_patterns/`` but crop types are under ``crop_types/``. There is no
``main-crop-types`` slug beside the other two and guessing one returns a 404.
"""

from __future__ import annotations

import logging
import re
from pathlib import Path

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

__all__ = ["CATALOGUE_BASE", "PRODUCTS", "load_manifest", "select_tiles", "tile_path"]

CATALOGUE_BASE = (
    "https://s3.waw3-1.cloudferro.com/swift/v1/CatalogueCSV/landcover_landuse/"
)

#: Product code -> catalogue sub-path. See the note above about the themes.
PRODUCTS: dict[str, str] = {
    "CTY": "crop_types/clms_vlcc_crop-types_europe_10m_yearly_v1",
    "CPMCE": "cropping_patterns/clms_vlcc_main-crop-emergence_europe_10m_yearly_v1",
    "CPMCH": "cropping_patterns/clms_vlcc_main-crop-harvest_europe_10m_yearly_v1",
}

_TILE_RE = re.compile(r"_R10m_(E\d+N\d+)_")


def load_manifest(products: dict[str, str] | None = None) -> pd.DataFrame:
    """Read every product's catalogue CSV and parse the bbox into lon/lat bounds.

    Args:
        products: Override of :data:`PRODUCTS`, for testing against a local copy.

    Returns:
        One row per published file, with ``product``, ``year``, ``tile`` and the
        four bound columns added to the catalogue's own fields.
    """
    frames = []
    for code, slug in (products or PRODUCTS).items():
        stem = slug.split("/")[-1]
        url = f"{CATALOGUE_BASE}{slug}/{stem}_cog.csv"
        frame = pd.read_csv(url, sep=";")
        frame["product"] = code
        frame["year"] = pd.to_datetime(frame["content_date_start"]).dt.year
        frame["tile"] = frame["name"].str.extract(_TILE_RE)
        corners = frame["bbox"].str.findall(r"(-?\d+\.?\d*)").apply(
            lambda v: np.asarray(v, dtype=float)
        )
        frame["lon_min"] = corners.apply(lambda a: a[0::2].min())
        frame["lon_max"] = corners.apply(lambda a: a[0::2].max())
        frame["lat_min"] = corners.apply(lambda a: a[1::2].min())
        frame["lat_max"] = corners.apply(lambda a: a[1::2].max())
        frames.append(frame)
        logger.info("%s: %d files, %.1f GB", code, len(frame),
                    frame["content_length"].sum() / 1e9)

    return pd.concat(frames, ignore_index=True)


def select_tiles(manifest: pd.DataFrame, lon: np.ndarray, lat: np.ndarray) -> list[str]:
    """Tiles holding at least one of the given cell centres.

    Args:
        manifest: Output of :func:`load_manifest`.
        lon: Cell-centre longitudes.
        lat: Cell-centre latitudes.

    Returns:
        Sorted tile identifiers, e.g. ``["E38N24", ...]``.
    """
    lon = np.asarray(lon)
    lat = np.asarray(lat)
    tiles = manifest.drop_duplicates("tile")[
        ["tile", "lon_min", "lon_max", "lat_min", "lat_max"]
    ]
    keep = [
        t.tile
        for t in tiles.itertuples()
        if (
            (lon >= t.lon_min) & (lon <= t.lon_max)
            & (lat >= t.lat_min) & (lat <= t.lat_max)
        ).any()
    ]
    logger.info("%d published tiles -> %d intersecting the cell set",
                manifest["tile"].nunique(), len(keep))
    return sorted(keep)


def tile_path(root: Path, product: str, year: int, tile: str) -> Path:
    """Local path of one fetched raster.

    The layout mirrors the product's own naming so a file is identifiable
    without its directory: ``CLMS_HRLVLCC_<product>_S<year>_R10m_<tile>_03035.tif``.
    """
    return (
        Path(root)
        / product
        / str(year)
        / f"CLMS_HRLVLCC_{product}_S{year}_R10m_{tile}_03035.tif"
    )
