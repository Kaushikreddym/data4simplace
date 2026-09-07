"""Compose the phenology stage: tiles in, validation tables out.

This is a **validation** dataset, not a model input. The pipeline's other stages
produce what a crop model consumes -- weather, soil, site, management -- and land
under the run directory as ``weather/``, ``soil/`` and so on. What this stage
writes is what a run is scored and calibrated *against*, so it lands under
``validation/phenology/`` beside them, from the same driver and the same run
config.

The work runs in two passes over the same tiles, because the winter/spring cut
cannot be chosen and applied in one:

``cuts``
    Reads ``CTY`` and ``CPMCE`` on a stride, accumulating the **unsplit**
    emergence histogram per country. Cheap -- a sixteenth of the pixels -- and it
    only has to resolve where two modes separate, not measure a median.

``stats``
    Reads all three layers at full resolution and accumulates the per-class
    histograms using the cuts the first pass fixed.

A single pass would have to assume the cut before it could measure it, which is
the assumption this design exists to remove.
"""

from __future__ import annotations

import argparse
import logging
import sys
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
import rasterio

from data4simplace.config import PipelineConfig, load_config
from data4simplace.grid import TargetGrid
from data4simplace.phenology import catalogue
from data4simplace.phenology.decode import CTY_CLASSES, N_BINS, days_to_doy, decode_yydoy
from data4simplace.phenology.reduce import merge_histograms, reduce_to_table
from data4simplace.phenology.seasons import (
    SPLIT_CROPS,
    SeasonCut,
    antimode_cut,
    default_cut_day,
    slug,
)
from data4simplace.phenology.zonal import accumulate_tile_year, histogram_frame
from data4simplace.phenology.zones import (
    admin_zone_raster,
    grid_zone_raster,
    load_admin_zones,
)

logger = logging.getLogger(__name__)

__all__ = ["PhenologyHandler", "build_parser", "main"]


@dataclass(slots=True)
class _TileZones:
    """Both zone labellings of one tile, built once and reused across its years."""

    admin: np.ndarray
    admin_ids: pd.DataFrame
    grid: np.ndarray
    country_of_zone: np.ndarray


class PhenologyHandler:
    """Runs one tile, or reduces what the tiles produced."""

    def __init__(self, config: PipelineConfig) -> None:
        self.config = config
        self.settings = config.phenology
        self.root = Path(config.paths.clms_root)
        self.out = Path(config.paths.output_dir) / "validation" / "phenology"
        self.grid = TargetGrid.from_config(config.grid)

    # -- layout ---------------------------------------------------------------

    @property
    def histogram_dir(self) -> Path:
        return self.out / "histograms"

    @property
    def cuts_dir(self) -> Path:
        return self.out / "cuts"

    @property
    def inventory_dir(self) -> Path:
        return self.root / "inventory"

    def tiles(self) -> list[str]:
        """Tile ids this run covers, from the cached selection or the catalogue."""
        cached = self.inventory_dir / "tiles_export.txt"
        if cached.is_file():
            return [t.strip() for t in cached.read_text().split() if t.strip()]
        return self.build_inventory()[1]

    def manifest(self) -> pd.DataFrame:
        """The tile manifest, from the local cache if it exists.

        Cached deliberately. The catalogue is three CSVs fetched over the
        network, and an array of 664 tasks each re-downloading them would be
        both slow and inconsiderate to a public service, for an answer that
        cannot differ between tasks of one run.
        """
        cached = self.inventory_dir / "clms_manifest_export.parquet"
        if cached.is_file():
            return pd.read_parquet(cached)
        return self.build_inventory()[0]

    def _selection_cells(self) -> tuple[np.ndarray, np.ndarray, str]:
        """The cell centres tile selection is made against.

        **The exported cell set, not the whole grid.** ``TargetGrid.cell_table``
        enumerates every cell in the bounding box, most of which is open sea or
        non-cropland: selecting against it keeps 862 tiles and ~105 GB, where the
        cells the export actually writes need 664 and 80.7 GB. A quarter of the
        download would be ocean.

        Falls back to the full grid when no export exists yet -- a first run has
        no ``site.csv`` to read -- and says which it used, because the two give
        different tile counts and a silent switch between them would look like
        the catalogue had changed.
        """
        site = Path(self.config.paths.output_dir) / "site" / "site.csv"
        if site.is_file():
            cells = pd.read_csv(site, usecols=["latitude", "longitude"])
            return cells["longitude"].to_numpy(), cells["latitude"].to_numpy(), str(site)
        table = self.grid.cell_table()
        logger.warning(
            "no %s yet: selecting tiles against the full grid, which is wider "
            "than the exported cell set. Re-run --stage inventory after the "
            "export to narrow it.", site,
        )
        return table["lon"].to_numpy(), table["lat"].to_numpy(), "full grid"

    def build_inventory(self, force: bool = False) -> tuple[pd.DataFrame, list[str]]:
        """Fetch the catalogues, select the tiles this run needs, and cache both.

        **A cached inventory is never widened by a rebuild.** Selection needs the
        exported cell set; without ``site.csv`` it can only fall back to the full
        grid, which is wider -- 862 tiles against 664, a quarter of them ocean. So
        rebuilding *after the export directory has been removed but before it has
        been rebuilt* would replace a good narrow list with a worse wide one, and
        send the fetch after ~200 tiles of sea. When there is nothing better to
        select against and a cache already exists, the cache wins.

        Args:
            force: Rebuild even when the selection would be no better informed.
        """
        cached = self.inventory_dir / "clms_manifest_export.parquet"
        site = Path(self.config.paths.output_dir) / "site" / "site.csv"
        if cached.is_file() and not site.is_file() and not force:
            tiles = self.tiles()
            logger.info(
                "keeping the cached inventory of %d tiles: %s does not exist yet, "
                "so a rebuild could only select against the full grid, which is "
                "wider. Re-run with force after the export to refresh it.",
                len(tiles), site,
            )
            return pd.read_parquet(cached), tiles

        full = catalogue.load_manifest()
        lon, lat, source = self._selection_cells()
        logger.info("selecting tiles against %s (%d cells)", source, lon.size)
        tiles = catalogue.select_tiles(full, lon, lat)
        selected = full[full["tile"].isin(tiles)]

        self.inventory_dir.mkdir(parents=True, exist_ok=True)
        full.to_parquet(self.inventory_dir / "clms_manifest_full.parquet", index=False)
        selected.to_parquet(
            self.inventory_dir / "clms_manifest_export.parquet", index=False
        )
        (self.inventory_dir / "tiles_export.txt").write_text("\n".join(tiles) + "\n")
        logger.info(
            "inventory: %d tiles, %d files, %.1f GB -> %s",
            len(tiles), len(selected), selected["content_length"].sum() / 1e9,
            self.inventory_dir,
        )
        return selected, tiles

    # -- pass 0: acquisition ---------------------------------------------------

    def run_fetch(self, tile: str) -> int:
        """Download every product and year of one tile.

        One array task owns one tile rather than one object: the 24 rasters of a
        tile share a manifest lookup and an S3 client, and 664 tasks is already
        ample parallelism against 15 936 tiny objects.

        Returns:
            Bytes now on disk for this tile.
        """
        from data4simplace.phenology.fetch import build_client, fetch_tile

        manifest = self.manifest()
        client = build_client()
        total = 0
        for year in self.settings.years:
            for product in catalogue.PRODUCTS:
                rows = manifest[
                    (manifest["tile"] == tile)
                    & (manifest["product"] == product)
                    & (manifest["year"] == year)
                ]
                if rows.empty:
                    logger.info("%s %s %d: not published", tile, product, year)
                    continue
                dest = catalogue.tile_path(self.root / "tiles", product, year, tile)
                fetch_tile(rows.iloc[0], dest, client=client)
                total += dest.stat().st_size
        logger.info("%s: %.1f MB on disk", tile, total / 1e6)
        return total

    # -- zones ----------------------------------------------------------------

    def _zones(self, profile) -> _TileZones:
        admin = load_admin_zones(
            self.settings.countries, Path(self.config.paths.cybench_polygon_root)
        )
        shape = (profile["height"], profile["width"])
        raster, ids = admin_zone_raster(admin, profile["transform"], shape, profile["crs"])
        grid = grid_zone_raster(profile["transform"], shape, profile["crs"], self.grid)
        countries = np.array([""] + list(ids["country"]), dtype=object)
        return _TileZones(raster, ids, grid, countries)

    # -- pass 1: where the seasons divide -------------------------------------

    def run_cuts(self, tile: str) -> Path:
        """Unsplit emergence histograms per country, on a stride."""
        stride = self.settings.cut_stride
        rows = []
        for year in self.settings.years:
            paths = {
                p: catalogue.tile_path(self.root / "tiles", p, year, tile)
                for p in ("CTY", "CPMCE")
            }
            if not all(p.is_file() for p in paths.values()):
                logger.info("%s %d: rasters absent, skipping", tile, year)
                continue
            with rasterio.open(paths["CTY"]) as src:
                zones = self._zones(src.profile)
                cty = src.read(1)[::stride, ::stride]
            with rasterio.open(paths["CPMCE"]) as src:
                emergence = decode_yydoy(src.read(1)[::stride, ::stride], year)
            admin = zones.admin[::stride, ::stride]

            for code in SPLIT_CROPS:
                mask = (cty == code) & np.isfinite(emergence) & (admin > 0)
                if not mask.any():
                    continue
                for zone in np.unique(admin[mask]):
                    sel = mask & (admin == zone)
                    counts = np.bincount(
                        emergence[sel].astype(np.int64), minlength=N_BINS
                    )[:N_BINS]
                    rows.append(
                        {
                            "country": zones.country_of_zone[zone],
                            "crop_code": int(code),
                            "year": int(year),
                            "counts": counts.astype(np.int64),
                        }
                    )
        frame = _explode_counts(rows)
        self.cuts_dir.mkdir(parents=True, exist_ok=True)
        path = self.cuts_dir / f"cuts_{tile}.parquet"
        frame.to_parquet(path, index=False)
        logger.info("%s: wrote %d cut-histogram rows", tile, len(frame))
        return path

    def reduce_cuts(self) -> Path:
        """Locate each country's antimode and write the cut table."""
        files = sorted(self.cuts_dir.glob("cuts_*.parquet"))
        if not files:
            raise FileNotFoundError(f"no cut histograms under {self.cuts_dir}")
        frame = pd.concat([pd.read_parquet(f) for f in files], ignore_index=True)
        merged = (
            frame.groupby(["country", "crop_code", "year", "bin"], observed=True)["count"]
            .sum()
            .reset_index()
        )

        records = []
        for (country, code, year), block in merged.groupby(
            ["country", "crop_code", "year"], observed=True
        ):
            counts = np.zeros(N_BINS, dtype=np.int64)
            counts[block["bin"].to_numpy()] = block["count"].to_numpy()
            cut: SeasonCut = antimode_cut(counts, int(year))
            records.append(
                {
                    "country": country,
                    "crop_code": int(code),
                    "year": int(year),
                    "cut_day": cut.day,
                    "cut_source": cut.source,
                    "cut_reason": cut.reason,
                    "winter_peak": cut.winter_peak,
                    "spring_peak": cut.spring_peak,
                    "trough": cut.trough,
                }
            )
        table = pd.DataFrame.from_records(records)
        self.out.mkdir(parents=True, exist_ok=True)
        path = self.out / "season_cuts.csv"
        table.to_csv(path, index=False)
        measured = int((table["cut_source"] == "antimode").sum())
        logger.info(
            "season cuts: %d rows, %d measured from the antimode, %d fell back to 1 Jan",
            len(table), measured, len(table) - measured,
        )
        return path

    # -- pass 2: the statistics -----------------------------------------------

    def _cut_raster(
        self, zones: _TileZones, year: int, cty: np.ndarray
    ) -> np.ndarray:
        """Per-pixel cut day, from the pixel's country **and its crop**.

        Both keys are needed. An earlier version took one cut per country, the
        median over its crops, on the theory that a single odd crop should not
        drag the classification. It did exactly the opposite: German oilseed
        rape is almost entirely autumn-sown, so it has no spring mode, falls
        back to 1 January -- and that fallback then pulled wheat's measured cut
        from 15 January to 1 January. Each crop keeps its own boundary; a crop
        that has none falls back alone.
        """
        path = self.out / "season_cuts.csv"
        fallback = default_cut_day(year)
        n_zones = len(zones.admin_ids) + 1
        width = int(max(SPLIT_CROPS)) + 1
        lut = np.full((n_zones, width), fallback, dtype=np.float32)

        if path.is_file():
            cuts = pd.read_csv(path)
            cuts = cuts[cuts["year"] == year]
            by_key = {
                (r.country, int(r.crop_code)): float(r.cut_day)
                for r in cuts.itertuples()
            }
            for i, country in enumerate(zones.admin_ids["country"], start=1):
                for code in SPLIT_CROPS:
                    if (country, code) in by_key:
                        lut[i, code] = by_key[(country, code)]

        codes = np.where(cty < width, cty, 0)
        return lut[zones.admin, codes]

    def run_stats(self, tile: str) -> list[Path]:
        """Full-resolution per-class histograms for one tile, all years."""
        written: list[Path] = []
        self.histogram_dir.mkdir(parents=True, exist_ok=True)
        zones: _TileZones | None = None

        for year in self.settings.years:
            paths = {
                p: catalogue.tile_path(self.root / "tiles", p, year, tile)
                for p in ("CTY", "CPMCE", "CPMCH")
            }
            if not all(p.is_file() for p in paths.values()):
                logger.info("%s %d: rasters absent, skipping", tile, year)
                continue

            with rasterio.open(paths["CTY"]) as src:
                if zones is None:      # geometry is year-independent: build once
                    zones = self._zones(src.profile)
                cty = src.read(1)
            with rasterio.open(paths["CPMCE"]) as src:
                cpmce = src.read(1)
            with rasterio.open(paths["CPMCH"]) as src:
                cpmch = src.read(1)

            cuts = self._cut_raster(zones, year, cty)
            for kind, raster, ids in (
                ("adm", zones.admin, zones.admin_ids),
                ("grid", zones.grid, None),
            ):
                if kind == "grid":
                    present = np.unique(raster[raster > 0])
                    index = np.zeros(int(raster.max()) + 1, dtype=np.int32)
                    index[present] = np.arange(1, present.size + 1)
                    labelled = index[raster]
                    ids = pd.DataFrame({"SimplaceID": present})
                else:
                    labelled = raster
                counts = accumulate_tile_year(
                    cty, cpmce, cpmch, labelled, len(ids), year, cuts,
                    block_rows=self.settings.block_rows,
                )
                frame = histogram_frame(counts, ids, kind, year, tile)
                if frame.empty:
                    continue
                path = self.histogram_dir / f"hist_{kind}_{tile}_{year}.parquet"
                frame.to_parquet(path, index=False)
                written.append(path)
        return written

    def reduce_stats(self) -> list[Path]:
        """Merge every tile's histograms into the two phenology tables."""
        cuts_path = self.out / "season_cuts.csv"
        cuts = None
        if cuts_path.is_file():
            raw = pd.read_csv(cuts_path)
            # Joined per crop, not per country: each split crop carries the cut
            # that actually classified it, so a row's provenance describes that
            # row. Both seasons of a crop share its boundary, hence two labels
            # per cut record.
            expanded = []
            for row in raw.itertuples():
                base = slug(CTY_CLASSES[int(row.crop_code)])
                for season in ("winter", "spring"):
                    expanded.append(
                        {
                            "country": row.country,
                            "year": int(row.year),
                            "crop": f"{base}_{season}",
                            # Both forms travel. `cut_doy` reads naturally, but
                            # it CANNOT be compared numerically against the
                            # emergence/harvest DOYs in the same row: the cut
                            # sits near New Year, so a spring emergence has a
                            # *smaller* DOY than the cut while being after it.
                            # `cut_day_anchored` is on the monotone
                            # days-since-1-Aug axis the split was actually made
                            # on, and is the one to compare with.
                            "cut_doy": days_to_doy(row.cut_day, int(row.year)),
                            "cut_day_anchored": int(row.cut_day),
                            "cut_source": row.cut_source,
                        }
                    )
            cuts = pd.DataFrame.from_records(expanded)

        written = []
        for kind, name in (("adm", "phenology_adm"), ("grid", "phenology_grid")):
            files = sorted(self.histogram_dir.glob(f"hist_{kind}_*.parquet"))
            if not files:
                logger.warning("no %s histograms under %s", kind, self.histogram_dir)
                continue

            # Reduced **one year at a time**. Merging every year at once is what
            # ran a 64 GB job out of memory on the grid: the admin key has 946
            # zones, the grid key has 70 705, and the long form is (zones x crops
            # x variables x bins). Years are independent -- no zone's histogram
            # spans two of them -- so this partitions exactly, with no effect on
            # the result, and bounds peak memory to a single year.
            by_year: dict[str, list[Path]] = {}
            for path in files:
                by_year.setdefault(path.stem.rsplit("_", 1)[-1], []).append(path)

            parts = []
            for year in sorted(by_year):
                merged = merge_histograms(by_year[year])
                parts.append(reduce_to_table(merged, cuts=cuts))
                del merged
                logger.info("%s %s: %d rows", kind, year, len(parts[-1]))

            table = pd.concat(parts, ignore_index=True) if parts else pd.DataFrame()
            path = self.out / f"{name}.parquet"
            table.to_parquet(path, index=False)
            logger.info("%s: %d rows -> %s", kind, len(table), path)
            written.append(path)
        return written


def _explode_counts(rows: list[dict]) -> pd.DataFrame:
    """Long form of the per-country cut histograms, non-zero bins only."""
    if not rows:
        return pd.DataFrame(columns=["country", "crop_code", "year", "bin", "count"])
    frames = []
    for row in rows:
        counts = row["counts"]
        bins = np.nonzero(counts)[0]
        if bins.size == 0:
            continue
        frames.append(
            pd.DataFrame(
                {
                    "country": row["country"],
                    "crop_code": row["crop_code"],
                    "year": row["year"],
                    "bin": bins.astype(np.int16),
                    "count": counts[bins],
                }
            )
        )
    return (
        pd.concat(frames, ignore_index=True)
        if frames
        else pd.DataFrame(columns=["country", "crop_code", "year", "bin", "count"])
    )


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="CLMS phenology: per-tile accumulation and reduction."
    )
    parser.add_argument("--config", required=True, type=Path)
    parser.add_argument(
        "--stage",
        required=True,
        choices=(
            "inventory",
            "clms-fetch",
            "cuts",
            "reduce-cuts",
            "stats",
            "reduce-stats",
            "count-tiles",
        ),
    )
    parser.add_argument(
        "--tile-index",
        type=int,
        help="Index into the tile list; SLURM_ARRAY_TASK_ID supplies it.",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    logging.basicConfig(
        level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s %(message)s"
    )
    config = load_config(args.config)
    if not config.flags.run_phenology_processing:
        logger.info("flags.run_phenology_processing is false; nothing to do")
        return 0

    handler = PhenologyHandler(config)
    tiles = handler.tiles()

    if args.stage == "count-tiles":
        print(len(tiles))
        return 0
    if args.stage == "inventory":
        handler.build_inventory()
        return 0
    if args.stage == "reduce-cuts":
        handler.reduce_cuts()
        return 0
    if args.stage == "reduce-stats":
        handler.reduce_stats()
        return 0

    if args.tile_index is None:
        raise SystemExit("--tile-index is required for the per-tile stages")
    if not 0 <= args.tile_index < len(tiles):
        raise SystemExit(f"--tile-index {args.tile_index} outside 0..{len(tiles) - 1}")
    tile = tiles[args.tile_index]
    logger.info("tile %s (%d of %d)", tile, args.tile_index + 1, len(tiles))

    if args.stage == "clms-fetch":
        handler.run_fetch(tile)
    elif args.stage == "cuts":
        handler.run_cuts(tile)
    else:
        handler.run_stats(tile)
    return 0


if __name__ == "__main__":
    sys.exit(main())
