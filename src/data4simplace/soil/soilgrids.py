"""SoilGrids fetcher, scale-factor un-scaler and CRS transformer.

SoilGrids v2 provides 250 m global soil property layers in the Homolosine
projection (``EPSG:152160``). Values are stored as scaled integers; the
official conversion factors must be applied before any calculation. This module
loads layers (from a local root or the public WCS service), un-scales them,
reprojects to ``EPSG:4326`` and aggregates onto the 10 km target grid.

WCS coverages are requested directly in EPSG:4326 by default (the service
reprojects on the fly and subsets in lon/lat), so no client-side Homolosine
transform is needed; set ``soil.wcs_crs`` to fetch in the native grid instead.
"""

from __future__ import annotations

import logging
import tempfile
from pathlib import Path

from typing import TYPE_CHECKING

import numpy as np
import requests
import rioxarray  # noqa: F401  (registers the .rio accessor)
import xarray as xr
from pyproj import CRS, Transformer

from data4simplace.config import PipelineConfig
from data4simplace.grid import TargetGrid

if TYPE_CHECKING:  # pragma: no cover - typing only
    from data4simplace.soil.multiclass import (
        PrimaryClassStatistics,
        TopClassAggregation,
    )

logger = logging.getLogger(__name__)

# SoilGrids uses the Interrupted Goode Homolosine projection. ISRIC labels it
# ``EPSG:152160``, but that code is not in the PROJ database, so define it
# explicitly for client-side coordinate transforms and CRS tagging. The WCS
# service itself still understands the 152160 code server-side.
_HOMOLOSINE_PROJ4 = (
    "+proj=igh +lat_0=0 +lon_0=0 +x_0=0 +y_0=0 +ellps=WGS84 +units=m +no_defs"
)

# Official SoilGrids v2 conversion factors: mapped_value / factor -> unit.
# Reference: https://www.isric.org/explore/soilgrids/faq-soilgrids
SCALE_FACTORS: dict[str, float] = {
    "clay": 10.0,      # g/kg -> % (0.1 %)
    "silt": 10.0,      # g/kg -> %
    "sand": 10.0,      # g/kg -> %
    "bdod": 100.0,     # cg/cm3 -> kg/dm3
    "soc": 10.0,       # dg/kg -> g/kg
    "phh2o": 10.0,     # pH*10 -> pH
    "nitrogen": 100.0,  # cg/kg -> g/kg
    "cec": 10.0,       # mmol(c)/kg -> cmol(c)/kg
    "cfvo": 10.0,      # cm3/dm3 -> %
    "wv0010": 10.0,    # 0.1 vol% -> vol% (water content at 10 kPa)
    "wv0033": 10.0,    # 0.1 vol% -> vol% (water content at 33 kPa)
    "wv1500": 10.0,    # 0.1 vol% -> vol% (water content at 1500 kPa)
}

# The SoilGrids volumetric water-content coverages, wettest suction first. They
# are predicted independently of one another, so the physical ordering
# ``theta(10 kPa) >= theta(33 kPa) >= theta(1500 kPa)`` is not guaranteed pixel
# by pixel; :meth:`SoilGridsHandler.harmonise_water_retention` enforces it.
WATER_RETENTION_LAYERS: tuple[str, ...] = ("wv0010", "wv0033", "wv1500")

# Layers where a stored ``0`` means "no data" rather than a measurement. The
# GeoTIFFs carry no nodata tag, so the sentinel has to be masked by value; see
# :meth:`SoilGridsHandler.mask_nodata`. ``cfvo`` is deliberately absent: a soil
# with no coarse fragments legitimately reads 0.
ZERO_IS_NODATA: frozenset[str] = frozenset(
    {
        "clay", "silt", "sand", "bdod", "soc", "phh2o", "nitrogen", "cec",
        # No real soil holds zero water at any of the three suctions, so a 0
        # here is the same water/ice/gap sentinel as for the other properties.
        "wv0010", "wv0033", "wv1500",
    }
)

# WCS base per property (SoilGrids v2 Web Coverage Service, Homolosine grid).
_WCS_BASE = "https://maps.isric.org/mapserv?map=/map/{layer}.map"


class SoilGridsHandler:
    """Load, un-scale, reproject and regrid SoilGrids layers.

    Parameters
    ----------
    config:
        The validated pipeline configuration.
    """

    def __init__(self, config: PipelineConfig) -> None:
        self._config = config
        self._grid = TargetGrid.from_config(config.grid)
        self._soil = config.soil
        self._root = config.paths.soilgrids_root
        self._cache_dir = self._resolve_cache_dir()
        self._homolosine: CRS | None = None
        # Populated by :meth:`load_dominant` when ``flags.write_soil_statistics``
        # is set; the pipeline writes it to the intermediate NetCDF/CSV files.
        self.class_statistics: "PrimaryClassStatistics | None" = None
        # Populated by :meth:`load_dominant` under ``soil.aggregation_method:
        # top3``: the per-class profiles, areas and heterogeneity metrics that
        # feed the intermediate rasters and the per-rank SIMPLACE CSVs.
        self.top_classes: "TopClassAggregation | None" = None

    def _resolve_cache_dir(self) -> Path:
        """Directory for cached WCS downloads (created lazily on first fetch)."""
        if self._soil.wcs_cache_dir is not None:
            return Path(self._soil.wcs_cache_dir)
        if self._root is not None:
            return Path(self._root) / "_wcs_cache"
        return Path(tempfile.gettempdir()) / "data4simplace_soilgrids"

    def _homolosine_crs(self) -> CRS:
        """Resolve (once) the SoilGrids Homolosine CRS for client-side use.

        Falls back to an explicit Interrupted Goode Homolosine proj4 definition
        when the configured code (e.g. ``EPSG:152160``) is unknown to PROJ. The
        result is cached so the failing lookup is not retried per coverage.
        """
        if self._homolosine is None:
            try:
                self._homolosine = CRS.from_user_input(self._soil.homolosine_crs)
            except Exception:  # noqa: BLE001 - any PROJ resolution failure -> proj4
                self._homolosine = CRS.from_proj4(_HOMOLOSINE_PROJ4)
        return self._homolosine

    # ------------------------------------------------------------------ #
    # Loading a single (layer, depth) coverage
    # ------------------------------------------------------------------ #
    def _local_path(self, layer: str, depth: str) -> Path | None:
        """Resolve a local GeoTIFF for ``layer``/``depth`` if a root is set."""
        if self._root is None:
            return None
        # Accept both ``clay_0-5cm_mean.tif`` and ``clay/clay_0-5cm.tif`` layouts.
        candidates = [
            Path(self._root) / f"{layer}_{depth}_mean.tif",
            Path(self._root) / f"{layer}_{depth}.tif",
            Path(self._root) / layer / f"{layer}_{depth}_mean.tif",
            Path(self._root) / layer / f"{layer}_{depth}.tif",
        ]
        for cand in candidates:
            if cand.is_file():
                return cand
        return None

    @staticmethod
    def _open_geotiff(path: Path) -> xr.DataArray:
        """Open a GeoTIFF as a single-band, chunked :class:`~xarray.DataArray`."""
        da = rioxarray.open_rasterio(path, masked=True, chunks={"x": 2048, "y": 2048})
        if isinstance(da, list):  # multi-file datasets are not expected here
            da = da[0]
        da = da.squeeze("band", drop=True) if "band" in da.dims else da
        return da  # type: ignore[return-value]

    def _open_coverage(self, layer: str, depth: str) -> xr.DataArray | None:
        """Open a coverage from a local tile, else fetch it from the WCS service."""
        path = self._local_path(layer, depth)
        if path is not None:
            return self._open_geotiff(path)

        if self._root is not None:
            logger.info("SoilGrids tile not found locally for %s %s", layer, depth)
        if self._soil.use_wcs:
            return self.fetch_wcs(layer, depth)

        logger.warning(
            "SoilGrids coverage unavailable for %s %s (no local tile; WCS disabled)",
            layer,
            depth,
        )
        return None

    # ------------------------------------------------------------------ #
    # WCS retrieval
    # ------------------------------------------------------------------ #
    def _wgs84_bbox(self) -> tuple[float, float, float, float]:
        """Target bounding box (+1-cell halo) as ``(min_lon, min_lat, max_lon, max_lat)``."""
        g = self._config.grid
        halo = g.resolution_deg
        return (g.min_lon - halo, g.min_lat - halo, g.max_lon + halo, g.max_lat + halo)

    def _homolosine_bbox(self) -> tuple[float, float, float, float]:
        """Target bounding box (+1-cell halo) projected to Homolosine X/Y."""
        g = self._config.grid
        halo = g.resolution_deg
        transformer = Transformer.from_crs(g.crs, self._homolosine_crs(), always_xy=True)
        lons = [g.min_lon - halo, g.max_lon + halo, g.max_lon + halo, g.min_lon - halo]
        lats = [g.min_lat - halo, g.min_lat - halo, g.max_lat + halo, g.max_lat + halo]
        xs, ys = transformer.transform(lons, lats)
        return min(xs), min(ys), max(xs), max(ys)

    def _wcs_subsets(self) -> tuple[list[tuple[str, str]], str]:
        """Bounding-box SUBSET params and CRS URI for a WCS GetCoverage request.

        For ``wcs_crs == EPSG:4326`` the coverage is requested directly in
        geographic coordinates (``Long``/``Lat`` subsets), so the response is
        already WGS84 and needs no client-side reprojection. Any other CRS is
        requested in the SoilGrids Homolosine grid (``X``/``Y`` subsets).
        """
        epsg = self._soil.wcs_crs.split(":")[-1]
        crs_uri = f"http://www.opengis.net/def/crs/EPSG/0/{epsg}"
        if epsg == "4326":
            min_lon, min_lat, max_lon, max_lat = self._wgs84_bbox()
            subsets = [
                ("SUBSET", f"Long({min_lon:.4f},{max_lon:.4f})"),
                ("SUBSET", f"Lat({min_lat:.4f},{max_lat:.4f})"),
            ]
        else:
            xmin, ymin, xmax, ymax = self._homolosine_bbox()
            subsets = [
                ("SUBSET", f"X({xmin:.1f},{xmax:.1f})"),
                ("SUBSET", f"Y({ymin:.1f},{ymax:.1f})"),
            ]
        return subsets, crs_uri

    def _wcs_get(
        self, map_layer: str, coverage_id: str, cache_file: Path
    ) -> xr.DataArray | None:
        """Fetch one coverage from the ISRIC SoilGrids WCS, with disk caching.

        Only the target bounding box is requested, so downloads stay small.
        Results are cached under :attr:`_cache_dir`; a cached tile is reused on
        later runs. Network or service errors are logged and yield ``None``
        rather than aborting the whole soil stage.
        """
        self._cache_dir.mkdir(parents=True, exist_ok=True)
        if cache_file.is_file() and cache_file.stat().st_size > 0:
            logger.debug("Using cached WCS tile %s", cache_file)
            return self._open_geotiff(cache_file)

        subsets, crs_uri = self._wcs_subsets()
        params = [
            ("SERVICE", "WCS"),
            ("VERSION", "2.0.1"),
            ("REQUEST", "GetCoverage"),
            ("COVERAGEID", coverage_id),
            ("FORMAT", "image/tiff"),
            *subsets,
            ("SUBSETTINGCRS", crs_uri),
            ("OUTPUTCRS", crs_uri),
        ]
        url = _WCS_BASE.format(layer=map_layer)
        try:
            with requests.get(
                url, params=params, stream=True, timeout=self._soil.wcs_timeout
            ) as resp:
                resp.raise_for_status()
                ctype = resp.headers.get("Content-Type", "")
                if "tif" not in ctype and "octet-stream" not in ctype:
                    # WCS reports errors as an XML ExceptionReport with HTTP 200.
                    logger.warning(
                        "WCS returned non-raster response for %s (%s): %s",
                        coverage_id, ctype, resp.text[:300],
                    )
                    return None
                tmp = cache_file.with_suffix(".tif.part")
                with tmp.open("wb") as handle:
                    for chunk in resp.iter_content(chunk_size=1 << 20):
                        handle.write(chunk)
                tmp.replace(cache_file)
        except requests.RequestException as exc:
            logger.warning("WCS fetch failed for %s: %s", coverage_id, exc)
            return None

        return self._open_geotiff(cache_file)

    def fetch_wcs(self, layer: str, depth: str) -> xr.DataArray | None:
        """Fetch a ``(layer, depth)`` soil property coverage from the WCS."""
        logger.info("Fetching %s %s from SoilGrids WCS", layer, depth)
        coverage = f"{layer}_{depth}_{self._soil.wcs_stat}"
        return self._wcs_get(layer, coverage, self._cache_dir / f"{coverage}.tif")

    def fetch_wrb(self) -> xr.DataArray | None:
        """Fetch the WRB Reference Soil Group coverage (``wrb`` map) from the WCS."""
        coverage = self._soil.wrb_layer
        logger.info("Fetching WRB %s from SoilGrids WCS", coverage)
        return self._wcs_get("wrb", coverage, self._cache_dir / f"wrb_{coverage}.tif")

    def _open_wrb(self) -> xr.DataArray | None:
        """Open the WRB MostProbable coverage from a local tile, else the WCS."""
        if self._root is not None:
            coverage = self._soil.wrb_layer
            for cand in (
                Path(self._root) / f"wrb_{coverage}.tif",
                Path(self._root) / "wrb" / f"{coverage}.tif",
            ):
                if cand.is_file():
                    return self._open_geotiff(cand)
        if self._soil.use_wcs:
            return self.fetch_wrb()
        logger.warning("WRB coverage unavailable (no local tile; WCS disabled)")
        return None

    # ------------------------------------------------------------------ #
    # Transformations
    # ------------------------------------------------------------------ #
    @staticmethod
    def unscale(da: xr.DataArray, layer: str) -> xr.DataArray:
        """Apply the official SoilGrids conversion factor for ``layer``."""
        factor = SCALE_FACTORS.get(layer)
        if factor is None:
            logger.warning("No scale factor known for layer %s; leaving unscaled", layer)
            return da
        return da / factor

    @staticmethod
    def mask_nodata(da: xr.DataArray, layer: str) -> xr.DataArray:
        """Mask the SoilGrids no-data sentinel for ``layer``.

        The GeoTIFFs (WCS subsets in particular) carry **no nodata tag** and
        write ``0`` over water, ice and other gaps, so ``masked=True`` on open
        does not catch them. Left in, such a pixel reads as a real measurement:
        a 0/0/0 texture pixel classifies as USDA *sand* and joins the majority
        vote, and a pH of 0 dominates the H+ mean (``10**-0 = 1``).

        Zero is masked for every layer where it is physically impossible; see
        :data:`ZERO_IS_NODATA`. ``cfvo`` is excluded because a soil genuinely
        free of coarse fragments is 0.
        """
        if layer not in ZERO_IS_NODATA:
            return da
        return da.where(da > 0)

    def reproject(self, da: xr.DataArray) -> xr.DataArray:
        """Reproject a coverage to the target CRS (WGS84), if needed.

        Coverages fetched directly in EPSG:4326 already match the target and are
        returned unchanged. A coverage lacking a CRS (e.g. a local Homolosine
        tile) is tagged with the resolved Homolosine definition first.
        """
        target = CRS.from_user_input(self._soil.target_crs)
        if da.rio.crs is not None and da.rio.crs.to_epsg() == target.to_epsg():
            return da
        if da.rio.crs is None:
            da = da.rio.write_crs(self._homolosine_crs())
        return da.rio.reproject(self._soil.target_crs)

    # ------------------------------------------------------------------ #
    # Public API
    # ------------------------------------------------------------------ #
    def load(self) -> xr.Dataset:
        """Load all configured layers as depth-resolved fields on the grid.

        Returns
        -------
        xarray.Dataset
            One variable per soil property with a ``depth`` dimension, all on
            the 10 km target grid in ``EPSG:4326``.
        """
        per_layer: dict[str, xr.DataArray] = {}

        for layer in self._soil.layers:
            depth_slices: list[xr.DataArray] = []
            loaded_depths: list[str] = []
            for depth in self._soil.depths:
                raw = self._open_coverage(layer, depth)
                if raw is None:
                    continue
                unscaled = self.mask_nodata(self.unscale(raw, layer), layer)
                wgs84 = self.reproject(unscaled)
                regridded = self._grid.regrid(wgs84, method="mean")
                depth_slices.append(regridded)
                loaded_depths.append(depth)

            if not depth_slices:
                logger.warning("No coverages loaded for layer %s", layer)
                continue

            stacked = xr.concat(depth_slices, dim="depth")
            stacked = stacked.assign_coords(depth=loaded_depths)
            per_layer[layer] = stacked
            logger.info("Loaded soil layer %s across %d depths", layer, len(loaded_depths))

        if not per_layer:
            raise RuntimeError(
                "No SoilGrids layers could be loaded. Provide local tiles via "
                "paths.soilgrids_root, or enable/allow the WCS fetch "
                "(soil.use_wcs) with network access to the ISRIC service."
            )

        return xr.Dataset(per_layer)

    # ------------------------------------------------------------------ #
    # Dominant soil-type workflow (250 m masking before aggregation)
    # ------------------------------------------------------------------ #
    def _load_fine(self) -> tuple[dict[str, dict[str, xr.DataArray]], xr.DataArray]:
        """Load every (layer, depth) coverage at native 250 m in EPSG:4326.

        Returns a ``{layer: {depth: DataArray}}`` mapping plus a reference grid
        (the first coverage loaded). All other coverages are ``reproject_match``-ed
        onto that reference so pixels are perfectly aligned for masking.

        The no-data sentinel is masked before reprojection (see
        :meth:`mask_nodata`), so gap pixels cannot vote on the soil class or
        enter an aggregate.
        """
        fine: dict[str, dict[str, xr.DataArray]] = {}
        reference: xr.DataArray | None = None

        for layer in self._soil.layers:
            fine[layer] = {}
            for depth in self._soil.depths:
                raw = self._open_coverage(layer, depth)
                if raw is None:
                    continue
                wgs84 = self.reproject(self.mask_nodata(self.unscale(raw, layer), layer))
                if reference is None:
                    reference = wgs84
                else:
                    wgs84 = wgs84.rio.reproject_match(reference)
                fine[layer][depth] = wgs84

        if reference is None:
            raise RuntimeError(
                "No SoilGrids layers could be loaded. Provide local tiles via "
                "paths.soilgrids_root, or enable/allow the WCS fetch "
                "(soil.use_wcs) with network access to the ISRIC service."
            )
        return fine, reference

    def load_dominant(self) -> tuple[xr.Dataset, xr.Dataset | None]:
        """Load soil via the dominant-soil-type 250 m workflow (CLAUDE.md).

        Cropland-filters 250 m pixels, selects the dominant soil class per 10 km
        cell (``usda`` texture class from the topsoil, or ``wrb`` Reference Soil
        Group), masks to that class, optionally computes pedotransfer functions
        **at 250 m**, then aggregates each variable with its rule (mean /
        geometric mean / pH H+ mean). Empty cells are optionally nearest-filled.

        With ``soil.aggregation_method: top3`` the same masking and aggregation
        runs once per primary class (Method B): the result is stored on
        :attr:`top_classes`, and rank 1 -- the dominant class -- is returned as
        ``soil``, so both methods export the identical primary profile.

        Returns
        -------
        (soil, hydraulic):
            ``soil`` is the depth-resolved soil dataset on the target grid.
            ``hydraulic`` holds PTF outputs when ``flags.compute_ptf`` is set and
            texture is available, else ``None``.
        """
        from data4simplace.grid import _standardise_lonlat_names
        from data4simplace.soil.classify import (
            bin_reduce,
            class_composition,
            dominant_class_per_cell,
            dominant_pixel_mask,
            fill_missing_cells,
            reducer_for,
        )
        from data4simplace.soil.ptf import saxton_rawls
        from data4simplace.spatial.area import pixel_area_km2
        from data4simplace.spatial.cropland_weights import CroplandWeights

        fine, reference = self._load_fine()

        # --- Cropland keep-mask at 250 m (None -> keep all pixels) ------------
        # Built while coverages still carry rioxarray x/y spatial dims.
        weights = CroplandWeights(self._config)
        cropland = weights.keep_mask(reference)

        # rioxarray stages are done; standardise to lat/lon for masking/aggregation.
        cropland = _standardise_lonlat_names(cropland) if cropland is not None else None
        fine = {
            layer: {d: _standardise_lonlat_names(da) for d, da in depths.items()}
            for layer, depths in fine.items()
        }

        # --- Dominant class per target cell (usda texture / wrb RSG) ----------
        class_fine = self._class_field(fine, reference)
        if cropland is not None:
            class_fine = class_fine.where(cropland, 0).astype("int16")

        multiclass = self._soil.aggregation_method == "top3"
        if multiclass:
            # Method B: rank the classes once, with latitude-aware pixel areas,
            # and derive the dominant mask from rank 1 rather than a second vote
            # (both break ties on the lower code, so they agree by construction).
            ranked, uncertainty = class_composition(
                class_fine,
                self._grid,
                self._soil.n_primary_classes,
                pixel_area=pixel_area_km2(class_fine),
            )
            dominant = ranked["class_code"].sel(rank=1, drop=True)
        else:
            ranked = uncertainty = None
            dominant = dominant_class_per_cell(class_fine, self._grid)
        keep = dominant_pixel_mask(class_fine, dominant)
        logger.info(
            "Dominant soil-type mask built (%s, method=%s); aggregating masked pixels",
            self._soil.dominant_mode,
            self._soil.aggregation_method,
        )

        # --- Optional: statistics for the n most frequent classes ------------
        # Computed before masking, since ranks 2..n need their own pixel sets.
        if self._config.flags.write_soil_statistics:
            self.class_statistics = self._primary_class_statistics(class_fine, fine)

        # --- Mask every fine coverage, aggregate per rule --------------------
        statistic = self._soil.export_statistic
        soil, masked_fine = self._aggregate_masked(fine, keep, statistic)

        # --- Method B: the same aggregation for ranks 2..n -------------------
        if multiclass:
            assert ranked is not None and uncertainty is not None  # set above
            self.top_classes = self._rank_aggregation(
                class_fine, fine, soil, ranked, uncertainty
            )

        # --- Optional PTFs: compute at 250 m, then aggregate the outputs -----
        hydraulic: xr.Dataset | None = None
        if self._config.flags.compute_ptf and {"sand", "clay"}.issubset(masked_fine):
            hydraulic = self._ptf_dominant(
                masked_fine, saxton_rawls, bin_reduce, reducer_for("theta", statistic)
            )

        # --- Nearest-neighbour gap fill for empty coastal/island cells -------
        # Bounded to the cropland cells: those are the cells that get exported,
        # and an unbounded search would carry land profiles out over the sea.
        if self._soil.fill_missing:
            cropland_cells = weights.cell_mask(self._grid)
            soil = fill_missing_cells(soil, within=cropland_cells)
            if hydraulic is not None:
                hydraulic = fill_missing_cells(hydraulic, within=cropland_cells)
            if self.top_classes is not None:
                # Fill the per-rank stack too, so soil_1.csv keeps matching
                # soil.csv cell for cell. The class composition itself is left
                # alone: a filled cell has no pixels of its own to attribute an
                # area or a class share to.
                self.top_classes.properties = fill_missing_cells(
                    self.top_classes.properties, within=cropland_cells
                )

        return soil, hydraulic

    def _aggregate_masked(
        self,
        fine: dict[str, dict[str, xr.DataArray]],
        keep: xr.DataArray,
        statistic: str,
    ) -> tuple[xr.Dataset, dict[str, dict[str, xr.DataArray]]]:
        """Aggregate every masked (layer, depth) coverage onto the target grid.

        Parameters
        ----------
        fine:
            ``{layer: {depth: DataArray}}`` of un-scaled 250 m coverages.
        keep:
            Boolean fine-grid mask (cropland pixels of one soil class).
        statistic:
            ``soil.export_statistic``, selecting the per-variable reducers.

        Returns
        -------
        (soil, masked_fine):
            The depth-stacked dataset on the target grid, and the masked 250 m
            coverages the PTF needs to run before aggregation.
        """
        from data4simplace.soil.classify import bin_reduce, reducer_for

        per_layer: dict[str, xr.DataArray] = {}
        masked_fine: dict[str, dict[str, xr.DataArray]] = {}
        for layer, depths in fine.items():
            reducer = reducer_for(layer, statistic)
            slices: list[xr.DataArray] = []
            loaded: list[str] = []
            masked_fine[layer] = {}
            for depth, da in depths.items():
                da_masked = da.where(keep)
                masked_fine[layer][depth] = da_masked
                slices.append(bin_reduce(da_masked, self._grid, reducer))
                loaded.append(depth)
            if not slices:
                continue
            per_layer[layer] = xr.concat(slices, dim="depth").assign_coords(depth=loaded)
        return xr.Dataset(per_layer), masked_fine

    def _rank_aggregation(
        self,
        class_fine: xr.DataArray,
        fine: dict[str, dict[str, xr.DataArray]],
        dominant_soil: xr.Dataset,
        ranked: xr.Dataset,
        uncertainty: xr.Dataset,
    ) -> "TopClassAggregation":
        """Aggregate ranks 2..n the way rank 1 was, and bundle the result.

        Each rank is masked to **its own** class before aggregation, so a rank-2
        profile is built from rank-2 pixels only -- never a blend. Rank 1 is the
        already-computed dominant dataset, which is what keeps Method A and
        Method B in exact agreement about the exported profile.

        Parameters
        ----------
        class_fine:
            The cropland-filtered 250 m class field.
        fine:
            ``{layer: {depth: DataArray}}`` of un-scaled 250 m coverages.
        dominant_soil:
            The rank-1 aggregate, reused rather than recomputed.
        ranked, uncertainty:
            The composition and heterogeneity datasets from
            :func:`~data4simplace.soil.classify.class_composition`.
        """
        from data4simplace.soil.classify import dominant_pixel_mask
        from data4simplace.soil.multiclass import TopClassAggregation

        ranks = np.asarray(ranked["rank"].values)
        statistic = self._soil.export_statistic
        logger.info("Aggregating %d additional soil class(es) per cell", ranks.size - 1)

        per_rank: list[xr.Dataset] = [dominant_soil]
        for rank in ranks[1:]:
            mask = dominant_pixel_mask(
                class_fine, ranked["class_code"].sel(rank=rank, drop=True)
            )
            aggregated, _ = self._aggregate_masked(fine, mask, statistic)
            per_rank.append(aggregated)

        properties = xr.concat(per_rank, dim="rank").assign_coords(rank=ranks)
        return TopClassAggregation(
            properties=properties,
            classes=ranked,
            uncertainty=uncertainty,
            mode=self._soil.dominant_mode,
        )

    def _primary_class_statistics(
        self,
        class_fine: xr.DataArray,
        fine: dict[str, dict[str, xr.DataArray]],
    ) -> "PrimaryClassStatistics":
        """Describe the ``soil.n_primary_classes`` most frequent classes per cell.

        For each rank the 250 m pixels of that class are summarised per property
        and depth (mean, median, std, excess kurtosis, pixel count), so the
        exported single profile can be put in context: rank 1 is what the CSV
        carries, ranks 2..n are the classes it leaves out.

        Parameters
        ----------
        class_fine:
            The 250 m class field, already cropland-filtered (``0`` elsewhere).
        fine:
            ``{layer: {depth: DataArray}}`` of un-scaled 250 m coverages.

        Returns
        -------
        PrimaryClassStatistics
            Ranked class codes/shares plus the per-class statistics.
        """
        from data4simplace.soil.classify import (
            bin_describe,
            dominant_pixel_mask,
            rank_classes_per_cell,
        )
        from data4simplace.soil.multiclass import PrimaryClassStatistics

        n_classes = self._soil.n_primary_classes
        ranked = rank_classes_per_cell(class_fine, self._grid, n_classes)
        logger.info(
            "Describing the %d most frequent %s class(es) per cell "
            "(mean/median/std/kurtosis)",
            n_classes,
            self._soil.dominant_mode,
        )

        # {variable: [per-rank DataArray]}, each per-rank array stacked over depth.
        collected: dict[str, list[xr.DataArray]] = {}
        for rank in ranked["rank"].values:
            mask = dominant_pixel_mask(class_fine, ranked["class_code"].sel(rank=rank))
            for layer, depths in fine.items():
                per_stat: dict[str, list[xr.DataArray]] = {}
                loaded: list[str] = []
                for depth, da in depths.items():
                    described = bin_describe(da.where(mask), self._grid)
                    for stat, values in described.data_vars.items():
                        per_stat.setdefault(stat, []).append(values)
                    loaded.append(depth)
                if not loaded:
                    continue
                for stat, slices in per_stat.items():
                    stacked = xr.concat(slices, dim="depth").assign_coords(depth=loaded)
                    collected.setdefault(f"{layer}_{stat}", []).append(stacked)

        stats = xr.Dataset(
            {
                name: xr.concat(per_rank, dim="rank").assign_coords(
                    rank=ranked["rank"].values
                )
                for name, per_rank in collected.items()
            }
        )
        return PrimaryClassStatistics(
            classes=ranked, stats=stats, mode=self._soil.dominant_mode
        )

    def _class_field(
        self, fine: dict[str, dict[str, xr.DataArray]], reference: xr.DataArray
    ) -> xr.DataArray:
        """Return the 250 m dominant-class field for the configured mode.

        ``usda`` derives the 12-class USDA texture class from the topsoil
        sand/silt/clay; ``usda_profile`` pairs that topsoil class with the class
        of the thickness-weighted rooting-zone texture (12 x 12 composite codes);
        ``wrb`` loads the SoilGrids WRB Reference Soil Group coverage. All are
        integer class codes on the standardised lat/lon grid.
        """
        from data4simplace.grid import _standardise_lonlat_names
        from data4simplace.soil.classify import (
            depth_bounds_cm,
            rootzone_mean,
            usda_profile_class,
            usda_texture_class,
        )

        mode = self._soil.dominant_mode
        if mode in ("usda", "usda_profile"):
            topsoil = self._soil.depths[0]
            if not {"sand", "silt", "clay"}.issubset(fine) or any(
                topsoil not in fine[t] for t in ("sand", "silt", "clay")
            ):
                raise RuntimeError(
                    f"Dominant-mode {mode!r} needs sand, silt and clay at the topsoil "
                    f"depth ({topsoil}); one or more are missing."
                )
            topsoil_class = usda_texture_class(
                fine["sand"][topsoil], fine["silt"][topsoil], fine["clay"][topsoil]
            )
            if mode == "usda":
                return topsoil_class

            # Rooting zone: from the bottom of the topsoil layer down to the
            # configured depth, thickness-weighted across the layers in between.
            top_cm = depth_bounds_cm(topsoil)[1]
            bottom_cm = self._soil.rootzone_bottom_cm
            try:
                rootzone_class = usda_texture_class(
                    *(
                        rootzone_mean(fine[texture], top_cm, bottom_cm)
                        for texture in ("sand", "silt", "clay")
                    )
                )
            except ValueError as exc:
                raise RuntimeError(
                    f"Dominant-mode 'usda_profile' needs sand/silt/clay depths inside "
                    f"the {top_cm:g}-{bottom_cm:g} cm rooting zone; configured depths "
                    f"are {self._soil.depths}."
                ) from exc
            return usda_profile_class(topsoil_class, rootzone_class)

        if mode == "wrb":
            raw = self._open_wrb()
            if raw is None:
                raise RuntimeError(
                    "Dominant-mode 'wrb' needs the WRB coverage, but it could not "
                    "be loaded (no local tile and WCS fetch failed/disabled)."
                )
            # Categorical layer: reproject (nearest) and align, no scale factor.
            wrb = self.reproject(raw).rio.reproject_match(reference)
            wrb = _standardise_lonlat_names(wrb)
            return wrb.fillna(0).astype("int16").rename("wrb_class")

        raise ValueError(
            f"Unsupported dominant_mode {mode!r}; use 'usda', 'usda_profile' or 'wrb'."
        )

    def _ptf_dominant(
        self,
        masked_fine: dict[str, dict[str, xr.DataArray]],
        saxton_rawls,
        bin_reduce,
        cell_mean,
    ) -> xr.Dataset:
        """Run Saxton-Rawls on masked 250 m pixels per depth, then aggregate.

        Non-linear PTFs must be evaluated before any spatial averaging, so each
        depth is computed at 250 m and only the outputs are binned to 10 km.
        """
        shared = sorted(
            set(masked_fine["sand"]) & set(masked_fine["clay"]),
            key=self._soil.depths.index,
        )
        soc_fine = masked_fine.get("soc", {})
        outputs: dict[str, list[xr.DataArray]] = {}
        for depth in shared:
            om = None
            if depth in soc_fine:
                om = soc_fine[depth] * 0.1 * 1.724  # g/kg SOC -> % organic matter
            ptf = saxton_rawls(masked_fine["sand"][depth], masked_fine["clay"][depth], om)
            for name, da in ptf.data_vars.items():
                outputs.setdefault(name, []).append(
                    bin_reduce(da, self._grid, cell_mean)
                )
        hydraulic = xr.Dataset(
            {name: xr.concat(slices, dim="depth").assign_coords(depth=shared)
             for name, slices in outputs.items()}
        )
        return hydraulic

    def load_processed(self) -> tuple[xr.Dataset, xr.Dataset | None]:
        """Load, aggregate and texture-normalise soil per the configured mode.

        Dispatches on ``soil.dominant_mode``: ``none`` uses the legacy plain-mean
        regrid (with PTFs computed after aggregation), otherwise the dominant
        soil-type 250 m workflow (PTFs computed at 250 m). Returns the ready-to-
        export ``(soil, hydraulic)`` pair; ``hydraulic`` is ``None`` unless
        ``flags.compute_ptf`` is set and texture is available.
        """
        if self._soil.dominant_mode == "none":
            soil = self.harmonise_water_retention(self.normalise_texture(self.load()))
            hydraulic: xr.Dataset | None = None
            if self._config.flags.compute_ptf and {"sand", "clay"}.issubset(soil.data_vars):
                from data4simplace.soil.ptf import saxton_rawls

                om = soil["soc"] * 0.1 * 1.724 if "soc" in soil.data_vars else None
                hydraulic = saxton_rawls(soil["sand"], soil["clay"], om)
            return soil, hydraulic

        soil, hydraulic = self.load_dominant()
        if self.top_classes is not None:
            # Every rank must get the same post-processing as the exported
            # profile, or a rank-2 CSV would carry un-normalised texture.
            self.top_classes.properties = self.harmonise_water_retention(
                self.normalise_texture(self.top_classes.properties)
            )
        return self.harmonise_water_retention(self.normalise_texture(soil)), hydraulic

    @staticmethod
    def harmonise_water_retention(ds: xr.Dataset) -> xr.Dataset:
        """Force the water-retention layers into their physical order.

        SoilGrids predicts ``wv0010``, ``wv0033`` and ``wv1500`` with separate
        models, so a cell can come out with, say, more water at 1500 kPa than at
        33 kPa — which would give SIMPLACE a negative plant-available water. The
        drier suction is clipped to the wetter one
        (``wv0010 >= wv0033 >= wv1500``), keeping the wettest layer authoritative.

        Applied per target cell **after** aggregation: the per-layer reducers are
        linear means, and clipping a mean is enough to guarantee the ordering the
        exporter relies on.
        """
        present = [layer for layer in WATER_RETENTION_LAYERS if layer in ds.data_vars]
        if len(present) < 2:
            return ds
        out = ds.copy()
        for wetter, drier in zip(present, present[1:]):
            # np.minimum propagates NaN, so a cell missing either layer stays
            # missing rather than silently inheriting the other one's value.
            out[drier] = np.minimum(out[drier], out[wetter]).astype("float32")
        return out

    @staticmethod
    def normalise_texture(ds: xr.Dataset) -> xr.Dataset:
        """Rescale clay/silt/sand so the three fractions sum to 100 %."""
        needed = {"clay", "silt", "sand"}
        if not needed.issubset(ds.data_vars):
            return ds
        total = ds["clay"] + ds["silt"] + ds["sand"]
        total = total.where(total > 0)
        out = ds.copy()
        for frac in needed:
            out[frac] = (ds[frac] / total * 100.0).astype("float32")
        return out
