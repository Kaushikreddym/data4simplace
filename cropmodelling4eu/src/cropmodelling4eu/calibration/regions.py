"""Calibration regions: the unit a parameter set is fitted on.

A region is not an administrative unit and not an environmental zone alone. It
is built in three layers, each of which earns its place:

===============  =============================================================
EnZ (13 zones)   agro-climate. **Not** EnS's 84 strata, which against the
                 observation density is hopelessly over-parameterised and which
                 PEP725 reaches maybe 15 of at all.
Crop calendar    splits a zone that straddles a season break — ATC spans
                 Ireland and SW France, and one thermal-time set cannot serve
                 both.
Yield level      proxies the management intensity LINTUL-5 does not represent.
                 A region mixing 3 t/ha and 9 t/ha districts fits neither.
===============  =============================================================

The calendar enters through a **circular encoding** (``sin``/``cos`` of ``sos``
and ``eos``), because the mean of DOY 300 and DOY 40 is not a date and a
Euclidean cluster on raw day-of-year would put the two ends of the same season
at opposite corners of the space.

Regions are the batching unit as well as the pooling unit, and that is forced
rather than chosen: ``torchcrop.calibration.paths.rebuild_table`` assembles an
``[N, 2]`` table from **scalar** ordinates, so a table parameter cannot be
``[B]``-shaped and broadcast per cell the way a scalar field can. Every table
in stages 1-3 — ``slatb``, ``fltb``, ``ruetb``, ``rdrltb``, ``dtsmtb``,
``vernrt`` — is calibrated, so one forward pass may carry exactly one region's
parameters. See :mod:`cropmodelling4eu.calibration.dataset`, which produces
region-homogeneous batches for that reason.
"""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Sequence

import numpy as np
import pandas as pd

from cropmodelling4eu.calibration.config import RegionConfig

logger = logging.getLogger(__name__)

__all__ = ["CalibrationRegions", "build_regions"]

_TWO_PI = 2.0 * np.pi

#: A snap beyond this is not a coastline any more. Reported, not enforced: a
#: cell has to land in some region to be run, and dropping it silently would be
#: worse than fitting it in a neighbouring zone and saying so.
_FAR_SNAP_KM: float = 50.0


@dataclass(frozen=True, slots=True)
class CalibrationRegions:
    """The cell -> region map and its provenance.

    Attributes:
        table: One row per cell — ``SimplaceID``, ``lon``, ``lat``, ``adm_id``,
            ``country``, ``enz``, ``enz_name``, ``region_id``, ``region``.
        summary: One row per region with its cell, unit and zone counts.
    """

    table: pd.DataFrame
    summary: pd.DataFrame

    @property
    def region_ids(self) -> np.ndarray:
        """Region identifiers, ascending."""
        return np.sort(self.summary["region_id"].to_numpy())

    def cells(self, region_id: int) -> np.ndarray:
        """``SimplaceID``s of one region."""
        return self.table.loc[
            self.table["region_id"] == region_id, "SimplaceID"
        ].to_numpy(np.int64)

    def units(self, region_id: int) -> np.ndarray:
        """Observation units (``adm_id``) a region covers."""
        sub = self.table[self.table["region_id"] == region_id]
        return sub["adm_id"].dropna().unique()

    def of_unit(self) -> pd.Series:
        """``adm_id -> region_id``, by the unit's most common cell region.

        A unit whose cells straddle two regions is assigned whole rather than
        split: the loss aggregates a unit's cells before comparing, so a split
        unit would contribute a partial mean to two different parameter sets.
        """
        counts = (
            self.table.dropna(subset=["adm_id"])
            .groupby(["adm_id", "region_id"], observed=True)
            .size()
            .rename("n")
            .reset_index()
            .sort_values(["adm_id", "n", "region_id"], ascending=[True, False, True])
        )
        return counts.drop_duplicates("adm_id").set_index("adm_id")["region_id"]

    def summarise(self) -> str:
        lines = [f"{len(self.summary)} calibration regions over {len(self.table)} cells"]
        for row in self.summary.itertuples():
            lines.append(
                f"  {row.region_id:>3}  {row.region:<12} "
                f"{row.n_cells:>6} cells  {row.n_units:>4} units"
            )
        return "\n".join(lines)


def _circular(doy: pd.Series) -> np.ndarray:
    """``[sin, cos]`` of a day-of-year, so DOY 365 and DOY 1 are neighbours."""
    angle = _TWO_PI * doy.to_numpy(float) / 365.0
    return np.column_stack([np.sin(angle), np.cos(angle)])


def _assign_enz(cells: pd.DataFrame, shapefile: Path) -> pd.DataFrame:
    """Label each cell centre with its EnZ zone (EPSG:3035 point-in-polygon)."""
    import geopandas as gpd

    zones = gpd.read_file(shapefile)
    points = gpd.GeoDataFrame(
        cells[["SimplaceID"]].copy(),
        geometry=gpd.points_from_xy(cells["lon"], cells["lat"]),
        crs="EPSG:4326",
    ).to_crs(zones.crs)

    joined = (
        gpd.sjoin(points, zones[["EnZ", "EnZ_name", "geometry"]], predicate="within")
        .drop(columns="index_right")
        .drop_duplicates("SimplaceID")
        .set_index("SimplaceID")
    )
    out = cells.copy()
    out["enz"] = out["SimplaceID"].map(joined["EnZ"])
    out["enz_name"] = out["SimplaceID"].map(joined["EnZ_name"])

    missing = out["enz"].isna()
    if missing.any():
        # A cell centre just off the EnZ coastline is ordinary; it takes the
        # zone of its nearest neighbour rather than being dropped, since the
        # export's own cropland mask already decided it is land worth running.
        #
        # The search is unbounded because every cell must land in some region to
        # be run at all — so the **distance is reported** instead. That is the
        # part worth reading: a 2 km snap is a coastline, a 200 km one is a cell
        # in a zone the layer does not cover, and it will be fitted against a
        # climate it does not have.
        near = gpd.sjoin_nearest(
            points[missing.to_numpy()],
            zones[["EnZ", "EnZ_name", "geometry"]],
            distance_col="_snap_m",
        ).drop_duplicates("SimplaceID").set_index("SimplaceID")
        out.loc[missing, "enz"] = out.loc[missing, "SimplaceID"].map(near["EnZ"])
        out.loc[missing, "enz_name"] = out.loc[missing, "SimplaceID"].map(near["EnZ_name"])

        distance_km = near["_snap_m"] / 1000.0
        logger.info(
            "%d of %d cells (%.0f%%) snapped to the nearest EnZ zone — "
            "distance median %.1f km, p95 %.1f km, max %.1f km",
            int(missing.sum()), len(out), 100.0 * missing.mean(),
            distance_km.median(), distance_km.quantile(0.95), distance_km.max(),
        )
        far = int((distance_km > _FAR_SNAP_KM).sum())
        if far:
            logger.warning(
                "%d cell(s) are more than %.0f km from any EnZ zone; they are "
                "still assigned to their nearest, so they will be fitted "
                "against an agro-climate the layer does not place them in",
                far, _FAR_SNAP_KM,
            )

    out["enz"] = out["enz"].astype("Int64")
    return out


def _silhouette_k(features: np.ndarray, max_k: int, seed: int) -> int:
    """Number of sub-clusters for one zone, by silhouette; 1 when none beats it."""
    from sklearn.cluster import KMeans
    from sklearn.metrics import silhouette_score

    n = features.shape[0]
    best_k, best_score = 1, -1.0
    for k in range(2, min(max_k, n - 1) + 1):
        labels = KMeans(n_clusters=k, n_init=10, random_state=seed).fit_predict(features)
        if len(np.unique(labels)) < 2:
            continue
        score = float(silhouette_score(features, labels))
        if score > best_score:
            best_k, best_score = k, score
    return best_k


def build_regions(
    cells: pd.DataFrame,
    calendar: pd.DataFrame,
    yields: pd.DataFrame,
    settings: RegionConfig | None = None,
    cache: Path | None = None,
) -> CalibrationRegions:
    """Cluster cells into calibration regions; see the module docstring.

    Args:
        cells: ``SimplaceID``, ``lon``, ``lat``, ``adm_id``, ``country`` — the
            export's runnable cells already joined to the CyBench units by
            :func:`cropmodelling4eu.evaluation.regions.assign_cells_to_regions`.
        calendar: CyBench ``crop_calendar`` — ``adm_id``, ``sos``, ``eos``.
        yields: CyBench yields — ``adm_id``, ``yield`` — averaged per unit here.
        settings: Clustering settings.
        cache: Parquet to read the cell table back from and write it to.

    Returns:
        The :class:`CalibrationRegions`.

    Raises:
        KeyError: If ``cells`` lacks a required column.
    """
    settings = settings or RegionConfig()
    required = {"SimplaceID", "lon", "lat", "adm_id"}
    missing = required - set(cells.columns)
    if missing:
        raise KeyError(f"cells lacks {sorted(missing)}")

    fingerprint = settings.model_dump(mode="json")
    if cache is not None and Path(cache).is_file():
        table = pd.read_parquet(cache)
        cached_settings = _read_fingerprint(cache)
        if set(table["SimplaceID"]) != set(cells["SimplaceID"]):
            logger.info("%s covers a different cell set; rebuilding", cache)
        elif cached_settings != fingerprint:
            # The cache keys on the settings as well as the cells: clustering
            # is what these settings *do*, so reusing a cache built under
            # different ones would silently ignore the change.
            logger.info("%s was built under different settings; rebuilding", cache)
        else:
            logger.info("calibration regions read from %s", cache)
            return CalibrationRegions(table, _summarise(table))

    table = _assign_enz(cells, settings.enz_shapefile)

    level = (
        yields.groupby("adm_id", observed=True)["yield"].mean().rename("mean_yield")
    )
    unit = calendar.set_index("adm_id")[["sos", "eos"]].join(level, how="outer")
    for col in ("sos", "eos"):
        table[col] = table["adm_id"].map(unit[col])
    table["mean_yield"] = table["adm_id"].map(unit["mean_yield"])

    # A cell whose unit publishes no calendar or no yield still has to land in
    # a region -- it takes its zone's median, so it clusters with the zone's
    # bulk instead of forming a hole.
    for col in ("sos", "eos", "mean_yield"):
        table[col] = table[col].fillna(
            table.groupby("enz", observed=True)[col].transform("median")
        ).fillna(table[col].median())

    table["region_id"] = -1
    table["region"] = pd.NA
    next_id = 1
    consumed: set[tuple[str, str]] = set()
    for enz, block in table.groupby("enz", observed=True, sort=True):
        name = str(block["enz_name"].iloc[0])
        features = np.column_stack(
            [
                _circular(block["sos"]),
                _circular(block["eos"]),
                _standardise(block["mean_yield"].to_numpy(float)),
            ]
        )
        k = (
            1
            if len(block) < 2 * settings.min_cells_per_region
            else _silhouette_k(features, settings.max_clusters_per_zone, settings.seed)
        )
        labels = _kmeans_labels(features, k, settings.seed)
        labels = _merge_small(labels, settings.min_cells_per_region)
        split = labels.max() + 1
        labels, origins, applied = _apply_merges(labels, name, settings.merge_regions)
        consumed |= applied

        for local in np.unique(labels):
            rows = block.index[labels == local]
            parts = origins[int(local)]
            table.loc[rows, "region_id"] = next_id
            table.loc[rows, "region"] = (
                name
                if len(origins) == 1 and len(parts) == 1
                else f"{name}-{'+'.join(str(i + 1) for i in parts)}"
            )
            next_id += 1
        if applied:
            logger.info(
                "EnZ %s: %d cells -> %d cluster(s), %d region(s) after %s",
                name, len(block), split, labels.max() + 1,
                ", ".join(f"{s}->{t}" for s, t in sorted(applied)),
            )
        else:
            logger.info(
                "EnZ %s: %d cells -> %d region(s)", name, len(block), labels.max() + 1
            )

    unconsumed = [p for p in settings.merge_regions if p not in consumed]
    if unconsumed:
        # A merge naming a zone the clustering never produced is a stale or
        # mistyped setting, and silently ignoring it would leave the run fitting
        # the regions the config says it merged.
        raise ValueError(
            f"merge_regions {unconsumed} named no EnZ zone in this domain; "
            f"zones present: {sorted(set(table['enz_name']))}"
        )

    table["region"] = table["region"].astype("string")
    summary = _summarise(table)
    lo, hi = settings.n_regions
    if not lo <= len(summary) <= hi:
        # Not fatal: the band is a design target, and a domain smaller than
        # Europe legitimately falls below it. Silence would not be defensible.
        logger.warning(
            "%d regions, outside the %d-%d target band — check "
            "regions.max_clusters_per_zone and min_cells_per_region",
            len(summary), lo, hi,
        )

    if cache is not None:
        Path(cache).parent.mkdir(parents=True, exist_ok=True)
        table.to_parquet(cache, index=False)
        _write_fingerprint(cache, fingerprint)
        logger.info("cached calibration regions to %s", cache)
    return CalibrationRegions(table, summary)


def _fingerprint_path(cache: Path) -> Path:
    return Path(cache).with_suffix(".settings.json")


def _read_fingerprint(cache: Path) -> dict | None:
    path = _fingerprint_path(cache)
    return json.loads(path.read_text()) if path.is_file() else None


def _write_fingerprint(cache: Path, fingerprint: dict) -> None:
    _fingerprint_path(cache).write_text(json.dumps(fingerprint, indent=2, default=str))


def _standardise(values: np.ndarray) -> np.ndarray:
    """Zero-mean, unit-sd column, so yield does not dominate the four angles."""
    sd = float(np.nanstd(values))
    centred = values - float(np.nanmean(values))
    return (centred / sd if sd > 0 else centred).reshape(-1, 1)


def _kmeans_labels(features: np.ndarray, k: int, seed: int) -> np.ndarray:
    if k <= 1:
        return np.zeros(features.shape[0], dtype=int)
    from sklearn.cluster import KMeans

    return KMeans(n_clusters=k, n_init=10, random_state=seed).fit_predict(features)


def _merge_small(labels: np.ndarray, minimum: int) -> np.ndarray:
    """Fold clusters below ``minimum`` cells into the zone's largest, renumbered."""
    counts = pd.Series(labels).value_counts()
    keep = counts[counts >= minimum].index.to_numpy()
    if keep.size == 0:
        return np.zeros_like(labels)
    largest = counts.idxmax()
    merged = np.where(np.isin(labels, keep), labels, largest)
    remap = {old: new for new, old in enumerate(np.unique(merged))}
    return np.vectorize(remap.get)(merged)


def _apply_merges(
    labels: np.ndarray, zone: str, merges: Sequence[tuple[str, str]]
) -> tuple[np.ndarray, dict[int, tuple[int, ...]], set[tuple[str, str]]]:
    """Fold named clusters of one zone together; see ``RegionConfig.merge_regions``.

    Returns the relabelled array, the original cluster indices each surviving
    label now carries, and the pairs this zone consumed. The origins are what
    keep the region *name* honest: a merged region is written ``PAN-1+3``, so
    renumbering never leaves an existing name pointing at a different region.
    """
    originals = [int(v) for v in np.unique(labels)]
    canonical = {local: local for local in originals}

    def resolve(local: int) -> int:
        while canonical[local] != local:
            local = canonical[local]
        return local

    applied: set[tuple[str, str]] = set()
    for source, target in merges:
        if source.rsplit("-", 1)[0] != zone:
            continue
        pair = []
        for entry in (source, target):
            index = int(entry.rsplit("-", 1)[1]) - 1
            if index not in canonical:
                raise ValueError(
                    f"merge_regions names {entry!r}, which EnZ {zone} did not "
                    f"produce (it has {[f'{zone}-{i + 1}' for i in originals]})"
                )
            pair.append(index)
        canonical[resolve(pair[0])] = resolve(pair[1])
        applied.add((source, target))

    merged = np.array([resolve(int(v)) for v in labels])
    remap = {old: new for new, old in enumerate(np.unique(merged))}
    origins: dict[int, tuple[int, ...]] = {}
    for local in originals:
        key = remap[resolve(local)]
        origins[key] = origins.get(key, ()) + (local,)
    return np.vectorize(remap.get)(merged), origins, applied


def _summarise(table: pd.DataFrame) -> pd.DataFrame:
    return (
        table.groupby(["region_id", "region"], observed=True)
        .agg(
            n_cells=("SimplaceID", "size"),
            n_units=("adm_id", "nunique"),
            enz=("enz_name", "first"),
            mean_yield=("mean_yield", "mean"),
        )
        .reset_index()
        .sort_values("region_id")
        .reset_index(drop=True)
    )
