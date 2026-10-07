"""PEP725 ground observations, as the independent check on the CLMS calendar.

The CLMS stage produces the phenology a crop model is scored against, and
nothing inside it can say whether ``CPMCE``/``CPMCH`` date the events they
claim to. PEP725 can: it is a human standing in a field recording a BBCH code,
so it is the only observation on this system that is not a satellite product.

The comparison is deliberately narrow. PEP725 is **94 % German** and covers
2.6-22.4 degrees E, 42.4-54.9 degrees N, so this module validates CLMS over
Central Europe and says nothing about the Mediterranean, Nordic or Atlantic
zones -- which are the zones CLMS was acquired for. That is not a flaw in the
design; it is the reason the check is worth running at all. Agreement in the
overlap is what licenses trusting CLMS outside it, and it is the only evidence
that can.

Two stage pairings survive scrutiny, and one does not:

===============  ==================  ===========================================
PEP725           CLMS                Reading
===============  ==================  ===========================================
BBCH 10          ``CPMCE``           Emergence. Both name the same event.
BBCH 100         ``CPMCH``           Harvest -- but ``CPMCH`` sits about a month
                                     earlier than every other harvest reference
                                     on disk, and this module is where that is
                                     measured rather than assumed.
BBCH 0           --                  Sowing. CLMS has no sowing layer; carried
                                     only to report the sowing-to-emergence lag.
===============  ==================  ===========================================

**Supports differ and that is the point of having two of them.** A PEP725 date
is one observer at one point; a CLMS date is the median over every wheat pixel
in a 10 km cell or an administrative unit. So a disagreement is a mixture of
product error and support mismatch, and the two are separated by comparing the
*same* PEP725 stations aggregated both ways -- which is why
:func:`match_to_grid` and :func:`match_to_admin` exist side by side rather than
one being derived from the other.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

__all__ = [
    "DAYS_IN_YEAR",
    "PHASES",
    "Phase",
    "circular_mean_doy",
    "doy_difference",
    "load_observations",
    "match_to_admin",
    "match_to_grid",
    "pair_admin",
    "pair_grid",
    "skill",
    "station_seasons",
]

DAYS_IN_YEAR: float = 365.25


#: A stage observed on or after this day-of-year belongs to the **next**
#: harvest year. It only ever applies to the autumn stages (see :class:`Phase`);
#: DOY 180 sits in the empty middle of their distribution -- sowing runs
#: DOY 260-299 and emergence DOY 272-316, with a handful of late-emerging
#: stragglers in January-March -- so nothing real lands near the cut.
AUTUMN_CUT_DOY: int = 180


@dataclass(frozen=True, slots=True)
class Phase:
    """One BBCH code and the CLMS column it is compared against.

    Attributes:
        code: PEP725 ``phase_id``.
        name: Column name this module writes the date under.
        clms_column: Matching column of the CLMS reducer output, or ``None``
            where CLMS publishes no counterpart.
        autumn: Whether the stage happens in the autumn *preceding* the harvest
            it belongs to. See :func:`station_seasons` -- this is the field that
            aligns PEP725's calendar year with CLMS's harvest year, and getting
            it wrong pairs two different seasons while still producing
            plausible-looking dates.
    """

    code: int
    name: str
    clms_column: str | None
    autumn: bool


#: The winter-wheat stages. Heading (BBCH 51) is deliberately absent: CLMS has
#: no counterpart, so it would be a column with nothing to pair against.
PHASES: tuple[Phase, ...] = (
    Phase(0, "sowing_doy", None, autumn=True),
    Phase(10, "emergence_doy", "emergence_doy_median", autumn=True),
    Phase(100, "harvest_doy", "harvest_doy_median", autumn=False),
)


def _files(root: Path) -> list[Path]:
    """Every PEP725 CSV export under ``root``, sorted."""
    paths = sorted(Path(root).glob("pep725_*.csv"))
    if not paths:
        raise FileNotFoundError(f"no pep725_*.csv under {root}")
    return paths


def load_observations(
    root: Path | str,
    years: tuple[int, int] = (2017, 2024),
    *,
    genus: str = "Triticum",
    winter_only: bool = True,
    extra_phases: tuple[int, ...] = (),
) -> pd.DataFrame:
    """PEP725 winter-wheat stage dates, one row per (station, year, phase).

    Args:
        root: Directory holding the ``pep725_*.csv`` exports.
        years: Inclusive ``(first, last)`` filter. The default is CLMS's own
            2017-2024 window, since a year outside it can never be paired.
        genus: Botanical genus to keep.
        extra_phases: BBCH codes to keep **in addition** to :data:`PHASES`.
            ``PHASES`` is the set with a CLMS counterpart, which is what this
            module pairs against; a caller that does not need a counterpart —
            a calibration scoring heading (BBCH 51) against the model's own
            anthesis date, say — asks for it here rather than being silently
            handed a frame without it. :func:`station_seasons` still pivots
            only the ``PHASES`` stages, so the extra codes are the caller's to
            handle.
        winter_only: Keep ``cult_season == 2`` only. **This is the strict
            reading and it is the right one here**: CLMS's ``wheat_winter`` is
            specifically the autumn-emerging mode, so admitting the
            ``cult_season == 0`` "unspecified" rows would compare a mixture
            against a split product and blame the difference on CLMS.

    Returns:
        ``s_id``, ``provider_id``, ``lon``, ``lat``, ``alt``, ``year``,
        ``phase_id``, ``doy``.

    Raises:
        FileNotFoundError: If no export is found under ``root``.
        ValueError: If the filters leave nothing.
    """
    frame = pd.concat(
        [pd.read_csv(path, sep=";") for path in _files(root)], ignore_index=True
    )
    n_all = len(frame)
    codes = [phase.code for phase in PHASES] + [int(c) for c in extra_phases]
    keep = (
        (frame["genus"] == genus)
        & frame["year"].between(*years)
        & frame["phase_id"].isin(codes)
    )
    if winter_only:
        keep &= frame["cult_season"] == 2
    frame = frame[keep]
    if frame.empty:
        raise ValueError(
            f"no {genus} rows in {years[0]}-{years[1]} for phases {codes}"
        )

    out = (
        frame[["s_id", "provider_id", "lon", "lat", "alt", "year", "phase_id", "day"]]
        .rename(columns={"day": "doy"})
        .astype({"year": "int64", "phase_id": "int64", "doy": "float64"})
        .reset_index(drop=True)
    )
    logger.info(
        "PEP725: %d of %d rows kept -- %d stations, %d providers, %d-%d",
        len(out), n_all, out["s_id"].nunique(), out["provider_id"].nunique(),
        out["year"].min(), out["year"].max(),
    )
    return out


def station_seasons(observations: pd.DataFrame) -> pd.DataFrame:
    """Pivot to one row per (station, **harvest year**) with a column per stage.

    **PEP725 dates by calendar year; CLMS dates by harvest year.** A PEP725 row
    for 2017 at one German station carries emergence on 9 October 2017 and
    harvest on 9 August 2017 -- two different seasons. Pivoting on the raw
    ``year`` therefore pairs the emergence of one season with the harvest of the
    previous one, and the trap is that it still produces a plausible season
    length (~304 d) and a plausible-looking near-zero emergence bias, while
    destroying the interannual correlation.

    So each stage is moved onto the harvest year of the season it belongs to,
    using :attr:`Phase.autumn` and :data:`AUTUMN_CUT_DOY`: an autumn stage
    observed in year Y belongs to harvest year Y+1, and harvest never moves.
    This is the same season the CLMS reducer means -- its
    :func:`~data4simplace.phenology.decode.anchor_for` runs a season from
    1 August of Y-1 to the end of Y.

    ``season_length_days`` is emergence to harvest, matching what CLMS
    measures, and is computed from the **real dates** rather than from a
    day-of-year wrap, so leap years are exact.

    Args:
        observations: :func:`load_observations` output.

    Returns:
        ``s_id``, ``lon``, ``lat``, ``harvest_year``, one column per
        :data:`PHASES` name, and ``season_length_days`` where both endpoints
        are observed.
    """
    autumn = {phase.code: phase.autumn for phase in PHASES}
    names = {phase.code: phase.name for phase in PHASES}

    obs = observations.copy()
    obs["stage"] = obs["phase_id"].map(names)
    is_autumn = obs["phase_id"].map(autumn) & (obs["doy"] >= AUTUMN_CUT_DOY)
    obs["harvest_year"] = obs["year"] + is_autumn.astype(int)
    obs["date"] = pd.to_datetime(obs["year"], format="%Y") + pd.to_timedelta(
        obs["doy"] - 1, unit="D"
    )
    moved = int(is_autumn.sum())
    logger.info(
        "season alignment: %d of %d rows moved to the following harvest year "
        "(%d autumn-stage rows stayed, e.g. a January emergence)",
        moved, len(obs),
        int((obs["phase_id"].map(autumn) & ~is_autumn).sum()),
    )

    # A station reporting a stage twice in one season is a data-entry accident;
    # the median of two is one of them and of three the middle one, which is
    # what a single observer would have written.
    wide = obs.pivot_table(
        index=["s_id", "harvest_year"], columns="stage", values="doy", aggfunc="median"
    ).reset_index()
    dates = obs.pivot_table(
        index=["s_id", "harvest_year"], columns="stage", values="date", aggfunc="min"
    )
    wide = wide.merge(
        observations[["s_id", "lon", "lat"]].drop_duplicates("s_id"),
        on="s_id", how="left",
    )
    for phase in PHASES:
        if phase.name not in wide.columns:
            wide[phase.name] = np.nan

    if {"emergence_doy", "harvest_doy"} <= set(dates.columns):
        span = (dates["harvest_doy"] - dates["emergence_doy"]).dt.days
        wide = wide.merge(
            span.rename("season_length_days").reset_index(),
            on=["s_id", "harvest_year"], how="left",
        )
        # A negative or absurd span means the two endpoints were filed to the
        # same harvest year by mistake -- drop the duration, keep the dates.
        wide.loc[~wide["season_length_days"].between(150, 400), "season_length_days"] = (
            np.nan
        )
    else:
        wide["season_length_days"] = np.nan

    logger.info(
        "%d station-seasons across %d stations; emergence on %d, harvest on %d, "
        "season length on %d", len(wide), wide["s_id"].nunique(),
        int(wide["emergence_doy"].notna().sum()),
        int(wide["harvest_doy"].notna().sum()),
        int(wide["season_length_days"].notna().sum()),
    )
    return wide


def match_to_grid(
    seasons: pd.DataFrame, grid, *, max_km: float | None = None
) -> pd.DataFrame:
    """Assign each station to the 0.1 degree cell that **contains** it.

    Containment, not nearest-neighbour, is the honest default: a CLMS grid row
    is the median over the wheat pixels of one cell, so the only station that
    observes the same ground is one standing inside it. ``max_km`` widens this
    to a nearest-centre match for a sensitivity check -- at 0.1 degrees the
    half-diagonal is about 7.9 km at the equator and less further north, so a
    radius above that is deliberately comparing a station to its neighbours.

    Args:
        seasons: :func:`station_seasons` output.
        grid: A :class:`~data4simplace.grid.TargetGrid`.
        max_km: When given, match to the nearest cell centre within this
            distance instead of by containment.

    Returns:
        ``seasons`` plus ``SimplaceID`` and ``distance_km`` (to the matched
        cell's centre), with unmatched stations dropped.
    """
    lon = seasons["lon"].to_numpy()
    lat = seasons["lat"].to_numpy()
    res = grid.resolution_deg
    n_lon = grid.lon_centers.size
    n_lat = grid.lat_centers.size

    col = np.floor((lon - grid.min_lon) / res).astype(np.int64)
    row = np.floor((grid.max_lat - lat) / res).astype(np.int64)
    inside = (col >= 0) & (col < n_lon) & (row >= 0) & (row < n_lat)

    out = seasons.copy()
    out["SimplaceID"] = np.where(inside, row * n_lon + col + 1, -1).astype("int64")
    centre_lon = np.where(inside, grid.lon_centers[np.clip(col, 0, n_lon - 1)], np.nan)
    centre_lat = np.where(inside, grid.lat_centers[np.clip(row, 0, n_lat - 1)], np.nan)
    out["distance_km"] = _haversine_km(lon, lat, centre_lon, centre_lat)

    if max_km is not None:
        out = out[inside & (out["distance_km"] <= max_km)]
        logger.info(
            "grid match within %.1f km: %d of %d station-seasons, %d cells",
            max_km, len(out), len(seasons), out["SimplaceID"].nunique(),
        )
    else:
        out = out[inside]
        logger.info(
            "grid match by containment: %d of %d station-seasons, %d cells "
            "(station-to-centre distance median %.1f km, max %.1f)",
            len(out), len(seasons), out["SimplaceID"].nunique(),
            out["distance_km"].median(), out["distance_km"].max(),
        )
    return out.reset_index(drop=True)


def match_to_admin(seasons: pd.DataFrame, admin) -> pd.DataFrame:
    """Assign each station to the CyBench administrative unit containing it.

    Args:
        seasons: :func:`station_seasons` output.
        admin: :func:`~data4simplace.phenology.zones.load_admin_zones` output.

    Returns:
        ``seasons`` plus ``country`` and ``adm_id``, with stations outside
        every polygon dropped.
    """
    import geopandas as gpd

    points = gpd.GeoDataFrame(
        seasons,
        geometry=gpd.points_from_xy(seasons["lon"], seasons["lat"]),
        crs="EPSG:4326",
    )
    joined = gpd.sjoin(
        points, admin[["country", "adm_id", "geometry"]], how="left", predicate="within"
    )
    # A station on a shared boundary lands in both polygons and sjoin emits it
    # twice; keeping the first is arbitrary but the alternative -- fanning the
    # station across two units -- double-counts it in every metric below.
    joined = joined[~joined.index.duplicated(keep="first")]
    out = pd.DataFrame(joined.drop(columns=["geometry", "index_right"]))
    matched = out[out["adm_id"].notna()].reset_index(drop=True)
    logger.info(
        "admin match: %d of %d station-seasons inside a CyBench unit, %d units",
        len(matched), len(seasons), matched["adm_id"].nunique(),
    )
    return matched


def _clms_columns(clms: pd.DataFrame) -> dict[str, str]:
    """CLMS median column -> the PEP725 stage name it pairs with."""
    pairs = {
        phase.clms_column: phase.name
        for phase in PHASES
        if phase.clms_column and phase.clms_column in clms.columns
    }
    if "season_length_days_median" in clms.columns:
        pairs["season_length_days_median"] = "season_length_days"
    return pairs


def _pair(
    matched: pd.DataFrame, clms: pd.DataFrame, keys: list[str]
) -> pd.DataFrame:
    """Aggregate stations onto ``keys`` and join the CLMS row for the same key.

    The station side is reduced with a **circular** mean for the dates and a
    plain mean for the duration, because two stations in one cell can straddle
    New Year on emergence and their arithmetic mean would land in July.
    """
    pairs = _clms_columns(clms)
    stage_names = list(pairs.values())

    aggregated = []
    for name in stage_names:
        if name not in matched.columns:
            continue
        block = matched[[*keys, name]].dropna()
        if block.empty:
            continue
        if name == "season_length_days":
            reduced = block.groupby(keys)[name].mean()
        else:
            reduced = block.groupby(keys)[name].apply(
                lambda s: circular_mean_doy(s.to_numpy())
            )
        aggregated.append(reduced.rename(f"pep_{name}"))
    if not aggregated:
        raise ValueError("no PEP725 stage survived the aggregation")

    obs = pd.concat(aggregated, axis=1).reset_index()
    counts = matched.groupby(keys).size().rename("n_stations").reset_index()
    obs = obs.merge(counts, on=keys)

    clms_side = clms.rename(columns={c: f"clms_{n}" for c, n in pairs.items()})
    # CLMS's `year` *is* the harvest year (decode.anchor_for), so the rename is
    # the whole alignment -- station_seasons has already moved PEP725 onto it.
    if "harvest_year" in keys and "harvest_year" not in clms_side.columns:
        clms_side = clms_side.rename(columns={"year": "harvest_year"})
    wanted = [*keys, *(f"clms_{n}" for n in pairs.values())]
    wanted += [c for c in ("n_pixels", "area_km2", "season_share") if c in clms_side]
    paired = obs.merge(clms_side[wanted], on=keys, how="inner")

    for name in pairs.values():
        pep_col, clms_col = f"pep_{name}", f"clms_{name}"
        if pep_col in paired and clms_col in paired:
            if name == "season_length_days":
                paired[f"residual_{name}"] = paired[clms_col] - paired[pep_col]
            else:
                paired[f"residual_{name}"] = doy_difference(
                    paired[clms_col].to_numpy(), paired[pep_col].to_numpy()
                )
    logger.info(
        "%d paired rows on %s (%d PEP725 rows, %d CLMS rows)",
        len(paired), keys, len(obs), len(clms),
    )
    return paired


def pair_grid(matched: pd.DataFrame, clms_grid: pd.DataFrame) -> pd.DataFrame:
    """Pair station observations against the CLMS 10 km rows, per (cell, year).

    Args:
        matched: :func:`match_to_grid` output.
        clms_grid: ``phenology_grid.parquet``, already filtered to one crop and
            through its quality gates.

    Returns:
        One row per ``(SimplaceID, year)`` on both sides, with ``pep_*``,
        ``clms_*`` and ``residual_*`` (CLMS minus PEP725) columns.
    """
    return _pair(matched, clms_grid, ["SimplaceID", "harvest_year"])


def pair_admin(matched: pd.DataFrame, clms_adm: pd.DataFrame) -> pd.DataFrame:
    """Pair station observations against the CLMS administrative rows.

    Args:
        matched: :func:`match_to_admin` output.
        clms_adm: ``phenology_adm.parquet``, filtered as above.

    Returns:
        One row per ``(adm_id, year)`` on both sides.
    """
    return _pair(matched, clms_adm, ["adm_id", "harvest_year"])


def circular_mean_doy(values: np.ndarray) -> float:
    """Mean day-of-year through the unit circle.

    A winter crop emerging on DOY 360 one season and DOY 5 the next has a
    linear mean of 182 -- the opposite side of the year from either.

    Args:
        values: Days of year. NaNs are dropped.

    Returns:
        Mean in ``[0, DAYS_IN_YEAR)``, or NaN if nothing is finite.
    """
    values = np.asarray(values, dtype=float)
    values = values[np.isfinite(values)]
    if values.size == 0:
        return float("nan")
    angle = 2.0 * np.pi * values / DAYS_IN_YEAR
    mean = np.arctan2(np.sin(angle).mean(), np.cos(angle).mean())
    return float((mean % (2.0 * np.pi)) * DAYS_IN_YEAR / (2.0 * np.pi))


def doy_difference(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """Signed shortest-path difference ``a - b`` in days, within +/- half a year.

    Args:
        a: Days of year.
        b: Days of year.

    Returns:
        Differences in ``(-DAYS_IN_YEAR/2, DAYS_IN_YEAR/2]``.
    """
    diff = (np.asarray(a, dtype=float) - np.asarray(b, dtype=float)) % DAYS_IN_YEAR
    return np.where(diff > DAYS_IN_YEAR / 2.0, diff - DAYS_IN_YEAR, diff)


def skill(
    paired: pd.DataFrame, stage: str, *, group_col: str = "SimplaceID"
) -> dict[str, float]:
    """Agreement between the two products for one stage.

    Reports the level *and* the two correlations separately, because they
    answer different questions: ``spatial_r`` asks whether the two products
    order the landscape the same way, ``anomaly_r`` whether they move together
    from season to season. A product can reproduce the continental gradient
    perfectly and carry no interannual information at all, and pooling the two
    hides exactly that.

    Args:
        paired: :func:`pair_grid` or :func:`pair_admin` output.
        stage: A stage name from :data:`PHASES`, or ``season_length_days``.
        group_col: The unit key anomalies are taken within.

    Returns:
        ``n``, ``units``, ``bias``, ``mae``, ``rmse``, ``pearson_r`` (pooled
        over raw pairs), ``spatial_r``, ``anomaly_r``, ``pep_anomaly_sd`` and
        ``clms_anomaly_sd``. The bias is ``CLMS - PEP725``.
    """
    pep_col, clms_col = f"pep_{stage}", f"clms_{stage}"
    frame = paired[[group_col, pep_col, clms_col]].dropna()
    if frame.empty:
        return {"n": 0.0, "units": 0.0}

    circular = stage != "season_length_days"
    residual = (
        doy_difference(frame[clms_col].to_numpy(), frame[pep_col].to_numpy())
        if circular
        else (frame[clms_col] - frame[pep_col]).to_numpy()
    )

    if circular:
        pep_mean = frame.groupby(group_col)[pep_col].apply(
            lambda s: circular_mean_doy(s.to_numpy())
        )
        clms_mean = frame.groupby(group_col)[clms_col].apply(
            lambda s: circular_mean_doy(s.to_numpy())
        )
        key = frame[group_col]
        pep_anom = doy_difference(
            frame[pep_col].to_numpy(), key.map(pep_mean).to_numpy()
        )
        clms_anom = doy_difference(
            frame[clms_col].to_numpy(), key.map(clms_mean).to_numpy()
        )
        # Correlation is linear, so the unit means need a linear
        # representation -- unwrapped about the PEP725 circular centre.
        centre = circular_mean_doy(pep_mean.to_numpy())
        spatial_pep = centre + doy_difference(pep_mean.to_numpy(), centre)
        spatial_clms = centre + doy_difference(clms_mean.to_numpy(), centre)
    else:
        pep_mean = frame.groupby(group_col)[pep_col].mean()
        clms_mean = frame.groupby(group_col)[clms_col].mean()
        key = frame[group_col]
        pep_anom = frame[pep_col].to_numpy() - key.map(pep_mean).to_numpy()
        clms_anom = frame[clms_col].to_numpy() - key.map(clms_mean).to_numpy()
        spatial_pep, spatial_clms = pep_mean.to_numpy(), clms_mean.to_numpy()

    # Pooled r over the raw (unit, season) pairs -- the statistic the package
    # CLAUDE.md's validation table reports, kept so the two are comparable.
    # It mixes the spatial and interannual terms below, which is exactly why
    # both of those are reported beside it rather than instead of it.
    pooled = pd.Series(
        centre + doy_difference(frame[pep_col].to_numpy(), centre)
        if circular else frame[pep_col].to_numpy()
    ).corr(pd.Series(
        centre + doy_difference(frame[clms_col].to_numpy(), centre)
        if circular else frame[clms_col].to_numpy()
    ))

    return {
        "n": float(len(frame)),
        "units": float(len(pep_mean)),
        "bias": float(np.mean(residual)),
        "mae": float(np.mean(np.abs(residual))),
        "rmse": float(np.sqrt(np.mean(residual**2))),
        "pearson_r": float(pooled),
        "spatial_r": float(pd.Series(spatial_pep).corr(pd.Series(spatial_clms))),
        "anomaly_r": float(pd.Series(pep_anom).corr(pd.Series(clms_anom))),
        "pep_anomaly_sd": float(pd.Series(pep_anom).std()),
        "clms_anomaly_sd": float(pd.Series(clms_anom).std()),
    }


def _haversine_km(
    lon1: np.ndarray, lat1: np.ndarray, lon2: np.ndarray, lat2: np.ndarray
) -> np.ndarray:
    """Great-circle distance in kilometres."""
    r = 6371.0088
    p1, p2 = np.radians(lat1), np.radians(lat2)
    dp = p2 - p1
    dl = np.radians(np.asarray(lon2, dtype=float) - np.asarray(lon1, dtype=float))
    h = np.sin(dp / 2.0) ** 2 + np.cos(p1) * np.cos(p2) * np.sin(dl / 2.0) ** 2
    return 2.0 * r * np.arcsin(np.sqrt(np.clip(h, 0.0, 1.0)))
