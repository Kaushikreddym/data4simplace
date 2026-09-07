"""CLMS HRL Croplands phenology as the observed crop calendar.

``data4simplace``'s phenology stage reduces the 10 m ``CPMCE`` (emergence) and
``CPMCH`` (harvest) rasters over the CyBench reporting units and the 10 km grid
and writes ``phenology_adm.parquet`` / ``phenology_grid.parquet`` — one row per
(zone, crop, year) with the pixel distribution's percentiles.

Why this replaces CyBench's ``crop_calendar`` for phenology:

============================  =====================  ======================
                              CyBench crop_calendar  CLMS HRL Croplands
============================  =====================  ======================
Quantity                      ``sos`` / ``eos``      emergence / harvest
Year dimension                **none** — static      2017-2024, per year
Winter/spring separation      none                   from the emergence year
Support                       reported unit          the wheat pixels in it
============================  =====================  ======================

The year dimension is the whole point. Against a static reference every unit's
interannual scatter is scored as error, so a model can only be penalised for
varying; against CLMS the anomaly correlation is a real measurement, which is
what :func:`split_skill` computes.

Two properties of the product decide how it must be read, both established in
``src/data4simplace/CLMS_DOWNLOAD.md``:

* ``CTY`` class 1110 is "wheat" with **no winter/spring split**. The reducer
  recovers it from the year encoded in ``CPMCE``'s ``YYDOY`` — autumn emergence
  is winter wheat — and ``split_validated`` records whether the two modes
  separated cleanly for that zone and year. Rows that failed are dropped here.
* ``CPMCH`` sits roughly a month earlier than every other harvest reference on
  disk. Whether it dates senescence rather than the combine is unresolved, so
  :func:`calendar_offset` measures the disagreement against CyBench ``eos``
  explicitly instead of leaving it inside a model bias.
"""

from __future__ import annotations

import logging
from pathlib import Path

import pandas as pd

from . import config, doy

__all__ = [
    "OBS_COLUMNS",
    "calendar_offset",
    "load_phenology",
    "split_skill",
]

logger = logging.getLogger(__name__)

#: Reducer column -> the name the evaluation uses. The **median** of the pixel
#: distribution is the point estimate, not the mean: the distribution inside a
#: unit is bounded by the product's own 6-day compositing step and is skewed by
#: the late tail of a re-sown field, so a median is what a single pass over the
#: unit's pixels would report.
OBS_COLUMNS: dict[str, str] = {
    "emergence_doy_median": "obs_emergence_doy",
    "harvest_doy_median": "obs_harvest_doy",
    "season_length_days_median": "obs_season_length_days",
}

#: Carried through so a pairing can show the within-unit spread beside the
#: point estimate — a unit whose wheat emerges across six weeks is a different
#: observation from one that emerges in four days.
_SPREAD_COLUMNS: tuple[str, ...] = (
    "emergence_doy_p10", "emergence_doy_p90",
    "harvest_doy_p10", "harvest_doy_p90",
    "season_length_days_p10", "season_length_days_p90",
)

_ZONE_KEY: dict[str, str] = {"adm": "adm_id", "grid": "SimplaceID"}


def load_phenology(
    countries: list[str] | None = None,
    *,
    zone: str = "adm",
    crop: str = config.CLMS_CROP,
    years: tuple[int, int] | None = None,
    min_pixels: int = config.CLMS_MIN_PIXELS,
    min_season_share: float = config.CLMS_MIN_SEASON_SHARE,
    require_split: bool = True,
    path: Path | str | None = None,
) -> pd.DataFrame:
    """The observed crop calendar, one row per (unit, year).

    Args:
        countries: Restrict to these country codes. ``None`` keeps all.
            Ignored for ``zone="grid"``, which carries no country column.
        zone: ``"adm"`` for the CyBench reporting units, ``"grid"`` for the
            10 km cells.
        crop: ``CTY`` label after the winter/spring split.
        years: Inclusive ``(first, last)`` filter on the **harvest** year,
            which is how the reducer labels a season and how both runs label
            theirs.
        min_pixels: Drop a row backed by fewer 10 m wheat pixels than this.
        min_season_share: Drop a row where this crop is a smaller share of the
            zone's wheat than this.
        require_split: Drop rows whose winter/spring separation the reducer
            could not validate.
        path: Override the parquet location.

    Returns:
        ``country`` (adm only), the zone key, ``year``, the three
        :data:`OBS_COLUMNS`, the p10/p90 spread, ``n_pixels``, ``area_km2``
        and ``season_share``.

    Raises:
        FileNotFoundError: If the parquet is absent.
        ValueError: If ``zone`` is unknown, or the gates leave nothing.
    """
    if zone not in _ZONE_KEY:
        raise ValueError(f"zone must be one of {sorted(_ZONE_KEY)}, got {zone!r}")
    key = _ZONE_KEY[zone]

    path = Path(path or config.CLMS_PHENOLOGY_DIR / config.CLMS_PHENOLOGY_FILES[zone])
    if not path.is_file():
        raise FileNotFoundError(
            f"no CLMS phenology at {path}. It is written by data4simplace's "
            "validation stage (submit/validation_reduce.sh); see "
            "src/data4simplace/CLMS_DOWNLOAD.md"
        )

    frame = pd.read_parquet(path)
    n_all = len(frame)
    frame = frame[frame["crop"] == crop]
    if frame.empty:
        raise ValueError(f"no {crop!r} rows in {path}")

    # Every gate is counted, not applied silently: the survivor count is the
    # honest denominator for everything scored downstream.
    gates: dict[str, pd.Series] = {}
    if require_split:
        gates["split not validated"] = ~frame["split_validated"].astype(bool)
    gates[f"< {min_pixels:,} pixels"] = frame["n_pixels"] < min_pixels
    gates[f"season share < {min_season_share}"] = frame["season_share"] < min_season_share
    if countries is not None and "country" in frame.columns:
        gates["outside the country set"] = ~frame["country"].isin(countries)
    if years is not None:
        lo, hi = years
        gates[f"outside {lo}-{hi}"] = ~frame["year"].between(lo, hi)

    dropped = pd.Series(False, index=frame.index)
    for label, mask in gates.items():
        newly = mask & ~dropped
        logger.info("CLMS %s: %d rows dropped (%s)", crop, int(newly.sum()), label)
        dropped |= mask
    frame = frame[~dropped]
    if frame.empty:
        raise ValueError(f"every {crop!r} row in {path} failed the quality gates")

    cols = [c for c in ("country", key, "year") if c in frame.columns]
    out = frame[[*cols, *OBS_COLUMNS, *_SPREAD_COLUMNS,
                 "n_pixels", "area_km2", "season_share"]].rename(columns=OBS_COLUMNS)
    out["year"] = out["year"].astype("int64")

    duplicated = int(out.duplicated(cols).sum())
    if duplicated:
        raise ValueError(
            f"{duplicated} duplicated {cols} rows in {path} — a join on them "
            "would fan out and every metric under it would be wrong"
        )

    logger.info(
        "CLMS %s (%s): %d of %d rows kept — %d zones x %d years",
        crop, zone, len(out), n_all, out[key].nunique(), out["year"].nunique(),
    )
    return out.reset_index(drop=True)


def calendar_offset(
    observed: pd.DataFrame,
    calendar: pd.DataFrame,
    *,
    obs_col: str = "obs_harvest_doy",
    ref_col: str = "eos",
) -> pd.DataFrame:
    """How far CLMS sits from CyBench's static calendar, per unit.

    Run this **before** reading any model bias against CLMS. The two references
    describe the same event and disagree on its level; that disagreement is a
    property of the products, and leaving it inside a model's bias would make a
    reference change look like a model finding.

    Args:
        observed: :func:`load_phenology` output at ``zone="adm"``.
        calendar: :func:`~cropmodelling4eu.evaluation.cybench.load_calendar`
            output, carrying ``adm_id`` and ``ref_col``.
        obs_col: The CLMS column.
        ref_col: The CyBench column describing the same event.

    Returns:
        One row per unit on both references, with both values and their
        difference (``clms - cybench``).
    """
    clms = observed.groupby("adm_id")[obs_col].mean()
    ref = calendar.drop_duplicates("adm_id").set_index("adm_id")[ref_col]
    out = pd.concat({obs_col: clms, ref_col: ref}, axis=1).dropna()
    out["difference"] = out[obs_col] - out[ref_col]
    logger.info(
        "%d units on both references; CLMS %s is %+.1f d from CyBench %s "
        "(median %+.0f, r %.2f)",
        len(out), obs_col, out["difference"].mean(), ref_col,
        out["difference"].median(), out[obs_col].corr(out[ref_col]),
    )
    return out


def _unit_means(frame: pd.DataFrame, col: str, group_col: str, circular: bool
                ) -> pd.Series:
    """Per-unit mean of ``col``, circular where the quantity is a date."""
    if not circular:
        return frame.groupby(group_col)[col].mean()
    return frame.groupby(group_col)[col].apply(
        lambda s: doy.circular_mean_doy(s.to_numpy())
    )


def split_skill(
    paired: pd.DataFrame,
    obs_col: str,
    sim_col: str,
    *,
    circular: bool = True,
    group_col: str = "adm_id",
) -> dict[str, float]:
    """Separate *where* skill from *when* skill.

    A static reference can only measure the first. With a year dimension the
    two come apart, and they answer different questions: the spatial term asks
    whether the continental gradient is right, the interannual term whether the
    model responds to the weather of a particular season. A model can be
    excellent at one and useless at the other, and pooling them hides it.

    Args:
        paired: One row per (unit, year) carrying both columns.
        obs_col: Observed column.
        sim_col: Simulated column.
        circular: Whether the quantity is a day-of-year. It matters: a winter
            crop's emergence straddles New Year, so a unit emerging at DOY 360
            one season and DOY 5 the next has a plain anomaly of −355 days and
            a true one of +10. Set ``False`` for a duration.
        group_col: The unit key anomalies are taken within.

    Returns:
        ``spatial_r`` over the per-unit means, ``anomaly_r`` over the
        deviations from them, the observed and simulated anomaly standard
        deviations (a model matching ``anomaly_r`` but not ``sim_anomaly_sd``
        has the right timing and the wrong amplitude), and the counts.
    """
    frame = paired[[group_col, obs_col, sim_col]].dropna()
    obs_mean = _unit_means(frame, obs_col, group_col, circular)
    sim_mean = _unit_means(frame, sim_col, group_col, circular)

    # Mapped rather than joined: a caller whose columns are already named
    # "obs"/"sim" would otherwise collide with the mean columns.
    key = frame[group_col]
    obs_at_unit = key.map(obs_mean).to_numpy()
    sim_at_unit = key.map(sim_mean).to_numpy()

    if circular:
        obs_anom = doy.doy_difference(frame[obs_col].to_numpy(), obs_at_unit)
        sim_anom = doy.doy_difference(frame[sim_col].to_numpy(), sim_at_unit)
        # Correlation is linear, so the unit means need a linear representation
        # too — unwrapped about the observed continental circular mean.
        centre = doy.circular_mean_doy(obs_mean.to_numpy())
        spatial_obs = doy.unwrap_doy(obs_mean.to_numpy(), centre)
        spatial_sim = doy.unwrap_doy(sim_mean.to_numpy(), centre)
    else:
        obs_anom = frame[obs_col].to_numpy() - obs_at_unit
        sim_anom = frame[sim_col].to_numpy() - sim_at_unit
        spatial_obs, spatial_sim = obs_mean.to_numpy(), sim_mean.to_numpy()

    obs_anom, sim_anom = pd.Series(obs_anom), pd.Series(sim_anom)
    return {
        "n": float(len(frame)),
        "units": float(len(obs_mean)),
        "spatial_r": float(pd.Series(spatial_obs).corr(pd.Series(spatial_sim))),
        "anomaly_r": float(obs_anom.corr(sim_anom)),
        "obs_anomaly_sd": float(obs_anom.std()),
        "sim_anomaly_sd": float(sim_anom.std()),
    }
