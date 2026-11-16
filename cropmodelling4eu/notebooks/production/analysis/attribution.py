"""Which weather extreme drives a unit's yield losses.

Three yield series are set against the same season indices of
:mod:`extremes`, all on the CyBench units: **observed** CyBench (dry matter),
and the two calibrated models' unit means. The pipeline:

1. **Detrend** each unit's yield with its own linear trend and express a season
   as the relative anomaly ``(y − ŷ) / ŷ`` in %. Observed yields carry a
   technology trend the models do not, so each series is detrended on its own;
   a *loss* is an anomaly at or below ``−LOSS_THRESHOLD``.
2. **Features** are each index's departure from the unit's own 1980–2024
   climatology — one fixed baseline shared by all three series — so the pooled
   model learns a response to an unusual season rather than the north–south
   gradient. Two climatological context variables (mean season
   temperature and pre-anthesis water balance) let that response differ between
   a Spanish and a Finnish unit; they are never permuted or occluded.
3. **Model** the anomaly with gradient boosting, cross-validated with whole
   years held out (``GroupKFold`` on year) — every unit of a season shares its
   weather, so a random split would leak the season into its own test set.
4. **Importance** is grouped permutation: all indices of one hazard are
   permuted together on the held-out year, and the loss of R² is the group's
   importance. Correlated indices inside a group no longer hide each other.
5. **Attribution** is occlusion on the held-out predictions: a group's
   contribution to a season is ``f(x) − f(x with that group at climatology)``,
   the part of the predicted anomaly that disappears if the hazard had been
   normal. A unit's **driver** is the hazard with the most negative mean
   contribution over its loss years.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import geopandas as gpd
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pymannkendall as mk
from matplotlib.patches import Patch
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.model_selection import GroupKFold

from cropmodelling4eu.evaluation import production as prod
from cropmodelling4eu.evaluation.plots import EUROPE_PROJECTION, _basemap
from cropmodelling4eu.evaluation.style import palette

from extremes import FEATURES, GROUPS, MODEL_FEATURES

DATASETS: dict[str, str] = {
    "observed": "CyBench (observed)",
    "simplace": "SIMPLACE",
    "torchcrop": "torchcrop",
}
HAZARDS: tuple[str, ...] = ("heat", "drought", "wet")
GROUP_LABELS: dict[str, str] = {
    "heat": "Heat", "drought": "Drought", "wet": "Heavy precipitation",
    "other": "Other weather",
}
#: Unit climatologies that let the pooled response differ between climates.
#: The water balance is used here as an aridity index; it is never permuted.
CONTEXT: dict[str, str] = {"clim_tmean": "tmean_season", "clim_cwb": "cwb_preanthesis"}

#: A season at least this far below the unit's trend (fraction) is a loss.
LOSS_THRESHOLD = 0.10
#: Fixed climatology every hazard anomaly is measured against, for all series.
BASELINE: tuple[int, int] = (1980, 2024)
#: Years all three series cover (CyBench ends in 2021, torchcrop starts in 2000).
COMMON_YEARS: tuple[int, int] = (2000, 2021)
#: Seasons a unit needs before its trend, and anything built on it, is used.
MIN_YEARS = 10
MIN_LOSS_YEARS = 2
#: Mean contribution (% of trend yield) a hazard needs in the loss years before
#: it is called a driver; weaker signals are "no clear signal".
MIN_CONTRIBUTION = 2.0
#: Heat and drought are "compound" when the weaker is at least this fraction of
#: the stronger: a hot spring is usually a dry one, and forcing a single label
#: on such a unit would report the correlation as a winner.
COMPOUND_RATIO = 0.6

DRIVERS: dict[str, str] = {
    "heat": "#c0392b",
    "compound heat + drought": "#8e44ad",
    "drought": "#d68910",
    "heavy precipitation": "#2471a3",
    "other weather": "#7f8c8d",
    "no clear signal": "#d5d8dc",
}


def unit_yields(runs: dict[str, pd.DataFrame], reference: pd.DataFrame,
                cell_to_adm: pd.DataFrame) -> pd.DataFrame:
    """``dataset, adm_id, year, yield_t_ha`` for the observed and both simulated series."""
    observed = reference.dropna(subset=["obs_yield_t_ha"])
    frames = [pd.DataFrame({"dataset": "observed", "adm_id": observed["adm_id"],
                            "year": observed["year"],
                            "yield_t_ha": observed["obs_yield_t_ha"]})]
    for model, frame in runs.items():
        units = prod.units_from_cells(frame[["SimplaceID", "year", "yield_t_ha"]], cell_to_adm)
        frames.append(units[["adm_id", "year", "yield_t_ha"]].assign(dataset=model))
    out = pd.concat(frames, ignore_index=True)
    return out.astype({"adm_id": "string", "year": int})


def unit_extremes(extremes: pd.DataFrame, cell_to_adm: pd.DataFrame) -> pd.DataFrame:
    """Equal-weight unit means of every index per ``(adm_id, year)``."""
    cells = extremes.merge(cell_to_adm[["SimplaceID", "adm_id"]], on="SimplaceID")
    out = cells.dropna(subset=["adm_id"]).groupby(["adm_id", "year"])[list(FEATURES)].mean()
    return out.reset_index().astype({"adm_id": "string", "year": int})


def detrend(yields: pd.DataFrame, min_years: int = MIN_YEARS) -> pd.DataFrame:
    """Adds ``trend``, ``anomaly_pct`` and ``loss`` per (dataset, unit) linear trend.

    Each series keeps its own trend: the observed one carries breeding and
    management gains the fixed-cultivar models do not, and a shared trend would
    leave a drift in the anomalies that lines up with the rising heat indices.
    """
    def one(g: pd.DataFrame) -> pd.DataFrame:
        if g["yield_t_ha"].notna().sum() < min_years:
            return g.iloc[:0]
        slope, intercept = np.polyfit(g["year"], g["yield_t_ha"], 1)
        trend = np.clip(intercept + slope * g["year"], 1e-3, None)
        return g.assign(trend=trend, anomaly_pct=100.0 * (g["yield_t_ha"] / trend - 1.0))

    out = pd.concat([one(g) for _, g in yields.dropna(subset=["yield_t_ha"])
                     .groupby(["dataset", "adm_id"])], ignore_index=True)
    return out.assign(loss=out["anomaly_pct"] <= -100.0 * LOSS_THRESHOLD)


def sen_trends(frame: pd.DataFrame, columns: list[str], by: str = "adm_id",
               per_years: int = 10, min_years: int = MIN_YEARS) -> pd.DataFrame:
    """Theil–Sen slope per ``per_years`` and Mann–Kendall p-value, long format."""
    rows = []
    for key, g in frame.sort_values("year").groupby(by):
        for col in columns:
            x = g[col].dropna()
            if len(x) < min_years or x.nunique() < 3:
                continue
            test = mk.original_test(x.to_numpy())
            rows.append({by: key, "variable": col, "slope": test.slope * per_years,
                         "p": test.p, "mean": float(x.mean())})
    return pd.DataFrame(rows)


@dataclass(slots=True)
class DriverModel:
    """Out-of-fold skill, grouped importance and per-season contributions of one series."""

    dataset: str
    r2: float
    rows: pd.DataFrame
    importance: pd.DataFrame
    contributions: pd.DataFrame = field(repr=False)


def hazard_anomalies(features: pd.DataFrame, baseline: tuple[int, int] = BASELINE,
                     detrended: bool = False) -> pd.DataFrame:
    """Index departures from each unit's ``baseline`` climatology, plus the context columns.

    The baseline is fixed and shared by every yield series, so one season has
    the same hazard anomaly whichever series it is paired with. ``detrended``
    measures the departure from the unit's baseline linear trend instead: shocks
    against shocks, consistent with the detrended yields, but a hazard's trend —
    and so any trend in its contribution — is removed by construction.
    """
    cols = list(FEATURES)
    ref = features[features["year"].between(*baseline)]
    clim = ref.groupby("adm_id")[cols].mean()
    out = features[["adm_id", "year"]].copy()
    if detrended:
        years = ref["year"].to_numpy(dtype=float)
        dx = years - ref.groupby("adm_id")["year"].transform("mean").to_numpy()
        centred = ref[cols] - ref.groupby("adm_id")[cols].transform("mean")
        slope = (centred.mul(dx, axis=0).groupby(ref["adm_id"]).sum()
                 .div(pd.Series(dx ** 2).groupby(ref["adm_id"].to_numpy()).sum(), axis=0))
        mid = ref.groupby("adm_id")["year"].mean()
        unit = features["adm_id"]
        trend = (clim.reindex(unit).to_numpy() + slope.reindex(unit).to_numpy()
                 * (features["year"].to_numpy() - mid.reindex(unit).to_numpy())[:, None])
        out[cols] = features[cols].to_numpy() - trend
    else:
        out[cols] = features[cols].to_numpy() - clim.reindex(features["adm_id"]).to_numpy()
    for name, source in CONTEXT.items():
        out[name] = clim[source].reindex(features["adm_id"]).to_numpy()
    return out


def common_sample(anomalies: pd.DataFrame, years: tuple[int, int] = COMMON_YEARS,
                  min_years: int = MIN_YEARS) -> pd.DataFrame:
    """Unit-years every series has inside ``years``, re-detrended on that window."""
    window = anomalies[anomalies["year"].between(*years)]
    keys = window.groupby(["adm_id", "year"])["dataset"].nunique()
    shared = keys[keys == anomalies["dataset"].nunique()].index
    window = window.set_index(["adm_id", "year"]).loc[shared].reset_index()
    return detrend(window[["dataset", "adm_id", "year", "yield_t_ha"]], min_years)


def design(anomalies: pd.DataFrame, hazards: pd.DataFrame) -> pd.DataFrame:
    """Yield anomalies joined to the hazard anomalies of :func:`hazard_anomalies`."""
    return anomalies.merge(hazards, on=["adm_id", "year"], how="inner").reset_index(drop=True)


def _model() -> HistGradientBoostingRegressor:
    # Shallow, large-leaf trees: CyBench anomalies are mostly non-weather noise,
    # and a deeper model doubled the observed series' held-out error.
    return HistGradientBoostingRegressor(
        max_iter=300, learning_rate=0.03, max_leaf_nodes=8, min_samples_leaf=300,
        l2_regularization=1.0, early_stopping=True, random_state=0)


def fit_driver_model(frame: pd.DataFrame, dataset: str, n_splits: int = 5,
                     n_repeats: int = 5, seed: int = 0) -> DriverModel:
    """Year-blocked cross-validated gradient boosting, grouped permutation and occlusion.

    ``rows`` carries, for every held-out season, the prediction and
    ``dse_<group>`` — the squared-error increase when that group is permuted —
    so importance can be re-pooled over any set of units afterwards.
    """
    columns = [*MODEL_FEATURES, *CONTEXT]
    x, y = frame[columns], frame["anomaly_pct"].to_numpy()
    rng = np.random.default_rng(seed)
    pred = np.full(len(frame), np.nan)
    dse = {g: np.zeros(len(frame)) for g in GROUPS}
    contrib = {g: np.zeros(len(frame)) for g in GROUPS}
    for train, test in GroupKFold(n_splits).split(x, y, frame["year"]):
        model = _model().fit(x.iloc[train], y[train])
        xt = x.iloc[test]
        pred[test] = model.predict(xt)
        base_se = (y[test] - pred[test]) ** 2
        for group, cols in GROUPS.items():
            for _ in range(n_repeats):
                shuffled = xt.copy()
                shuffled[cols] = xt[cols].to_numpy()[rng.permutation(len(test))]
                dse[group][test] += ((y[test] - model.predict(shuffled)) ** 2 - base_se) / n_repeats
            neutral = xt.copy()
            neutral[cols] = 0.0
            contrib[group][test] = pred[test] - model.predict(neutral)

    rows = frame[["adm_id", "year", "anomaly_pct", "loss"]].assign(
        predicted=pred, **{f"dse_{g}": v for g, v in dse.items()})
    contributions = frame[["adm_id", "year", "anomaly_pct", "loss"]].assign(**contrib)
    r2 = 1.0 - np.mean((y - pred) ** 2) / np.var(y)
    return DriverModel(dataset, float(r2), rows, grouped_importance(rows), contributions)


def grouped_importance(rows: pd.DataFrame, by: str | None = None) -> pd.DataFrame:
    """Loss of R² per hazard group, pooled over all rows or per ``by``."""
    def one(g: pd.DataFrame) -> pd.Series:
        var = g["anomaly_pct"].var(ddof=0)
        return pd.Series({grp: g[f"dse_{grp}"].mean() / var for grp in GROUPS}
                         | {"n": len(g)})

    return one(rows).to_frame().T if by is None else rows.groupby(by).apply(one, include_groups=False)


def per_feature_importance(frame: pd.DataFrame, n_splits: int = 5, n_repeats: int = 5,
                           seed: int = 0) -> pd.Series:
    """Loss of R² when each index alone is permuted on the held-out years."""
    columns = [*MODEL_FEATURES, *CONTEXT]
    x, y = frame[columns], frame["anomaly_pct"].to_numpy()
    rng = np.random.default_rng(seed)
    out = pd.Series(0.0, index=MODEL_FEATURES)
    for train, test in GroupKFold(n_splits).split(x, y, frame["year"]):
        model = _model().fit(x.iloc[train], y[train])
        xt, yt = x.iloc[test], y[test]
        base = np.mean((yt - model.predict(xt)) ** 2)
        for col in MODEL_FEATURES:
            for _ in range(n_repeats):
                shuffled = xt.copy()
                shuffled[col] = xt[col].to_numpy()[rng.permutation(len(test))]
                out[col] += (np.mean((yt - model.predict(shuffled)) ** 2) - base) * len(test)
    return out / (n_repeats * len(frame) * np.var(y))


def classify_drivers(contributions: pd.DataFrame) -> pd.DataFrame:
    """Per unit: mean contribution of each group over its loss years and the driver label."""
    losses = contributions[contributions["loss"]]
    summary = losses.groupby("adm_id")[[*GROUPS, "anomaly_pct"]].mean()
    summary["loss_years"] = losses.groupby("adm_id").size()
    summary["seasons"] = contributions.groupby("adm_id").size()
    summary = summary[summary["loss_years"] >= MIN_LOSS_YEARS]

    def label(row: pd.Series) -> str:
        ranked = row[list(GROUPS)].sort_values()
        top, value = ranked.index[0], ranked.iloc[0]
        if value > -MIN_CONTRIBUTION:
            return "no clear signal"
        hazards = row[list(HAZARDS)].sort_values()
        first, second = hazards.index[:2]
        if (top in ("heat", "drought") and {first, second} == {"heat", "drought"}
                and hazards.iloc[1] <= COMPOUND_RATIO * hazards.iloc[0]):
            return "compound heat + drought"
        return {"heat": "heat", "drought": "drought", "wet": "heavy precipitation",
                "other": "other weather"}[top]

    summary["driver"] = summary.apply(label, axis=1)
    return summary.reset_index()


def label_agreement(a: pd.DataFrame, b: pd.DataFrame) -> float:
    """Percent of units labelled in both :func:`classify_drivers` outputs that share a driver."""
    both = pd.concat([a.set_index("adm_id")["driver"], b.set_index("adm_id")["driver"]],
                     axis=1, join="inner")
    return float(100 * (both.iloc[:, 0] == both.iloc[:, 1]).mean()) if len(both) else np.nan


def yearly_contributions(contributions: pd.DataFrame) -> pd.DataFrame:
    """Mean anomaly and mean contribution of each group per year over all units."""
    return contributions.groupby("year")[["anomaly_pct", *GROUPS]].mean()


def category_choropleth(polygons: gpd.GeoDataFrame, labels: pd.Series, ax: plt.Axes,
                        title: str = "", categories: dict[str, str] = DRIVERS,
                        mode: str = "light", linewidth: float = 0.15) -> None:
    """Units filled by a class label, with each class' share of labelled units in the legend."""
    p = palette(mode)
    frame = polygons.assign(label=polygons["adm_id"].map(labels)).to_crs(
        EUROPE_PROJECTION.proj4_init)
    _basemap(ax, p)
    ax.grid(False)
    frame[frame["label"].isna()].plot(ax=ax, color=p["land"], edgecolor=p["surface"],
                                      linewidth=linewidth, zorder=2)
    for name, colour in categories.items():
        part = frame[frame["label"] == name]
        if len(part):
            part.plot(ax=ax, color=colour, edgecolor=p["surface"], linewidth=linewidth, zorder=2)
    counts = labels.value_counts()
    total = int(counts.sum()) or 1
    ax.legend(handles=[Patch(facecolor=c, label=f"{n}  ({100 * counts.get(n, 0) / total:.0f} %)")
                       for n, c in categories.items()],
              loc="upper left", frameon=False, labelcolor=p["secondary"],
              fontsize=plt.rcParams["font.size"] - 1.5)
    ax.set_title(title, color=p["primary"])
