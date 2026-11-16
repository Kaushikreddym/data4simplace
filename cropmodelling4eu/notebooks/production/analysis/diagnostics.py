"""Why observed yield anomalies are harder to predict than simulated ones.

The same gradient-boosting model of :mod:`attribution` is fitted to three
targets on **one panel** — the unit-years all three series share in
:data:`~attribution.COMMON_YEARS` — with identical predictors and folds:

* ``B`` — SIMPLACE and torchcrop anomalies (emulation skill),
* ``C`` — CyBench observed anomalies (data-driven skill).

Every anomaly uses one definition (relative, % of a per-unit trend) and the
trend method is a switch. Under leave-one-year-out the yield trend **and** the
hazard climatology are refitted inside every fold on the training years only,
so the held-out season never shapes its own reference; ``in_fold=False``
reproduces the full-series detrending of the attribution notebook, and the
difference between the two is the leakage. Under leave-one-country-out a test
unit contributes no training rows at all, so its trend and climatology come
from its own series — a per-unit target definition that never reaches the
model being scored.

Skill is ``1 − SSE / Σy²``: the reference forecast is the trend itself (an
anomaly of zero), which is what a weather model has to beat. It is reported
pooled over all held-out rows and as the mean ± sd over folds.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
from joblib import Parallel, delayed, parallel_config
from scipy import stats
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.impute import SimpleImputer
from sklearn.linear_model import RidgeCV
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import LeaveOneGroupOut
from statsmodels.nonparametric.smoothers_lowess import lowess

import attribution as att
from extremes import FEATURES, GROUPS, MODEL_FEATURES
from stage_indices import STAGES, VARIABLES

TARGETS: dict[str, str] = {
    "simplace": "B · SIMPLACE",
    "torchcrop": "B · torchcrop",
    "observed": "C · observed",
}
DETRENDS: tuple[str, ...] = ("linear", "quadratic", "loess")
SCHEMES: dict[str, str] = {"year": "leave-one-year-out", "country": "leave-one-country-out"}
LOESS_FRAC = 0.75
#: Training seasons a unit needs inside a fold before its trend is fitted.
MIN_FIT_YEARS = 8
#: A trend is floored at this fraction of the unit's mean so that a quadratic
#: extrapolated one year past its fit cannot turn a normal season into ±500 %.
TREND_FLOOR = 0.25
#: Step 1e flags. A robust |z| above OUTLIER_Z is reported, never assumed to be
#: an error: 2003 and 2018 are genuine weather and the point of the model.
OUTLIER_Z = 4.0
STEP_ALPHA = 0.001
SEED = 0
N_JOBS = 12
INNER_THREADS = 6

#: Stage-index columns enter the panel with this prefix: several share a name
#: with a season index of :mod:`extremes` (``edd_grainfill``, ``prec_grainfill``).
STAGE_PREFIX = "s_"
STANDARDISED: tuple[str, ...] = ("s_cwb_", "s_sm_")
#: ``name -> ((column, sign), (column, sign))``: the product of the signed
#: in-fold z-scores, so a large positive value is the named sequence.
SEQUENCES: dict[str, tuple[tuple[str, int], ...]] = {
    "seq_dry_spring_then_heat": (("s_prec_stem", -1), ("s_edd_grainfill", 1)),
    "seq_warm_autumn_wet_spring": (("s_tmean_autumn", 1), ("s_prec_stem", 1)),
    "seq_wet_winter_dry_spring": (("s_prec_winter", 1), ("s_prec_stem", -1)),
    "seq_mild_winter_wet_spring": (("s_tmean_winter", 1), ("s_wet_stem", 1)),
    "seq_dull_wet_grainfill": (("s_prec_grainfill", 1), ("s_rad_grainfill", -1)),
    "seq_hot_dry_grainfill": (("s_tmax_grainfill", 1), ("s_sm_grainfill", -1)),
}


def build_panel(yields: pd.DataFrame, features: pd.DataFrame, unit_country: pd.Series,
                years: tuple[int, int] = att.COMMON_YEARS,
                min_years: int = att.MIN_YEARS) -> pd.DataFrame:
    """Raw yields of every target and raw indices on the unit-years all targets share."""
    wide = (yields[yields["year"].between(*years)]
            .pivot_table(index=["adm_id", "year"], columns="dataset", values="yield_t_ha")
            .dropna(subset=list(TARGETS)).reset_index())
    wide = wide[wide.groupby("adm_id")["year"].transform("size") >= min_years]
    panel = wide.merge(features, on=["adm_id", "year"], how="inner")
    panel["country"] = panel["adm_id"].map(unit_country)
    return panel.sort_values(["adm_id", "year"]).reset_index(drop=True)


def _trend(x_fit: np.ndarray, y_fit: np.ndarray, x_eval: np.ndarray, method: str) -> np.ndarray:
    match method:
        case "linear" | "quadratic":
            centre = x_fit.mean()
            coef = np.polyfit(x_fit - centre, y_fit, 1 if method == "linear" else 2)
            return np.polyval(coef, x_eval - centre)
        case "loess":
            fit = lowess(y_fit, x_fit, frac=LOESS_FRAC, it=1, return_sorted=True)
            return np.interp(x_eval, fit[:, 0], fit[:, 1])
        case _:
            raise ValueError(f"unknown detrending method {method!r}")


def anomalies(panel: pd.DataFrame, target: str, method: str,
              train: np.ndarray | None = None) -> np.ndarray:
    """Relative anomaly (%) of every row against a per-unit trend fitted on ``train`` rows."""
    train = np.ones(len(panel), bool) if train is None else train
    years, y = panel["year"].to_numpy(float), panel[target].to_numpy(float)
    out = np.full(len(panel), np.nan)
    for idx in panel.groupby("adm_id").indices.values():
        fit = idx[train[idx]]
        if len(fit) < MIN_FIT_YEARS:
            continue
        trend = np.clip(_trend(years[fit], y[fit], years[idx], method),
                        TREND_FLOOR * y[fit].mean(), None)
        out[idx] = 100.0 * (y[idx] / trend - 1.0)
    return out


def hazards(panel: pd.DataFrame, train: np.ndarray | None = None,
            columns: list[str] | tuple[str, ...] = tuple(MODEL_FEATURES)) -> pd.DataFrame:
    """Index departures from each unit's mean over ``train`` rows, plus the context columns.

    Water-balance and soil-moisture columns are standardised by the unit's
    in-fold sd (a per-unit SPEI-like z-score); ``seq_*`` columns are products
    of the signed in-fold z-scores of :data:`SEQUENCES`. ``country_code`` is
    passed through for the region-aware model structures.
    """
    train = np.ones(len(panel), bool) if train is None else train
    plain = [c for c in columns if not c.startswith("seq_") and c != "country_code"]
    needed = sorted(set(plain) | {c for q in columns if q in SEQUENCES for c, _ in SEQUENCES[q]}
                    | set(att.CONTEXT.values()))
    ref = panel.loc[train].groupby("adm_id")[needed]
    mean = ref.mean().reindex(panel["adm_id"])
    sd = ref.std().reindex(panel["adm_id"]).replace(0.0, np.nan)
    dev = pd.DataFrame(panel[needed].to_numpy() - mean.to_numpy(), columns=needed, index=panel.index)
    z = dev / sd.to_numpy()
    out = dev[plain].copy()
    standard = [c for c in plain if c.startswith(STANDARDISED)]
    out[standard] = z[standard]
    for name in (c for c in columns if c.startswith("seq_")):
        out[name] = np.prod([sign * z[c].to_numpy() for c, sign in SEQUENCES[name]], axis=0)
    if "country_code" in columns:
        out["country_code"] = panel["country_code"].to_numpy()
    for name, source in att.CONTEXT.items():
        out[name] = mean[source].to_numpy()
    return out


@dataclass(slots=True)
class CVResult:
    """Held-out predictions of one (target, detrending, scheme) combination."""

    target: str
    method: str
    scheme: str
    in_fold: bool
    y: np.ndarray
    pred: np.ndarray
    fold: np.ndarray

    def scores(self) -> dict[str, float]:
        ok = np.isfinite(self.y) & np.isfinite(self.pred)
        y, p, f = self.y[ok], self.pred[ok], self.fold[ok]
        per_fold = [skill(y[f == k], p[f == k]) for k in np.unique(f) if (f == k).sum() >= 10]
        return {"R²": skill(y, p), "R² fold mean": float(np.mean(per_fold)),
                "R² fold sd": float(np.std(per_fold)), "RMSE (%)": float(np.sqrt(np.mean((y - p) ** 2))),
                "rows": int(ok.sum()), "folds": len(per_fold)}


def skill(y: np.ndarray, pred: np.ndarray) -> float:
    """``1 − SSE / Σy²``: skill against the trend-only forecast (anomaly 0)."""
    return float(1.0 - np.sum((y - pred) ** 2) / np.sum(y ** 2))


def cross_validate(panel: pd.DataFrame, target: str, method: str = "linear",
                   scheme: str = "year", in_fold: bool = True,
                   extra: pd.DataFrame | None = None) -> CVResult:
    """Leave-one-group-out gradient boosting on the panel, refitting references per fold.

    ``extra`` adds columns (e.g. a crop model's anomaly) that are used as given;
    anything in it must already be free of the held-out fold.
    """
    groups = panel[scheme].to_numpy()
    refit = in_fold and scheme == "year"
    y_full = None if refit else anomalies(panel, target, method)
    x_full = None if refit else hazards(panel)
    pred, y_out = np.full(len(panel), np.nan), np.full(len(panel), np.nan)
    fold = np.empty(len(panel), dtype=object)
    for train, test in LeaveOneGroupOut().split(panel, groups=groups):
        mask = np.zeros(len(panel), bool)
        mask[train] = True
        y = anomalies(panel, target, method, mask) if refit else y_full
        x = hazards(panel, mask) if refit else x_full
        if extra is not None:
            x = pd.concat([x, extra], axis=1)
        fit = train[np.isfinite(y[train])]
        model = att._model().fit(x.iloc[fit], y[fit])
        pred[test] = model.predict(x.iloc[test])
        y_out[test] = y[test]
        fold[test] = groups[test]
    return CVResult(target, method, scheme, in_fold, y_out, pred, fold)


def audit_grid(panel: pd.DataFrame, targets=tuple(TARGETS), methods=DETRENDS) -> pd.DataFrame:
    """Step 1 table: every target × detrending, LOYO in-fold and full-series, and LOCO."""
    runs = [(t, m, "year", True) for t in targets for m in methods]
    runs += [(t, m, "year", False) for t in targets for m in methods]
    runs += [(t, m, "country", False) for t in targets for m in methods]
    rows = []
    for target, method, scheme, in_fold in runs:
        result = cross_validate(panel, target, method, scheme, in_fold)
        rows.append({"target": TARGETS[target], "detrending": method,
                     "cv": SCHEMES[scheme] + ("" if scheme == "country" else
                                              " (in-fold refs)" if in_fold else " (full-series refs)"),
                     **result.scores()})
    return pd.DataFrame(rows)


def flag_series(yields: pd.DataFrame, years: tuple[int, int] = att.COMMON_YEARS) -> pd.DataFrame:
    """Per unit: robust outliers, a level step and gaps in the raw CyBench series."""
    rows = []
    observed = yields[(yields["dataset"] == "observed") & yields["year"].between(*years)]
    for unit, g in observed.dropna(subset=["yield_t_ha"]).sort_values("year").groupby("adm_id"):
        x, y = g["year"].to_numpy(float), g["yield_t_ha"].to_numpy(float)
        if len(y) < att.MIN_YEARS:
            continue
        r = y - np.polyval(np.polyfit(x, y, 1), x)
        mad = 1.4826 * np.median(np.abs(r - np.median(r))) or np.nan
        z = np.abs(r - np.median(r)) / mad
        splits = range(4, len(r) - 3)
        p_step = min(stats.ttest_ind(r[:k], r[k:], equal_var=False).pvalue for k in splits)
        rows.append({
            "adm_id": unit, "n": len(y), "max_robust_z": float(np.nanmax(z)),
            "outlier_year": int(x[np.nanargmax(z)]),
            "outlier": bool(np.nanmax(z) > OUTLIER_Z),
            "step": bool(p_step * len(splits) < STEP_ALPHA),
            "gap": bool(len(y) < x.max() - x.min() + 1),
        })
    out = pd.DataFrame(rows)
    out["flagged"] = out[["outlier", "step", "gap"]].any(axis=1)
    return out


# --------------------------------------------------------------------------- #
# Step 2 — baselines on shared folds
# --------------------------------------------------------------------------- #

MODELS: tuple[str, ...] = ("simplace", "torchcrop")
#: Crop-model outputs offered to the hybrid, as unit-year means. torchcrop's
#: stress factors are read from the raw Parquet because ``load_runs`` drops them;
#: SIMPLACE durations are left out because of the (0, 365] wrap.
SIM_OUTPUTS: dict[str, tuple[str, ...]] = {
    "simplace": ("biomass_g_m2", "max_lai"),
    "torchcrop": ("biomass_g_m2", "max_lai", "tranrf_mean", "nni_mean",
                  "heat_stress_factor", "days_to_anthesis"),
}


def unit_outputs(run_dir, cell_to_adm: pd.DataFrame, years: tuple[int, int]) -> pd.DataFrame:
    """Unit-year means of :data:`SIM_OUTPUTS`, columns ``<model>__<output>``."""
    frames = []
    for model, cols in SIM_OUTPUTS.items():
        path = run_dir / model / f"{model}_europe.parquet"
        cells = pd.read_parquet(path, columns=["SimplaceID", "year", *cols])
        cells = cells[cells["year"].between(*years)].merge(
            cell_to_adm[["SimplaceID", "adm_id"]], on="SimplaceID").dropna(subset=["adm_id"])
        means = cells.groupby(["adm_id", "year"])[list(cols)].mean()
        frames.append(means.add_prefix(f"{model}__"))
    out = pd.concat(frames, axis=1).reset_index()
    return out.astype({"adm_id": "string", "year": int})


@dataclass(slots=True, frozen=True)
class Fold:
    """``reference`` rows fit trends and climatologies; ``train`` rows fit the model."""

    name: str
    reference: np.ndarray
    train: np.ndarray
    test: np.ndarray


def loyo_folds(panel: pd.DataFrame) -> list[Fold]:
    """Leave one year out; references and model both see every other year."""
    folds = []
    for year in sorted(panel["year"].unique()):
        held = (panel["year"] == year).to_numpy()
        folds.append(Fold(str(year), ~held, np.flatnonzero(~held), np.flatnonzero(held)))
    return folds


def blocked_folds(panel: pd.DataFrame, k_country: int = 5, k_year: int = 5,
                  seed: int = SEED) -> list[Fold]:
    """Hold out a country block *and* a year block together.

    A model trained on other countries in the test year sees that season's
    shared shock through correlated weather and yields — the leave-one-country
    -out leak of Step 1. Here every training row is outside both blocks, and the
    references of every unit are fitted on the years outside the year block.
    """
    rng = np.random.default_rng(seed)
    sizes = panel["country"].value_counts()
    country_block, load = {}, np.zeros(k_country)
    for country in sizes.index:
        block = int(np.argmin(load))
        country_block[country], load[block] = block, load[block] + sizes[country]
    years = rng.permutation(sorted(panel["year"].unique()))
    year_block = {y: i % k_year for i, y in enumerate(years)}
    cb = panel["country"].map(country_block).to_numpy()
    yb = panel["year"].map(year_block).to_numpy()
    folds = []
    for i in range(k_country):
        for j in range(k_year):
            test = (cb == i) & (yb == j)
            train = (cb != i) & (yb != j)
            folds.append(Fold(f"c{i}y{j}", yb != j, np.flatnonzero(train), np.flatnonzero(test)))
    return folds


@dataclass(slots=True, frozen=True)
class Spec:
    """One model on the shared folds.

    ``target`` is ``observed``, a crop model, or ``residual:<model>`` (observed
    minus simulated anomaly). ``features`` are the hazard columns (``()`` for
    none); ``hybrid`` adds the crop-model outputs. ``direct`` skips the ML: the
    prediction is that crop model's own anomaly, optionally rescaled in-fold.
    ``model`` picks the structure of :func:`fit_predict`.
    """

    name: str
    target: str
    features: tuple[str, ...] = tuple(MODEL_FEATURES)
    hybrid: bool = False
    direct: str | None = None
    scaled: bool = False
    model: str = "gbm"


@dataclass(slots=True)
class SpecResult:
    """Held-out predictions of one :class:`Spec`, and its summed permutation losses."""

    spec: Spec
    scheme: str
    y: np.ndarray
    pred: np.ndarray
    fold: np.ndarray
    observed: np.ndarray
    simulated: np.ndarray | None
    dsse: dict[str, float]

    def scores(self) -> dict[str, float]:
        ok = np.isfinite(self.y) & np.isfinite(self.pred)
        y, p, f = self.y[ok], self.pred[ok], self.fold[ok]
        per_fold = [skill(y[f == k], p[f == k]) for k in np.unique(f) if (f == k).sum() >= 10]
        out = {"R²": skill(y, p), "R² fold mean": float(np.mean(per_fold)),
               "R² fold sd": float(np.std(per_fold)), "r": float(np.corrcoef(y, p)[0, 1]),
               "RMSE (%)": float(np.sqrt(np.mean((y - p) ** 2))), "rows": int(ok.sum())}
        if self.simulated is not None:
            out["R² on observed"] = skill(self.observed[ok], self.simulated[ok] + p)
        return out

    def importance(self) -> pd.Series:
        ok = np.isfinite(self.y) & np.isfinite(self.pred)
        return pd.Series(self.dsse, dtype=float) / float(np.sum(self.y[ok] ** 2))


def fold_targets(panel: pd.DataFrame, reference: np.ndarray, method: str
                 ) -> tuple[dict[str, np.ndarray], pd.DataFrame]:
    """In-fold targets and hybrid features (crop-model output anomalies) for one fold."""
    y = {t: anomalies(panel, t, method, reference) for t in TARGETS}
    y |= {f"residual:{m}": y["observed"] - y[m] for m in MODELS}
    output_cols = [c for c in panel if "__" in c]
    clim = panel.loc[reference].groupby("adm_id")[output_cols].mean().reindex(panel["adm_id"])
    hybrid = pd.DataFrame(panel[output_cols].to_numpy() - clim.to_numpy(),
                          columns=output_cols, index=panel.index)
    for m in MODELS:
        hybrid[f"{m}__yield_anomaly"] = y[m]
    return y, hybrid


_STAGE_GROUP = {"tmax": "heat", "edd": "heat", "prec": "water deficit", "cdd": "water deficit",
                "sm": "water deficit", "cwb": "water deficit", "wet": "wet", "waterlog": "wet",
                "vpd": "VPD", "tmean": "other", "rad": "other", "frost": "other",
                "winterkill": "other"}
_V1_GROUP = {"heat": "heat", "drought": "water deficit", "wet": "wet", "other": "other"}
IMPORTANCE_ORDER: tuple[str, ...] = ("heat", "water deficit", "wet", "VPD", "other", "sequence",
                                     "context", "region", "hybrid")


def group_of(column: str) -> str:
    """Importance group of a model column."""
    if "__" in column:
        return "hybrid"
    if column in att.CONTEXT:
        return "context"
    if column == "country_code":
        return "region"
    if column.startswith("seq_"):
        return "sequence"
    if column.startswith(STAGE_PREFIX):
        return _STAGE_GROUP[column[len(STAGE_PREFIX):].split("_")[0]]
    return _V1_GROUP[FEATURES[column][0]]


def importance_groups(columns: list[str]) -> dict[str, list[str]]:
    groups: dict[str, list[str]] = {}
    for c in columns:
        groups.setdefault(group_of(c), []).append(c)
    return groups


def _monotone(column: str) -> int:
    """Sign constraint of the ``monotonic`` structure: only where agronomy is unambiguous."""
    group = group_of(column)
    if group == "heat" or column.startswith(("s_frost_", "s_winterkill", "frost_days")):
        return -1
    if column.startswith(("s_sm_", "s_cwb_stem", "s_cwb_grainfill")) or column.endswith("yield_anomaly"):
        return 1
    if column.startswith(("s_cdd_", "cdd_")):
        return -1
    return 0


class _PerCountry:
    """A pooled model plus one per country with enough training rows."""

    def __init__(self, min_rows: int = 400):
        self.min_rows, self.models = min_rows, {}

    def fit(self, x: pd.DataFrame, y: np.ndarray, groups: np.ndarray) -> "_PerCountry":
        self.pooled = att._model().fit(x, y)
        for g in np.unique(groups):
            if (groups == g).sum() >= self.min_rows:
                self.models[g] = att._model().fit(x[groups == g], y[groups == g])
        return self

    def predict(self, x: pd.DataFrame, groups: np.ndarray) -> np.ndarray:
        out = self.pooled.predict(x)
        for g, model in self.models.items():
            if (groups == g).any():
                out[groups == g] = model.predict(x[groups == g])
        return out


def fit_predict(kind: str, x_train: pd.DataFrame, y_train: np.ndarray, g_train: np.ndarray,
                g_test: np.ndarray):
    """Fit one model structure; return ``predict(x_test)``."""
    columns = list(x_train.columns)
    match kind:
        case "gbm":
            model = att._model().fit(x_train, y_train)
        case "gbm_country":
            model = att._model().set_params(
                categorical_features=[columns.index("country_code")]).fit(x_train, y_train)
        case "monotonic":
            model = att._model().set_params(
                monotonic_cst=[_monotone(c) for c in columns]).fit(x_train, y_train)
        case "ridge":
            model = make_pipeline(SimpleImputer(strategy="constant", fill_value=0.0),
                                  StandardScaler(), RidgeCV(alphas=np.logspace(-1, 5, 25)))
            model.fit(x_train, y_train)
        case "per_country":
            per = _PerCountry().fit(x_train, y_train, g_train)
            return lambda x: per.predict(x, g_test)
        case _:
            raise ValueError(f"unknown model structure {kind!r}")
    return model.predict


def _run_fold(panel: pd.DataFrame, specs: list[Spec], fold: Fold, index: int, method: str,
              n_repeats: int, seed: int) -> list[tuple]:
    rng = np.random.default_rng([seed, index])
    y, hybrid = fold_targets(panel, fold.reference, method)
    country = panel["country"].to_numpy()
    feature_cache: dict[tuple[str, ...], pd.DataFrame] = {}
    out = []
    for s in specs:
        target = y[s.target]
        sim = y[s.target.split(":")[1]][fold.test] if s.target.startswith("residual:") else None
        dsse: dict[str, float] = {}
        if s.direct is not None:
            pred = y[s.direct][fold.test]
            if s.scaled:
                d = y[s.direct]
                tr = fold.train[np.isfinite(d[fold.train]) & np.isfinite(target[fold.train])]
                b, a = np.polyfit(d[tr], target[tr], 1)
                pred = a + b * pred
        else:
            feats = s.features + (("country_code",) if s.model == "gbm_country" else ())
            parts = []
            if feats:
                if feats not in feature_cache:
                    feature_cache[feats] = hazards(panel, fold.reference, feats)
                parts.append(feature_cache[feats])
            if s.hybrid:
                parts.append(hybrid)
            x = pd.concat(parts, axis=1)
            tr = fold.train[np.isfinite(target[fold.train])]
            te = fold.test
            predict = fit_predict(s.model, x.iloc[tr], target[tr], country[tr], country[te])
            xt = x.iloc[te]
            pred = predict(xt)
            ok = np.isfinite(target[te])
            base_se = np.sum((target[te][ok] - pred[ok]) ** 2)
            for group, cols in importance_groups(list(x.columns)).items():
                for _ in range(n_repeats):
                    shuffled = xt.copy()
                    shuffled[cols] = xt[cols].to_numpy()[rng.permutation(len(te))]
                    delta = np.sum((target[te][ok] - predict(shuffled)[ok]) ** 2) - base_se
                    dsse[group] = dsse.get(group, 0.0) + delta / n_repeats
        out.append((s.name, fold.test, target[fold.test], pred, y["observed"][fold.test], sim, dsse))
    return out


def run_specs(panel: pd.DataFrame, specs: list[Spec], folds: list[Fold], scheme: str,
              method: str = "loess", n_repeats: int = 3, seed: int = SEED,
              n_jobs: int = N_JOBS) -> dict[str, SpecResult]:
    """Every spec on the same folds (in parallel), references refitted per fold."""
    with parallel_config(backend="loky", inner_max_num_threads=INNER_THREADS):
        parts = Parallel(n_jobs=n_jobs)(
            delayed(_run_fold)(panel, specs, fold, i, method, n_repeats, seed)
            for i, fold in enumerate(folds))
    n = len(panel)
    store = {s.name: dict(y=np.full(n, np.nan), pred=np.full(n, np.nan),
                          fold=np.empty(n, dtype=object), obs=np.full(n, np.nan),
                          sim=np.full(n, np.nan), dsse={}) for s in specs}
    for fold, results in zip(folds, parts):
        for name, test, y, pred, obs, sim, dsse in results:
            st = store[name]
            st["y"][test], st["pred"][test], st["obs"][test] = y, pred, obs
            st["fold"][test] = fold.name
            if sim is not None:
                st["sim"][test] = sim
            for g, v in dsse.items():
                st["dsse"][g] = st["dsse"].get(g, 0.0) + v
    return {s.name: SpecResult(s, scheme, store[s.name]["y"], store[s.name]["pred"],
                               store[s.name]["fold"], store[s.name]["obs"],
                               store[s.name]["sim"] if s.target.startswith("residual:") else None,
                               store[s.name]["dsse"]) for s in specs}


NO_FEATURES: tuple[str, ...] = ()
STEP2_SPECS: list[Spec] = [
    Spec("B · SIMPLACE (ML)", "simplace"),
    Spec("B · torchcrop (ML)", "torchcrop"),
    Spec("SIMPLACE vs observed (no ML)", "observed", NO_FEATURES, direct="simplace"),
    Spec("torchcrop vs observed (no ML)", "observed", NO_FEATURES, direct="torchcrop"),
    Spec("SIMPLACE vs observed (rescaled)", "observed", NO_FEATURES, direct="simplace", scaled=True),
    Spec("torchcrop vs observed (rescaled)", "observed", NO_FEATURES, direct="torchcrop", scaled=True),
    Spec("C · observed (ML)", "observed"),
    Spec("residual · obs − SIMPLACE (ML)", "residual:simplace"),
    Spec("residual · obs − torchcrop (ML)", "residual:torchcrop"),
    Spec("hybrid C · hazards + crop models (ML)", "observed", hybrid=True),
    Spec("crop-model outputs only (ML)", "observed", NO_FEATURES, hybrid=True),
]


# --------------------------------------------------------------------------- #
# Step 3 — feature ablations
# --------------------------------------------------------------------------- #

STAGE_COLUMNS: list[str] = [f"{STAGE_PREFIX}{v}_{s}" for s in STAGES for v in VARIABLES] + [
    f"{STAGE_PREFIX}winterkill"]
STAGE_EXTREMES: tuple[str, ...] = tuple(f"{STAGE_PREFIX}{c}" for c in (
    "edd_flowering", "cdd_stem", "frost_stem", "frost_flowering", "tmean_autumn",
    "wet_stem", "waterlog_winter", "waterlog_stem", "winterkill"))
MOISTURE_ATMOSPHERE: tuple[str, ...] = tuple(f"{STAGE_PREFIX}{c}" for c in (
    "vpd_stem", "vpd_flowering", "vpd_grainfill", "cwb_stem", "cwb_grainfill",
    "sm_stem", "sm_grainfill", "rad_stem"))
FEATURE_SETS: dict[str, tuple[str, ...]] = {
    "S0 · 12 season indices": tuple(MODEL_FEATURES),
    "S1 · + stage extremes": tuple(MODEL_FEATURES) + STAGE_EXTREMES,
    "S2 · + VPD, SPEI, soil moisture": tuple(MODEL_FEATURES) + STAGE_EXTREMES + MOISTURE_ATMOSPHERE,
    "S3 · + sequences": tuple(MODEL_FEATURES) + STAGE_EXTREMES + MOISTURE_ATMOSPHERE + tuple(SEQUENCES),
    "FULL · stage × variable": tuple(STAGE_COLUMNS),
}
ABLATION_TARGETS: dict[str, str] = {
    "C": "observed", "B · SIMPLACE": "simplace", "B · torchcrop": "torchcrop",
    "residual · SIMPLACE": "residual:simplace", "residual · torchcrop": "residual:torchcrop",
}


def with_stage(panel: pd.DataFrame, stage: pd.DataFrame) -> pd.DataFrame:
    """``panel`` with the stage-index columns of one window/weighting variant, row order kept."""
    keep = panel.drop(columns=[c for c in panel if c.startswith(STAGE_PREFIX)])
    renamed = stage.rename(columns={c: f"{STAGE_PREFIX}{c}" for c in stage
                                    if c not in ("adm_id", "year")})
    out = keep.merge(renamed, on=["adm_id", "year"], how="left")
    assert (out[["adm_id", "year"]].to_numpy() == panel[["adm_id", "year"]].to_numpy()).all()
    return out


def shuffle_stages(panel: pd.DataFrame, seed: int) -> pd.DataFrame:
    """Scramble each row's stage order, the same permutation for every variable.

    The within-stage covariance and every season total survive; only *when*
    in the season each condition occurred is destroyed.
    """
    rng = np.random.default_rng(seed)
    out = panel.copy()
    order = np.argsort(rng.random((len(panel), len(STAGES))), axis=1)
    for v in VARIABLES:
        cols = [f"{STAGE_PREFIX}{v}_{s}" for s in STAGES]
        values = panel[cols].to_numpy()
        out[cols] = np.take_along_axis(values, order, axis=1)
    return out


def ablation_specs(features: tuple[str, ...], targets: dict[str, str] = ABLATION_TARGETS,
                   model: str = "gbm") -> list[Spec]:
    return [Spec(label, target, features, model=model) for label, target in targets.items()]


# --------------------------------------------------------------------------- #
# Step 5 — residual forensics
# --------------------------------------------------------------------------- #

#: Seasons documented in the literature as wheat-damaging, by country, to link
#: the largest misses to. A candidate label, not a diagnosis: the link is
#: confirmed (or not) by the stage anomalies printed beside each miss.
KNOWN_EVENTS: dict[tuple[int, str], str] = {
    **{(2003, c): "2003 heatwave and drought" for c in ("FR", "IT", "DE", "AT", "HU", "RO", "CZ", "PT", "ES")},
    **{(2007, c): "2007 SE-European heat and drought" for c in ("RO", "HU", "EL", "IT")},
    **{(2011, c): "2011 spring drought, NW Europe" for c in ("FR", "DE", "BE", "NL")},
    **{(2012, c): "2012 February frost / winterkill, summer drought" for c in ("FR", "DE", "PL", "HU", "RO")},
    **{(2016, c): "2016 wet, dull spring and early summer (disease, low radiation)" for c in ("FR", "BE", "DE", "NL")},
    **{(2018, c): "2018 northern drought and heat" for c in ("DE", "PL", "DK", "SE", "LT", "LV", "EE", "NL", "BE", "FI", "CZ")},
    **{(2005, c): "2005 Iberian drought" for c in ("ES", "PT")},
    **{(2012, "PT"): "2012 Iberian drought", (2020, "RO"): "2020 Romanian drought"},
    **{(2012, "ES"): "2012 Iberian drought",
       (2017, "ES"): "2017 Iberian drought", (2017, "IT"): "2017 Italian drought",
       (2020, "FR"): "2020 wet sowing autumn, dry spring"},
}


def residual_frame(panel: pd.DataFrame, results: dict[str, SpecResult]) -> pd.DataFrame:
    """Per unit-year: observed anomaly and observed − predicted for each result."""
    out = panel[["adm_id", "year", "country"]].copy()
    first = next(iter(results.values()))
    out["observed"] = first.observed
    for name, r in results.items():
        out[f"pred · {name}"] = r.pred
        out[f"resid · {name}"] = r.observed - r.pred
    return out


def top_misses(residuals: pd.DataFrame, column: str, stage_z: pd.DataFrame, n: int = 15,
               min_units: int = 3) -> pd.DataFrame:
    """Country-years with the most negative mean residual, the stage indices most out of
    their normal range there, and any candidate known event."""
    grouped = residuals.groupby(["country", "year"])
    table = pd.DataFrame({"units": grouped.size(), "observed anomaly (%)": grouped["observed"].mean(),
                          "mean residual (%)": grouped[column].mean()})
    table = table[table["units"] >= min_units].sort_values("mean residual (%)").head(n)
    z = stage_z.groupby(["country", "year"]).mean()
    table["most anomalous stage indices (z)"] = [
        ", ".join(f"{c[len(STAGE_PREFIX):]} {z.loc[k, c]:+.1f}"
                  for c in z.loc[k].abs().sort_values(ascending=False).index[:3])
        if k in z.index else "" for k in table.index]
    table["candidate event"] = [KNOWN_EVENTS.get((y, c), "—") for c, y in table.index]
    return table
