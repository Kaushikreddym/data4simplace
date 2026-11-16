"""A LightGBM emulator of torchcrop's yield anomaly, explained with TreeSHAP.

The target is the season's yield as % departure from the cell's ``historical`` 1985–2014
mean in the same model. Predictors are departures of the season indices from that same
baseline, CO₂ in ppm, and static context (a cell's climate and nitrogen rate), so one
pooled model can respond differently in Finland and in Spain.

SHAP explains **the emulator**, which explains torchcrop, not the field. CO₂ and warming
rise together, and SHAP splits credit between correlated inputs by the model's arbitrary
choice among them, so the two are only separable because ``hist-aer`` holds CO₂ at 1850
while its climate varies. Leaving that experiment out of the fit (:func:`score_splits`) tests it.
"""

from __future__ import annotations

import ast
from dataclasses import dataclass

import lightgbm as lgb
import numpy as np
import pandas as pd
import yaml
from sklearn.metrics import r2_score

import features as ft
import scenarios as sc

GROUP_LABELS = {"co2": "CO₂", "temperature": "Mean temperature", "water": "Water supply", "vpd": "Atmospheric demand (VPD)",
                "heat": "Heat extremes", "radiation": "Radiation", "wet": "Heavy precipitation", "frost": "Frost",
                "context": "Cell context"}
CO2 = "co2_ppm"
CONTEXT = ["clim_tmean_grainfill", "clim_tmean_winter", "clim_prec_preanthesis", "n_applied_g_m2", "irri"]
ANOMALY_FEATURES = [f"d_{f}" for f in ft.FEATURES]
PREDICTORS = [*ANOMALY_FEATURES, CO2, *CONTEXT]
GROUP_OF = {f"d_{f}": spec[0] for f, spec in ft.FEATURES.items()} | {CO2: "co2"} | {c: "context" for c in CONTEXT}
#: Below this a % anomaly explodes on a near-zero denominator.
MIN_BASE_YIELD = 1.0


def panel() -> pd.DataFrame:
    """Season indices joined to the yield anomaly, CO₂ and cell context."""
    frame = pd.read_parquet(ft.CACHE)
    yields, cell_ids = sc.cube("yield_t_ha"), frame["SimplaceID"].unique()
    long = (yields.sel(cell=np.sort(cell_ids)).sel(year=slice(ft.FIRST_YEAR, None)).to_series().dropna().rename("yield_t_ha")
            .rename_axis(["model", "experiment", "year", "SimplaceID"]).reset_index())
    frame = frame.merge(long, on=["model", "experiment", "year", "SimplaceID"])
    frame["year"] = frame["year"].astype(int)
    base = frame[(frame["experiment"] == "historical") & frame["year"].between(*sc.BASELINE)]
    means = base.groupby(["model", "SimplaceID"])[[*ft.FEATURES, "yield_t_ha"]].mean()
    joined = frame.join(means.add_prefix("base_"), on=["model", "SimplaceID"])
    joined = joined[joined["base_yield_t_ha"] >= MIN_BASE_YIELD]
    joined["anomaly_pct"] = 100.0 * (joined["yield_t_ha"] / joined["base_yield_t_ha"] - 1.0)
    for f in ft.FEATURES:
        joined[f"d_{f}"] = joined[f] - joined[f"base_{f}"]
    context = {"clim_tmean_grainfill": "base_tmean_grainfill", "clim_tmean_winter": "base_tmean_winter",
               "clim_prec_preanthesis": "base_prec_preanthesis"}
    joined = joined.assign(**{k: joined[v] for k, v in context.items()})
    co2 = sc.co2_series().rename_axis("year").reset_index().melt("year", var_name="experiment", value_name=CO2).dropna()
    joined = joined.merge(co2, on=["experiment", "year"])
    cells = sc.load_cells().frame[["lat", "lon", "ar6", "adm_country"]]
    static = (pd.concat({v: sc.cube(v).sel(experiment="historical", year=2000).mean("model").to_series() for v in ("n_applied_g_m2", "irri")}, axis=1)
              .rename_axis("SimplaceID"))
    joined = joined.join(cells, on="SimplaceID").join(static, on="SimplaceID")
    keep = ["model", "experiment", "year", "SimplaceID", "lat", "lon", "ar6", "adm_country", "yield_t_ha", "base_yield_t_ha",
            "anomaly_pct", *PREDICTORS]
    return joined[keep].dropna(subset=PREDICTORS + ["anomaly_pct"]).reset_index(drop=True)


def _regressor(seed: int = 0) -> lgb.LGBMRegressor:
    return lgb.LGBMRegressor(n_estimators=500, learning_rate=0.05, num_leaves=63, min_child_samples=100,
                             subsample=0.5, subsample_freq=1, colsample_bytree=0.8, n_jobs=48, random_state=seed, verbose=-1)


def _subsample(frame: pd.DataFrame, n: int, seed: int = 0) -> pd.DataFrame:
    return frame if len(frame) <= n else frame.sample(n, random_state=seed)


def fit(frame: pd.DataFrame, n: int = 1_500_000, seed: int = 0, predictors: list[str] | None = None) -> lgb.LGBMRegressor:
    train = _subsample(frame, n, seed)
    return _regressor(seed).fit(train[predictors or PREDICTORS], train["anomaly_pct"])


@dataclass(slots=True, frozen=True)
class Split:
    name: str
    held_out: pd.Series


def splits(frame: pd.DataFrame, seed: int = 0) -> list[Split]:
    """Held-out schemes in rising order of difficulty."""
    years = np.sort(frame["year"].unique())
    held_years = np.random.default_rng(seed).choice(years, len(years) // 5, replace=False)
    return [
        Split("random 20 % of years", frame["year"].isin(held_years)),
        *[Split(f"leave out model {m}", frame["model"] == m) for m in sc.MODELS],
        *[Split(f"leave out experiment {e}", frame["experiment"] == e) for e in sc.EXPERIMENTS],
    ]


def score_splits(frame: pd.DataFrame) -> pd.DataFrame:
    """R² and RMSE on each held-out set; leave-experiment-out is the extrapolation test."""
    rows = []
    for split in splits(frame):
        model = fit(frame[~split.held_out])
        test = _subsample(frame[split.held_out], 500_000)
        predicted = model.predict(test[PREDICTORS])
        rows.append({"held out": split.name, "rows": int(split.held_out.sum()),
                     "R²": r2_score(test["anomaly_pct"], predicted),
                     "RMSE (%)": float(np.sqrt(np.mean((test["anomaly_pct"] - predicted) ** 2))),
                     "bias (%)": float((predicted - test["anomaly_pct"]).mean())})
    return pd.DataFrame(rows)


def contributions(model: lgb.LGBMRegressor, frame: pd.DataFrame) -> pd.DataFrame:
    """TreeSHAP per predictor, with ``base`` the expected value, on ``frame``'s index."""
    values = model.booster_.predict(frame[PREDICTORS], pred_contrib=True)
    return pd.DataFrame(values, columns=[*PREDICTORS, "base"], index=frame.index, dtype=np.float32)


def by_group(shap: pd.DataFrame) -> pd.DataFrame:
    """Sum of TreeSHAP over each :data:`GROUP_LABELS` group; additive, so sums are exact."""
    groups = pd.Series(GROUP_OF).reindex(shap.columns.drop("base"))
    return shap.drop(columns="base").T.groupby(groups).sum().T


def counterfactual(model: lgb.LGBMRegressor, frame: pd.DataFrame, group: str) -> np.ndarray:
    """Prediction minus the prediction with ``group`` at its baseline normal.

    Anomalies go to zero and CO₂ to its 1985–2014 mean. A check on SHAP that assigns
    credit by a different rule: disagreement flags a split that the data do not pin down.
    """
    altered = frame[PREDICTORS].copy()
    columns = [c for c, g in GROUP_OF.items() if g == group]
    if group == "co2":
        altered[CO2] = float(sc.period(sc.co2_series()["historical"].to_xarray(), sc.BASELINE).mean())
    else:
        altered[columns] = 0.0
    return model.predict(frame[PREDICTORS]) - model.predict(altered)


def cotb_table() -> tuple[np.ndarray, np.ndarray]:
    """The RUE multiplier against CO₂ (``COTB``), checked to be the same in all regional crops."""
    def find(node):
        if isinstance(node, dict):
            if "cotb" in node and isinstance(node["cotb"], list):
                return node["cotb"]
            return next((hit for hit in map(find, node.values()) if hit is not None), None)
        return None

    tables = {str(find(yaml.safe_load(p.read_text()))) for p in sorted((sc.CALIBRATION / "yield" / "crops").glob("crop_region_*.yaml"))}
    if len(tables) != 1:
        raise ValueError(f"regional crops disagree on COTB: {tables}")
    table = np.array(ast.literal_eval(tables.pop()), dtype=float)
    return table[:, 0], table[:, 1]


def label_periods(frame: pd.DataFrame) -> pd.Series:
    """The analysis period of each row, or NaN between periods."""
    spans = {"early": sc.EARLY, "baseline": sc.BASELINE, "near": sc.NEAR, "far": sc.FAR}
    out = pd.Series(pd.NA, index=frame.index, dtype="object")
    for name, span in spans.items():
        out[frame["year"].between(*span)] = name
    return out
