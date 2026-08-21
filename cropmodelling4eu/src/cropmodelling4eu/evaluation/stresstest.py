"""Read and plot a stress-test run — SIMPLACE against torchcrop, same inputs.

The inputs are what ``submit/submit_stresstest.py`` writes under
``<output_dir>/stresstest``. That run removes every difference between the two
models that is not the model itself (crop parameters, sowing date, spin-up,
irrigation, CO2 — see the script's docstring), so unlike
:mod:`~cropmodelling4eu.evaluation.germany` **there is no observation here**:
neither model is right, and every number in this module is a *difference*
between two simulations of the same site, season and crop.

Three consequences for how the tables read:

* The sign convention is ``torchcrop − simplace`` throughout, so a negative
  yield bias means torchcrop is the lower of the two. Naming one "simulated"
  and the other "observed" would be a claim neither run supports.
* ``sowing_doy`` must agree **exactly**: torchcrop is latched to SIMPLACE's own
  simulated date. :func:`check_latches` asserts that rather than assuming it,
  because a silent fallback to the site table's date would make every later
  panel a comparison of two different seasons.
* A scenario is a pair of runs, not a treatment applied to observations, so
  ``potential`` minus ``limited`` is read per model (:func:`scenario_effect`)
  rather than pooled across them.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
import yaml

from .germany import DAILY_LABELS, _metrics, _series, daily_envelope, load_simplace_daily

logger = logging.getLogger(__name__)

__all__ = [
    "COMPARED",
    "DEFAULT_ROOT",
    "Compared",
    "agreement",
    "check_latches",
    "load_audit",
    "load_daily",
    "load_fertilizer_check",
    "load_provenance",
    "load_run",
    "pair",
    "plot_agreement",
    "plot_cell_pairs",
    "plot_daily",
    "plot_divergence",
    "plot_scenario_effect",
    "scenario_effect",
    "summary",
]

DEFAULT_ROOT = Path(
    "/data01/FDS/muduchuru/Data/SIMPLACE/cropmodelling4eu/stresstest"
)


@dataclass(frozen=True, slots=True)
class Compared:
    """One quantity both models report, with what a difference in it means."""

    key: str
    label: str
    unit: str
    note: str


#: The comparable outputs, ordered from the result to the diagnostics that
#: explain it. ``sowing_doy`` is first because it is a *check*, not a result:
#: it is an input to both models and any difference invalidates the rest.
COMPARED: tuple[Compared, ...] = (
    Compared("sowing_doy", "Sowing", "DOY",
             "an input to both models — must be identical, see check_latches"),
    Compared("yield_t_ha", "Yield", "t/ha", "grain dry matter, the headline result"),
    Compared("biomass_g_m2", "Biomass", "g/m²",
             "total above-ground dry matter; a yield gap with matching biomass "
             "is partitioning, not growth"),
    Compared("max_lai", "Peak LAI", "m²/m²",
             "the canopy that intercepted the light; the usual root of a "
             "biomass difference"),
    Compared("days_to_maturity", "Season length", "days",
             "sowing to DVS=2; both run the same TSUM1/TSUM2, so a difference "
             "here is the temperature the phenology integrated, not the crop"),
    Compared("tranrf_mean", "TRANRF", "1 = unstressed",
             "season-mean water stress; applied to growth in both models"),
    Compared("nni_mean", "NNI", "1 = unstressed",
             "season-mean nitrogen index; only binds at iopt=3"),
    Compared("n_applied_g_m2", "N applied", "g N/m²",
             "one schedule read by two routes — a difference is a conversion "
             "bug, not a model result"),
)

_MODEL_ORDER = ("simplace", "torchcrop")


# --------------------------------------------------------------------------- #
# Loading
# --------------------------------------------------------------------------- #


def load_provenance(root: Path = DEFAULT_ROOT) -> dict:
    """The ``stresstest`` block of the workspace config the run was driven by.

    That file *is* the run's input (``submit_stresstest.py`` writes it, loads it
    back and runs from it), so what it records — the removed inputs, the crop
    parameter source, the known asymmetries — describes the run rather than
    summarising it after the fact.
    """
    path = Path(root) / "torchcrop" / "config.yaml"
    if not path.is_file():
        logger.warning("no workspace config at %s", path)
        return {}
    return yaml.safe_load(path.read_text()).get("stresstest", {})


def _read_simplace(root: Path) -> pd.DataFrame:
    """SIMPLACE summaries, topped up from the per-scenario run directories.

    The combined ``stresstest_simplace.parquet`` holds only the scenarios of the
    *last* invocation, while ``simplace/<scenario>/<year>/`` survives from every
    one. A scenario missing from the combined file is therefore read from its own
    collected Parquet instead of being dropped — the alternative is a notebook
    that silently shows one scenario when two were run.
    """
    frames: list[pd.DataFrame] = []
    combined = root / "stresstest_simplace.parquet"
    if combined.is_file():
        frames.append(pd.read_parquet(combined))
    seen = set(frames[0]["scenario"]) if frames else set()

    for path in sorted(root.glob("simplace/*/*/simplace_europe.parquet")):
        scenario = path.parents[1].name
        if scenario in seen:
            continue
        frames.append(pd.read_parquet(path).assign(scenario=scenario))
        logger.info("scenario %r recovered from %s", scenario, path.parent)
    if not frames:
        raise FileNotFoundError(f"no SIMPLACE output under {root}")
    return pd.concat(frames, ignore_index=True)


def load_run(root: Path = DEFAULT_ROOT) -> pd.DataFrame:
    """Both models' per-(scenario, cell, season) summaries in one long frame.

    Columns are harmonised, not renamed away: ``yield_t_ha`` is derived where
    only ``yield_g_m2`` exists, and a quantity one model does not report stays
    absent so :func:`pair` drops it rather than comparing it against a zero.
    """
    root = Path(root)
    torchcrop_path = root / "torchcrop" / "stresstest_torchcrop.parquet"
    if not torchcrop_path.is_file():
        raise FileNotFoundError(f"no torchcrop output at {torchcrop_path}")

    frames = {
        "simplace": _read_simplace(root),
        "torchcrop": pd.read_parquet(torchcrop_path),
    }
    out = []
    for model, frame in frames.items():
        if "yield_t_ha" not in frame:
            frame = frame.assign(yield_t_ha=frame["yield_g_m2"] / 100.0)
        keep = ["scenario", "SimplaceID", "year", "lon", "lat"]
        keep += [v.key for v in COMPARED if v.key in frame]
        out.append(frame[keep].assign(model=model))

    run = pd.concat(out, ignore_index=True)
    logger.info(
        "%d cells, seasons %s, scenarios %s",
        run["SimplaceID"].nunique(), sorted(run["year"].unique()),
        sorted(run["scenario"].unique()),
    )
    return run


def load_audit(root: Path = DEFAULT_ROOT) -> pd.DataFrame:
    """The crop-parameter audit written before the run."""
    return pd.read_csv(Path(root) / "crop_parameter_audit.csv")


def load_fertilizer_check(root: Path = DEFAULT_ROOT) -> pd.DataFrame:
    """Per-(scenario, cell, season) nitrogen applied by each model."""
    return pd.read_csv(Path(root) / "fertilizer_check.csv")


def _first_season(daily: pd.DataFrame) -> pd.DataFrame:
    """Keep the first contiguous run of crop days per (model, cell, season).

    A stress-test SIMPLACE run ends on 31 December, by which time the solution
    has sown the *following* autumn's crop — a second block of ``0 < DVS < 2``
    whose last day falls in the same calendar year, so
    :func:`~cropmodelling4eu.evaluation.germany.load_simplace_daily` labels it
    with the same harvest year. Left in, it appears as a spurious second canopy
    past day ~380 of a season that matured at day ~330.
    """
    keys = ["model", "scenario", "SimplaceID", "year"]
    days = daily[keys + ["das"]].drop_duplicates().sort_values(keys + ["das"])
    later = days.groupby(keys)["das"].diff().gt(1).groupby(
        [days[key] for key in keys]
    ).cummax()
    kept = daily.merge(days[~later.to_numpy()], on=keys + ["das"])
    if len(kept) < len(daily):
        logger.info(
            "dropped %d daily rows belonging to a second crop in the same year",
            len(daily) - len(kept),
        )
    return kept


def load_daily(
    root: Path = DEFAULT_ROOT,
    scenarios: list[str] | None = None,
    years: list[int] | None = None,
    variables: tuple[str, ...] = ("LAI", "TRANRF", "NNI"),
) -> pd.DataFrame:
    """Daily trajectories from both models, in one long frame.

    torchcrop's are read from the Parquet the run wrote; SIMPLACE's are parsed
    out of each ``simplace/<scenario>/<year>/out/daily/`` directory, which is
    also where the scenario label comes from — the CSVs themselves carry none.
    Both are trimmed to one season by :func:`_first_season`.
    """
    root = Path(root)
    torchcrop = pd.read_parquet(root / "torchcrop" / "stresstest_torchcrop_daily.parquet")
    if scenarios is not None:
        torchcrop = torchcrop[torchcrop["scenario"].isin(scenarios)]
    if years is not None:
        torchcrop = torchcrop[torchcrop["year"].isin(years)]

    frames = [torchcrop]
    for out_dir in sorted(root.glob("simplace/*/*/out")):
        scenario, year = out_dir.parents[1].name, int(out_dir.parent.name)
        if (scenarios is not None and scenario not in scenarios) or (
            years is not None and year not in years
        ):
            continue
        frames.append(
            load_simplace_daily(out_dir, years=[year], variables=variables)
            .assign(scenario=scenario)
        )
    daily = pd.concat(frames, ignore_index=True)
    daily = _first_season(daily[daily["variable"].isin(variables)])
    logger.info(
        "%d daily rows, models %s, variables %s",
        len(daily), sorted(daily["model"].unique()), sorted(daily["variable"].unique()),
    )
    return daily


# --------------------------------------------------------------------------- #
# Pairing & metrics
# --------------------------------------------------------------------------- #


def pair(run: pd.DataFrame) -> pd.DataFrame:
    """One row per (scenario, cell, season) with both models side by side.

    Adds ``<key>_delta`` (``torchcrop − simplace``) for every compared quantity
    and ``yield_ratio``, which is the number that travels: a bias in t/ha is not
    comparable between a 4 t/ha cell and a 9 t/ha one.
    """
    keys = ["scenario", "SimplaceID", "year"]
    sides = {
        model: frame.drop(columns=["model"]).set_index(keys)
        for model, frame in run.groupby("model")
    }
    missing = set(_MODEL_ORDER) - set(sides)
    if missing:
        raise ValueError(f"the run has no rows for {sorted(missing)}")

    shared = [v.key for v in COMPARED if all(v.key in f for f in sides.values())]
    paired = sides["simplace"][["lon", "lat"]].join(
        sides["simplace"][shared].add_suffix("_simplace"), how="inner"
    ).join(sides["torchcrop"][shared].add_suffix("_torchcrop"), how="inner")

    for key in shared:
        paired[f"{key}_delta"] = paired[f"{key}_torchcrop"] - paired[f"{key}_simplace"]
    paired["yield_ratio"] = (
        paired["yield_t_ha_torchcrop"] / paired["yield_t_ha_simplace"].replace(0, np.nan)
    )
    dropped = {v.key for v in COMPARED} - set(shared)
    if dropped:
        logger.info("not comparable (one model does not report it): %s", sorted(dropped))
    return paired.reset_index()


def check_latches(paired: pd.DataFrame, tolerance: int = 0) -> pd.DataFrame:
    """The rows where the two models did **not** start from the same day.

    An empty frame is the pass. Anything in it invalidates every later panel for
    those cells: torchcrop fell back to the export's calendar instead of taking
    SIMPLACE's simulated ``PlantingDOY``, so the two ran different seasons.
    """
    off = paired[paired["sowing_doy_delta"].abs() > tolerance]
    (logger.info if off.empty else logger.warning)(
        "sowing latch: %d of %d cell-seasons differ by more than %d day(s)",
        len(off), len(paired), tolerance,
    )
    return off[["scenario", "SimplaceID", "year", "sowing_doy_simplace",
                "sowing_doy_torchcrop", "sowing_doy_delta"]]


def agreement(paired: pd.DataFrame, by: list[str] | None = None) -> pd.DataFrame:
    """Agreement metrics per compared quantity, optionally split by scenario.

    ``_metrics`` is reused unchanged, so ``bias``/``rmse``/``r`` mean what they
    do in every other notebook — but with SIMPLACE in the "observed" slot for
    the arithmetic only. Neither model is a reference.
    """
    by = by or ["scenario"]
    rows = []
    for group, frame in paired.groupby(by, dropna=False):
        labels = dict(zip(by, group if isinstance(group, tuple) else (group,)))
        for variable in COMPARED:
            simplace, torchcrop = (
                f"{variable.key}_simplace", f"{variable.key}_torchcrop",
            )
            if simplace not in frame:
                continue
            rows.append(
                labels
                | {"variable": variable.label, "unit": variable.unit}
                | _metrics(frame[torchcrop].to_numpy(float),
                           frame[simplace].to_numpy(float))
                | {
                    "simplace_mean": frame[simplace].mean(),
                    "torchcrop_mean": frame[torchcrop].mean(),
                    "ratio": frame[torchcrop].mean() / frame[simplace].mean()
                    if frame[simplace].mean() else np.nan,
                }
            )
    columns = by + ["variable", "unit", "n", "simplace_mean", "torchcrop_mean",
                    "bias", "ratio", "mae", "rmse", "r"]
    return pd.DataFrame(rows)[columns]


def scenario_effect(run: pd.DataFrame, key: str = "yield_t_ha") -> pd.DataFrame:
    """``potential − limited`` **within** each model, per cell and season.

    Read per model, never pooled: the two scenarios differ by ``iopt`` alone, so
    this is each model's own nutrient-limitation response, and the question is
    whether they respond by a similar amount — not whether they agree in level,
    which :func:`agreement` already answers.
    """
    wide = run.pivot_table(
        index=["model", "SimplaceID", "year"], columns="scenario", values=key
    )
    if not {"potential", "limited"} <= set(wide.columns):
        logger.warning(
            "only scenario(s) %s were run, so there is no effect to take",
            list(wide.columns),
        )
        return pd.DataFrame(columns=["model", "SimplaceID", "year", "effect"])
    # A model that ran only one of the two scenarios has no effect to report;
    # keeping its half-empty rows would put NaNs into every later panel.
    complete = wide.dropna(subset=["potential", "limited"])
    for model in set(wide.index.get_level_values("model")) - set(
        complete.index.get_level_values("model")
    ):
        logger.warning("%s did not run both scenarios; dropped from the effect", model)
    wide = complete
    return (
        wide.assign(
            effect=wide["potential"] - wide["limited"],
            effect_percent=100 * (wide["potential"] / wide["limited"] - 1),
        )
        .reset_index()
    )


def summary(paired: pd.DataFrame) -> pd.Series:
    """The stress test in one column: does it agree, and where does it not."""
    yields = paired["yield_ratio"].replace([np.inf, -np.inf], np.nan)
    return pd.Series(
        {
            "cell-seasons": len(paired),
            "scenarios": paired["scenario"].nunique(),
            "sowing latch mismatches": int((paired["sowing_doy_delta"] != 0).sum()),
            "N applied max |diff| (g/m²)": round(
                paired["n_applied_g_m2_delta"].abs().max(), 5
            ) if "n_applied_g_m2_delta" in paired else np.nan,
            "yield simplace mean (t/ha)": round(paired["yield_t_ha_simplace"].mean(), 2),
            "yield torchcrop mean (t/ha)": round(paired["yield_t_ha_torchcrop"].mean(), 2),
            "yield ratio (median)": round(yields.median(), 2),
            "yield ratio (10-90 %)": (
                f"{yields.quantile(0.1):.2f}-{yields.quantile(0.9):.2f}"
            ),
            "peak LAI ratio (median)": round(
                (paired["max_lai_torchcrop"] / paired["max_lai_simplace"]).median(), 2
            ),
            "season length diff (median days)": round(
                paired["days_to_maturity_delta"].median(), 1
            ) if "days_to_maturity_delta" in paired else np.nan,
            "cells torchcrop < 25 % of simplace": int((yields < 0.25).sum()),
        },
        name="value",
    )


# --------------------------------------------------------------------------- #
# Figures
# --------------------------------------------------------------------------- #


def _annotate(ax, frame: pd.DataFrame, simplace: str, torchcrop: str) -> None:
    stats = _metrics(frame[torchcrop].to_numpy(float), frame[simplace].to_numpy(float))
    ax.text(
        0.04, 0.96,
        f"n={stats['n']:.0f}\nbias={stats['bias']:.2f}\n"
        f"RMSE={stats['rmse']:.2f}\nr={stats['r']:.2f}",
        transform=ax.transAxes, va="top", ha="left", fontsize=7.5,
        bbox={"facecolor": "white", "alpha": 0.7, "edgecolor": "none", "pad": 2},
    )


def plot_agreement(
    paired: pd.DataFrame,
    variables: tuple[str, ...] = ("yield_t_ha", "biomass_g_m2", "max_lai",
                                 "days_to_maturity"),
    palette: dict | None = None,
):
    """torchcrop against SIMPLACE, one panel per quantity and scenario.

    The 1:1 line is the only reference line drawn: there is no observation to
    regress on, so a fitted line would suggest a skill the run cannot measure.
    """
    import matplotlib.pyplot as plt

    labels = {v.key: v for v in COMPARED}
    scenarios = list(dict.fromkeys(paired["scenario"]))
    years = sorted(paired["year"].unique())
    colours = dict(zip(years, _series(palette) * 3))

    fig, axes = plt.subplots(
        len(scenarios), len(variables),
        figsize=(3.1 * len(variables), 3.2 * len(scenarios)), squeeze=False,
    )
    for row, scenario in enumerate(scenarios):
        block = paired[paired["scenario"] == scenario]
        for column, key in enumerate(variables):
            ax = axes[row][column]
            simplace, torchcrop = f"{key}_simplace", f"{key}_torchcrop"
            if simplace not in block:
                ax.set_axis_off()
                continue
            for year in years:
                season = block[block["year"] == year]
                ax.scatter(season[simplace], season[torchcrop], s=18, alpha=0.65,
                           color=colours[year],
                           label=str(year) if row == 0 and column == 0 else None)
            values = pd.concat([block[simplace], block[torchcrop]]).to_numpy(float)
            lo, hi = np.nanmin(values), np.nanmax(values)
            pad = 0.06 * (hi - lo or 1.0)
            ax.plot([lo - pad, hi + pad], [lo - pad, hi + pad], color="black",
                    lw=1, ls="--", zorder=0)
            ax.set_xlim(lo - pad, hi + pad)
            ax.set_ylim(lo - pad, hi + pad)
            _annotate(ax, block, simplace, torchcrop)
            variable = labels[key]
            ax.set_xlabel(f"SIMPLACE {variable.label} [{variable.unit}]", fontsize=8)
            if column == 0:
                ax.set_ylabel(f"{scenario}\ntorchcrop [{variable.unit}]", fontsize=8)
            else:
                ax.set_ylabel(f"torchcrop [{variable.unit}]", fontsize=8)
            ax.grid(alpha=0.3)

    handles, legend_labels = axes[0][0].get_legend_handles_labels()
    fig.tight_layout(rect=(0, 0, 1, 0.92))
    fig.legend(handles, legend_labels, frameon=False, ncol=len(years),
               loc="upper center", bbox_to_anchor=(0.5, 1.0), title="season")
    return fig, axes


def plot_cell_pairs(
    paired: pd.DataFrame, key: str = "yield_t_ha", palette: dict | None = None
):
    """Both models' value for every cell, joined by a rule, sorted by SIMPLACE.

    A scatter hides *which* cells disagree; this does not. A long rule on one
    cell and none on its neighbours is a site problem (soil, a failed
    establishment); a systematic fan is a model difference.
    """
    import matplotlib.pyplot as plt

    variable = {v.key: v for v in COMPARED}[key]
    panels = list(paired.groupby(["scenario", "year"]))
    simplace_colour, torchcrop_colour = _series(palette)[:2]

    fig, axes = plt.subplots(
        1, len(panels), figsize=(4.2 * len(panels), 5.0), sharey=True, squeeze=False,
    )
    for ax, ((scenario, year), block) in zip(axes[0], panels):
        block = block.sort_values(f"{key}_simplace").reset_index(drop=True)
        position = np.arange(len(block))
        ax.hlines(position, block[f"{key}_simplace"], block[f"{key}_torchcrop"],
                  color="grey", lw=1, alpha=0.6, zorder=0)
        ax.scatter(block[f"{key}_simplace"], position, s=22, color=simplace_colour,
                   label="simplace", zorder=2)
        ax.scatter(block[f"{key}_torchcrop"], position, s=22, color=torchcrop_colour,
                   label="torchcrop", zorder=2)
        ax.set_yticks(position)
        ax.set_yticklabels(block["SimplaceID"].astype(str), fontsize=6)
        ax.set_xlabel(f"{variable.label} [{variable.unit}]", fontsize=9)
        ax.set_title(f"{scenario} {year}", fontsize=10)
        ax.grid(alpha=0.3, axis="x")
    axes[0][0].set_ylabel("SimplaceID (sorted by the SIMPLACE value)", fontsize=8)
    axes[0][0].legend(frameon=False, fontsize=8, loc="lower right")
    fig.tight_layout()
    return fig, axes


def plot_daily(
    daily: pd.DataFrame,
    scenario: str | None = None,
    years: list[int] | None = None,
    variables: tuple[str, ...] = ("LAI", "TRANRF", "NNI"),
    palette: dict | None = None,
):
    """Median and interquartile range across cells, per model, on days after sowing.

    The x-axis is ``das`` as it is in the smoke test, but here it is the *same*
    day for both models by construction — the sowing latch — so a horizontal
    offset between the curves is a difference in development rate, not in the
    day the crop went in.
    """
    import matplotlib.pyplot as plt

    if scenario is not None:
        daily = daily[daily["scenario"] == scenario]
    years = years or sorted(daily["year"].unique())
    envelope = daily_envelope(daily)
    models = [m for m in _MODEL_ORDER if m in set(daily["model"])]
    colours = dict(zip(models, _series(palette)))

    fig, axes = plt.subplots(
        len(variables), len(years), figsize=(5.4 * len(years), 2.7 * len(variables)),
        sharex=True, squeeze=False,
    )
    for row, variable in enumerate(variables):
        for column, year in enumerate(years):
            ax = axes[row][column]
            for model in models:
                series = envelope[
                    (envelope["model"] == model) & (envelope["year"] == year)
                    & (envelope["variable"] == variable)
                ].sort_values("das")
                if series.empty:
                    continue
                ax.fill_between(series["das"], series["lo"], series["hi"],
                                color=colours[model], alpha=0.18, lw=0)
                ax.plot(series["das"], series["median"], color=colours[model], lw=1.6,
                        label=model if row == 0 and column == 0 else None)
            if variable in ("TRANRF", "NNI"):
                ax.axhline(1.0, color="black", lw=0.8, ls=":", zorder=0)
                ax.set_ylim(-0.05, 1.15)
            ax.grid(alpha=0.3)
            if row == 0:
                ax.set_title(f"{year} season", fontsize=11)
            if column == 0:
                ax.set_ylabel(DAILY_LABELS.get(variable, variable), fontsize=9)
            if row == len(variables) - 1:
                ax.set_xlabel("days after sowing (the same day in both models)")

    handles, labels = axes[0][0].get_legend_handles_labels()
    title = "median and IQR over cells" + (f" — {scenario}" if scenario else "")
    fig.tight_layout(rect=(0, 0, 1, 0.93))
    fig.legend(handles, labels, frameon=False, ncol=len(models), loc="upper center",
               bbox_to_anchor=(0.5, 1.0), title=title)
    return fig, axes


def plot_divergence(
    paired: pd.DataFrame,
    diagnostics: tuple[str, ...] = ("max_lai", "tranrf_mean", "nni_mean",
                                    "days_to_maturity"),
    palette: dict | None = None,
):
    """The yield ratio against the difference in each state variable.

    Which of these the ratio tracks is the answer the summary tables cannot
    give: a ratio that falls with the LAI difference is a canopy that never
    built, one that falls with ``tranrf_mean`` is a water balance that closed
    the stomata, and one that tracks nothing is a partitioning or phenology
    difference the season means do not resolve.
    """
    import matplotlib.pyplot as plt

    labels = {v.key: v for v in COMPARED}
    present = [key for key in diagnostics if f"{key}_delta" in paired]
    scenarios = list(dict.fromkeys(paired["scenario"]))
    colours = dict(zip(scenarios, _series(palette)))

    fig, axes = plt.subplots(1, len(present), figsize=(3.3 * len(present), 3.4),
                             sharey=True, squeeze=False)
    for ax, key in zip(axes[0], present):
        for scenario in scenarios:
            block = paired[paired["scenario"] == scenario]
            ax.scatter(block[f"{key}_delta"], block["yield_ratio"], s=18, alpha=0.65,
                       color=colours[scenario], label=scenario)
        finite = paired[[f"{key}_delta", "yield_ratio"]].replace(
            [np.inf, -np.inf], np.nan
        ).dropna()
        correlation = (
            finite.corr().iloc[0, 1] if len(finite) > 2 else np.nan
        )
        ax.axhline(1.0, color="black", lw=0.8, ls="--", zorder=0)
        ax.axvline(0.0, color="black", lw=0.8, ls=":", zorder=0)
        variable = labels[key]
        ax.set_xlabel(f"Δ {variable.label} [{variable.unit}]", fontsize=8.5)
        ax.set_title(f"r = {correlation:.2f}", fontsize=9)
        ax.grid(alpha=0.3)
    axes[0][0].set_ylabel("yield ratio  torchcrop / SIMPLACE", fontsize=9)
    axes[0][0].legend(frameon=False, fontsize=8)
    fig.tight_layout()
    return fig, axes


def plot_scenario_effect(effect: pd.DataFrame, palette: dict | None = None):
    """Each model's own ``potential − limited`` response, as a distribution.

    Nothing is compared against the other model's *level* here; the question is
    whether both react to ``iopt`` by a similar amount.
    """
    import matplotlib.pyplot as plt

    if effect.empty:
        raise ValueError("only one scenario was run, so there is no effect to plot")

    models = [m for m in _MODEL_ORDER if m in set(effect["model"])]
    colours = dict(zip(models, _series(palette)))
    fig, axes = plt.subplots(1, 2, figsize=(10, 3.8))
    for offset, model in enumerate(models):
        block = effect[effect["model"] == model]
        axes[0].boxplot(
            block["effect"].dropna(), positions=[offset], widths=0.55,
            patch_artist=True, boxprops={"facecolor": colours[model], "alpha": 0.6},
            medianprops={"color": "black"},
        )
        axes[1].scatter(block["limited"], block["potential"], s=20, alpha=0.7,
                        color=colours[model], label=model)

    axes[0].set_xticks(range(len(models)))
    axes[0].set_xticklabels(models)
    axes[0].axhline(0, color="black", lw=0.8)
    axes[0].set(ylabel="potential − limited [t/ha]",
                title="Nutrient-limitation response, per cell and season")

    values = pd.concat([effect["limited"], effect["potential"]]).to_numpy(float)
    limit = (np.nanmin(values), np.nanmax(values))
    axes[1].plot(limit, limit, color="black", lw=1, ls="--", zorder=0)
    axes[1].set(xlabel="limited yield [t/ha]", ylabel="potential yield [t/ha]",
                title="Per model, both scenarios")
    axes[1].legend(frameon=False, fontsize=8)
    for ax in axes:
        ax.grid(alpha=0.3)
    fig.tight_layout()
    return fig, axes
