"""Score a German smoke run against CyBench yields and PEP725 phenology.

The two references are not the same kind of thing, and the difference decides
how far a result can be pushed:

**CyBench** is an administrative statistic over a whole NUTS-3 region, reported
at market moisture (~13.5 %), while a simulated cell is one point inside it
reporting grain dry matter. Roughly 15 % of the reference is therefore water,
and **no correction is applied** — so part of any negative bias is the moisture
convention rather than the model.

**PEP725** is a point observation at a volunteer's station, matched to the
nearest simulated cell. Neither pairing is exact, and the spread the mismatch
adds is not model error.

**The two runs no longer sow the same way.** SIMPLACE runs the rule-based
solution (``templates/brandenburg/``): it sows on the first day inside the
cell's planting window on which a weather rule holds, so its sowing row is a
result. torchcrop still latches one ``sowing_doy`` per cell from the site table,
so its sowing row is an input echoed back and measures an offset, not skill.
Read the two rows differently, and check ``sowing_forced`` before crediting
SIMPLACE with the spread — a season the window deadline sowed is a constant in
disguise. Sowing is reported either way because it propagates into every later
stage.

Used by ``evaluation/germany_smoke_evaluation.ipynb`` and by
``scripts/validate_germany.py``, which is a thin CLI over it.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

__all__ = [
    "PHASES",
    "Phase",
    "common_window",
    "load_pep725",
    "load_runs",
    "plot_cells",
    "plot_phenology",
    "plot_yield",
    "validate_phenology",
    "validate_yield",
]

CYBENCH_DE = Path(
    "/data01/FDS/muduchuru/Data/Agri/cybench/cybench-data/wheat/DE/yield_wheat_DE.csv"
)
PEP725_FILES = [
    Path("/data01/FDS/muduchuru/Data/Agri/PEP725/pep725_20260807_0.csv"),
    Path("/data01/FDS/muduchuru/Data/Agri/PEP725/pep725_20260807_400000.csv"),
]

#: A station this far from a cell centre is still matched to it. Half a grid
#: diagonal is ~7 km; 25 km keeps enough stations per cell to average over
#: without crossing a climate gradient.
MATCH_KM = 25.0


@dataclass(frozen=True, slots=True)
class Phase:
    """A PEP725 BBCH phase, the simulated column it pairs with, and the caveat."""

    column: str
    label: str
    caveat: str


#: Codes above 99 are PEP725's own additions.
PHASES: dict[int, Phase] = {
    0: Phase("sowing_doy", "Sowing (BBCH 00)", "rule-based in SIMPLACE, a constant input in torchcrop"),
    10: Phase("emergence_doy", "Emergence (BBCH 10)", "first leaf through the sheath"),
    51: Phase("anthesis_doy", "Heading (BBCH 51)", "heading precedes anthesis by ~5-10 d"),
    85: Phase("maturity_doy", "Soft dough (BBCH 85)", "precedes full ripeness (BBCH 89)"),
    100: Phase("maturity_doy", "Harvest (PEP725 100)", "at or after maturity"),
}


def _haversine_km(lon1, lat1, lon2, lat2) -> np.ndarray:
    """Great-circle distance [km] between two arrays of points."""
    lon1, lat1, lon2, lat2 = map(np.radians, (lon1, lat1, lon2, lat2))
    a = (
        np.sin((lat2 - lat1) / 2) ** 2
        + np.cos(lat1) * np.cos(lat2) * np.sin((lon2 - lon1) / 2) ** 2
    )
    return 6371.0 * 2 * np.arcsin(np.sqrt(a))


def _metrics(sim: np.ndarray, obs: np.ndarray) -> dict[str, float]:
    """RMSE, MAE, bias and Pearson r, where each is defined."""
    keep = np.isfinite(sim) & np.isfinite(obs)
    sim, obs = sim[keep], obs[keep]
    if sim.size < 2:
        return dict(n=float(sim.size), rmse=np.nan, mae=np.nan, bias=np.nan, r=np.nan)
    residual = sim - obs
    return {
        "n": float(sim.size),
        "rmse": float(np.sqrt((residual**2).mean())),
        "mae": float(np.abs(residual).mean()),
        "bias": float(residual.mean()),
        # A correlation needs both sides to vary; a constant input has no variance.
        "r": (
            float(np.corrcoef(sim, obs)[0, 1])
            if sim.std() > 1e-9 and obs.std() > 1e-9
            else np.nan
        ),
    }


def _normalise(frame: pd.DataFrame) -> pd.DataFrame:
    """Ensure a run carries ``maturity_doy`` and ``yield_t_ha``.

    SIMPLACE reports the date; torchcrop reports ``days_to_maturity`` from its
    sowing latch, so the date is rebuilt and wrapped back into the year — a crop
    sown on DOY 270 and maturing 272 days later matures on DOY 177 of the
    *following* calendar year, which is the harvest year the row is labelled
    with. Without this the two models cannot share a phenology table at all.
    """
    out = frame
    if "yield_t_ha" not in out and "yield_g_m2" in out:
        out = out.assign(yield_t_ha=out["yield_g_m2"] / 100.0)
    if "maturity_doy" not in out and "days_to_maturity" in out:
        out = out.assign(
            maturity_doy=(out["sowing_doy"] + out["days_to_maturity"] - 1) % 365 + 1
        )
    return out


def load_runs(**paths: Path) -> dict[str, pd.DataFrame]:
    """Load each named run's Parquet, filling in the derivable columns.

    A run that is not on disk is skipped with a warning rather than raised on:
    the notebook is useful with SIMPLACE alone before torchcrop has been run.
    """
    runs = {}
    for name, path in paths.items():
        if not Path(path).is_file():
            logger.warning("%s: no run at %s, skipping it", name, path)
            continue
        runs[name] = _normalise(pd.read_parquet(path))
    if not runs:
        raise FileNotFoundError(f"none of {list(paths.values())} exists")
    return runs


def common_window(runs: dict[str, pd.DataFrame]) -> tuple[int, int]:
    """The harvest years every run covers, so the models are scored on one span."""
    return (
        max(int(f["year"].min()) for f in runs.values()),
        min(int(f["year"].max()) for f in runs.values()),
    )


def validate_yield(
    runs: dict[str, pd.DataFrame], cells: pd.DataFrame
) -> pd.DataFrame:
    """Compare simulated yields with the CyBench NUTS-3 statistics."""
    observed = (
        pd.read_csv(CYBENCH_DE)
        .rename(columns={"harvest_year": "year", "yield": "obs_t_ha"})
        [["adm_id", "year", "obs_t_ha"]]
        .dropna()
    )
    rows = []
    for model, frame in runs.items():
        joined = frame.merge(cells[["SimplaceID", "adm_id"]], on="SimplaceID").merge(
            observed, on=["adm_id", "year"]
        )
        if joined.empty:
            logger.warning("%s: no (region, year) pair overlaps CyBench", model)
            continue
        rows.append(
            _metrics(joined["yield_t_ha"].to_numpy(), joined["obs_t_ha"].to_numpy())
            | {
                "model": model,
                "sim_mean": joined["yield_t_ha"].mean(),
                "obs_mean": joined["obs_t_ha"].mean(),
                "years": f"{int(joined['year'].min())}-{int(joined['year'].max())}",
                "regions": joined["adm_id"].nunique(),
            }
        )
    return pd.DataFrame(rows)


def load_pep725(cells: pd.DataFrame, years: tuple[int, int]) -> pd.DataFrame:
    """PEP725 wheat observations matched to the nearest simulated cell."""
    pep = pd.concat(
        [pd.read_csv(path, sep=";") for path in PEP725_FILES], ignore_index=True
    )
    pep = pep[
        (pep["genus"] == "Triticum")
        & pep["year"].between(*years)
        & pep["phase_id"].isin(PHASES)
    ]
    # cult_season 2 is the winter crop, 0 unspecified. Keeping the rest would
    # mix a spring-sown series into a winter-wheat comparison.
    if "cult_season" in pep:
        pep = pep[pep["cult_season"].isin([0, 2])]

    stations = pep[["s_id", "lon", "lat"]].drop_duplicates()
    distance = _haversine_km(
        stations["lon"].to_numpy()[:, None], stations["lat"].to_numpy()[:, None],
        cells["lon"].to_numpy()[None, :], cells["lat"].to_numpy()[None, :],
    )
    stations = stations.assign(
        SimplaceID=cells["SimplaceID"].to_numpy()[distance.argmin(axis=1)],
        distance_km=distance.min(axis=1),
    )
    stations = stations[stations["distance_km"] <= MATCH_KM]
    logger.info(
        "PEP725: %d stations within %.0f km of a simulated cell (of %d)",
        len(stations), MATCH_KM, distance.shape[0],
    )
    return pep.merge(stations[["s_id", "SimplaceID", "distance_km"]], on="s_id")


def validate_phenology(
    runs: dict[str, pd.DataFrame], observed: pd.DataFrame
) -> pd.DataFrame:
    """Compare simulated stage dates with PEP725, per stage and model."""
    rows = []
    for code, phase in PHASES.items():
        # One observed date per (cell, year): the median over the matched
        # stations, which is robust to a single mis-typed entry.
        obs = (
            observed[observed["phase_id"] == code]
            .groupby(["SimplaceID", "year"])["day"]
            .median()
            .rename("obs_doy")
            .reset_index()
        )
        for model, frame in runs.items():
            if phase.column not in frame:
                continue
            joined = (
                frame[["SimplaceID", "year", phase.column]]
                .merge(obs, on=["SimplaceID", "year"])
                .dropna()
            )
            if joined.empty:
                continue
            rows.append(
                _metrics(joined[phase.column].to_numpy(), joined["obs_doy"].to_numpy())
                | {
                    "model": model,
                    "phase": code,
                    "stage": phase.label,
                    "caveat": phase.caveat,
                    "sim_mean": joined[phase.column].mean(),
                    "obs_mean": joined["obs_doy"].mean(),
                }
            )
    return pd.DataFrame(rows)


# --------------------------------------------------------------------------- #
# Figures
# --------------------------------------------------------------------------- #


def _series(palette: dict | None) -> list[str]:
    """The project's categorical colours, in order, with a usable fallback."""
    palette = palette or {}
    return [
        palette.get(f"series_{i}", default)
        for i, default in enumerate(["#4c72b0", "#dd8452", "#55a868"], start=1)
    ]


def plot_cells(cells: pd.DataFrame, palette: dict | None = None):
    """Where the sampled cells sit, over Germany."""
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(5.5, 6.5))
    ax.scatter(
        cells["lon"], cells["lat"], s=60, zorder=3,
        color=(palette or {}).get("primary", "#4c72b0"),
        edgecolor="white", linewidth=0.8,
    )
    for _, cell in cells.iterrows():
        ax.annotate(
            str(cell.get("adm_id", "")), (cell["lon"], cell["lat"]),
            fontsize=6, xytext=(4, 3), textcoords="offset points", alpha=0.75,
        )
    ax.set(
        xlabel="longitude [°E]", ylabel="latitude [°N]",
        title=f"{len(cells)} smoke-test cells, {cells['adm_id'].nunique()} NUTS-3 regions",
    )
    ax.set_aspect(1 / np.cos(np.radians(cells["lat"].mean())))
    ax.grid(alpha=0.3)
    fig.tight_layout()
    return fig, ax


def plot_yield(runs: dict[str, pd.DataFrame], cells: pd.DataFrame, palette: dict | None = None):
    """Simulated against observed yield, and both as a national time series."""
    import matplotlib.pyplot as plt

    observed = (
        pd.read_csv(CYBENCH_DE)
        .rename(columns={"harvest_year": "year", "yield": "obs_t_ha"})
        [["adm_id", "year", "obs_t_ha"]]
        .dropna()
    )
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.6))
    colours = _series(palette)

    for colour, (model, frame) in zip(colours, runs.items()):
        joined = frame.merge(cells[["SimplaceID", "adm_id"]], on="SimplaceID").merge(
            observed, on=["adm_id", "year"]
        )
        if joined.empty:
            continue
        axes[0].scatter(
            joined["obs_t_ha"], joined["yield_t_ha"], s=14, alpha=0.5,
            label=model, color=colour,
        )
        annual = joined.groupby("year")[["yield_t_ha", "obs_t_ha"]].mean()
        axes[1].plot(annual.index, annual["yield_t_ha"], marker="o", ms=4,
                     label=f"{model} (sim)", color=colour)

    if not observed.empty:
        national = (
            observed[observed["adm_id"].isin(cells["adm_id"])]
            .groupby("year")["obs_t_ha"].mean()
        )
        axes[1].plot(national.index, national.values, color="black", lw=2,
                     marker="s", ms=4, label="CyBench (obs)")

    limit = 12.0
    axes[0].plot([0, limit], [0, limit], color="black", lw=1, ls="--", zorder=0)
    axes[0].set(xlim=(0, limit), ylim=(0, limit),
                xlabel="CyBench yield [t/ha]", ylabel="simulated yield [t/ha]",
                title="Per region and year")
    axes[1].set(xlabel="harvest year", ylabel="yield [t/ha]",
                title="Mean over the sampled regions")
    for ax in axes:
        ax.legend(frameon=False, fontsize=8)
        ax.grid(alpha=0.3)
    fig.tight_layout()
    return fig, axes


def plot_phenology(phenology: pd.DataFrame, palette: dict | None = None):
    """Stage-date bias per model, with the not-a-prediction rows marked."""
    import matplotlib.pyplot as plt

    if phenology.empty:
        raise ValueError("no stage could be paired, so there is nothing to plot")

    stages = list(dict.fromkeys(phenology["stage"]))
    models = list(dict.fromkeys(phenology["model"]))
    colours = _series(palette)
    positions = np.arange(len(stages))
    width = 0.8 / max(len(models), 1)

    fig, ax = plt.subplots(figsize=(9, 4.4))
    for offset, (colour, model) in enumerate(zip(colours, models)):
        rows = phenology[phenology["model"] == model].set_index("stage")
        bias = [rows["bias"].get(stage, np.nan) for stage in stages]
        error = [rows["rmse"].get(stage, np.nan) for stage in stages]
        ax.bar(
            positions + offset * width - 0.4 + width / 2, bias, width * 0.92,
            yerr=error, capsize=3, label=model, color=colour,
            error_kw={"alpha": 0.4, "lw": 1},
        )

    ax.axhline(0, color="black", lw=1)
    ax.set_xticks(positions)
    ax.set_xticklabels(
        [s.replace(" (", "\n(") for s in stages], fontsize=8
    )
    ax.set(ylabel="bias [days]  (simulated − observed)",
           title="Stage dates vs PEP725; bars are bias, whiskers RMSE")
    ax.legend(frameon=False, fontsize=8)
    ax.grid(alpha=0.3, axis="y")
    fig.text(
        0.01, -0.04,
        "Sowing is a constant input, not a prediction. Soft dough precedes the "
        "full ripeness the models report, so a positive bias there is expected.",
        fontsize=7.5, alpha=0.75,
    )
    fig.tight_layout()
    return fig, ax
