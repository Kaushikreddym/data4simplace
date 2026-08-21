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
    "DAILY_LABELS",
    "PHASES",
    "Phase",
    "add_season_doy",
    "common_window",
    "daily_envelope",
    "load_pep725",
    "load_runs",
    "load_simplace_daily",
    "plot_cells",
    "plot_daily",
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


#: Daily series both models can report, with the axis label a plot needs.
DAILY_LABELS: dict[str, str] = {
    "LAI": "LAI [m² m⁻²]",
    "AGB": "AGB [t ha⁻¹]",
    "TRANRF": "TRANRF  (1 = unstressed)",
    "NNI": "NNI  (1 = unstressed)",
    "DVS": "DVS  (2 = mature)",
}


def load_simplace_daily(
    out_dir: Path,
    ids: np.ndarray | None = None,
    years: list[int] | None = None,
    variables: tuple[str, ...] = ("LAI", "AGB", "NNI", "TRANRF"),
    model: str = "simplace",
) -> pd.DataFrame:
    """SIMPLACE's per-cell ``out/daily/<id>_daily.csv`` files, in long form.

    Returns the same columns :func:`cropmodelling4eu.torchcrop.run.daily_batch`
    does, so the two models concatenate into one frame and plot from one code
    path.

    ``model`` labels every row (default ``"simplace"``); the smoke test's own
    IOPT sweep calls this once per solution build and passes
    ``model=f"simplace_iopt{i}"`` so :func:`daily_envelope`'s
    ``groupby("model")`` tells the three builds apart instead of averaging
    them together.

    **The season, not the calendar year, is the unit.** A daily file is one
    continuous run over the whole window with a crop sown every autumn, so a
    season is cut out by ``DVS``: a contiguous block of ``0 < DVS < 2``, labelled
    with the harvest year its **last** day falls in. That matches how torchcrop's
    rows are labelled and needs no assumption about the sowing date — which is a
    *result* under the rule-based solution, not an input.

    ``AGB``, ``NNI`` and ``TRANRF`` are only present if the solution's daily
    output declares them; older runs carry ``LAI`` and ``DevStage`` alone and
    the missing series are reported as such rather than silently dropped.
    """
    out_dir = Path(out_dir)
    daily_dir = out_dir / "daily" if (out_dir / "daily").is_dir() else out_dir
    paths = (
        sorted(daily_dir.glob("*_daily.csv"))
        if ids is None
        else [daily_dir / f"{int(i)}_daily.csv" for i in ids]
    )

    frames: list[pd.DataFrame] = []
    missing_files: list[int] = []
    missing_columns: set[str] = set()
    for path in paths:
        if not path.is_file():
            missing_files.append(int(path.stem.split("_")[0]))
            continue
        raw = pd.read_csv(path, sep=";", parse_dates=["DATE"], dayfirst=True)
        available = [v for v in variables if v in raw.columns]
        missing_columns |= set(variables) - set(available)

        dvs = raw["DevStage"].to_numpy(float)
        crop = (dvs > 0.0) & (dvs < 2.0)
        if not crop.any():
            continue
        # A season is one contiguous run of crop days; the cumulative count of
        # block starts numbers them without a groupby over dates.
        season = np.cumsum(crop & ~np.r_[False, crop[:-1]])
        block = raw.loc[crop].assign(_season=season[crop])
        harvest_year = block.groupby("_season")["DATE"].transform("max").dt.year

        base = pd.DataFrame(
            {
                "model": model,
                "SimplaceID": int(path.stem.split("_")[0]),
                "year": harvest_year.to_numpy(np.int16),
                "date": block["DATE"].to_numpy(),
                "doy": block["DOY"].to_numpy(np.int16),
            },
            index=block.index,
        )
        if years is not None:
            base = base[base["year"].isin(list(years))]
            block = block.loc[base.index]
        if base.empty:
            continue
        # Days after sowing, from the block's own first crop day: the solution
        # sows on a rule, so this is the model's date and not the site table's.
        base["das"] = (
            base["date"] - base.groupby("year")["date"].transform("min")
        ).dt.days.to_numpy(np.int16)
        frames.append(
            pd.concat(
                [
                    base.assign(variable=name,
                                value=block[name].to_numpy(np.float32))
                    for name in available
                ],
                ignore_index=True,
            )
        )

    if missing_files:
        logger.warning(
            "no daily file for %d of %d cells (e.g. %s) -- SIMPLACE writes none "
            "for a cell that never matures", len(missing_files), len(paths),
            missing_files[:3],
        )
    if missing_columns:
        logger.warning(
            "the daily output does not declare %s; add them to the solution's "
            "Daily_crop_growth header and re-run to get them",
            sorted(missing_columns),
        )
    if not frames:
        raise FileNotFoundError(f"no usable daily output under {daily_dir}")
    return pd.concat(frames, ignore_index=True)


def add_season_doy(daily: pd.DataFrame, days_in_year: float = 365.0) -> pd.DataFrame:
    """A day-of-year axis that stays continuous across a season's New Year.

    ``doy`` wraps at 365/1, so a winter-wheat season sown in autumn and
    harvested the following summer would fold its second half back onto the
    start of the axis instead of continuing past it. Rows dated in a season's
    second calendar year get ``days_in_year`` added, so one (model, cell,
    year) season sorts and plots as one continuous run.
    """
    season_start = daily.groupby(["model", "SimplaceID", "year"])["date"].transform("min")
    bumped = daily["doy"].astype(float) + np.where(
        daily["date"].dt.year > season_start.dt.year, days_in_year, 0.0
    )
    return daily.assign(season_doy=bumped)


def daily_envelope(daily: pd.DataFrame, by: str = "das") -> pd.DataFrame:
    """Median and interquartile range across cells, per model, year and day.

    Cells are aggregated on **days after sowing** by default (``by="das"``),
    not on the calendar date: the two models used to disagree about the
    sowing day itself, and a mean over calendar dates would then smear that
    timing difference into an apparent difference in level.

    Now that torchcrop is latched to SIMPLACE's own realized sowing date (see
    ``sowing_table``), the two agree on day zero closely enough that
    ``by="season_doy"`` is meaningful too -- useful for reading a stress
    episode against the actual calendar rather than each model's own count
    from sowing. Call :func:`add_season_doy` on ``daily`` first to add that
    column.
    """
    grouped = daily.groupby(["model", "year", "variable", by], observed=True)
    envelope = grouped["value"].quantile([0.25, 0.5, 0.75]).unstack()
    envelope.columns = ["lo", "median", "hi"]
    return envelope.join(grouped["value"].size().rename("n_cells")).reset_index()


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


#: Fallback categorical colours, keyed ``series_1``... in the project palette.
#: Sized for a handful of model *families* (simplace, torchcrop, ...), not one
#: per run: ``_styles`` below assigns colour by family and cycles past this
#: list rather than erroring if a third or fourth model joins the notebook.
_SERIES_DEFAULTS: tuple[str, ...] = (
    "#4c72b0", "#dd8452", "#55a868", "#c44e52", "#8172b2", "#937860", "#da8bc3",
)

#: Marker and linestyle cycles. Distinct from colour so a run stays legible
#: greyscale, and so runs of the *same* model at different settings (e.g. an
#: IOPT sweep, all "torchcrop") don't need distinct colours to stay tellable
#: apart on an overlapping line plot -- they get one shared colour and their
#: own marker/linestyle instead.
_MARKERS: tuple[str, ...] = ("o", "s", "^", "D", "v", "P", "X", "*")
_LINESTYLES: tuple[str | tuple, ...] = ("-", "--", ":", "-.", (0, (3, 1, 1, 1)))
#: Bar-chart counterpart of marker/linestyle: a bar has no line to dash, so a
#: within-family run is told apart by hatch pattern instead (plot_phenology).
_HATCHES: tuple[str, ...] = ("", "///", "...", "xxx", "\\\\\\", "ooo")

#: Shared figure styling -- bumped from matplotlib's defaults (~10pt text,
#: 1.5pt lines) so a figure is legible read from a notebook cell at normal
#: zoom, not shrunk to a paper column. Every ``plot_*`` function below runs
#: inside ``plt.rc_context(_RC)`` so titles, axis labels, ticks, legends and
#: any line/marker that does not set its own size all pick this up uniformly.
_RC: dict = {
    "font.size": 12,
    "axes.titlesize": 15,
    "axes.labelsize": 13,
    "xtick.labelsize": 11,
    "ytick.labelsize": 11,
    "legend.fontsize": 11,
    "lines.linewidth": 2.2,
    "lines.markersize": 6,
}


def _family(model: str) -> str:
    """The model family a run name belongs to, for colour grouping.

    Run names look like ``simplace_iopt2`` or ``torchcrop_iopt1`` (or bare
    ``simplace``/``torchcrop`` outside an IOPT sweep); the family is the part
    before the first underscore. Every IOPT of one model then shares one
    colour and only the marker/linestyle tell them apart -- so the figure
    reads as "two models" rather than "six unrelated series".
    """
    return model.split("_", 1)[0]


def _series(palette: dict | None, n: int = 3) -> list[str]:
    """The project's categorical colours, in order, with a usable fallback."""
    palette = palette or {}
    return [
        palette.get(f"series_{i}", _SERIES_DEFAULTS[(i - 1) % len(_SERIES_DEFAULTS)])
        for i in range(1, n + 1)
    ]


def _styles(models: list[str], palette: dict | None = None) -> dict[str, dict]:
    """Colour by family, marker and linestyle by run, stable across figures.

    Colour is assigned to each *family* (see :func:`_family`) so an IOPT sweep
    on one side of the comparison is visually "one model, several settings"
    rather than several unrelated colours; marker and linestyle are assigned
    by position *within* a family so its runs stay individually legible.
    Assignment is by first-occurrence order, so the same run gets the same
    look in every plot of a notebook rather than each figure re-deriving its
    own cycle from whichever models happen to be present.
    """
    families = list(dict.fromkeys(_family(m) for m in models))
    family_colour = dict(zip(families, _series(palette, n=len(families))))
    seen: dict[str, int] = {}
    styles = {}
    for model in models:
        family = _family(model)
        i = seen.get(family, 0)
        seen[family] = i + 1
        styles[model] = {
            "color": family_colour[family],
            "marker": _MARKERS[i % len(_MARKERS)],
            "ls": _LINESTYLES[i % len(_LINESTYLES)],
            "hatch": _HATCHES[i % len(_HATCHES)],
        }
    return styles


def plot_cells(cells: pd.DataFrame, palette: dict | None = None):
    """Where the sampled cells sit, over Germany."""
    import matplotlib.pyplot as plt

    with plt.rc_context(_RC):
        fig, ax = plt.subplots(figsize=(5.5, 6.5))
        ax.scatter(
            cells["lon"], cells["lat"], s=70, zorder=3,
            color=(palette or {}).get("primary", "#4c72b0"),
            edgecolor="white", linewidth=1.0,
        )
        for _, cell in cells.iterrows():
            ax.annotate(
                str(cell.get("adm_id", "")), (cell["lon"], cell["lat"]),
                fontsize=9, xytext=(4, 3), textcoords="offset points", alpha=0.75,
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
    with plt.rc_context(_RC):
        fig, axes = plt.subplots(1, 2, figsize=(11, 4.6))
        styles = _styles(list(runs), palette)

        for model, frame in runs.items():
            style = styles[model]
            joined = frame.merge(cells[["SimplaceID", "adm_id"]], on="SimplaceID").merge(
                observed, on=["adm_id", "year"]
            )
            if joined.empty:
                continue
            axes[0].scatter(
                joined["obs_t_ha"], joined["yield_t_ha"], s=22, alpha=0.5,
                label=model, color=style["color"], marker=style["marker"],
            )
            annual = joined.groupby("year")[["yield_t_ha", "obs_t_ha"]].mean()
            axes[1].plot(annual.index, annual["yield_t_ha"], marker=style["marker"],
                         ls=style["ls"], ms=7, lw=2.4, label=f"{model} (sim)",
                         color=style["color"])

        if not observed.empty:
            national = (
                observed[observed["adm_id"].isin(cells["adm_id"])]
                .groupby("year")["obs_t_ha"].mean()
            )
            axes[1].plot(national.index, national.values, color="black", lw=2.8,
                         marker="s", ms=6, label="CyBench (obs)")

        limit = 12.0
        axes[0].plot([0, limit], [0, limit], color="black", lw=1.6, ls="--", zorder=0)
        axes[0].set(xlim=(0, limit), ylim=(0, limit),
                    xlabel="CyBench yield [t/ha]", ylabel="simulated yield [t/ha]",
                    title="Per region and year")
        axes[1].set(xlabel="harvest year", ylabel="yield [t/ha]",
                    title="Mean over the sampled regions")
        for ax in axes:
            # Below the axes, not "best": both panels are densely scattered
            # everywhere, including the corners, so any in-axes legend covers
            # data at this font size. ncol=3 keeps the 7-entry box short.
            ax.legend(frameon=False, ncol=3, loc="upper center",
                      bbox_to_anchor=(0.5, -0.14), fontsize=_RC["legend.fontsize"] - 1)
            ax.grid(alpha=0.3)
        fig.tight_layout()
    return fig, axes


def plot_daily(
    daily: pd.DataFrame,
    years: list[int],
    variables: tuple[str, ...] = ("LAI", "AGB", "TRANRF", "NNI"),
    palette: dict | None = None,
    by: str = "das",
):
    """One column per season, one row per variable; median and IQR over cells.

    The band is the spread **across cells**, not an uncertainty: 30 cells from
    the Rhine to Mecklenburg do not have the same season, and a narrow band
    where the other model's is wide is itself the result.

    ``by`` selects the x-axis: ``"das"`` (default, days after each model's own
    realized sowing) or ``"season_doy"`` (the actual calendar day-of-year,
    continuous across New Year -- see :func:`add_season_doy`, run
    automatically here if the column is missing). ``"season_doy"`` reads a
    stress episode against the calendar itself rather than each model's own
    count from sowing, and ticks are formatted as month/day.
    """
    import matplotlib.pyplot as plt
    from matplotlib.ticker import FuncFormatter

    from .doy import doy_to_month_day

    if by == "season_doy" and "season_doy" not in daily.columns:
        daily = add_season_doy(daily)

    envelope = daily_envelope(daily, by=by)
    models = list(dict.fromkeys(daily["model"]))
    styles = _styles(models, palette)

    with plt.rc_context(_RC):
        fig, axes = plt.subplots(
            len(variables), len(years), figsize=(6.2 * len(years), 3.1 * len(variables)),
            sharex=True, squeeze=False,
        )
        for row, variable in enumerate(variables):
            for column, year in enumerate(years):
                ax = axes[row][column]
                for model in models:
                    style = styles[model]
                    series = envelope[
                        (envelope["model"] == model)
                        & (envelope["year"] == year)
                        & (envelope["variable"] == variable)
                    ].sort_values(by)
                    if series.empty:
                        continue
                    ax.fill_between(series[by], series["lo"], series["hi"],
                                    color=style["color"], alpha=0.15, lw=0)
                    # markevery, not every day: a marker on each of ~200 daily
                    # points would be a solid line of dots, defeating the point.
                    ax.plot(series[by], series["median"], color=style["color"],
                            lw=2.4, ls=style["ls"], marker=style["marker"], ms=6,
                            markevery=14,
                            label=model if row == 0 and column == 0 else None)
                if variable in ("TRANRF", "NNI"):
                    ax.axhline(1.0, color="black", lw=1.2, ls=":", zorder=0)
                    ax.set_ylim(-0.05, 1.15)
                ax.grid(alpha=0.3)
                if row == 0:
                    ax.set_title(f"{year} season", fontsize=14)
                if column == 0:
                    ax.set_ylabel(DAILY_LABELS.get(variable, variable), fontsize=12)
                if row == len(variables) - 1:
                    if by == "season_doy":
                        ax.set_xlabel("date")
                        ax.xaxis.set_major_formatter(
                            FuncFormatter(lambda v, _: doy_to_month_day(v))
                        )
                    else:
                        ax.set_xlabel("days after sowing")

        handles, labels = axes[0][0].get_legend_handles_labels()
        fig.legend(handles, labels, frameon=False, ncol=len(models),
                   loc="upper center", bbox_to_anchor=(0.5, 1.02))
        fig.tight_layout()
    return fig, axes


def plot_phenology(phenology: pd.DataFrame, palette: dict | None = None):
    """Stage-date bias per model, with the not-a-prediction rows marked."""
    import matplotlib.pyplot as plt

    if phenology.empty:
        raise ValueError("no stage could be paired, so there is nothing to plot")

    stages = list(dict.fromkeys(phenology["stage"]))
    models = list(dict.fromkeys(phenology["model"]))
    styles = _styles(models, palette)
    positions = np.arange(len(stages))
    width = 0.8 / max(len(models), 1)

    with plt.rc_context(_RC):
        fig, ax = plt.subplots(figsize=(10, 5))
        for offset, model in enumerate(models):
            rows = phenology[phenology["model"] == model].set_index("stage")
            bias = [rows["bias"].get(stage, np.nan) for stage in stages]
            error = [rows["rmse"].get(stage, np.nan) for stage in stages]
            # A bar has no line to dash, so within one colour (one model
            # family) a run is told apart by hatch instead of marker/linestyle
            # -- see _HATCHES. edgecolor is what the hatch lines draw in.
            ax.bar(
                positions + offset * width - 0.4 + width / 2, bias, width * 0.92,
                yerr=error, capsize=4, label=model, color=styles[model]["color"],
                hatch=styles[model]["hatch"], edgecolor="black", linewidth=0.7,
                error_kw={"alpha": 0.5, "lw": 1.5},
            )

        ax.axhline(0, color="black", lw=1.6)
        ax.set_xticks(positions)
        ax.set_xticklabels([s.replace(" (", "\n(") for s in stages])
        ax.set(ylabel="bias [days]  (simulated − observed)",
               title="Stage dates vs PEP725; bars are bias, whiskers RMSE")
        ax.legend(frameon=False)
        ax.grid(alpha=0.3, axis="y")
        fig.text(
            0.01, -0.04,
            "Sowing is a constant input, not a prediction. Soft dough precedes the "
            "full ripeness the models report, so a positive bias there is expected.",
            fontsize=10, alpha=0.75,
        )
        fig.tight_layout()
    return fig, ax
