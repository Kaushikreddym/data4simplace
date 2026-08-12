"""Publication figures.

Every function returns a :class:`~matplotlib.figure.Figure` and draws nothing
to screen, so a notebook cell can save it, restyle it or drop it. Colours come
from :mod:`utils.style` by role.

Two conventions run through the module:

* **Identity is never colour with 23 countries on screen.** The categorical
  palette stops at eight; past that a hue is not separable. Country identity is
  carried by position instead — a sorted axis, a small-multiple panel, a direct
  label on the extremes.
* **Dates are unwrapped before plotting.** A scatter axis is linear, so a
  day-of-year is shifted by whole years about the observed circular mean first
  (:func:`utils.doy.unwrap_doy`); the ticks are then relabelled as calendar
  dates so the reader never sees the shifted number.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Sequence

import cartopy.crs as ccrs
import cartopy.feature as cfeature
import geopandas as gpd
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.colors import BoundaryNorm, ListedColormap
from matplotlib.figure import Figure
from matplotlib.lines import Line2D
from matplotlib.patches import Patch
from matplotlib.ticker import FuncFormatter, MaxNLocator

from .config import COUNTRY_NAMES, FIGURE_DIR
from .doy import circular_mean_doy, doy_to_month_day, unwrap_doy
from .metrics import METRIC_LABELS
from .style import DIVERGING_CMAP, SEQUENTIAL_CMAP, diverging_norm, palette

logger = logging.getLogger(__name__)

__all__ = [
    "EUROPE_EXTENT",
    "EUROPE_PROJECTION",
    "bias_bars",
    "category_map",
    "cell_map",
    "component_bars",
    "country_choropleth",
    "metric_table",
    "residual_panels",
    "save",
    "scatter_density",
    "scatter_one_to_one",
    "scatter_small_multiples",
    "timeseries_pair",
    "timeseries_small_multiples",
]

#: ETRS89-LAEA, the Eurostat standard for pan-European maps. Equal-area, so a
#: Finnish cell and a Greek one occupy the area they should — which matters
#: here, where the reader is judging how much of the domain a bias covers.
EUROPE_PROJECTION = ccrs.LambertAzimuthalEqualArea(
    central_longitude=10.0, central_latitude=52.0
)

#: ``(lon_min, lon_max, lat_min, lat_max)`` in plate carrée.
EUROPE_EXTENT: tuple[float, float, float, float] = (-11.0, 33.0, 34.0, 70.0)


# --------------------------------------------------------------------------- #
# Helpers
# --------------------------------------------------------------------------- #


def save(fig: Figure, name: str, directory: Path | None = None) -> Path:
    """Write a figure to the project figure directory as PNG and PDF.

    Both formats on purpose: the PNG for the notebook and for review, the
    vector PDF for the manuscript.

    Args:
        fig: The figure.
        name: Base name, without extension.
        directory: Target directory; :data:`utils.config.FIGURE_DIR` by default.

    Returns:
        Path of the written PNG.
    """
    directory = directory or FIGURE_DIR
    directory.mkdir(parents=True, exist_ok=True)
    png = directory / f"{name}.png"
    fig.savefig(png)
    fig.savefig(directory / f"{name}.pdf")
    logger.info("wrote %s(.png/.pdf)", directory / name)
    return png


def _country_label(code: str) -> str:
    """``"DE"`` -> ``"Germany (DE)"``, falling back to the bare code."""
    name = COUNTRY_NAMES.get(code)
    return f"{name} ({code})" if name else str(code)


def _emptiest_corner(
    ax: plt.Axes, x: np.ndarray, y: np.ndarray
) -> tuple[float, float, str, str]:
    """Pick the corner of ``ax`` holding the fewest points.

    A fixed corner is wrong for this project: an under-predicting model fills
    the lower right of a 1:1 scatter, and a constant-valued simulation (the
    DOY 270 sowing latch) fills a whole horizontal band. Counting first puts the
    metric block where it does not sit on the data.

    Returns:
        ``(x_frac, y_frac, ha, va)`` for an axes-fraction ``ax.text`` call.
    """
    corners = {
        (0.03, 0.97, "left", "top"): (0.0, 0.5, 0.5, 1.0),
        (0.97, 0.97, "right", "top"): (0.5, 1.0, 0.5, 1.0),
        (0.03, 0.03, "left", "bottom"): (0.0, 0.5, 0.0, 0.5),
        (0.97, 0.03, "right", "bottom"): (0.5, 1.0, 0.0, 0.5),
    }
    (x0, x1), (y0, y1) = ax.get_xlim(), ax.get_ylim()
    if x1 == x0 or y1 == y0 or x.size == 0:
        return (0.97, 0.03, "right", "bottom")

    fx, fy = (x - x0) / (x1 - x0), (y - y0) / (y1 - y0)
    counts = {
        anchor: int(np.sum((fx >= lo_x) & (fx <= hi_x) & (fy >= lo_y) & (fy <= hi_y)))
        for anchor, (lo_x, hi_x, lo_y, hi_y) in corners.items()
    }
    # Ties keep dict order, so the lower right stays the default when the data
    # leaves every corner equally free.
    return min(counts, key=lambda a: (counts[a], list(corners).index(a)))


def _annotate_metrics(
    ax: plt.Axes,
    stats: dict[str, float],
    keys: Sequence[str],
    p: dict[str, str],
    anchor: tuple[float, float, str, str] = (0.97, 0.03, "right", "bottom"),
) -> None:
    """Put a small metric block in one corner of an axes.

    A block, not a title: the numbers belong to the marks, and a reader
    comparing panels wants them in the same place on every one. It sits on a
    surface-coloured patch so it stays legible if the corner is not quite empty.

    Args:
        ax: Target axes.
        stats: Metric dict.
        keys: Metrics to show, in order.
        p: Palette.
        anchor: ``(x_frac, y_frac, ha, va)``, usually from
            :func:`_emptiest_corner`.
    """
    lines = []
    for key in keys:
        value = stats.get(key, float("nan"))
        if key == "n":
            lines.append(f"n = {int(value)}")
        elif np.isfinite(value):
            lines.append(f"{METRIC_LABELS.get(key, key)} = {value:.2f}")
    x_frac, y_frac, ha, va = anchor
    ax.text(
        x_frac, y_frac, "\n".join(lines), transform=ax.transAxes,
        ha=ha, va=va, fontsize=plt.rcParams["font.size"] - 1.5,
        color=p["secondary"], linespacing=1.45, zorder=6,
        bbox={"facecolor": p["surface"], "edgecolor": "none",
              "alpha": 0.82, "pad": 2.5},
    )


def _square_limits(ax: plt.Axes, values: np.ndarray, pad: float = 0.06) -> None:
    """Give both axes the same range, so the 1:1 line is the diagonal."""
    finite = values[np.isfinite(values)]
    if finite.size == 0:
        return
    lo, hi = float(finite.min()), float(finite.max())
    margin = (hi - lo) * pad or 1.0
    ax.set_xlim(lo - margin, hi + margin)
    ax.set_ylim(lo - margin, hi + margin)
    ax.set_aspect("equal", adjustable="box")


def _doy_axis(ax: plt.Axes) -> None:
    """Relabel a linear axis of unwrapped days-of-year as calendar dates."""
    formatter = FuncFormatter(lambda v, _: doy_to_month_day(v))
    ax.xaxis.set_major_locator(MaxNLocator(5))
    ax.yaxis.set_major_locator(MaxNLocator(5))
    ax.xaxis.set_major_formatter(formatter)
    ax.yaxis.set_major_formatter(formatter)


def _prepare_pair(
    frame: pd.DataFrame, obs_col: str, sim_col: str, circular: bool
) -> tuple[np.ndarray, np.ndarray, float | None]:
    """Observed/simulated arrays ready to plot, unwrapped if they are dates.

    Returns:
        ``(obs, sim, centre)``; ``centre`` is the observed circular mean the
        dates were unwrapped about, or ``None`` for a linear quantity.
    """
    obs = frame[obs_col].to_numpy(dtype=float)
    sim = frame[sim_col].to_numpy(dtype=float)
    keep = np.isfinite(obs) & np.isfinite(sim)
    obs, sim = obs[keep], sim[keep]
    if not circular or obs.size == 0:
        return obs, sim, None
    centre = circular_mean_doy(obs)
    return unwrap_doy(obs, centre), unwrap_doy(sim, centre), centre


# --------------------------------------------------------------------------- #
# Scatter
# --------------------------------------------------------------------------- #


def scatter_one_to_one(
    frame: pd.DataFrame,
    obs_col: str,
    sim_col: str,
    stats: dict[str, float] | None = None,
    metric_keys: Sequence[str] = ("n", "bias", "rmse", "r2", "pearson_r"),
    title: str = "",
    xlabel: str = "Observed",
    ylabel: str = "Simulated",
    circular: bool = False,
    mode: str = "light",
    figsize: tuple[float, float] = (4.6, 4.6),
    highlight: Sequence[str] | None = None,
    group_col: str = "country",
) -> Figure:
    """Observed against simulated, with the 1:1 line.

    One series, one hue: with every country-year pooled into a single cloud,
    colour has no identity job to do, so it is spent on density instead
    (translucent marks with a surface-coloured ring, so overlaps read).

    Args:
        frame: Paired rows.
        obs_col: Observation column.
        sim_col: Simulation column.
        stats: Metric dict to annotate; nothing is drawn if ``None``.
        metric_keys: Which metrics to show, in order.
        title: Axes title.
        xlabel: X label, including units.
        ylabel: Y label, including units.
        circular: ``True`` when the columns are days-of-year.
        mode: Style mode.
        figsize: Figure size in inches.
        highlight: Group values to label directly — the extremes, not every
            point. Their marks are drawn in the second categorical hue.
        group_col: Column ``highlight`` refers to.

    Returns:
        The figure.
    """
    p = palette(mode)
    obs, sim, _ = _prepare_pair(frame, obs_col, sim_col, circular)

    fig, ax = plt.subplots(figsize=figsize)
    ax.grid(True, linewidth=0.7)

    ax.scatter(
        obs, sim, s=16, c=p["series_1"], alpha=0.45,
        linewidths=0.6, edgecolors=p["surface"], zorder=2, label="Country-year",
    )

    both = np.concatenate([obs, sim]) if obs.size else np.array([0.0, 1.0])
    _square_limits(ax, both)
    lo, hi = ax.get_xlim()
    ax.plot([lo, hi], [lo, hi], color=p["baseline"], linewidth=1.2, zorder=1)
    ax.annotate(
        "1:1", xy=(hi, hi), xytext=(-4, 4), textcoords="offset points",
        ha="right", va="bottom", color=p["muted"],
        fontsize=plt.rcParams["font.size"] - 1.5,
    )

    if highlight is not None and group_col in frame.columns:
        picked = frame[frame[group_col].isin(list(highlight))]
        if not picked.empty:
            h_obs, h_sim, _ = _prepare_pair(picked, obs_col, sim_col, circular)
            ax.scatter(
                h_obs, h_sim, s=22, c=p["series_2"], alpha=0.85,
                linewidths=0.8, edgecolors=p["surface"], zorder=3,
                label=", ".join(map(str, highlight)),
            )
            ax.legend(loc="upper left", labelcolor=p["secondary"])

    if circular:
        _doy_axis(ax)
    if stats:
        _annotate_metrics(ax, stats, metric_keys, p, _emptiest_corner(ax, obs, sim))

    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    fig.tight_layout()
    return fig


def scatter_density(
    frame: pd.DataFrame,
    obs_col: str,
    sim_col: str,
    stats: dict[str, float] | None = None,
    metric_keys: Sequence[str] = ("n", "bias", "rmse", "r2", "pearson_r"),
    title: str = "",
    xlabel: str = "Observed",
    ylabel: str = "Simulated",
    circular: bool = False,
    mode: str = "light",
    figsize: tuple[float, float] = (5.0, 4.6),
    gridsize: int = 55,
    log_counts: bool = True,
    cbar_label: str = "Cell-years per bin",
) -> Figure:
    """Observed against simulated as a binned density, with the 1:1 line.

    The gridded comparisons pair tens of thousands of cell-years, and at that
    count a scatter is a silhouette: the marks saturate, and the eye reads the
    outline of the cloud rather than where the mass sits. Binning the plane and
    encoding the count restores that — and the count is a magnitude, so it gets
    the single-hue sequential ramp.

    Args:
        frame: Paired rows.
        obs_col: Observation column.
        sim_col: Simulation column.
        stats: Metric dict to annotate; nothing is drawn if ``None``.
        metric_keys: Which metrics to show, in order.
        title: Axes title.
        xlabel: X label, including units.
        ylabel: Y label, including units.
        circular: ``True`` when the columns are days-of-year.
        mode: Style mode.
        figsize: Figure size in inches.
        gridsize: Hexagons across the x range. Higher resolves more structure
            and empties more bins.
        log_counts: Colour on a log count scale. On by default: cell-year
            densities span three orders of magnitude, and a linear ramp would
            paint everything but the mode as empty.
        cbar_label: Colour-bar label.

    Returns:
        The figure.
    """
    p = palette(mode)
    obs, sim, _ = _prepare_pair(frame, obs_col, sim_col, circular)

    fig, ax = plt.subplots(figsize=figsize)
    ax.grid(True, linewidth=0.7)

    both = np.concatenate([obs, sim]) if obs.size else np.array([0.0, 1.0])
    _square_limits(ax, both)
    extent = (*ax.get_xlim(), *ax.get_ylim())

    hexes = ax.hexbin(
        obs, sim, gridsize=gridsize, extent=extent, cmap=SEQUENTIAL_CMAP,
        bins="log" if log_counts else None, mincnt=1, linewidths=0.0, zorder=2,
    )

    lo, hi = ax.get_xlim()
    ax.plot([lo, hi], [lo, hi], color=p["baseline"], linewidth=1.2, zorder=3)
    ax.annotate(
        "1:1", xy=(hi, hi), xytext=(-4, 4), textcoords="offset points",
        ha="right", va="bottom", color=p["muted"],
        fontsize=plt.rcParams["font.size"] - 1.5, zorder=4,
    )

    bar = fig.colorbar(hexes, ax=ax, shrink=0.85, pad=0.02, aspect=26)
    bar.set_label(cbar_label, color=p["secondary"])
    bar.outline.set_visible(False)
    bar.ax.tick_params(color=p["muted"], labelcolor=p["secondary"])

    if circular:
        _doy_axis(ax)
    if stats:
        # The density fills the diagonal band, so the emptiest corner is one of
        # the off-diagonal two whatever the bias sign.
        _annotate_metrics(ax, stats, metric_keys, p, _emptiest_corner(ax, obs, sim))

    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    fig.tight_layout()
    return fig


def scatter_small_multiples(
    frame: pd.DataFrame,
    obs_col: str,
    sim_col: str,
    metrics: pd.DataFrame | None = None,
    group_col: str = "country",
    order: Sequence[str] | None = None,
    n_cols: int = 6,
    circular: bool = False,
    title: str = "",
    xlabel: str = "Observed",
    ylabel: str = "Simulated",
    mode: str = "light",
    panel_size: float = 1.75,
    annotate: Sequence[str] = ("bias", "rmse"),
) -> Figure:
    """One 1:1 panel per country — the honest alternative to 23 hues.

    Every panel shares one hue and one pair of axis limits, so the panels are
    comparable at a glance and a country's offset from the diagonal is the
    signal.

    Args:
        frame: Paired rows.
        obs_col: Observation column.
        sim_col: Simulation column.
        metrics: Per-country metric table (from
            :func:`utils.metrics.metrics_by_group`) used for the per-panel
            annotation and, when ``order`` is ``None``, for the panel order.
        group_col: Column holding the country code.
        order: Explicit panel order. ``None`` sorts by the metric table's index
            order if given, else alphabetically.
        n_cols: Panels per row.
        circular: ``True`` for days-of-year.
        title: Figure suptitle.
        xlabel: Shared X label.
        ylabel: Shared Y label.
        mode: Style mode.
        panel_size: Side of one panel in inches.
        annotate: Metrics printed in each panel.

    Returns:
        The figure.
    """
    p = palette(mode)
    if order is None:
        order = (
            [c for c in metrics.index if c != "ALL"]
            if metrics is not None
            else sorted(frame[group_col].dropna().unique())
        )
    order = [c for c in order if c in set(frame[group_col])]

    n_rows = int(np.ceil(len(order) / n_cols))
    fig, axes = plt.subplots(
        n_rows, n_cols,
        figsize=(n_cols * panel_size + 1.0, n_rows * panel_size + 1.0),
        squeeze=False, sharex=True, sharey=True,
    )

    # One shared range across every panel; per-panel limits would let a
    # low-yielding country look as well-fitted as a high-yielding one.
    all_obs, all_sim, _ = _prepare_pair(frame, obs_col, sim_col, circular)
    span = np.concatenate([all_obs, all_sim]) if all_obs.size else np.array([0.0, 1.0])

    for ax, code in zip(axes.ravel(), order):
        block = frame[frame[group_col] == code]
        obs, sim, _ = _prepare_pair(block, obs_col, sim_col, circular)
        ax.scatter(
            obs, sim, s=11, c=p["series_1"], alpha=0.6,
            linewidths=0.5, edgecolors=p["surface"], zorder=2,
        )
        _square_limits(ax, span)
        lo, hi = ax.get_xlim()
        ax.plot([lo, hi], [lo, hi], color=p["baseline"], linewidth=1.0, zorder=1)
        ax.set_title(str(code), fontsize=plt.rcParams["font.size"] - 0.5)
        if metrics is not None and code in metrics.index:
            _annotate_metrics(ax, metrics.loc[code].to_dict(), annotate, p,
                              _emptiest_corner(ax, obs, sim))
        if circular:
            _doy_axis(ax)
            ax.tick_params(labelrotation=30)

    for ax in axes.ravel()[len(order):]:
        ax.set_visible(False)

    fig.supxlabel(xlabel, color=p["secondary"], fontsize=plt.rcParams["font.size"])
    fig.supylabel(ylabel, color=p["secondary"], fontsize=plt.rcParams["font.size"])
    if title:
        fig.suptitle(title, x=0.01, ha="left", color=p["primary"],
                     fontsize=plt.rcParams["font.size"] + 2)
    fig.tight_layout()
    return fig


# --------------------------------------------------------------------------- #
# Bias
# --------------------------------------------------------------------------- #


def bias_bars(
    metrics: pd.DataFrame,
    value_col: str = "bias",
    error_col: str | None = None,
    title: str = "",
    xlabel: str = "Bias",
    mode: str = "light",
    figsize: tuple[float, float] | None = None,
    label_extremes: int = 3,
    drop_pooled: bool = True,
) -> Figure:
    """Country bias as a sorted horizontal bar chart.

    Bias has a meaningful zero and a meaningful sign, so the fill carries the
    **sign** — cool for under-prediction, warm for over-prediction — as two
    flat colours from the diverging poles. Not a continuous ramp: the bar's
    length already encodes magnitude, and shading it darker-where-longer would
    spend the colour channel restating what the axis says. Sorting turns the
    country axis into the ordering channel, which is what makes 23 categories
    readable without 23 hues.

    Args:
        metrics: Per-country metric table, indexed by country code.
        value_col: Column to plot.
        error_col: Optional column drawn as a symmetric whisker — pass
            ``"rmse"`` to show the scatter behind the offset.
        title: Axes title.
        xlabel: X label, including units.
        mode: Style mode.
        figsize: Overridden by default to fit the number of countries.
        label_extremes: How many bars at each end carry a value label. Labelling
            every bar restates the axis and goes unread.
        drop_pooled: Drop the pooled ``ALL`` row, which is not a country.

    Returns:
        The figure.
    """
    p = palette(mode)
    table = metrics.copy()
    if drop_pooled and "ALL" in table.index:
        table = table.drop(index="ALL")
    table = table[table[value_col].notna()].sort_values(value_col)

    values = table[value_col].to_numpy(dtype=float)
    # The two poles of the diverging ramp, taken at a step that clears contrast
    # on both surfaces. Sign only — magnitude is the bar.
    under, over = DIVERGING_CMAP(0.18), DIVERGING_CMAP(0.86)
    colors = [over if v >= 0 else under for v in values]
    y = np.arange(len(table))

    errors = (
        table[error_col].to_numpy(dtype=float)
        if error_col and error_col in table.columns else None
    )

    fig, ax = plt.subplots(figsize=figsize or (5.6, 0.24 * len(table) + 1.6))
    ax.barh(y, values, height=0.62, color=colors, zorder=2)
    if errors is not None:
        ax.errorbar(
            values, y, xerr=errors, fmt="none", ecolor=p["muted"],
            elinewidth=0.9, capsize=2.0, zorder=3,
        )

    ax.axvline(0.0, color=p["baseline"], linewidth=1.0, zorder=1)
    ax.set_yticks(y, [str(c) for c in table.index])
    ax.set_ylim(-0.8, len(table) - 0.2)
    ax.grid(True, axis="x", linewidth=0.7)
    ax.grid(False, axis="y")

    legend_handles = [
        Patch(facecolor=over, label="Over-predicted"),
        Patch(facecolor=under, label="Under-predicted"),
    ]
    if errors is not None:
        legend_handles.append(
            Line2D([0], [0], color=p["muted"], linewidth=0.9,
                   label=f"± {METRIC_LABELS.get(error_col, error_col)}")
        )
    ax.legend(handles=legend_handles, loc="lower left", labelcolor=p["secondary"],
              ncol=len(legend_handles), bbox_to_anchor=(0.0, 1.005))

    if label_extremes:
        # Labels ride the whisker end, not the bar end, so they never sit on
        # top of the error bar they are meant to be read beside.
        tips = values if errors is None else values + np.sign(values) * errors
        edges = list(range(min(label_extremes, len(table)))) + list(
            range(max(0, len(table) - label_extremes), len(table))
        )
        for i in sorted(set(edges)):
            ax.annotate(
                f"{values[i]:+.2f}", xy=(tips[i], y[i]),
                xytext=(5 if values[i] >= 0 else -5, 0), textcoords="offset points",
                ha="left" if values[i] >= 0 else "right", va="center",
                color=p["secondary"], fontsize=plt.rcParams["font.size"] - 1.5,
            )
        ax.margins(x=0.12)

    ax.set_xlabel(xlabel)
    # Extra pad so the title clears the legend strip sitting above the axes.
    ax.set_title(title, pad=24.0)
    fig.tight_layout()
    return fig


def component_bars(
    frame: pd.DataFrame,
    columns: Sequence[str],
    labels: Sequence[str] | None = None,
    sort_by: str | None = None,
    title: str = "",
    xlabel: str = "",
    mode: str = "light",
    figsize: tuple[float, float] | None = None,
    drop_pooled: bool = False,
) -> Figure:
    """Two or three signed quantities per group, as grouped horizontal bars.

    Written for an error *decomposition*, where the point is that the columns
    add up: a simulated harvest date is wrong because the crop was sown on the
    wrong day, because it grew for the wrong number of days, or both, and the
    reader needs to see which. Here identity is only two or three categories,
    so it can be carried by the categorical hues.

    Args:
        frame: One row per group, indexed by the group label.
        columns: Columns to draw, in order. Two or three; past that the bars
            are too thin to compare.
        labels: Legend labels; the column names by default.
        sort_by: Column to order the groups by. ``None`` keeps the frame order.
        title: Axes title.
        xlabel: X label, including units.
        mode: Style mode.
        figsize: Overridden by default to fit the number of groups.
        drop_pooled: Drop an ``ALL`` row, which is not a group.

    Returns:
        The figure.
    """
    p = palette(mode)
    table = frame.copy()
    if drop_pooled and "ALL" in table.index:
        table = table.drop(index="ALL")
    if sort_by is not None:
        table = table.sort_values(sort_by)

    labels = list(labels or columns)
    n_series = len(columns)
    height = 0.8 / n_series
    y = np.arange(len(table))

    fig, ax = plt.subplots(
        figsize=figsize or (5.8, 0.30 * n_series * len(table) / 2 + 1.8)
    )
    for i, (col, label) in enumerate(zip(columns, labels)):
        # Top-to-bottom in legend order: the offset runs negative so the first
        # series sits at the top of each group, matching the legend's reading.
        offset = (n_series - 1) / 2.0 - i
        ax.barh(
            y + offset * height, table[col].to_numpy(dtype=float), height=height,
            color=p[f"series_{i + 1}"], label=label, zorder=2,
        )

    ax.axvline(0.0, color=p["baseline"], linewidth=1.0, zorder=1)
    ax.set_yticks(y, [str(i) for i in table.index])
    ax.set_ylim(-0.7, len(table) - 0.3)
    ax.grid(True, axis="x", linewidth=0.7)
    ax.grid(False, axis="y")
    ax.legend(loc="lower left", labelcolor=p["secondary"], ncol=n_series,
              bbox_to_anchor=(0.0, 1.005))
    ax.set_xlabel(xlabel)
    ax.set_title(title, pad=24.0)
    fig.tight_layout()
    return fig


def residual_panels(
    frame: pd.DataFrame,
    residual_col: str,
    obs_col: str,
    year_col: str = "year",
    group_col: str = "country",
    title: str = "",
    ylabel: str = "Residual (simulated - observed)",
    xlabel_obs: str = "Observed",
    mode: str = "light",
    figsize: tuple[float, float] = (10.0, 3.4),
    n_bins: int = 12,
) -> Figure:
    """Residuals against magnitude, against time, and as a distribution.

    Three views because they fail differently: panel 1 catches a slope error
    (residuals trending with the observation), panel 2 catches drift or a bad
    year, panel 3 shows whether the offset is a shift or a spread. A binned
    median rides panels 1 and 2 — with thousands of translucent points, the
    eye reads density, not centre.

    Args:
        frame: Paired rows carrying the residual.
        residual_col: Residual column (simulated - observed).
        obs_col: Observation column for the x axis of panel 1.
        year_col: Year column for panel 2.
        group_col: Unused for colour; kept so callers can pass a full frame.
        title: Figure suptitle.
        ylabel: Shared Y label.
        xlabel_obs: X label of panel 1.
        mode: Style mode.
        figsize: Figure size in inches.
        n_bins: Bins for the median line of panel 1.

    Returns:
        The figure.
    """
    p = palette(mode)
    fig, axes = plt.subplots(1, 3, figsize=figsize,
                             gridspec_kw={"width_ratios": [1.0, 1.0, 0.6]})

    resid = frame[residual_col].to_numpy(dtype=float)

    for i, (ax, x_col, xlabel) in enumerate(((axes[0], obs_col, xlabel_obs),
                                             (axes[1], year_col, "Harvest year"))):
        x = frame[x_col].to_numpy(dtype=float)
        keep = np.isfinite(x) & np.isfinite(resid)
        ax.scatter(x[keep], resid[keep], s=13, c=p["series_1"], alpha=0.35,
                   linewidths=0.5, edgecolors=p["surface"], zorder=2)
        ax.axhline(0.0, color=p["baseline"], linewidth=1.0, zorder=1)

        if keep.sum() > n_bins:
            edges = np.quantile(x[keep], np.linspace(0, 1, n_bins + 1))
            edges = np.unique(edges)
            if edges.size > 2:
                idx = np.clip(np.digitize(x[keep], edges[1:-1]), 0, edges.size - 2)
                centres = 0.5 * (edges[:-1] + edges[1:])
                medians = np.array([
                    np.median(resid[keep][idx == b]) if np.any(idx == b) else np.nan
                    for b in range(edges.size - 1)
                ])
                ax.plot(centres, medians, color=p["series_2"], linewidth=2.0,
                        zorder=4, label="Binned median")
                # One legend for the figure, not one per panel: both panels
                # carry the same two marks.
                if i == 0:
                    ax.legend(loc="best", labelcolor=p["secondary"])
        ax.set_xlabel(xlabel)

    finite = resid[np.isfinite(resid)]
    axes[2].hist(finite, bins=30, orientation="horizontal",
                 color=p["series_1"], alpha=0.75, zorder=2)
    axes[2].axhline(0.0, color=p["baseline"], linewidth=1.0, zorder=1)
    if finite.size:
        axes[2].axhline(float(np.mean(finite)), color=p["series_2"],
                        linewidth=2.0, zorder=3)
        axes[2].annotate(
            f"mean {np.mean(finite):+.2f}", xy=(0.95, float(np.mean(finite))),
            xycoords=("axes fraction", "data"), xytext=(0, 6),
            textcoords="offset points", ha="right", color=p["secondary"],
            fontsize=plt.rcParams["font.size"] - 1.5,
        )
    axes[2].set_xlabel("Count")
    axes[2].grid(False, axis="y")

    axes[0].set_ylabel(ylabel)
    for ax in axes[1:]:
        ax.tick_params(labelleft=False)
    if title:
        fig.suptitle(title, x=0.01, ha="left", color=p["primary"],
                     fontsize=plt.rcParams["font.size"] + 2)
    fig.tight_layout()
    return fig


def timeseries_pair(
    frame: pd.DataFrame,
    obs_col: str,
    sim_col: str,
    year_col: str = "year",
    spread: tuple[float, float] | None = (0.25, 0.75),
    stats: dict[str, float] | None = None,
    metric_keys: Sequence[str] = ("n", "bias", "rmse", "pearson_r"),
    title: str = "",
    ylabel: str = "",
    mode: str = "light",
    figsize: tuple[float, float] = (7.2, 3.4),
) -> Figure:
    """Domain-mean observed and simulated through time, with a spread band.

    The band is the *spatial* spread across cells in that year, not a
    confidence interval, and it is drawn because the two means can agree while
    the fields behind them do not overlap at all. Both series are labelled at
    their last point as well as in the legend, so the figure survives being read
    in greyscale.

    Args:
        frame: Paired rows, several per year.
        obs_col: Observation column.
        sim_col: Simulation column.
        year_col: Year column.
        spread: Quantile pair drawn as a band around each series. ``None``
            draws the means only.
        stats: Metric dict to annotate; nothing is drawn if ``None``.
        metric_keys: Which metrics to show, in order.
        title: Axes title.
        ylabel: Y label, including units.
        mode: Style mode.
        figsize: Figure size in inches.

    Returns:
        The figure.
    """
    p = palette(mode)
    grouped = frame.groupby(year_col, sort=True)
    years = np.asarray(sorted(frame[year_col].dropna().unique()), dtype=float)

    fig, ax = plt.subplots(figsize=figsize)
    ax.grid(True, linewidth=0.7)

    for col, colour, label in (
        (obs_col, p["series_1"], "Observed"),
        (sim_col, p["series_2"], "Simulated"),
    ):
        means = grouped[col].mean().reindex(sorted(grouped.groups)).to_numpy(dtype=float)
        if spread is not None:
            lower = grouped[col].quantile(spread[0]).to_numpy(dtype=float)
            upper = grouped[col].quantile(spread[1]).to_numpy(dtype=float)
            ax.fill_between(years, lower, upper, color=colour, alpha=0.16,
                            linewidth=0.0, zorder=1)
        ax.plot(years, means, color=colour, linewidth=2.0, zorder=3, label=label)
        if means.size and np.isfinite(means[-1]):
            ax.annotate(
                label, xy=(years[-1], means[-1]), xytext=(6, 0),
                textcoords="offset points", ha="left", va="center",
                color=p["secondary"], fontsize=plt.rcParams["font.size"] - 1.5,
            )

    ax.legend(loc="lower left", labelcolor=p["secondary"], ncol=2,
              bbox_to_anchor=(0.0, 1.005))
    ax.margins(x=0.08)
    if stats:
        _annotate_metrics(ax, stats, metric_keys, p)
    ax.set_xlabel("Harvest year")
    ax.set_ylabel(ylabel)
    ax.set_title(title, pad=24.0)
    fig.tight_layout()
    return fig


def timeseries_small_multiples(
    frame: pd.DataFrame,
    obs_col: str,
    sim_col: str,
    year_col: str = "year",
    group_col: str = "country",
    order: Sequence[str] | None = None,
    n_cols: int = 6,
    title: str = "",
    ylabel: str = "",
    mode: str = "light",
    panel_size: tuple[float, float] = (1.9, 1.35),
) -> Figure:
    """Observed and simulated time series, one panel per country.

    Two series, so a legend is present; it sits once at the figure level rather
    than in every panel.

    Args:
        frame: Paired rows.
        obs_col: Observation column.
        sim_col: Simulation column.
        year_col: Year column.
        group_col: Country column.
        order: Panel order; alphabetical by default.
        n_cols: Panels per row.
        title: Figure suptitle.
        ylabel: Shared Y label.
        mode: Style mode.
        panel_size: ``(width, height)`` of one panel in inches.

    Returns:
        The figure.
    """
    p = palette(mode)
    order = list(order or sorted(frame[group_col].dropna().unique()))
    n_rows = int(np.ceil(len(order) / n_cols))
    fig, axes = plt.subplots(
        n_rows, n_cols,
        figsize=(n_cols * panel_size[0] + 1.0, n_rows * panel_size[1] + 1.2),
        squeeze=False, sharex=True, sharey=True,
    )

    handles = None
    for ax, code in zip(axes.ravel(), order):
        block = frame[frame[group_col] == code].sort_values(year_col)
        (obs_line,) = ax.plot(block[year_col], block[obs_col],
                              color=p["series_1"], linewidth=1.6, label="Observed")
        (sim_line,) = ax.plot(block[year_col], block[sim_col],
                              color=p["series_2"], linewidth=1.6, label="Simulated")
        handles = [obs_line, sim_line]
        ax.set_title(str(code), fontsize=plt.rcParams["font.size"] - 0.5)
        ax.tick_params(labelrotation=30)

    for ax in axes.ravel()[len(order):]:
        ax.set_visible(False)

    if handles:
        fig.legend(handles=handles, loc="lower center", ncol=2,
                   labelcolor=p["secondary"], bbox_to_anchor=(0.5, -0.01))
    fig.supylabel(ylabel, color=p["secondary"], fontsize=plt.rcParams["font.size"])
    if title:
        fig.suptitle(title, x=0.01, ha="left", color=p["primary"],
                     fontsize=plt.rcParams["font.size"] + 2)
    fig.tight_layout()
    return fig


# --------------------------------------------------------------------------- #
# Maps
# --------------------------------------------------------------------------- #


def _basemap(ax: plt.Axes, p: dict[str, str], extent=EUROPE_EXTENT) -> None:
    """Recessive land/ocean/coastline context under the data."""
    ax.set_extent(extent, crs=ccrs.PlateCarree())
    ax.add_feature(cfeature.OCEAN.with_scale("50m"), facecolor=p["ocean"], zorder=0)
    ax.add_feature(cfeature.LAND.with_scale("50m"), facecolor=p["land"], zorder=0)
    ax.coastlines("50m", linewidth=0.4, color=p["muted"], zorder=4)
    ax.spines["geo"].set_visible(False)


def country_choropleth(
    polygons: gpd.GeoDataFrame,
    values: pd.Series,
    title: str = "",
    cbar_label: str = "",
    diverging: bool = True,
    vmax: float | None = None,
    mode: str = "light",
    figsize: tuple[float, float] = (5.6, 6.0),
    annotate: bool = False,
) -> Figure:
    """Fill each national footprint with a per-country value.

    Args:
        polygons: Country footprints from
            :func:`utils.regions.load_country_polygons`.
        values: Value per country, indexed by country code.
        title: Axes title.
        cbar_label: Colour-bar label, including units.
        diverging: ``True`` for a signed quantity (bias) — a zero-centred
            blue-to-red ramp. ``False`` for a magnitude — the single-hue ramp.
        vmax: Symmetric limit for the diverging ramp.
        mode: Style mode.
        figsize: Figure size in inches.
        annotate: Print the value on each country. Off by default; it collides
            badly in the Benelux and turns the map into a table.

    Returns:
        The figure.
    """
    p = palette(mode)
    frame = polygons.copy()
    frame["value"] = frame["country"].map(values)

    fig, ax = plt.subplots(figsize=figsize,
                           subplot_kw={"projection": EUROPE_PROJECTION})
    _basemap(ax, p)
    ax.grid(False)

    cmap = DIVERGING_CMAP if diverging else SEQUENTIAL_CMAP
    norm = diverging_norm(frame["value"], vmax) if diverging else None

    plotted = frame.to_crs(EUROPE_PROJECTION.proj4_init)
    plotted.plot(
        ax=ax, column="value", cmap=cmap, norm=norm,
        edgecolor=p["surface"], linewidth=0.6, zorder=2,
        missing_kwds={"color": p["land"], "edgecolor": p["surface"],
                      "linewidth": 0.6, "hatch": None},
        legend=True,
        legend_kwds={"label": cbar_label, "orientation": "horizontal",
                     "shrink": 0.62, "pad": 0.02, "aspect": 30},
    )

    if annotate:
        for _, row in plotted.iterrows():
            if np.isfinite(row["value"]):
                point = row.geometry.representative_point()
                ax.annotate(f"{row['value']:+.1f}", (point.x, point.y),
                            ha="center", va="center", color=p["primary"],
                            fontsize=plt.rcParams["font.size"] - 2.5, zorder=5)

    ax.set_title(title, color=p["primary"])
    fig.tight_layout()
    return fig


def cell_map(
    frame: pd.DataFrame,
    value_col: str,
    lon_col: str = "lon",
    lat_col: str = "lat",
    cmap: str | ListedColormap | None = None,
    norm: BoundaryNorm | None = None,
    vmin: float | None = None,
    vmax: float | None = None,
    title: str = "",
    cbar_label: str = "",
    mode: str = "light",
    figsize: tuple[float, float] = (6.5, 6.0),
    extent: tuple[float, float, float, float] = EUROPE_EXTENT,
    projection: ccrs.Projection = EUROPE_PROJECTION,
) -> Figure:
    """Gridded spatial cell map over Europe.

    Args:
        frame: Dataframe containing coordinates and values per grid cell.
        value_col: Name of the column to map.
        lon_col: Longitude column name.
        lat_col: Latitude column name.
        cmap: Colormap instance or registered colormap name. Defaults to
            :data:`utils.style.SEQUENTIAL_CMAP`.
        norm: Optional normalization instance (e.g., :class:`~matplotlib.colors.BoundaryNorm`).
            Ignored if ``vmin`` or ``vmax`` are explicitly passed alongside it.
        vmin: Minimum value for color scaling.
        vmax: Maximum value for color scaling.
        title: Axes title.
        cbar_label: Colorbar label including units.
        mode: Style mode.
        figsize: Figure size in inches.
        extent: Map spatial bounds ``(lon_min, lon_max, lat_min, lat_max)``.
        projection: Cartopy CRS projection.

    Returns:
        The figure.
    """
    p = palette(mode)
    cmap = cmap or SEQUENTIAL_CMAP

    fig, ax = plt.subplots(
        figsize=figsize,
        subplot_kw={"projection": projection},
    )

    ax.set_extent(extent, crs=ccrs.PlateCarree())
    ax.add_feature(cfeature.BORDERS, linewidth=0.4, edgecolor=p["muted"], zorder=3)
    ax.add_feature(cfeature.COASTLINE, linewidth=0.5, edgecolor=p["secondary"], zorder=3)
    ax.add_feature(cfeature.LAND, facecolor=p["surface"], zorder=0)

    # Pivot grid coordinates for spatial plotting
    grid = frame.pivot(index=lat_col, columns=lon_col, values=value_col)
    lons, lats = np.meshgrid(grid.columns.values, grid.index.values)

    mesh = ax.pcolormesh(
        lons,
        lats,
        grid.values,
        transform=ccrs.PlateCarree(),
        cmap=cmap,
        norm=norm,
        vmin=vmin,
        vmax=vmax,
        zorder=2,
    )

    bar = fig.colorbar(mesh, ax=ax, shrink=0.75, pad=0.03, aspect=24)
    if cbar_label:
        bar.set_label(cbar_label, color=p["secondary"])
    bar.outline.set_visible(False)
    bar.ax.tick_params(color=p["muted"], labelcolor=p["secondary"])

    if title:
        ax.set_title(title, pad=12.0)

    fig.tight_layout()
    return fig


def category_map(
    cells: pd.DataFrame,
    value_col: str,
    categories: Sequence[str],
    colors: Sequence[str] | None = None,
    title: str = "",
    mode: str = "light",
    figsize: tuple[float, float] = (5.6, 6.0),
    overlay: gpd.GeoDataFrame | None = None,
    legend_title: str = "",
) -> Figure:
    """The 0.5° field drawn as a small number of named classes.

    Used for the one question a continuous bias map cannot answer: is the
    simulated date *inside* the reference's window, or before it, or after it.
    That has three states and no meaningful distance between them, so it gets
    discrete fills and a legend rather than a colour bar.

    The default fills take the two poles of the diverging ramp for the signed
    outer classes and a neutral grey for the middle one, which keeps the
    encoding consistent with :func:`bias_bars` — cool is early, warm is late,
    grey is agreement.

    Args:
        cells: One row per cell with ``lon``, ``lat`` and ``value_col``.
        value_col: Column holding the class label.
        categories: Class labels in display order.
        colors: One colour per class. ``None`` uses the diverging poles for
            three classes and the categorical hues otherwise.
        title: Axes title.
        mode: Style mode.
        figsize: Figure size in inches.
        overlay: Optional boundaries drawn over the field.
        legend_title: Heading above the legend entries.

    Returns:
        The figure.

    Raises:
        ValueError: If ``colors`` is given with a different length than
            ``categories``.
    """
    p = palette(mode)
    categories = list(categories)
    if colors is None:
        colors = (
            [DIVERGING_CMAP(0.18), "#b9b8b0", DIVERGING_CMAP(0.86)]
            if len(categories) == 3
            else [p[f"series_{i + 1}"] for i in range(len(categories))]
        )
    if len(colors) != len(categories):
        raise ValueError(
            f"{len(colors)} colours for {len(categories)} categories"
        )

    lon = np.round(cells["lon"].to_numpy(dtype=float), 4)
    lat = np.round(cells["lat"].to_numpy(dtype=float), 4)
    lons, lon_idx = np.unique(lon, return_inverse=True)
    lats, lat_idx = np.unique(lat, return_inverse=True)

    codes = pd.Categorical(cells[value_col], categories=categories).codes.astype(float)
    codes[codes < 0] = np.nan
    grid = np.full((lats.size, lons.size), np.nan)
    grid[lat_idx, lon_idx] = codes

    fig, ax = plt.subplots(figsize=figsize,
                           subplot_kw={"projection": EUROPE_PROJECTION})
    _basemap(ax, p)
    ax.grid(False)

    cmap = ListedColormap(list(colors))
    norm = BoundaryNorm(np.arange(-0.5, len(categories), 1.0), cmap.N)
    ax.pcolormesh(
        lons, lats, np.ma.masked_invalid(grid), cmap=cmap, norm=norm,
        transform=ccrs.PlateCarree(), shading="nearest", zorder=2,
    )

    if overlay is not None:
        overlay.to_crs(EUROPE_PROJECTION.proj4_init).boundary.plot(
            ax=ax, color=p["muted"], linewidth=0.45, zorder=3
        )

    counts = pd.Series(cells[value_col]).value_counts()
    total = int(counts.sum()) or 1
    ax.legend(
        handles=[
            Patch(facecolor=colour,
                  label=f"{name}  ({100.0 * counts.get(name, 0) / total:.0f} %)")
            for name, colour in zip(categories, colors)
        ],
        loc="upper left", labelcolor=p["secondary"], title=legend_title or None,
        frameon=False,
    )
    ax.set_title(title, color=p["primary"])
    fig.tight_layout()
    return fig


# --------------------------------------------------------------------------- #
# Tables
# --------------------------------------------------------------------------- #


def metric_table(
    metrics: pd.DataFrame,
    columns: Sequence[str],
    precision: int = 2,
    gradient_on: Sequence[str] = ("bias",),
    caption: str = "",
    country_names: bool = True,
):
    """Render a metric table as a styled DataFrame for the notebook.

    The table is the accessible twin of every figure: each number a bar or a
    fill encodes is readable here without relying on colour. A light diverging
    gradient rides only the signed columns, where it means the same thing as in
    :func:`bias_bars`.

    Args:
        metrics: Metric table indexed by country code.
        columns: Columns to show, in order.
        precision: Decimal places.
        gradient_on: Signed columns that get the zero-centred gradient.
        caption: Table caption.
        country_names: Prefix the index with the full country name.

    Returns:
        A :class:`pandas.io.formats.style.Styler`.
    """
    present = [c for c in columns if c in metrics.columns]
    table = metrics[present].copy()
    if country_names:
        table.index = [
            "All countries" if c == "ALL" else _country_label(str(c))
            for c in table.index
        ]
    table = table.rename(columns={c: METRIC_LABELS.get(c, c) for c in present})

    styler = (
        table.style.format(precision=precision, na_rep="—",
                           formatter={METRIC_LABELS["n"]: "{:.0f}"}
                           if METRIC_LABELS["n"] in table.columns else None)
        .set_caption(caption)
    )
    for col in gradient_on:
        label = METRIC_LABELS.get(col, col)
        if label in table.columns and table[label].notna().any():
            limit = float(np.nanmax(np.abs(table[label].to_numpy(dtype=float))))
            styler = styler.background_gradient(
                cmap=DIVERGING_CMAP, subset=[label],
                vmin=-limit, vmax=limit,
            )
    return styler
