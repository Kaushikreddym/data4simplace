"""Combine the torchcrop shards and render spatial maps of the Europe run.

Reads every ``torchcrop_shard_*.parquet`` written by
:mod:`cropmodelling4eu.torchcrop.run`, checks
the shard set is complete, rasterises the per-cell summaries back onto the
0.1 deg target grid, and writes:

* ``torchcrop_europe.parquet``  — every (cell, year) row, concatenated
* ``torchcrop_europe_grid.nc``  — ``(year, lat, lon)`` cubes per variable plus
  the across-year climatology, so the run is queryable without re-gridding
* ``maps/*.png``                — the map figures

Maps use a single sequential hue per magnitude variable and a diverging pair
only where the variable has a genuine midpoint (the yield anomaly). Colour is
never the only encoding: every panel carries a labelled colourbar.
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import xarray as xr

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import cartopy.crs as ccrs       # noqa: E402
import cartopy.feature as cfeature  # noqa: E402

from cropmodelling4eu.config import GridConfig, RunConfig, load_config  # noqa: E402
from cropmodelling4eu.export.cells import id_to_rowcol  # noqa: E402

logger = logging.getLogger("cropmodelling4eu.torchcrop.maps")

PLATE = ccrs.PlateCarree()

#: One panel per row: (column, title, unit, colormap, robust-percentile clip).
#: Sequential single-hue ramps for magnitude; the anomaly panel is handled
#: separately because it is the only variable with a real midpoint.
MAP_PANELS = [
    ("yield_t_ha", "Grain yield", "t ha⁻¹", "YlGnBu", (2, 98)),
    ("yield_cv", "Interannual variability", "CV [-]", "YlOrRd", (2, 98)),
    ("max_lai", "Peak leaf area index", "m² m⁻²", "YlGn", (2, 98)),
    ("tranrf_mean", "Water stress (1 = unstressed)", "TRANRF [-]", "Blues", (2, 98)),
    ("nni_mean", "N nutrition index (1 = unstressed)", "NNI [-]", "BuPu", (2, 98)),
    ("n_applied_g_m2", "N applied", "g N m⁻²", "PuRd", (2, 98)),
    ("days_to_maturity", "Season length to maturity", "days", "viridis", (2, 98)),
    ("biomass_g_m2", "Above-ground biomass", "g m⁻²", "YlGnBu", (2, 98)),
]

#: Variables carried through to the gridded NetCDF, per year and as a mean.
GRID_VARS = [
    "yield_t_ha", "adjusted_yield_t_ha", "biomass_g_m2", "max_lai",
    "final_dvs", "days_to_maturity", "n_applied_g_m2", "tranrf_mean",
    "nni_mean", "heat_stress_factor",
]


# --------------------------------------------------------------------------- #
# Load & grid
# --------------------------------------------------------------------------- #


def load_shards(shard_dir: Path, n_shards: int) -> pd.DataFrame:
    """Concatenate every shard, failing loudly on an incomplete set.

    A missing shard would silently blank ~1/n of the domain — and because
    shards are dealt round-robin the hole is a speckle across all of Europe,
    not an obviously empty corner. So it is an error, not a warning.
    """
    paths = sorted(shard_dir.glob("torchcrop_shard_*.parquet"))
    found = {int(p.stem.rsplit("_", 1)[1]) for p in paths}
    missing = sorted(set(range(n_shards)) - found)
    if missing:
        raise FileNotFoundError(
            f"{len(missing)} of {n_shards} shards missing from {shard_dir}: "
            f"{missing}. Re-run them with "
            f"'./submit/submit_torchcrop.sh --retry' before mapping."
        )
    logger.info("loading %d shards from %s", len(paths), shard_dir)
    frame = pd.concat((pd.read_parquet(p) for p in paths), ignore_index=True)
    frame["yield_t_ha"] = frame["yield_g_m2"] / 100.0
    frame["adjusted_yield_t_ha"] = frame["adjusted_yield_g_m2"] / 100.0
    return frame


def to_grid(frame: pd.DataFrame, grid: GridConfig) -> xr.Dataset:
    """Scatter the per-(cell, year) rows onto the ``(year, lat, lon)`` grid.

    Placement is arithmetic — ``SimplaceID`` encodes the row and column
    directly — so this is an index assignment, not a spatial join. The grid
    comes from the run configuration rather than a module constant, since the
    same id means different coordinates under different grid bounds.
    """
    years = np.sort(frame["year"].unique())
    year_idx = pd.Series(np.arange(len(years)), index=years)
    row, col = id_to_rowcol(frame["SimplaceID"].to_numpy(), grid)
    yr = year_idx.reindex(frame["year"].to_numpy()).to_numpy()

    res, n_lat, n_lon = grid.resolution_deg, grid.n_lat, grid.n_lon
    lats = grid.max_lat - res / 2.0 - np.arange(n_lat) * res
    lons = grid.min_lon + res / 2.0 + np.arange(n_lon) * res

    data = {}
    for name in GRID_VARS:
        if name not in frame.columns:
            continue
        cube = np.full((len(years), n_lat, n_lon), np.nan, dtype=np.float32)
        cube[yr, row, col] = frame[name].to_numpy(np.float32)
        data[name] = (("year", "lat", "lon"), cube)

    ds = xr.Dataset(data, coords={"year": years, "lat": lats, "lon": lons})

    # Climatology: the mean over years, plus the coefficient of variation as
    # the interannual-risk layer. CV is undefined where the mean is ~0, so
    # those cells stay NaN rather than exploding.
    #
    # The suffix is `_clim`, not `_mean`: two of the per-year variables are
    # already named `tranrf_mean`/`nni_mean` (means over the *growing season*),
    # so a `_mean` suffix would make `tranrf_mean` ambiguous between the 3-D
    # per-year cube and its 2-D climatology.
    mean = ds.mean("year", skipna=True)
    std = ds["yield_t_ha"].std("year", skipna=True)
    for name, arr in mean.data_vars.items():
        ds[f"{name}_clim"] = arr
    ds["yield_std"] = std
    ds["yield_cv"] = xr.where(
        mean["yield_t_ha"] > 0.1, std / mean["yield_t_ha"], np.nan
    )
    ds["n_years"] = ds["yield_t_ha"].notnull().sum("year")

    ds.attrs["description"] = "torchcrop / LINTUL-5 winter wheat, SIMPLACE Europe"
    ds["lat"].attrs.update(units="degrees_north", standard_name="latitude")
    ds["lon"].attrs.update(units="degrees_east", standard_name="longitude")
    return ds


# --------------------------------------------------------------------------- #
# Maps
# --------------------------------------------------------------------------- #


def _basemap(ax) -> None:
    """Recessive coastline/border context under the data."""
    ax.add_feature(cfeature.OCEAN.with_scale("50m"), facecolor="#eef2f5", zorder=0)
    ax.add_feature(cfeature.LAND.with_scale("50m"), facecolor="#f7f7f5", zorder=0)
    ax.coastlines("50m", linewidth=0.4, color="0.45", zorder=3)
    ax.add_feature(
        cfeature.BORDERS.with_scale("50m"), linewidth=0.3, edgecolor="0.65", zorder=3
    )
    gl = ax.gridlines(draw_labels=False, linewidth=0.3, color="0.85", zorder=1)
    gl.top_labels = gl.right_labels = False


def _clim_key(ds: xr.Dataset, name: str) -> str | None:
    """The 2-D field to map for a panel: its climatology if one exists.

    Panels name the *quantity* (``yield_t_ha``, ``tranrf_mean``); `to_grid`
    stores the across-year mean under a ``_clim`` suffix. Variables computed
    directly as 2-D (``yield_cv``) have no ``_clim`` twin and are used as-is.
    """
    for candidate in (f"{name}_clim", name):
        if candidate in ds and "year" not in ds[candidate].dims:
            return candidate
    return None


def _draw(ax, ds: xr.Dataset, name: str, title: str, unit: str, cmap: str,
          clip: tuple[int, int], extent, diverging: bool = False) -> None:
    """One map panel: pcolormesh + a labelled colourbar."""
    field = ds[name]
    values = field.values
    finite = values[np.isfinite(values)]
    if finite.size == 0:
        ax.set_title(f"{title} — no data", loc="left", fontsize=10)
        return
    if diverging:
        limit = float(np.nanpercentile(np.abs(finite), clip[1]))
        vmin, vmax = -limit, limit
    else:
        vmin, vmax = (float(np.nanpercentile(finite, clip[0])),
                      float(np.nanpercentile(finite, clip[1])))

    ax.set_extent(extent, crs=PLATE)
    _basemap(ax)
    mesh = ax.pcolormesh(
        ds["lon"], ds["lat"], values, transform=PLATE,
        cmap=cmap, vmin=vmin, vmax=vmax, shading="nearest", zorder=2,
    )
    ax.set_title(title, loc="left", fontsize=10)
    cbar = plt.colorbar(mesh, ax=ax, orientation="vertical", shrink=0.85,
                        aspect=28, pad=0.02, extend="both")
    cbar.set_label(unit, fontsize=8)
    cbar.ax.tick_params(labelsize=7)


def _extent(ds: xr.Dataset, pad: float = 1.0) -> list[float]:
    """Bounding box of the cells that actually carry data, padded."""
    valid = ds["yield_t_ha_clim"].notnull()
    lons = ds["lon"].where(valid.any("lat"), drop=True)
    lats = ds["lat"].where(valid.any("lon"), drop=True)
    if lons.size == 0 or lats.size == 0:
        return [GRID_MIN_LON, GRID_MIN_LON + N_LON * GRID_RES,
                GRID_MAX_LAT - N_LAT * GRID_RES, GRID_MAX_LAT]
    return [float(lons.min()) - pad, float(lons.max()) + pad,
            float(lats.min()) - pad, float(lats.max()) + pad]


def _panel_size(extent, height: float = 3.4) -> tuple[float, float]:
    """Panel (width, height) in inches matching the domain's aspect ratio.

    A PlateCarree GeoAxes holds an equal aspect in degrees, so a panel whose
    box does not match ``lon_span : lat_span`` draws the map as a strip inside
    a much larger axes — and the colourbar, which sizes to the *axes*, then
    towers over it. Sizing the figure from the extent keeps both in scale.
    """
    lon_span = max(extent[1] - extent[0], 1e-6)
    lat_span = max(extent[3] - extent[2], 1e-6)
    # Clamp so a pathologically thin subset (a smoke test, one latitude band)
    # cannot demand a 20:1 figure.
    ratio = float(np.clip(lon_span / lat_span, 0.5, 4.0))
    return height * ratio, height


def _grid_shape(n_panels: int, extent, target: float = 1.4) -> tuple[int, int]:
    """(nrow, ncol) whose overall figure aspect sits closest to ``target``.

    A fixed 2x4 grid is wrong for a wide domain: Europe is ~2:1, so four
    columns of 2:1 panels give a 5.5:1 figure that no screen shows legibly.
    Choosing the column count from the panel aspect keeps the sheet roughly
    landscape whatever the domain.
    """
    pw, ph = _panel_size(extent)
    best = None
    for ncol in range(1, n_panels + 1):
        nrow = int(np.ceil(n_panels / ncol))
        # 1.28 mirrors the per-panel colourbar allowance in the callers.
        aspect = (ncol * pw * 1.28) / (nrow * ph)
        score = abs(np.log(aspect / target))
        if best is None or score < best[0]:
            best = (score, nrow, ncol)
    return best[1], best[2]


def plot_overview(ds: xr.Dataset, out_dir: Path) -> Path:
    """Eight-panel climatology overview of the whole run."""
    extent = _extent(ds)
    pw, ph = _panel_size(extent)
    nrow, ncol = _grid_shape(len(MAP_PANELS), extent)
    # x1.28 on width leaves room for each panel's colourbar.
    fig, axes = plt.subplots(
        nrow, ncol, figsize=(pw * ncol * 1.28, ph * nrow), layout="constrained",
        subplot_kw={"projection": PLATE},
    )
    axes = np.atleast_1d(axes)
    for ax in axes.ravel()[len(MAP_PANELS):]:
        ax.set_visible(False)
    for ax, (name, title, unit, cmap, clip) in zip(axes.ravel(), MAP_PANELS):
        key = _clim_key(ds, name)
        if key is None:
            ax.set_visible(False)
            continue
        _draw(ax, ds, key, title, unit, cmap, clip, extent)

    years = ds["year"].values
    fig.suptitle(
        f"torchcrop / LINTUL-5 winter wheat — SIMPLACE Europe, "
        f"{years.min()}–{years.max()} mean of {len(years)} seasons",
        x=0.01, ha="left", fontsize=14,
    )
    path = out_dir / "map_overview.png"
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    logger.info("wrote %s", path)
    return path


def plot_single(ds: xr.Dataset, out_dir: Path) -> list[Path]:
    """One standalone figure per climatology variable."""
    extent = _extent(ds)
    written = []
    for name, title, unit, cmap, clip in MAP_PANELS:
        key = _clim_key(ds, name)
        if key is None:
            continue
        pw, ph = _panel_size(extent, height=7.0)
        fig, ax = plt.subplots(figsize=(pw * 1.22, ph), layout="constrained",
                               subplot_kw={"projection": PLATE})
        _draw(ax, ds, key, title, unit, cmap, clip, extent)
        path = out_dir / f"map_{name}.png"
        fig.savefig(path, dpi=150, bbox_inches="tight")
        plt.close(fig)
        written.append(path)
    logger.info("wrote %d single-variable maps", len(written))
    return written


def plot_year_panels(ds: xr.Dataset, out_dir: Path, max_years: int = 24) -> Path | None:
    """Per-season yield anomaly against the run's own climatology.

    Anomaly, not raw yield, because the between-year signal is what a per-year
    panel is for — the north–south gradient would otherwise dominate every
    panel and hide it. Diverging ramp on a symmetric scale about zero.
    """
    years = ds["year"].values
    if years.size < 2:
        return None
    if years.size > max_years:
        years = years[-max_years:]
        logger.info("per-year panel limited to the last %d seasons", max_years)

    anomaly = ds["yield_t_ha"].sel(year=years) - ds["yield_t_ha_clim"]
    limit = float(np.nanpercentile(np.abs(anomaly.values[np.isfinite(anomaly.values)]), 98))
    extent = _extent(ds)

    nrow, ncol = _grid_shape(len(years), extent)
    pw, ph = _panel_size(extent, height=2.7)
    fig, axes = plt.subplots(
        nrow, ncol, figsize=(pw * ncol, ph * nrow + 0.9), layout="constrained",
        subplot_kw={"projection": PLATE},
    )
    axes = np.atleast_1d(axes).ravel()
    mesh = None
    for ax, year in zip(axes, years):
        ax.set_extent(extent, crs=PLATE)
        _basemap(ax)
        mesh = ax.pcolormesh(
            ds["lon"], ds["lat"], anomaly.sel(year=year).values, transform=PLATE,
            cmap="RdBu", vmin=-limit, vmax=limit, shading="nearest", zorder=2,
        )
        ax.set_title(str(int(year)), loc="left", fontsize=9)
    for ax in axes[len(years):]:
        ax.set_visible(False)

    if mesh is not None:
        cbar = fig.colorbar(mesh, ax=axes[: len(years)].tolist(),
                            orientation="horizontal", shrink=0.5, aspect=45,
                            extend="both")
        cbar.set_label("yield anomaly vs. run climatology [t ha⁻¹]", fontsize=9)
    fig.suptitle("Per-season yield anomaly", x=0.01, ha="left", fontsize=13)
    path = out_dir / "map_yield_anomaly_by_year.png"
    fig.savefig(path, dpi=140, bbox_inches="tight")
    plt.close(fig)
    logger.info("wrote %s", path)
    return path


# --------------------------------------------------------------------------- #
# CLI
# --------------------------------------------------------------------------- #


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--shard-dir", required=True, help="directory of shard parquets")
    p.add_argument("--out-dir", required=True)
    p.add_argument("--n-shards", type=int, default=10)
    p.add_argument("--config", type=Path, help="Run configuration YAML.")
    p.add_argument(
        "--export-dir",
        type=Path,
        help="Export directory, to read the grid back from its frozen config "
        "(instead of --config).",
    )
    p.add_argument("--no-single-maps", action="store_true")
    args = p.parse_args(argv)

    logging.basicConfig(
        level=logging.INFO, format="%(asctime)s %(levelname)-7s %(message)s",
        datefmt="%H:%M:%S", stream=sys.stdout,
    )

    if args.config is not None:
        grid = load_config(args.config).grid
    elif args.export_dir is not None:
        grid = RunConfig.from_export(args.export_dir).grid
    else:
        # The default Europe grid. Logged rather than assumed silently, since a
        # mismatch places every cell at the wrong coordinates on every map.
        grid = GridConfig()
        logger.warning(
            "No --config or --export-dir; gridding onto the default Europe grid "
            "(%s..%s E, %s..%s N at %s deg). Pass one if the run used another.",
            grid.min_lon, grid.max_lon, grid.min_lat, grid.max_lat,
            grid.resolution_deg,
        )

    out_dir = Path(args.out_dir)
    map_dir = out_dir / "maps"
    map_dir.mkdir(parents=True, exist_ok=True)

    frame = load_shards(Path(args.shard_dir), args.n_shards)
    logger.info(
        "%d rows, %d cells, seasons %d-%d",
        len(frame), frame["SimplaceID"].nunique(),
        frame["year"].min(), frame["year"].max(),
    )

    combined = out_dir / "torchcrop_europe.parquet"
    frame.to_parquet(combined, index=False, compression="zstd")
    logger.info("wrote %s", combined)

    ds = to_grid(frame, grid)
    grid_path = out_dir / "torchcrop_europe_grid.nc"
    encoding = {v: {"zlib": True, "complevel": 4} for v in ds.data_vars}
    ds.to_netcdf(grid_path, encoding=encoding)
    logger.info("wrote %s", grid_path)

    plot_overview(ds, map_dir)
    plot_year_panels(ds, map_dir)
    if not args.no_single_maps:
        plot_single(ds, map_dir)

    summary = (
        frame.groupby("year")["yield_t_ha"]
        .agg(["count", "mean", "std", "min", "max"])
        .round(2)
    )
    print("\n--- yield [t/ha] by season ---")
    print(summary.to_string())
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
