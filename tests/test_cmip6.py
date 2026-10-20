"""CMIP6-BC climate source, the base-export cell set and scenario linking.

The fixtures mimic climdata's output as found on disk: ``latitude``/
``longitude`` axes running south to north over a regional frame, the MSWX
``<VAR>/YYYYDDD.nc`` daily layout, and the one-file-per-year cache.
"""

from __future__ import annotations

import json

import numpy as np
import pandas as pd
import pytest
import xarray as xr

from data4simplace.climate.mswx_handler import MSWXHandler
from data4simplace.config import PipelineConfig
from data4simplace.grid import TargetGrid
from data4simplace.scenario import link_base_export
from data4simplace.site import load_co2_series
from data4simplace.spatial import base_export_cells, export_cell_mask

LAT = np.round(np.arange(50.05, 56.0, 0.1), 2)  # ascending, as climdata writes it
LON = np.round(np.arange(5.05, 11.0, 0.1), 2)
DAYS = pd.date_range("1990-12-30", "1991-01-02", freq="D") + pd.Timedelta(hours=12)
EXP = "TOY_hist-aer"


def _field(day: int) -> np.ndarray:
    """A frame whose value encodes the day and the position, so misalignment shows."""
    return (day * 1000 + np.add.outer(np.arange(LAT.size), np.arange(LON.size))).astype("float32")


def _write(path, name, times, frames) -> None:
    xr.Dataset(
        {name: (("time", "latitude", "longitude"), np.stack(frames))},
        coords={"time": times, "latitude": LAT, "longitude": LON},
    ).to_netcdf(path, engine="h5netcdf")


@pytest.fixture()
def cmip_root(tmp_path):
    root = tmp_path / "CMIP6_MSWX_BC"
    for folder, name in (("TAS", "tas"), ("PR", "pr")):
        var = root / EXP / folder
        var.mkdir(parents=True)
        for i, t in enumerate(DAYS):
            _write(var / f"{t.year}{t.dayofyear:03d}.nc", name, [t], [_field(i)])
        cache = root / "_yearly_cache" / EXP / "mbcn_pr-tas"
        cache.mkdir(parents=True, exist_ok=True)
        for year in (1990, 1991):
            sel = [(i, t) for i, t in enumerate(DAYS) if t.year == year]
            _write(cache / f"BCSD_{name}_model_scenario_{year}.nc", name,
                   [t for _, t in sel], [_field(i) for i, _ in sel])
    return root


def _config(root, out=None, grid=None, **climate) -> PipelineConfig:
    return PipelineConfig.model_validate({
        "flags": {"run_climate_processing": True, "export_simplace_weather": True},
        "grid": grid or {"resolution_deg": 0.1, "min_lon": 7.0, "max_lon": 9.0,
                         "min_lat": 52.0, "max_lat": 54.0},
        "time": {"start": "1990-12-31", "end": "1991-01-02"},
        "paths": {"mswx_root": "/nonexistent", "cmip6_bc_root": str(root),
                  "output_dir": str(out or root / "out")},
        "climate": {"source": "cmip6_bc", "experiment": EXP, "read_workers": 1,
                    "variables": {"TAS": "tas", "PR": "pr"}, **climate},
    })


def test_daily_cmip_layout_loads_with_latitude_names(cmip_root):
    ds = MSWXHandler(_config(cmip_root)).load()
    assert ds is not None and set(ds.data_vars) == {"tas", "pr"}
    assert ds.sizes["time"] == 3
    # Day index 1 is 1990-12-31; the target grid runs north to south.
    first = ds["tas"].isel(time=0).values
    assert first.min() >= 1000 and first.max() < 2000
    assert ds["lat"].values[0] > ds["lat"].values[-1]


def test_yearly_layout_matches_daily(cmip_root):
    daily = MSWXHandler(_config(cmip_root)).load()
    yearly = MSWXHandler(_config(cmip_root, layout="yearly")).load()
    xr.testing.assert_allclose(daily.load(), yearly.load())
    assert pd.DatetimeIndex(yearly["time"].values).hour.max() == 0


def test_tile_outside_regional_frame_loads_nothing(cmip_root):
    grid = {"resolution_deg": 0.1, "min_lon": 40.0, "max_lon": 42.0,
            "min_lat": 52.0, "max_lat": 54.0}
    assert MSWXHandler(_config(cmip_root, grid=grid)).load() is None


def test_missing_required_variable_refused_before_reading(cmip_root):
    cfg = _config(cmip_root, variables={"TAS": "tas", "RSDS": "rsds"}, required=["rsds"])
    handler = MSWXHandler(cfg)
    assert handler.missing_required() == ["rsds"]
    with pytest.raises(ValueError, match="rsds"):
        handler.load()


def test_required_must_be_loaded_variable(cmip_root):
    with pytest.raises(ValueError, match="sfcWind"):
        _config(cmip_root, required=["sfcWind"])


def test_cmip_source_needs_experiment(cmip_root):
    with pytest.raises(ValueError, match="experiment"):
        _config(cmip_root, experiment=None)


def test_co2_hold_year_flattens_series(tmp_path):
    path = tmp_path / "co2.csv"
    pd.DataFrame({"year": [1849, 1850, 1851, 2014], "co2_ppm": [284.0, 284.3, 284.6, 397.5]}
                 ).to_csv(path, index=False)
    series, source = load_co2_series(path, range(1849, 2016), hold_year=1850)
    assert np.allclose(series.values, 284.3)
    assert series.index[0] == 1849 and series.index[-1] == 2015
    assert source == "file, held at 1850"
    with pytest.raises(ValueError, match="outside"):
        load_co2_series(path, range(1849, 2016), hold_year=1700)


def _base_export(tmp_path, grid_cfg, ids) -> "Path":
    base = tmp_path / "base"
    (base / "soil").mkdir(parents=True)
    (base / "site").mkdir()
    (base / "management").mkdir()
    pd.DataFrame({"location": ids, "clay_1": 20.0}).to_csv(base / "soil" / "soil.csv", index=False)
    pd.DataFrame({"location": ids}).to_csv(base / "site" / "site.csv", index=False)
    for name in ("fertilizer_winter_wheat.csv", "fertilizer_winter_wheat_long.csv"):
        (base / "management" / name).write_text("location\n1\n")
    (base / "soil" / "_shards").mkdir()
    return base


def test_base_cells_restrict_export_mask(cmip_root, tmp_path):
    cfg = _config(cmip_root)
    grid = TargetGrid.from_config(cfg.grid)
    base = _base_export(tmp_path, cfg.grid, [1, 5, 22])
    cfg = cfg.model_copy(update={"paths": cfg.paths.model_copy(update={"cell_set_from": base})})
    within = base_export_cells(cfg)
    assert int(within.sum()) == 3
    # A sub-grid sharing the cell centres picks up only its own cells.
    sub = TargetGrid(min_lon=7.0, max_lon=7.5, min_lat=53.5, max_lat=54.0, resolution_deg=0.1)
    mask = export_cell_mask(cfg, sub, None, soil_export_fallback=False, within=within)
    # Ids 1, 5 and 22 on the 20-column grid are (0,0), (0,4) and (1,1).
    assert mask.shape == sub.shape and int(mask.sum()) == 3
    assert mask.values.ravel()[[0, 4, 6]].all()
    assert grid.shape == (20, 20)


def test_base_cells_require_a_soil_export(cmip_root, tmp_path):
    cfg = _config(cmip_root)
    cfg = cfg.model_copy(update={"paths": cfg.paths.model_copy(update={"cell_set_from": tmp_path})})
    with pytest.raises(FileNotFoundError):
        base_export_cells(cfg)


def test_link_base_export(cmip_root, tmp_path):
    base = _base_export(tmp_path, None, [1, 5])
    out = tmp_path / "scenario"
    (out / "weather" / "0").mkdir(parents=True)
    (out / "weather" / "0" / "daily_mean_RES1_C0R0.csv.gz").write_bytes(b"")
    co2 = tmp_path / "co2.csv"
    pd.DataFrame({"year": [1850, 2014], "co2_ppm": [284.3, 397.5]}).to_csv(co2, index=False)
    cfg = _config(cmip_root, out=out)
    cfg = cfg.model_copy(update={
        "paths": cfg.paths.model_copy(update={"cell_set_from": base, "co2_file": co2}),
        "site": cfg.site.model_copy(update={"co2_hold_year": 1850}),
    })

    with pytest.raises(RuntimeError, match="unfinished"):
        link_base_export(cfg, tiles_expected=1)
    record = link_base_export(cfg)

    assert (out / "soil" / "soil.csv").resolve() == (base / "soil" / "soil.csv").resolve()
    assert not (out / "soil" / "_shards").exists()
    assert (out / "management" / "fertilizer_winter_wheat.csv").is_symlink()
    assert not (out / "management" / "fertilizer_winter_wheat_long.csv").exists()
    assert (out / "site" / "site.csv").is_symlink()
    written = pd.read_csv(out / "site" / "co2.csv", comment="#")
    assert np.allclose(written["co2_ppm"], 284.3)
    assert "held at 1850" in (out / "site" / "co2.csv").read_text().splitlines()[0]
    assert record["weather_cells"] == 1
    assert json.loads((out / "scenario.json").read_text())["experiment"] == EXP
    # Idempotent: a second call replaces its own links.
    link_base_export(cfg)


def test_absent_variable_filled_from_mswx_climatology(cmip_root, tmp_path):
    """sfcWind, absent from the CMIP source, takes MSWX's value for the same day of year."""
    mswx = tmp_path / "MSWX" / "SFCWIND"
    mswx.mkdir(parents=True)
    lat, lon = np.round(np.arange(55.95, 50.0, -0.1), 2), LON
    for stem, value in (("1979001", 1.0), ("1979002", 2.0), ("1979365", 365.0)):
        xr.Dataset(
            {"wind_speed": (("time", "lat", "lon"), np.full((1, lat.size, lon.size), value, "float32"))},
            coords={"time": [0], "lat": lat, "lon": lon},
        ).to_netcdf(mswx / f"{stem}.nc", engine="h5netcdf")

    cfg = _config(
        cmip_root,
        variables={"TAS": "tas", "SFCWIND": "sfcWind"},
        required=["tas", "sfcWind"],
        climatology_fill=["sfcWind"],
        climatology_years=[1979, 1979],
        climatology_window_days=1,
        climatology_cache=str(tmp_path / "clim"),
    )
    cfg = cfg.model_copy(update={"paths": cfg.paths.model_copy(update={"mswx_root": mswx.parent})})
    handler = MSWXHandler(cfg)
    assert handler.missing_required() == [] and handler.filled() == ["sfcWind"]

    ds = handler.load()
    # 1990-12-31, 1991-01-01, 1991-01-02 -> day of year 365, 1, 2.
    assert np.allclose(ds["sfcWind"].isel(lat=0, lon=0).values, [365.0, 1.0, 2.0])
    assert len(list((tmp_path / "clim").glob("sfcWind_1979-1979_w1_*.nc"))) == 1
    # The cached climatology is reused, not recomputed.
    for f in mswx.iterdir():
        f.unlink()
    assert np.allclose(MSWXHandler(cfg).load()["sfcWind"].isel(lat=0, lon=0).values, [365.0, 1.0, 2.0])


def test_real_variable_wins_over_fill(cmip_root):
    cfg = _config(cmip_root, climatology_fill=["pr"])
    assert MSWXHandler(cfg).filled() == []
