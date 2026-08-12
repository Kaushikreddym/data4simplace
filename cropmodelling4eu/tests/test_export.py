"""Tests for the export reader layer."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from cropmodelling4eu.config import GridConfig, RunConfig
from cropmodelling4eu.export import resolve_export
from cropmodelling4eu.export.cells import (
    cell_frame,
    id_to_lonlat,
    id_to_rowcol,
    rowcol_to_id,
    shard_cells,
    weather_ids,
    weather_path,
)
from cropmodelling4eu.export.management import (
    irrigation_flags,
    load_composition,
    read_fertilizer_plans,
)
from cropmodelling4eu.export.site import read_co2, read_site
from cropmodelling4eu.export.soil import read_soil
from cropmodelling4eu.export.weather import (
    load_season_block,
    load_seasons,
    season_window,
    sowing_date,
)

from .conftest import LAYER_BOTTOMS_M, TEST_CELLS, TEST_GRID


# --------------------------------------------------------------------------- #
# Cell identity
# --------------------------------------------------------------------------- #


def test_id_rowcol_round_trip(grid):
    ids = np.array([1, 2, 11, 35, 100])
    row, col = id_to_rowcol(ids, grid)
    assert rowcol_to_id(row, col, grid).tolist() == ids.tolist()


def test_id_to_lonlat_places_the_first_cell_at_the_north_west_corner(grid):
    lon, lat = id_to_lonlat(np.array([1]), grid)
    # Half a cell in from the corner, and latitude runs north to south.
    assert lon[0] == pytest.approx(TEST_GRID["min_lon"] + 0.05)
    assert lat[0] == pytest.approx(TEST_GRID["max_lat"] - 0.05)

    # The next row down is one resolution step south.
    _, lat_next = id_to_lonlat(np.array([1 + grid.n_lon]), grid)
    assert lat_next[0] == pytest.approx(lat[0] - grid.resolution_deg)


def test_grid_drives_the_decoding_not_a_constant():
    """The same id decodes differently under a different grid."""
    europe = GridConfig(resolution_deg=0.1, min_lon=-17.0, max_lon=52.0,
                        min_lat=34.0, max_lat=72.0)
    other = GridConfig(**TEST_GRID)
    assert id_to_rowcol(700, europe)[0] != id_to_rowcol(700, other)[0]


def test_weather_path_and_discovery(export_dir, grid):
    path = weather_path(export_dir, TEST_CELLS[0], grid)
    assert path.is_file() and path.name == "daily_mean_RES1_C0R0.csv.gz"
    assert weather_ids(export_dir, grid).tolist() == sorted(TEST_CELLS)


def test_weather_discovery_reports_an_empty_export(tmp_path, grid):
    (tmp_path / "weather").mkdir()
    with pytest.raises(FileNotFoundError, match="No weather files"):
        weather_ids(tmp_path, grid)


def test_weather_reads_the_row_nested_layout(tmp_path, grid):
    """data4simplace nests ``weather/<row>/<file>``; the flat one still works.

    Both layouts exist on disk — every export written before 2026-08 is flat —
    so discovery and path resolution must not depend on which.
    """
    from cropmodelling4eu.export.cells import _is_nested

    from .conftest import TEST_CELLS, _write_weather

    _write_weather(tmp_path / "weather", grid, TEST_CELLS, nested=True)
    _is_nested.cache_clear()

    path = weather_path(tmp_path, TEST_CELLS[0], grid)
    row = (TEST_CELLS[0] - 1) // grid.n_lon
    assert path.is_file() and path.parent.name == str(row)
    # The id comes from the filename, so a nested export enumerates identically.
    assert weather_ids(tmp_path, grid).tolist() == sorted(TEST_CELLS)


def test_shard_cells_deals_round_robin():
    ids = np.arange(1, 11)
    assert shard_cells(ids, 0, 3).tolist() == [1, 4, 7, 10]
    assert shard_cells(ids, 1, 3).tolist() == [2, 5, 8]
    # Every cell lands in exactly one shard.
    dealt = np.concatenate([shard_cells(ids, s, 3) for s in range(3)])
    assert sorted(dealt.tolist()) == ids.tolist()
    with pytest.raises(ValueError, match="out of range"):
        shard_cells(ids, 3, 3)


def test_cell_frame_carries_coordinates(grid):
    frame = cell_frame(np.array(TEST_CELLS), grid)
    assert list(frame.columns) == ["SimplaceID", "row", "col", "lon", "lat"]
    assert len(frame) == len(TEST_CELLS)


# --------------------------------------------------------------------------- #
# Soil
# --------------------------------------------------------------------------- #


def test_soil_reads_layer_geometry_from_the_file(export_dir):
    soil = read_soil(export_dir, layout="wide")
    assert soil.bottoms_m.tolist() == LAYER_BOTTOMS_M
    assert soil.tops_m.tolist() == [0.0, *LAYER_BOTTOMS_M[:-1]]
    assert soil.n_layers == 6
    assert soil.ids.tolist() == sorted(TEST_CELLS)


def test_soil_falls_back_to_the_default_geometry_with_a_warning(tmp_path, caplog):
    soil_dir = tmp_path / "soil"
    soil_dir.mkdir()
    frame = pd.DataFrame({"location": [1], **{f"clay_{n}": 20.0 for n in range(1, 7)}})
    frame.to_csv(soil_dir / "soil.csv", index=False)

    soil = read_soil(tmp_path, layout="wide")
    assert soil.bottoms_m.tolist() == [0.1, 0.3, 0.5, 0.7, 1.0, 2.0]
    assert "declares no layer depths" in caplog.text


def test_both_layouts_describe_the_same_soil(export_dir):
    wide = read_soil(export_dir, layout="wide")
    long_ = read_soil(export_dir, layout="long")

    assert wide.bottoms_m.tolist() == long_.bottoms_m.tolist()
    assert wide.ids.tolist() == long_.ids.tolist()
    for stem in ("clay", "sand", "bulkdensity", "soilwater_fc", "soilwater_wp"):
        assert long_.values[stem] == pytest.approx(wide.values[stem])
    # carbon differs by a unit factor in the file and must not after reading.
    assert long_.values["carbon"] == pytest.approx(wide.values["carbon"])


def test_auto_layout_prefers_the_wide_file(export_dir):
    auto = read_soil(export_dir)
    # The wide fixture carries mineral N; the long one does not.
    assert "ammonium" in auto


def test_depth_mean_and_sum(export_dir):
    soil = read_soil(export_dir, layout="wide")
    # A uniform profile: the thickness-weighted mean is the layer value.
    assert soil.depth_mean("clay", 0.0, 1.0) == pytest.approx(20.0)
    # A stock is additive: the rooting zone holds the first five layers whole.
    assert soil.depth_sum("ammonium", 0.0, 1.0) == pytest.approx(12.0 * 5)
    # Half of the last layer is half its stock.
    assert soil.depth_sum("ammonium", 1.0, 1.5) == pytest.approx(12.0 * 0.5)


def test_depth_window_outside_the_profile_is_an_error(export_dir):
    soil = read_soil(export_dir, layout="wide")
    with pytest.raises(ValueError, match="overlaps none"):
        soil.depth_mean("clay", 3.0, 4.0)


def test_select_orders_and_reports_missing_cells(export_dir):
    soil = read_soil(export_dir, layout="wide")
    picked = soil.select(np.array([TEST_CELLS[2], TEST_CELLS[0]]))
    assert picked.ids.tolist() == [TEST_CELLS[2], TEST_CELLS[0]]

    with pytest.raises(KeyError, match="no soil profile"):
        soil.select(np.array([TEST_CELLS[0], 999999]))


def test_ragged_long_file_is_rejected(export_dir):
    """A long file must be rectangular for SIMPLACE to assemble its arrays."""
    path = export_dir / "soil" / "soil_long.csv"
    frame = pd.read_csv(path)
    frame.drop(index=frame.index[-1]).to_csv(path, index=False)
    with pytest.raises(ValueError, match="do not have 6 layers"):
        read_soil(export_dir, layout="long")


# --------------------------------------------------------------------------- #
# Management
# --------------------------------------------------------------------------- #


def test_composition_reads_elemental_contents(export_dir):
    composition = load_composition(export_dir / "management" / "fertilizer_composition.xml")
    assert composition.loc["KAS", "N"] == pytest.approx(0.27)
    assert composition.loc["P", "P"] == pytest.approx(0.4364)


def test_plans_convert_products_to_nutrients(export_dir):
    plans = read_fertilizer_plans(
        export_dir / "management" / "fertilizer_winter_wheat.csv",
        export_dir / "management" / "fertilizer_composition.xml",
    )
    plan = plans[TEST_CELLS[0]]
    assert plan.shape == (5, 4)
    # Sorted by DVS, so the two DVS 0.001 events lead.
    assert plan[:, 0].tolist() == sorted(plan[:, 0].tolist())
    # 64 g of KAS at 27 % N = 17.28 g N/m2 across the three dressings.
    assert plan[:, 1].sum() == pytest.approx(64.0 * 0.27)
    assert plan[:, 2].sum() == pytest.approx(40.0 * 0.4364)
    assert plan[:, 3].sum() == pytest.approx(40.0 * 0.8302)


def test_wide_schedule_without_a_composition_file_is_an_error(export_dir):
    with pytest.raises(ValueError, match="fertilizer_composition.xml"):
        read_fertilizer_plans(
            export_dir / "management" / "fertilizer_winter_wheat.csv", None
        )


def test_unknown_carrier_is_named(export_dir):
    path = export_dir / "management" / "fertilizer_winter_wheat.csv"
    frame = pd.read_csv(path)
    frame.loc[0, "vType"] = "Mystery"
    frame.to_csv(path, index=False)
    with pytest.raises(ValueError, match="Mystery"):
        read_fertilizer_plans(path, export_dir / "management" / "fertilizer_composition.xml")


def test_long_schedule_needs_no_composition(tmp_path):
    """A long schedule's amounts are nutrients already."""
    path = tmp_path / "fertilizer_long.csv"
    pd.DataFrame(
        {
            "Location": [1, 1, 2],
            "ENZ": -99,
            "vCrop": "winter_wheat",
            "Year": 2000,
            "vIRRIGATION": 0,
            "Number": [1, 2, 1],
            "DVS": [0.25, 0.9, 0.25],
            "Amount": [9.0, 9.0, 12.0],
        }
    ).to_csv(path, index=False)

    plans = read_fertilizer_plans(path)
    assert plans[1][:, 1].sum() == pytest.approx(18.0)
    assert plans[2].shape == (1, 4)


def test_irrigation_flags(export_dir):
    flags = irrigation_flags(export_dir / "management" / "fertilizer_winter_wheat.csv")
    assert flags.loc[TEST_CELLS[-1]] == 1
    assert flags.loc[TEST_CELLS[0]] == 0


# --------------------------------------------------------------------------- #
# Site and CO2
# --------------------------------------------------------------------------- #


def test_site_table_is_read_per_cell(export_dir):
    site = read_site(export_dir)
    assert not site.is_fallback
    ids = np.array(TEST_CELLS)
    assert site.sowing_doy(ids).tolist() == [280, 280, 295, 295]
    assert site.altitude_m(ids).tolist() == [10.0, 20.0, 300.0, 1200.0]
    assert "sowing DOY 280-295" in site.summarise()


def test_missing_site_table_falls_back_and_says_so(tmp_path, caplog):
    site = read_site(tmp_path)
    assert site.is_fallback
    assert site.sowing_doy(np.array([1, 2])).tolist() == [270, 270]
    assert site.altitude_m(np.array([1])).tolist() == [0.0]
    assert "FALLBACK" in site.summarise()
    assert "largest single error" not in caplog.text  # that line is the CLI's
    assert "constant sowing DOY 270" in caplog.text


def test_site_fills_cells_absent_from_the_table(export_dir):
    site = read_site(export_dir)
    assert site.sowing_doy(np.array([999999])).tolist() == [270]


def test_co2_is_read_and_extended(export_dir):
    series = read_co2(export_dir, [2000, 2001, 2002])
    assert series.loc[2000] == pytest.approx(369.6)
    # A year past the series is carried from its end, not dropped.
    extended = read_co2(export_dir, [2002, 2010])
    assert extended.loc[2010] == pytest.approx(373.3)


# --------------------------------------------------------------------------- #
# Weather
# --------------------------------------------------------------------------- #


def test_sowing_date_picks_the_right_calendar_year():
    # An autumn sowing precedes its harvest year.
    assert sowing_date(2001, 280).year == 2000
    # A spring sowing is in the harvest year itself.
    assert sowing_date(2001, 90).year == 2001


def test_season_window_opens_before_sowing():
    start, end = season_window(2001, 280, spinup_months=6)
    assert start < sowing_date(2001, 280)
    # Two years from the window's start, so a late season is never truncated.
    assert (end - start).days >= 700


def test_load_seasons_slices_the_record_once(export_dir, grid):
    seasons = load_seasons(export_dir, TEST_CELLS[0], [2000, 2001], 280, grid)
    assert set(seasons) == {2000, 2001}
    block = seasons[2000]
    assert block.ndim == 2 and block.shape[1] == 8
    # Radiation is converted from a W/m2 flux to MJ/m2/d.
    assert block[:, 4].max() < 30.0
    # Vapour pressure is rebuilt and physically plausible [kPa].
    assert 0.1 < np.nanmedian(block[:, 6]) < 4.0
    # Wind is the real column, not the fallback.
    assert block[:, 7] == pytest.approx(2.5)


def test_years_the_record_cannot_cover_are_dropped(export_dir, grid):
    # The fixture record ends in 2002, so a 2003 harvest has no window.
    seasons = load_seasons(export_dir, TEST_CELLS[0], [2001, 2003], 280, grid)
    assert set(seasons) == {2001}


def test_load_season_block_shares_one_time_axis(export_dir, grid):
    ids = np.array(TEST_CELLS)
    blocks = load_season_block(export_dir, ids, [2000, 2001], 280, grid, workers=2)
    for year, (array, start_doy) in blocks.items():
        assert array.shape[0] == len(ids)
        assert array.shape[2] == 8
        assert int(array[0, 0, 0]) == start_doy


def test_missing_wind_column_falls_back(export_dir, grid):
    """A pre-fix export carries -99.9 in every Windspeed row."""
    import gzip

    path = weather_path(export_dir, TEST_CELLS[0], grid)
    with gzip.open(path, "rt") as handle:
        frame = pd.read_csv(handle, sep="\t")
    frame["Windspeed"] = -99.9
    with gzip.open(path, "wt") as handle:
        frame.to_csv(handle, sep="\t", index=False)

    block = load_seasons(export_dir, TEST_CELLS[0], [2001], 280, grid)[2001]
    assert block[:, 7] == pytest.approx(2.0)  # the documented constant


# --------------------------------------------------------------------------- #
# Bundle resolution
# --------------------------------------------------------------------------- #


def test_resolve_export_intersects_the_inputs(run_config):
    bundle = resolve_export(run_config)
    assert bundle.ids.tolist() == sorted(TEST_CELLS)
    assert bundle.soil.ids.tolist() == sorted(TEST_CELLS)
    assert set(bundle.plans) == set(TEST_CELLS)
    assert not bundle.site.is_fallback
    assert len(bundle.co2) == len(run_config.season.years)


def test_resolve_export_drops_cells_without_a_plan(run_config, export_dir):
    path = export_dir / "management" / "fertilizer_winter_wheat.csv"
    frame = pd.read_csv(path)
    frame[frame["location"] != TEST_CELLS[0]].to_csv(path, index=False)

    bundle = resolve_export(run_config)
    assert TEST_CELLS[0] not in bundle.ids.tolist()
    assert len(bundle.ids) == len(TEST_CELLS) - 1


def test_resolve_export_keeps_cells_without_a_site_row(run_config, export_dir):
    """Site never narrows the cell set: a missing row takes the fallback."""
    path = export_dir / "site" / "site.csv"
    frame = pd.read_csv(path)
    frame[frame["location"] != TEST_CELLS[0]].to_csv(path, index=False)

    bundle = resolve_export(run_config)
    assert TEST_CELLS[0] in bundle.ids.tolist()
    assert bundle.site.sowing_doy(np.array([TEST_CELLS[0]])).tolist() == [270]


def test_empty_intersection_names_every_input(run_config, export_dir):
    path = export_dir / "management" / "fertilizer_winter_wheat.csv"
    pd.read_csv(path).head(0).to_csv(path, index=False)
    with pytest.raises(RuntimeError, match="have weather.*have soil.*fertilizer plan"):
        resolve_export(run_config)


def test_potential_run_needs_no_schedule(run_config, export_dir):
    (export_dir / "management" / "fertilizer_winter_wheat.csv").unlink()
    bundle = resolve_export(run_config, require_management=False)
    assert bundle.ids.tolist() == sorted(TEST_CELLS)
    assert bundle.plans == {}


def test_config_reads_the_grid_back_from_the_export(export_dir):
    config = RunConfig.from_export(export_dir)
    assert config.grid.min_lon == TEST_GRID["min_lon"]
    assert config.grid.n_lon == 10


def test_config_warns_when_the_export_has_no_frozen_grid(tmp_path, caplog):
    config = RunConfig.from_export(tmp_path)
    assert "falling back to the default Europe grid" in caplog.text
    assert config.grid.n_lon == 690
