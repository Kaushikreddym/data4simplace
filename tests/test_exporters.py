"""Tests for the reference parser and exporters."""

from __future__ import annotations

import numpy as np
import pandas as pd
import xarray as xr

from data4simplace.exporters.base_exporter import parse_reference_csv
from data4simplace.exporters.soil_export import SoilExporter, remap_depth_weighted
from data4simplace.grid import TargetGrid


def test_parse_reference_detects_semicolon_and_missing(tmp_path):
    ref = tmp_path / "ref.csv"
    ref.write_text(
        "SimplaceID;DEPTH_TOP;DEPTH_BOTTOM;CLAY_0-5cm;CLAY_5-15cm\n"
        "1;0;5;20.0;-99\n"
        "1;5;15;-99;22.0\n"
    )
    spec = parse_reference_csv(ref)
    assert spec.delimiter == ";"
    assert spec.missing_value == "-99"
    assert spec.columns[0] == "SimplaceID"
    assert spec.depths == ["0-5cm", "5-15cm"]


def test_remap_depth_weighted_overlap():
    # Two source layers 0-10 / 10-20 with values 10 and 20; a destination layer
    # 0-20 must be the thickness-weighted mean (15), and 5-15 also 15.
    values = np.array([[10.0], [20.0]])  # (n_src, 1)
    src = [(0.0, 10.0), (10.0, 20.0)]
    out = remap_depth_weighted(values, src, [(0.0, 20.0), (5.0, 15.0)])
    assert np.allclose(out[:, 0], [15.0, 15.0])
    # A destination layer beyond all source coverage is NaN.
    out2 = remap_depth_weighted(values, src, [(20.0, 30.0)])
    assert np.isnan(out2[0, 0])


def _cells_over_fixture(config, soil_dataset):
    grid = TargetGrid.from_config(config.grid)
    cells = grid.cell_table().iloc[:2].copy()
    cells["row"] = [0, 1]
    cells["col"] = [0, 1]
    cells["lat"] = soil_dataset["lat"].values
    cells["lon"] = soil_dataset["lon"].values
    return cells


def test_soil_exporter_writes_wide_file(tmp_path, config, soil_dataset):
    cells = _cells_over_fixture(config, soil_dataset)

    exporter = SoilExporter(config, reference_path=None)  # fallback wide schema
    out = exporter.export(soil_dataset, cells, tmp_path)
    assert out.is_file() and out.name == "soil.csv"

    written = pd.read_csv(out)
    # One row per cell, wide format with per-layer columns.
    assert len(written) == 2
    assert "location" in written.columns
    for col in ("clay_1", "sand_1", "bulkdensity_1", "carbon_1", "PH_1"):
        assert col in written.columns
    # Uniform fixture -> clay stays 20 % after depth remap.
    assert np.allclose(written["clay_1"], 20.0)
    assert set(written["location"]) == set(cells["SimplaceID"])


def test_soil_exporter_conforms_and_carries_constants(tmp_path, config, soil_dataset):
    # Reference with a derivable column (clay_1), a per-layer non-derivable
    # column (CaCO3_1) and a single-value constant (soiltype).
    ref = tmp_path / "soil_ref.csv"
    ref.write_text("location,soiltype,clay_1,CaCO3_1\n49612,AP5,7.7,202\n")

    cells = _cells_over_fixture(config, soil_dataset)
    exporter = SoilExporter(config, reference_path=ref)
    out = exporter.export(soil_dataset, cells, tmp_path)
    written = pd.read_csv(out)

    # Exactly the reference columns, in order.
    assert list(written.columns) == ["location", "soiltype", "clay_1", "CaCO3_1"]
    # clay_1 overwritten from SoilGrids (20), constants carried from the template.
    assert np.allclose(written["clay_1"], 20.0)
    assert set(written["soiltype"]) == {"AP5"}
    assert set(written["CaCO3_1"]) == {202}


def test_soil_exporter_writes_water_retention_from_soilgrids(tmp_path, config, soil_dataset):
    cells = _cells_over_fixture(config, soil_dataset)
    exporter = SoilExporter(config, reference_path=None)
    written = pd.read_csv(exporter.export(soil_dataset, cells, tmp_path))

    # vol% -> m3/m3, straight from wv0010/wv0033/wv1500, no PTF involved.
    assert np.allclose(written["soilwater_sat_1"], 0.42)
    assert np.allclose(written["soilwater_fc_1"], 0.31)
    assert np.allclose(written["soilwater_wp_1"], 0.14)
    # Initial water content starts at field capacity.
    assert np.allclose(written["soilwater_init_1"], written["soilwater_fc_1"])


def test_water_content_layers_take_precedence_over_the_ptf(tmp_path, config, soil_dataset):
    # A PTF result that disagrees with the wv* layers must not win.
    hydraulic = xr.Dataset(
        {
            name: (("depth", "lat", "lon"), np.full((6, 2, 2), value, dtype="float32"))
            for name, value in (
                ("theta_fc", 0.99), ("theta_wp", 0.98),
                ("theta_sat", 0.97), ("theta_paw", 0.01),
            )
        },
        coords=soil_dataset.coords,
    )
    cells = _cells_over_fixture(config, soil_dataset)
    exporter = SoilExporter(config, reference_path=None)
    written = pd.read_csv(exporter.export(soil_dataset, cells, tmp_path, hydraulic=hydraulic))

    assert np.allclose(written["soilwater_fc_1"], 0.31)
    # The PTF still fills a column the water-content layers cannot cover.
    no_water = soil_dataset.drop_vars(["wv0010", "wv0033", "wv1500"])
    fallback = pd.read_csv(
        exporter.export(no_water, cells, tmp_path, hydraulic=hydraulic)
    )
    assert np.allclose(fallback["soilwater_fc_1"], 0.99)


def test_mineral_nitrogen_from_total_n_and_bulk_density(tmp_path, config, soil_dataset):
    cells = _cells_over_fixture(config, soil_dataset)
    exporter = SoilExporter(config, reference_path=None)
    written = pd.read_csv(exporter.export(soil_dataset, cells, tmp_path))

    # Layer 1 is 0-10 cm: 1.2 g/kg * 1.4 kg/dm3 * 100 * 10 cm = 1680 kg N/ha
    # total, of which 1 % is mineral, split 30/70 ammonium:nitrate.
    mineral = 1.2 * 1.4 * 100 * 10 * config.soil.mineral_n_fraction
    assert np.allclose(written["ammonium_1"], mineral * 0.3)
    assert np.allclose(written["nitrate_1"], mineral * 0.7)
    # Layer 6 spans 100-200 cm, so the same concentration gives a 10x stock.
    assert np.allclose(written["nitrate_6"], mineral * 0.7 * 10)


def test_mineral_nitrogen_off_falls_back_to_the_reference_constant(tmp_path, config_dict, soil_dataset):
    from data4simplace.config import PipelineConfig

    config_dict["soil"] = {"mineral_n_fraction": 0.0}
    config = PipelineConfig.model_validate(config_dict)

    ref = tmp_path / "soil_ref.csv"
    ref.write_text("location,clay_1,nitrate_1\n49612,7.7,202\n")

    cells = _cells_over_fixture(config, soil_dataset)
    exporter = SoilExporter(config, reference_path=ref)
    written = pd.read_csv(exporter.export(soil_dataset, cells, tmp_path))
    assert set(written["nitrate_1"]) == {202}


# --------------------------------------------------------------------------- #
# Weather exporter
# --------------------------------------------------------------------------- #


def _climate_dataset(config, **variables) -> xr.Dataset:
    """A tiny (time, lat, lon) climate cube on the fixture grid."""
    grid = TargetGrid.from_config(config.grid)
    lats, lons = grid.lat_centers[:2], grid.lon_centers[:2]
    times = pd.date_range("1979-01-01", periods=3)
    return xr.Dataset(
        {
            name: (
                ("time", "lat", "lon"),
                np.full((times.size, lats.size, lons.size), value, dtype="float32"),
            )
            for name, value in variables.items()
        },
        coords={"time": times, "lat": lats, "lon": lons},
    )


def _weather_cells(config) -> pd.DataFrame:
    grid = TargetGrid.from_config(config.grid)
    cells = grid.cell_table().iloc[:1].copy()
    cells["row"], cells["col"] = 0, 0
    return cells


def test_weather_exporter_writes_wind_from_sfcwind(tmp_path, config):
    """``sfcWind`` must reach the ``Windspeed`` column, converted to 2 m.

    Regression test: ``SFCWIND`` was present in ``climate.variables`` and loaded
    by the handler, but absent from the exporter's canonical->SIMPLACE map, so
    every exported file carried the ``-99.9`` sentinel. Nothing caught it.
    """
    from data4simplace.exporters.weather_export import WeatherExporter

    climate = _climate_dataset(config, tas=8.0, sfcWind=10.0)
    exporter = WeatherExporter(config, reference_path=None)
    written = exporter.export(climate, _weather_cells(config), tmp_path)
    assert len(written) == 1

    frame = pd.read_csv(written[0], sep="\t")
    assert "Windspeed" in frame.columns
    # FAO-56 eq. 47 at z = 10 m: u2 = u10 * 4.87 / ln(67.8*10 - 5.42).
    expected = 10.0 * 4.87 / np.log(67.8 * 10.0 - 5.42)
    assert np.allclose(frame["Windspeed"], round(expected, 2))
    # The conversion must not be a no-op, or the 10 m product leaks through.
    assert frame["Windspeed"].iloc[0] < 10.0
    assert np.allclose(frame["TempMean"], 8.0)


def test_weather_exporter_sentinels_columns_without_a_source(tmp_path, config):
    """A reference column MSWX cannot supply stays at the sentinel."""
    from data4simplace.exporters.weather_export import WeatherExporter

    climate = _climate_dataset(config, tas=8.0)
    exporter = WeatherExporter(config, reference_path=None)
    written = exporter.export(climate, _weather_cells(config), tmp_path)

    frame = pd.read_csv(written[0], sep="\t")
    # No sfcWind in the dataset, and RefET has no MSWX source at all.
    assert set(frame["Windspeed"]) == {-99.9}
    assert set(frame["RefET"]) == {-99.9}


def test_weather_exporter_drops_unmapped_variables(tmp_path, config):
    """A loaded variable with no entry in the map must not invent a column."""
    from data4simplace.exporters.weather_export import WeatherExporter

    climate = _climate_dataset(config, tas=8.0, sfcWindmax=30.0)
    exporter = WeatherExporter(config, reference_path=None)
    written = exporter.export(climate, _weather_cells(config), tmp_path)

    frame = pd.read_csv(written[0], sep="\t")
    assert "sfcWindmax" not in frame.columns
