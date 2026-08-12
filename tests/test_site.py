"""Tests for the site stage: crop calendar, elevation, CO2 and the exporter."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest
import xarray as xr

from data4simplace.config import PipelineConfig
from data4simplace.exporters.site_export import SiteExporter
from data4simplace.grid import TargetGrid
from data4simplace.site.calendar import (
    calendar_path,
    load_calendar,
    resolve_calendar_crop,
)
from data4simplace.site.co2 import FALLBACK_CO2_PPM, load_co2_series, write_co2_series
from data4simplace.site.elevation import load_elevation, resolve_dem_variable
from data4simplace.site.handler import SiteHandler, fill_calendar_gaps


# --------------------------------------------------------------------------- #
# Fixtures
# --------------------------------------------------------------------------- #


#: Target grid for the site tests: 10.5-12.5 E, 51.5-53.5 N at 0.1 deg, so a
#: 20x20 target grid sits under a 4x4 grid of 0.5 deg calendar cells and
#: different target cells sample different source cells.
SITE_GRID = {
    "resolution_deg": 0.1,
    "min_lon": 10.5,
    "max_lon": 12.5,
    "min_lat": 51.5,
    "max_lat": 53.5,
    "crs": "EPSG:4326",
}


@pytest.fixture()
def sage_file(tmp_path):
    """A 4x4 SAGE-shaped calendar at 0.5 deg covering :data:`SITE_GRID`.

    The south-west source cell is NaN so the "product does not cover this cell"
    path is exercised, and one covered cell has a NaN ``index`` so the product's
    own extrapolation flag is exercised too.
    """
    lat = np.array([51.75, 52.25, 52.75, 53.25])
    lon = np.array([10.75, 11.25, 11.75, 12.25])
    shape = (lat.size, lon.size)

    plant = np.full(shape, 280.0)
    plant[0, 0] = np.nan  # a cell the crop mask does not cover
    index = np.ones(shape)
    index[1, 1] = np.nan  # SAGE extrapolated this one from a neighbour

    dataset = xr.Dataset(
        {
            "plant": (("latitude", "longitude"), plant),
            "plant.start": (("latitude", "longitude"), np.full(shape, 265.0)),
            "plant.end": (("latitude", "longitude"), np.full(shape, 300.0)),
            "harvest": (("latitude", "longitude"), np.full(shape, 210.0)),
            "tot.days": (
                ("latitude", "longitude"),
                np.full(shape, 295, dtype="timedelta64[D]").astype("timedelta64[ns]"),
            ),
            "index": (("latitude", "longitude"), index),
        },
        coords={"latitude": lat, "longitude": lon},
    )
    path = tmp_path / "Wheat.Winter.crop.calendar.fill.nc"
    dataset.to_netcdf(path)
    return path


@pytest.fixture()
def dem_file(tmp_path):
    """A 0.05 deg terrain DEM rising west to east, finer than the target grid."""
    lat = np.arange(51.475, 53.55, 0.05)
    lon = np.arange(10.475, 12.55, 0.05)
    elevation = np.tile(np.arange(lon.size, dtype="float64") * 10.0, (lat.size, 1))
    dataset = xr.Dataset(
        {"elevation": (("lat", "lon"), elevation)},
        coords={"lat": lat, "lon": lon},
    )
    path = tmp_path / "dem.nc"
    dataset.to_netcdf(path)
    return path


@pytest.fixture()
def site_config(config_dict, sage_file, dem_file) -> PipelineConfig:
    """The shared config fixture, on :data:`SITE_GRID`, with the site stage on."""
    config_dict = dict(config_dict)
    config_dict["flags"] = {
        **config_dict["flags"],
        "run_site_processing": True,
        "export_simplace_site": True,
    }
    config_dict["grid"] = dict(SITE_GRID)
    config_dict["paths"] = {
        **config_dict["paths"],
        "calendar_root": str(sage_file.parent),
        "dem_path": str(dem_file),
    }
    config_dict["npk"] = {"simplace_crop": "winter_wheat"}
    return PipelineConfig.model_validate(config_dict)


# --------------------------------------------------------------------------- #
# Config
# --------------------------------------------------------------------------- #


def test_site_export_requires_the_stage(config_dict):
    config_dict["flags"] = {**config_dict["flags"], "export_simplace_site": True}
    with pytest.raises(ValueError, match="requires flags.run_site_processing"):
        PipelineConfig.model_validate(config_dict)


def test_site_stage_requires_its_inputs(config_dict):
    config_dict["flags"] = {**config_dict["flags"], "run_site_processing": True}
    with pytest.raises(ValueError, match="calendar_root.*dem_path"):
        PipelineConfig.model_validate(config_dict)


def test_calendar_crop_resolution(site_config):
    assert resolve_calendar_crop(site_config) == "Wheat.Winter"

    unknown = site_config.model_copy(
        update={"npk": site_config.npk.model_copy(update={"simplace_crop": "quinoa"})}
    )
    with pytest.raises(ValueError, match="site.calendar_crop"):
        resolve_calendar_crop(unknown)

    override = unknown.model_copy(
        update={"site": unknown.site.model_copy(update={"calendar_crop": "Barley"})}
    )
    assert resolve_calendar_crop(override) == "Barley"


def test_calendar_path_hints_at_a_gzipped_file(tmp_path):
    (tmp_path / "Oats.crop.calendar.fill.nc.gz").write_bytes(b"")
    with pytest.raises(FileNotFoundError, match="gunzip"):
        calendar_path(tmp_path, "Oats")


# --------------------------------------------------------------------------- #
# Calendar
# --------------------------------------------------------------------------- #


def test_calendar_samples_nearest_and_flags_coverage(site_config):
    grid = TargetGrid.from_config(site_config.grid)
    calendar = load_calendar(site_config, grid)

    assert calendar.sizes == {"lat": 20, "lon": 20}
    # Nearest sampling, never averaging: a date is one of the source values.
    sampled = calendar["sowing_doy"].values
    assert set(np.unique(sampled[np.isfinite(sampled)])) == {280.0}
    assert float(calendar["season_length_days"].max()) == 295.0

    # The uncovered source cell leaves its target cells flagged not-present,
    # and their dates NaN rather than zero.
    present = calendar["calendar_present"].values
    assert present.min() == 0.0 and present.max() == 1.0
    assert np.isnan(sampled[present == 0.0]).all()

    # The product's own extrapolation flag survives the sampling.
    assert float(calendar["calendar_filled"].max()) == 1.0


def test_calendar_gap_fill_is_bounded_by_the_mask(site_config):
    grid = TargetGrid.from_config(site_config.grid)
    site = load_calendar(site_config, grid)

    # Unbounded: every gap is reachable, so all cells end up with a date.
    filled = fill_calendar_gaps(site, within=None)
    assert np.isfinite(filled["sowing_doy"].values).all()
    assert (filled["calendar_source"].values == 2).any()  # 2 = nearest

    # Bounded to the cells that already have a date: nothing is borrowed.
    within = site["calendar_present"] == 1.0
    bounded = fill_calendar_gaps(site, within=within)
    assert (bounded["calendar_source"].values == 2).sum() == 0
    gaps = site["calendar_present"].values == 0.0
    assert np.isnan(bounded["sowing_doy"].values[gaps]).all()


# --------------------------------------------------------------------------- #
# Elevation
# --------------------------------------------------------------------------- #


def test_elevation_binned_mean(site_config):
    grid = TargetGrid.from_config(site_config.grid)
    altitude = load_elevation(site_config, grid)

    assert altitude.name == "altitude_m"
    assert altitude.sizes == {"lat": 20, "lon": 20}
    # The DEM rises 10 m per 0.05 deg eastwards, so each 0.1 deg column is the
    # mean of two source columns and successive columns differ by 20 m.
    row = altitude.isel(lat=0).values
    assert np.allclose(np.diff(row), 20.0)
    # Rows are identical: the source varies only with longitude.
    assert np.allclose(altitude.values - altitude.values[0], 0.0)


def test_geoid_variable_is_rejected():
    dataset = xr.Dataset({"geoid_altitude": (("lat", "lon"), np.zeros((2, 2)))})
    with pytest.raises(ValueError, match="geoid/datum offset"):
        resolve_dem_variable(dataset, "geoid_altitude")
    # It is not picked up by auto-detection either.
    with pytest.raises(ValueError, match="No elevation variable"):
        resolve_dem_variable(dataset, None)


def test_named_dem_variable_must_exist():
    dataset = xr.Dataset({"elevation": (("lat", "lon"), np.zeros((2, 2)))})
    assert resolve_dem_variable(dataset, None) == "elevation"
    with pytest.raises(ValueError, match="not in the DEM"):
        resolve_dem_variable(dataset, "dem")


# --------------------------------------------------------------------------- #
# CO2
# --------------------------------------------------------------------------- #


def test_co2_fallback_interpolates_between_anchors():
    series, source = load_co2_series(None, [1995, 2000, 2005])
    assert source == "fallback"
    assert series[2000] == FALLBACK_CO2_PPM[2000]
    # 1995 sits halfway between the 1990 and 2000 anchors.
    assert series[1995] == pytest.approx(
        (FALLBACK_CO2_PPM[1990] + FALLBACK_CO2_PPM[2000]) / 2.0, abs=0.3
    )


def test_co2_file_is_read_and_years_outside_it_are_carried(tmp_path, caplog):
    path = tmp_path / "co2.csv"
    path.write_text("year,co2_ppm\n2000,370.0\n2002,374.0\n")

    series, source = load_co2_series(path, [2000, 2001, 2002, 2010])
    assert source == "file"
    assert series[2001] == pytest.approx(372.0)   # interpolated inside the range
    assert series[2010] == pytest.approx(374.0)   # carried from the last year
    assert "outside the CO2 series" in caplog.text


def test_missing_co2_file_is_an_error_not_a_fallback(tmp_path):
    with pytest.raises(FileNotFoundError):
        load_co2_series(tmp_path / "absent.csv")


def test_write_co2_series_records_its_provenance(tmp_path):
    series, source = load_co2_series(None, [2000, 2001])
    out = write_co2_series(series, source, tmp_path)

    assert out.is_file() and out.name == "co2.csv"
    assert out.read_text().splitlines()[0] == "# atmospheric CO2 [ppm], source: fallback"
    frame = pd.read_csv(out, comment="#")
    assert list(frame.columns) == ["year", "co2_ppm"]
    assert len(frame) == 2


# --------------------------------------------------------------------------- #
# Exporter
# --------------------------------------------------------------------------- #


def _cell_table(config) -> pd.DataFrame:
    return TargetGrid.from_config(config.grid).cell_table()


def test_site_exporter_writes_one_row_per_cell(tmp_path, site_config):
    site = SiteHandler(site_config).load()
    site = fill_calendar_gaps(site, within=None)
    cells = _cell_table(site_config)

    out = SiteExporter(site_config).export(site, cells, tmp_path)
    assert out.is_file() and out.name == "site.csv"

    written = pd.read_csv(out)
    assert len(written) == len(cells)
    assert written["location"].tolist() == cells["SimplaceID"].tolist()
    assert list(written.columns)[:4] == [
        "location", "latitude", "longitude", "altitude_m",
    ]
    # Sampled dates are whole days, and the altitude follows the DEM.
    assert (written["sowing_doy"] % 1 == 0).all()
    assert written["altitude_m"].between(0, 400).all()
    assert written["calendar_product"].eq("Wheat.Winter").all()


def test_uncovered_cells_take_the_fallback_and_say_so(tmp_path, site_config):
    """A cell no calendar reaches must be marked, not silently defaulted."""
    site = SiteHandler(site_config).load()
    # Bound the fill to the covered cells, so the uncovered ones stay gaps.
    site = fill_calendar_gaps(site, within=site["calendar_present"] == 1.0)
    cells = _cell_table(site_config)

    frame = SiteExporter(site_config).build_frame(site, cells)
    fallback = frame[frame["calendar_source"] == "fallback"]
    assert not fallback.empty
    assert (fallback["sowing_doy"] == site_config.site.fallback_sowing_doy).all()
    assert (frame[frame["calendar_source"] == "product"]["sowing_doy"] == 280).all()


def test_site_cell_set_matches_the_soil_cell_set(tmp_path, site_config, soil_dataset):
    """site.csv and soil.csv must key on exactly the same cells."""
    from data4simplace.exporters.soil_export import SoilExporter

    cells = _cell_table(site_config).iloc[:2].copy()
    cells["row"] = [0, 1]
    cells["col"] = [0, 1]
    cells["lat"] = soil_dataset["lat"].values
    cells["lon"] = soil_dataset["lon"].values

    site = fill_calendar_gaps(SiteHandler(site_config).load(), within=None)
    site_frame = SiteExporter(site_config).build_frame(site, cells)
    soil_frame = SoilExporter(site_config, reference_path=None).build_frame(
        soil_dataset, cells
    )
    assert set(site_frame["location"]) == set(soil_frame["location"])
