"""Tests for the irrigated / rainfed cell classification."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest
import xarray as xr

from data4simplace.config import PipelineConfig
from data4simplace.grid import TargetGrid
from data4simplace.management.irrigation import (
    SOURCE_ECIRA,
    SOURCE_MIRCA,
    SOURCE_NONE,
    IrrigationClassification,
    IrrigationClassifier,
    classify,
    conservative_regrid,
    irrigated_fraction,
    resolve_crop_group,
)


@pytest.fixture()
def irr_config(tmp_path):
    """A config on a 2x2 target grid with the irrigation stage enabled."""

    def _build(**irrigation):
        return PipelineConfig.model_validate(
            {
                "flags": {
                    "run_npk_processing": True,
                    "export_simplace_management": True,
                    "run_irrigation_classification": True,
                },
                "grid": {
                    "resolution_deg": 0.1,
                    "min_lon": 11.2,
                    "max_lon": 11.4,
                    "min_lat": 52.1,
                    "max_lat": 52.3,
                },
                "time": {"start": "1979-01-01", "end": "1979-01-03"},
                "paths": {
                    "mswx_root": str(tmp_path),
                    "npk_root": str(tmp_path),
                    "mirca_root": str(tmp_path / "mirca"),
                    "ecira_root": str(tmp_path / "ecira"),
                    "output_dir": str(tmp_path / "out"),
                },
                "irrigation": irrigation,
            }
        )

    return _build


# --------------------------------------------------------------------------- #
# The rule
# --------------------------------------------------------------------------- #
def test_irrigated_fraction_needs_a_minimum_crop_area():
    irrigated = np.array([[8.0, 60.0]])
    total = np.array([[8.0, 100.0]])  # first cell is all-irrigated but tiny
    fraction = irrigated_fraction(irrigated, total, min_crop_area_ha=10.0)
    assert np.isnan(fraction[0, 0])
    assert fraction[0, 1] == pytest.approx(0.6)


def test_irrigated_fraction_is_nan_where_the_crop_is_absent():
    fraction = irrigated_fraction(np.zeros((1, 1)), np.zeros((1, 1)), 0.0)
    assert np.isnan(fraction[0, 0])


def test_classify_is_strict_and_sends_unclassified_to_zero():
    fraction = np.array([[0.51, 0.5, 0.49, np.nan]])
    # Exactly at the threshold counts as rainfed: the rule is "> threshold".
    assert classify(fraction, 0.5).tolist() == [[1, 0, 0, 0]]
    assert classify(fraction, 0.5).dtype == np.int8


def test_classify_honours_a_non_default_threshold():
    fraction = np.array([[0.25, 0.45]])
    assert classify(fraction, 0.2).tolist() == [[1, 1]]
    assert classify(fraction, 0.8).tolist() == [[0, 0]]


# --------------------------------------------------------------------------- #
# Crop groups
# --------------------------------------------------------------------------- #
def test_winter_wheat_resolves_to_the_cereal_group(irr_config):
    # ECIRA has no wheat class, so a wheat run is classified on cereals.
    assert resolve_crop_group(irr_config()) == "cereals"


def test_crop_group_override_wins(irr_config):
    assert resolve_crop_group(irr_config(crop_group="maize")) == "maize"


def test_unknown_simplace_crop_is_an_error_not_a_guess(irr_config):
    config = irr_config()
    config.npk.simplace_crop = "sugar_beet"
    with pytest.raises(ValueError, match="irrigation.crop_group"):
        resolve_crop_group(config)


# --------------------------------------------------------------------------- #
# Regridding
# --------------------------------------------------------------------------- #
def test_conservative_regrid_conserves_the_total(irr_config):
    grid = TargetGrid.from_config(irr_config().grid)
    # A 6x6 source at 1/30 deg covering exactly the 2x2 target box.
    src_lat = np.arange(52.1 + 1 / 60, 52.3, 1 / 30)
    src_lon = np.arange(11.2 + 1 / 60, 11.4, 1 / 30)
    values = np.arange(src_lat.size * src_lon.size, dtype="float64").reshape(
        src_lat.size, src_lon.size
    )
    out = conservative_regrid(values, src_lat, src_lon, grid)
    assert out.shape == grid.shape
    assert out.sum() == pytest.approx(values.sum())


def test_conservative_regrid_follows_the_grids_north_to_south_rows(irr_config):
    grid = TargetGrid.from_config(irr_config().grid)
    assert grid.lat_centers[0] > grid.lat_centers[-1]  # the grid runs north -> south
    src_lat = np.array([52.125, 52.175, 52.225, 52.275])
    src_lon = np.array([11.225, 11.275, 11.325, 11.375])
    values = np.zeros((4, 4))
    values[3, 0] = 100.0  # northernmost, westernmost source cell
    out = conservative_regrid(values, src_lat, src_lon, grid)
    assert out[0, 0] == pytest.approx(100.0)  # -> the grid's first (northern) row
    assert out.sum() == pytest.approx(100.0)


def test_conservative_regrid_accepts_a_descending_source_latitude(irr_config):
    grid = TargetGrid.from_config(irr_config().grid)
    src_lat = np.array([52.275, 52.225, 52.175, 52.125])  # descending, as MIRCA-OS
    src_lon = np.array([11.225, 11.275, 11.325, 11.375])
    values = np.zeros((4, 4))
    values[0, 0] = 100.0  # again the northernmost, westernmost cell
    out = conservative_regrid(values, src_lat, src_lon, grid)
    assert out[0, 0] == pytest.approx(100.0)


# --------------------------------------------------------------------------- #
# Source selection
# --------------------------------------------------------------------------- #
def _classifier(config, mirca=None, ecira=None):
    """A classifier with the two product readers stubbed out."""
    classifier = IrrigationClassifier(config)
    if mirca is not None:
        classifier._mirca_fraction = lambda: mirca  # type: ignore[method-assign]
    if ecira is not None:
        classifier._ecira_fraction = lambda: ecira  # type: ignore[method-assign]
    return classifier


def test_merged_prefers_ecira_and_falls_back_to_mirca(irr_config):
    # Cell 0: both classify and disagree -> ECIRA wins.
    # Cell 1: only MIRCA classifies      -> MIRCA fills in.
    # Cell 2: neither                    -> unclassified, written as 0.
    # Cell 3: only ECIRA classifies.
    mirca = np.array([[0.9, 0.8], [np.nan, np.nan]])
    ecira = np.array([[0.1, np.nan], [np.nan, 0.6]])
    result = _classifier(irr_config(source="merged"), mirca, ecira).classify()

    assert result.virr.tolist() == [[0, 1], [0, 1]]
    assert result.source_id.tolist() == [
        [SOURCE_ECIRA, SOURCE_MIRCA],
        [SOURCE_NONE, SOURCE_ECIRA],
    ]
    assert result.fraction[0, 0] == pytest.approx(0.1)
    assert np.isnan(result.fraction[1, 0])


def test_single_source_modes_ignore_the_other_product(irr_config):
    mirca = np.array([[0.9, 0.9], [0.9, 0.9]])
    ecira = np.array([[0.1, 0.1], [0.1, 0.1]])

    only_mirca = _classifier(irr_config(source="mirca"), mirca, ecira).classify()
    assert only_mirca.virr.tolist() == [[1, 1], [1, 1]]
    assert set(np.unique(only_mirca.source_id)) == {SOURCE_MIRCA}

    only_ecira = _classifier(irr_config(source="ecira"), mirca, ecira).classify()
    assert only_ecira.virr.tolist() == [[0, 0], [0, 0]]
    assert set(np.unique(only_ecira.source_id)) == {SOURCE_ECIRA}


def test_single_source_mode_does_not_read_the_other_product(irr_config):
    """``source: mirca`` must not touch ECIRA, whose root may be unset."""

    def _boom():
        raise AssertionError("ECIRA must not be read in mirca mode")

    classifier = IrrigationClassifier(irr_config(source="mirca"))
    classifier._mirca_fraction = lambda: np.full((2, 2), 0.9)  # type: ignore[method-assign]
    classifier._ecira_fraction = _boom  # type: ignore[method-assign]
    assert classifier.classify().n_irrigated == 4


# --------------------------------------------------------------------------- #
# The result object
# --------------------------------------------------------------------------- #
def _classification(virr):
    virr = np.asarray(virr, dtype="int8")
    return IrrigationClassification(
        virr=virr,
        fraction=np.where(virr == 1, 0.9, 0.1).astype("float32"),
        source_id=np.full(virr.shape, SOURCE_MIRCA, dtype="int8"),
        crop_group="cereals",
        source="mirca",
        threshold=0.5,
        min_crop_area_ha=10.0,
        year=2015,
    )


def test_column_gathers_by_row_and_col(irr_config):
    result = _classification([[1, 0], [0, 1]])
    cells = TargetGrid.from_config(irr_config().grid).cell_table()
    assert result.column(cells).tolist() == [1, 0, 0, 1]

    # A subset of cells, out of order, still lines up with its own rows.
    subset = cells.iloc[[3, 0]]
    assert result.column(subset).tolist() == [1, 1]


def test_column_of_an_empty_cell_table_is_empty():
    assert _classification([[1]]).column(pd.DataFrame()).size == 0


def test_to_dataset_carries_the_rule_and_the_grid(irr_config):
    grid = TargetGrid.from_config(irr_config().grid)
    data = _classification([[1, 0], [0, 1]]).to_dataset(grid)
    assert isinstance(data, xr.Dataset)
    assert set(data.data_vars) == {"vIRR", "irrigated_fraction", "source_id"}
    assert np.array_equal(data["lat"].values, grid.lat_centers)
    assert "0.5" in data["vIRR"].attrs["rule"]


# --------------------------------------------------------------------------- #
# MIRCA-OS reading
# --------------------------------------------------------------------------- #
def _write_mirca(folder, stem, year, system, monthly, lat, lon):
    """Write a MIRCA-OS-shaped monthly file, coordinates on the cell corners."""
    folder.mkdir(parents=True, exist_ok=True)
    xr.Dataset(
        {"harvested_area": (("month", "latitude", "longitude"), monthly)},
        # MIRCA-OS stores each cell's east/south edge, and its months are sorted
        # as strings (1, 10, 11, 12, 2, ...).
        coords={
            "month": [1, 10, 11, 12, 2, 3, 4, 5, 6, 7, 8, 9],
            "latitude": lat,
            "longitude": lon,
        },
    ).to_netcdf(folder / f"MIRCA-OS_{stem}_{year}_{system}.nc")


def test_mirca_harvested_area_sums_subcrop_peaks(irr_config, tmp_path):
    """The annual area is the sum over sub-crops of each one's peak month."""
    config = irr_config(source="mirca", crop_group="wheat", year=2015)
    grid = TargetGrid.from_config(config.grid)
    res = config.grid.resolution_deg
    # Cell corners: centre + half a cell east / minus half a cell south.
    lat = np.sort(grid.lat_centers)[::-1] - res / 2.0
    lon = grid.lon_centers + res / 2.0

    folder = tmp_path / "mirca" / "2015"
    for stem, peak in (("Wheat_1", 30.0), ("Wheat_2", 12.0)):
        monthly = np.zeros((12, lat.size, lon.size))
        monthly[4] = peak  # month index 4 is calendar month 2 before sorting
        _write_mirca(folder, stem, 2015, "ir", monthly, lat, lon)

    area = IrrigationClassifier(config)._mirca_harvested_area("ir")
    # 30 + 12 ha in every source cell, and the source is the target grid.
    assert area.shape == grid.shape
    assert np.allclose(area, 42.0)


def test_mirca_half_cell_correction_puts_area_in_the_right_cell(irr_config, tmp_path):
    """Reading the corner coordinates as centres would shift the grid half a cell."""
    config = irr_config(source="mirca", crop_group="maize", year=2015)
    grid = TargetGrid.from_config(config.grid)
    res = config.grid.resolution_deg
    lat = np.sort(grid.lat_centers)[::-1] - res / 2.0
    lon = grid.lon_centers + res / 2.0

    monthly = np.zeros((12, lat.size, lon.size))
    monthly[0, 0, 0] = 100.0  # north-west source cell only
    _write_mirca(tmp_path / "mirca" / "2015", "Maize", 2015, "ir", monthly, lat, lon)

    area = IrrigationClassifier(config)._mirca_harvested_area("ir")
    assert area[0, 0] == pytest.approx(100.0)
    assert area.sum() == pytest.approx(100.0)


def test_missing_mirca_file_names_the_path(irr_config, tmp_path):
    config = irr_config(source="mirca", crop_group="maize", year=2015)
    (tmp_path / "mirca").mkdir(parents=True, exist_ok=True)
    with pytest.raises(FileNotFoundError, match="MIRCA-OS_Maize_2015_ir.nc"):
        IrrigationClassifier(config)._mirca_harvested_area("ir")


# --------------------------------------------------------------------------- #
# Configuration
# --------------------------------------------------------------------------- #
def test_merged_requires_both_roots(tmp_path):
    def _config(paths):
        return PipelineConfig.model_validate(
            {
                "flags": {"run_irrigation_classification": True},
                "grid": {"min_lon": 11.2, "max_lon": 11.4, "min_lat": 52.1, "max_lat": 52.3},
                "time": {"start": "1979-01-01", "end": "1979-01-03"},
                "paths": {"mswx_root": str(tmp_path), **paths},
                "irrigation": {"source": "merged"},
            }
        )

    with pytest.raises(ValueError, match="paths.ecira_root"):
        _config({"mirca_root": str(tmp_path)})
    with pytest.raises(ValueError, match="paths.mirca_root"):
        _config({"ecira_root": str(tmp_path)})
    # Both present: valid.
    assert _config({"mirca_root": str(tmp_path), "ecira_root": str(tmp_path)})


def test_single_source_only_requires_its_own_root(tmp_path):
    config = PipelineConfig.model_validate(
        {
            "flags": {"run_irrigation_classification": True},
            "grid": {"min_lon": 11.2, "max_lon": 11.4, "min_lat": 52.1, "max_lat": 52.3},
            "time": {"start": "1979-01-01", "end": "1979-01-03"},
            "paths": {"mswx_root": str(tmp_path), "ecira_root": str(tmp_path)},
            "irrigation": {"source": "ecira", "threshold": 0.7},
        }
    )
    assert config.irrigation.threshold == 0.7
    assert config.paths.mirca_root is None


def test_threshold_defaults_to_half_and_is_bounded(tmp_path):
    base = {
        "flags": {},
        "grid": {"min_lon": 11.2, "max_lon": 11.4, "min_lat": 52.1, "max_lat": 52.3},
        "time": {"start": "1979-01-01", "end": "1979-01-03"},
        "paths": {"mswx_root": str(tmp_path)},
    }
    assert PipelineConfig.model_validate(base).irrigation.threshold == 0.5
    with pytest.raises(ValueError):
        PipelineConfig.model_validate({**base, "irrigation": {"threshold": 1.5}})
