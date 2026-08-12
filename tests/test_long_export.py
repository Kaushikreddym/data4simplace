"""Tests for the long (row-per-depth) export layout."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest
import xarray as xr

from data4simplace.config import PipelineConfig
from data4simplace.exporters.layout import (
    ERA5_SOIL,
    MANAGEMENT_DIALECTS,
    SOIL_DIALECTS,
    SUSTAG_SOIL,
    detect_layout,
    select_dialect,
)
from data4simplace.exporters.mgmt_export import LongManagementExporter, ManagementExporter
from data4simplace.exporters.soil_export import LongSoilExporter, SoilExporter
from data4simplace.grid import TargetGrid


# --------------------------------------------------------------------------- #
# Fixtures
# --------------------------------------------------------------------------- #


def _cells(config, soil_dataset) -> pd.DataFrame:
    cells = TargetGrid.from_config(config.grid).cell_table().iloc[:2].copy()
    cells["row"] = [0, 1]
    cells["col"] = [0, 1]
    cells["lat"] = soil_dataset["lat"].values
    cells["lon"] = soil_dataset["lon"].values
    return cells


@pytest.fixture()
def long_config(config_dict) -> PipelineConfig:
    """The shared config with both layouts enabled and simplace depths."""
    config_dict = dict(config_dict)
    config_dict["export"] = {"layout": "both"}
    config_dict["soil"] = {"long_depths": "simplace"}
    return PipelineConfig.model_validate(config_dict)


@pytest.fixture()
def npk_dataset(config) -> xr.Dataset:
    """N/P2O5/K2O rates in kg/ha on the 2x2 fixture grid."""
    lat = np.array([52.25, 52.15])
    lon = np.array([11.25, 11.35])

    def layer(value):
        return (("lat", "lon"), np.full((2, 2), value, dtype="float64"))

    return xr.Dataset(
        {"N": layer(180.0), "P2O5": layer(60.0), "K2O": layer(90.0)},
        coords={"lat": lat, "lon": lon},
    )


# --------------------------------------------------------------------------- #
# Layout detection and dialect selection
# --------------------------------------------------------------------------- #


def test_detect_layout_reads_the_depth_axis():
    wide = ["location", "clay_1", "clay_2", "clay_3", "soilwater_fc_1"]
    assert detect_layout(wide) == "wide"

    long_columns = ["location", "Depth", "LL", "DUL", "SAT", "NH4_mg_kg"]
    assert detect_layout(long_columns) == "long"

    # A depth column is not enough on its own: the Brandenburg reference has
    # depth_1..depth_6 as well as its per-layer block.
    assert detect_layout(["location", "depth_1", "clay_1", "clay_2", "clay_3"]) == "wide"


def test_select_dialect_matches_on_column_overlap():
    assert select_dialect(SUSTAG_SOIL.column_names, SOIL_DIALECTS) is SUSTAG_SOIL
    assert select_dialect(ERA5_SOIL.column_names, SOIL_DIALECTS) is ERA5_SOIL


def test_select_dialect_warns_when_nothing_matches(caplog):
    dialect = select_dialect(["alpha", "beta"], SOIL_DIALECTS)
    assert dialect.name in SOIL_DIALECTS
    assert "No known soil long dialect matches" in caplog.text


def test_every_dialect_column_is_resolvable():
    """A dialect column must name a real source, a derived key or a constant."""
    from data4simplace.exporters.layout import DERIVED_COLUMNS

    for dialects in (SOIL_DIALECTS, MANAGEMENT_DIALECTS):
        for dialect in dialects.values():
            for column in dialect.columns:
                assert (
                    column.source is not None
                    or column.constant is not None
                    or column.derive in DERIVED_COLUMNS
                ), f"{dialect.name}.{column.name} resolves to nothing"


# --------------------------------------------------------------------------- #
# The tidy profile table
# --------------------------------------------------------------------------- #


def test_profile_table_is_one_row_per_cell_and_depth(long_config, soil_dataset):
    cells = _cells(long_config, soil_dataset)
    exporter = SoilExporter(long_config, reference_path=None)

    native = exporter.build_profile_table(soil_dataset, cells, depths="native")
    # Native keeps SoilGrids' six horizons, un-remapped.
    assert len(native) == len(cells) * soil_dataset.sizes["depth"]
    assert native["depth_bottom_cm"].iloc[:6].tolist() == [5, 15, 30, 60, 100, 200]

    # A location's layers stay contiguous and in depth order.
    first = native[native["location"] == native["location"].iloc[0]]
    assert first["depth_top_cm"].is_monotonic_increasing

    simplace = exporter.build_profile_table(soil_dataset, cells, depths="simplace")
    assert simplace["depth_bottom_cm"].iloc[:6].tolist() == [10, 30, 50, 70, 100, 200]


def test_profile_table_carries_canonical_units(long_config, soil_dataset):
    cells = _cells(long_config, soil_dataset)
    table = SoilExporter(long_config, reference_path=None).build_profile_table(
        soil_dataset, cells
    )
    # The fixture is 20 % clay, bulk density 1.4, SOC 15 g/kg, and the water
    # contents are un-scaled from vol% to m3/m3.
    assert table["clay"].eq(20.0).all()
    assert table["bulkdensity"].eq(1.4).all()
    assert table["carbon"].eq(15.0).all()
    assert table["soilwater_fc"].round(4).eq(0.31).all()
    assert table["soilwater_sat"].round(4).eq(0.42).all()


def test_profile_scalars_are_thickness_weighted(long_config, soil_dataset):
    cells = _cells(long_config, soil_dataset)
    table = SoilExporter(long_config, reference_path=None).build_profile_table(
        soil_dataset, cells
    )
    # The fixture is uniform with depth, so the profile mean is the layer value
    # and the scalar is repeated on every row of the location.
    assert table["soilwater_fc_global"].round(4).eq(0.31).all()
    assert table.groupby("location")["soilwater_sat_global"].nunique().eq(1).all()


def test_profile_table_rejects_an_unknown_depth_mode(long_config, soil_dataset):
    cells = _cells(long_config, soil_dataset)
    with pytest.raises(ValueError, match="native.*simplace"):
        SoilExporter(long_config, reference_path=None).build_profile_table(
            soil_dataset, cells, depths="soilgrids"
        )


# --------------------------------------------------------------------------- #
# Long soil export
# --------------------------------------------------------------------------- #


def test_long_soil_file_has_the_dialect_columns(tmp_path, long_config, soil_dataset):
    cells = _cells(long_config, soil_dataset)
    exporter = LongSoilExporter(long_config, reference_path=None)

    out = exporter.export(soil_dataset, cells, tmp_path)
    assert out.is_file() and out.name == "soil_long.csv"

    written = pd.read_csv(out)
    assert list(written.columns) == SUSTAG_SOIL.column_names
    assert len(written) == len(cells) * 6
    assert written["location"].nunique() == len(cells)


def test_long_and_wide_describe_the_same_soil(long_config, soil_dataset):
    """The two layouts are two serialisations of one computation."""
    cells = _cells(long_config, soil_dataset)
    wide = SoilExporter(long_config, reference_path=None).build_frame(soil_dataset, cells)
    long_frame = LongSoilExporter(long_config, reference_path=None).build_frame(
        soil_dataset, cells
    )

    for location in wide["location"]:
        wide_row = wide[wide["location"] == location].iloc[0]
        rows = long_frame[long_frame["location"] == location].reset_index(drop=True)
        for layer in range(1, 7):
            assert rows.loc[layer - 1, "clay"] == pytest.approx(
                float(wide_row[f"clay_{layer}"]), abs=1e-4
            )
            assert rows.loc[layer - 1, "DUL"] == pytest.approx(
                float(wide_row[f"soilwater_fc_{layer}"]), abs=1e-4
            )
            assert rows.loc[layer - 1, "LL"] == pytest.approx(
                float(wide_row[f"soilwater_wp_{layer}"]), abs=1e-4
            )
            assert rows.loc[layer - 1, "BD"] == pytest.approx(
                float(wide_row[f"bulkdensity_{layer}"]), abs=1e-4
            )


def test_unit_conversions_are_applied_not_copied(long_config, soil_dataset):
    """A column with a unit factor must differ from its canonical source."""
    cells = _cells(long_config, soil_dataset)
    long_frame = LongSoilExporter(long_config, reference_path=None).build_frame(
        soil_dataset, cells
    )
    # carbon is 15 g/kg; SUSTAg's OC is g/100g, so 1.5.
    assert long_frame["OC"].round(4).eq(1.5).all()
    # C:N from 15 g/kg carbon over 1.2 g/kg nitrogen.
    assert long_frame["CN"].round(2).eq(12.5).all()


def test_mineral_n_round_trips_through_the_concentration(long_config, soil_dataset):
    """``NH4_mg_kg`` must invert the wide file's kg/ha stock exactly."""
    cells = _cells(long_config, soil_dataset)
    profile = SoilExporter(long_config, reference_path=None).build_profile_table(
        soil_dataset, cells, depths="simplace"
    )
    long_frame = LongSoilExporter(long_config, reference_path=None).build_frame(
        soil_dataset, cells
    )
    thickness = profile["depth_bottom_cm"] - profile["depth_top_cm"]
    back = long_frame["NH4_mg_kg"].to_numpy() * thickness.to_numpy() * profile[
        "bulkdensity"
    ].to_numpy() / 100.0
    assert back == pytest.approx(profile["ammonium"].to_numpy(), rel=1e-9)


def test_unmapped_properties_are_named_not_silently_dropped(
    long_config, soil_dataset, caplog
):
    cells = _cells(long_config, soil_dataset)
    # The ERA5 dialect carries no silt column, so silt must be reported.
    config = long_config.model_copy(
        update={
            "reference": long_config.reference.model_copy(
                update={"soil_file_long": None}
            )
        }
    )
    exporter = LongSoilExporter(config, reference_path=None)
    exporter._dialect = ERA5_SOIL  # force the narrower dialect
    exporter.build_frame(soil_dataset, cells)
    assert "has no column for" in caplog.text
    assert "silt" in caplog.text


def test_non_derivable_columns_carry_the_reference_constants(
    tmp_path, config_dict, soil_dataset
):
    """A real long reference declares more than SoilGrids can supply.

    ``alfa``, ``n``, ``ksat``, ``RootingDepth`` and the rest have no SoilGrids
    source; filled with the missing sentinel the file would not run, so they
    come from the reference's own first row exactly as the wide exporter does.
    """
    reference = tmp_path / "slim_soil.csv"
    reference.write_text(
        "location,Depth,LL,DUL,SAT,BD,OC,alfa,n,ksat,RootingDepth\n"
        "site_a,0.3,0.12,0.30,0.45,1.39,0.77,6,1.5,9e-05,1.3\n"
    )
    config_dict = dict(config_dict)
    config_dict["export"] = {"layout": "long"}
    config_dict["reference"] = {"soil_file_long": str(reference)}
    config = PipelineConfig.model_validate(config_dict)

    cells = _cells(config, soil_dataset)
    out = LongSoilExporter(config, reference_path=None).export(
        soil_dataset, cells, tmp_path
    )
    written = pd.read_csv(out)

    # The output conforms to the reference's exact columns and order ...
    assert list(written.columns) == list(pd.read_csv(reference).columns)
    # ... with the non-derivable ones carried, not sentinelled.
    assert written["alfa"].eq(6).all()
    assert written["ksat"].eq(9e-05).all()
    assert written["RootingDepth"].eq(1.3).all()
    # ... and the derivable ones actually derived.
    assert written["BD"].eq(1.4).all()  # the fixture, not the reference's 1.39


def test_era5_dialect_uses_its_own_delimiter(tmp_path, long_config, soil_dataset):
    cells = _cells(long_config, soil_dataset)
    exporter = LongSoilExporter(long_config, reference_path=None)
    exporter._dialect = ERA5_SOIL
    exporter._spec = None  # the spec follows the dialect

    out = exporter.export(soil_dataset, cells, tmp_path)
    header = out.read_text().splitlines()[0]
    assert ";" in header and "," not in header
    assert header.split(";")[0] == "soiltype"


# --------------------------------------------------------------------------- #
# Long management export
# --------------------------------------------------------------------------- #


def test_long_schedule_has_the_dialect_columns(tmp_path, long_config, npk_dataset):
    cells = _cells(long_config, npk_dataset)
    exporter = LongManagementExporter(long_config, reference_path=None)

    out = exporter.export(npk_dataset, cells, tmp_path)
    assert out.name.endswith("_long.csv")

    written = pd.read_csv(out)
    assert list(written.columns) == MANAGEMENT_DIALECTS["sustag"].column_names
    # No vType column: the amount is the nutrient, not a product.
    assert "vType" not in written.columns
    # The config fixture's window is a single year.
    assert written["Year"].nunique() == 1


def test_long_amounts_are_nutrients_not_products(long_config, npk_dataset):
    """The nutrient basis must undo the wide schedule's carrier division."""
    cells = _cells(long_config, npk_dataset)
    wide = ManagementExporter(long_config, reference_path=None).build_frame(
        npk_dataset, cells
    )
    long_frame = LongManagementExporter(long_config, reference_path=None).build_frame(
        npk_dataset, cells
    )

    # N is 180 kg/ha = 18 g N/m2, split across the reference's three dressings.
    n_total = long_frame[long_frame["Location"] == long_frame["Location"].iloc[0]]
    # One year, so the per-cell nutrient total is the whole rate for each
    # nutrient; the N events must sum to 18 g/m2.
    assert n_total["Amount"].sum() == pytest.approx(18.0 + 60.0 * 0.4364 * 0.1
                                                    + 90.0 * 0.8302 * 0.1, abs=0.05)
    # The wide file carries product grams, which are strictly larger for KAS.
    assert wide["Amount"].sum() > long_frame["Amount"].sum()


def test_product_basis_is_refused_without_a_type_column(long_config, npk_dataset):
    config = long_config.model_copy(
        update={"npk": long_config.npk.model_copy(update={"long_amount_basis": "product"})}
    )
    cells = _cells(config, npk_dataset)
    with pytest.raises(ValueError, match="needs a fertilizer-type column"):
        LongManagementExporter(config, reference_path=None).build_frame(npk_dataset, cells)


def test_long_schedule_repeats_per_year(config_dict, npk_dataset):
    config_dict = dict(config_dict)
    config_dict["export"] = {"layout": "long"}
    config_dict["time"] = {"start": "2000-01-01", "end": "2002-12-31"}
    config = PipelineConfig.model_validate(config_dict)
    cells = _cells(config, npk_dataset)

    frame = LongManagementExporter(config, reference_path=None).build_frame(
        npk_dataset, cells
    )
    assert sorted(frame["Year"].unique()) == [2000, 2001, 2002]
    # Each year carries the same events, since NPKGRIDS has no time axis.
    per_year = frame.groupby("Year").size()
    assert per_year.nunique() == 1


# --------------------------------------------------------------------------- #
# Config
# --------------------------------------------------------------------------- #


def test_layout_resolution_and_overrides(config_dict):
    config_dict = dict(config_dict)
    config_dict["export"] = {"layout": "wide", "soil_layout": "both"}
    config = PipelineConfig.model_validate(config_dict)

    assert config.export.writes("soil", "wide")
    assert config.export.writes("soil", "long")
    assert config.export.writes("management", "wide")
    assert not config.export.writes("management", "long")


def test_default_layout_is_wide(config):
    assert config.export.layout == "wide"
    assert config.export.writes("soil", "wide")
    assert not config.export.writes("soil", "long")
