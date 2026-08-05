"""Tests for the top-N multi-class aggregation products (Method B)."""

from __future__ import annotations

import json

import numpy as np
import pandas as pd
import pytest
import xarray as xr

from data4simplace.exporters.soil_export import TopSoilExporter
from data4simplace.grid import TargetGrid
from data4simplace.soil.multiclass import METADATA_COLUMNS, NETCDF_SUBDIR


def _cells(config, soil_dataset) -> pd.DataFrame:
    grid = TargetGrid.from_config(config.grid)
    cells = grid.cell_table().iloc[:2].copy()
    cells["row"] = [0, 1]
    cells["col"] = [0, 1]
    cells["lat"] = soil_dataset["lat"].values
    cells["lon"] = soil_dataset["lon"].values
    return cells


# --------------------------------------------------------------------------- #
# Metadata
# --------------------------------------------------------------------------- #
def test_metadata_frame_carries_class_area_and_uncertainty(config, soil_dataset, top_classes):
    cells = _cells(config, soil_dataset)
    meta = top_classes.metadata_frame(1, cells)

    assert list(meta.columns) == METADATA_COLUMNS
    assert list(meta["soil_class_id"]) == [7, 10]
    assert list(meta["class_name"]) == ["loam", "sandy_loam"]
    np.testing.assert_allclose(meta["area_km2"], [60.0, 100.0])
    np.testing.assert_allclose(meta["area_fraction"], [0.6, 1.0])
    np.testing.assert_allclose(meta["cell_dominance_ratio"], [0.4, 0.0])
    np.testing.assert_allclose(meta["cell_shannon_entropy"], [0.83, 0.0], rtol=1e-3)
    np.testing.assert_array_equal(meta["SimplaceID"], cells["SimplaceID"])


def test_metadata_frame_marks_cells_without_that_rank(config, soil_dataset, top_classes):
    meta = top_classes.metadata_frame(2, _cells(config, soil_dataset))
    # The uniform cell has no second class.
    assert list(meta["soil_class_id"]) == [10, 0]
    assert meta["class_name"].iloc[1] == "unclassified"


# --------------------------------------------------------------------------- #
# Intermediate NetCDFs
# --------------------------------------------------------------------------- #
def test_write_produces_the_three_intermediate_rasters(tmp_path, soil_dataset, top_classes):
    written = top_classes.write(tmp_path, soil=soil_dataset)

    names = [p.name for p in written]
    assert names == [
        "intermediate_soil_properties.nc",
        "intermediate_top3_classes.nc",
        "intermediate_soil_uncertainty.nc",
    ]
    assert all(p.parent.name == NETCDF_SUBDIR for p in written)
    assert all(p.parent.parent.name == "soil" for p in written)

    with xr.open_dataset(written[0]) as properties:
        # The 10 km property stack that soil.csv is written from: no rank dim.
        assert "rank" not in properties.dims
        assert "clay" in properties.data_vars

    with xr.open_dataset(written[1]) as top:
        assert set(top["clay"].dims) == {"rank", "depth", "lat", "lon"}
        assert top.sizes["rank"] == 3
        # Class identity travels with the properties, codes and names alike.
        assert int(top["class_code"].sel(rank=1).values[0, 0]) == 7
        lookup = dict(
            zip(top["class_code_value"].values.tolist(),
                [str(n) for n in top["class_code_name"].values])
        )
        assert lookup[7] == "loam"
        assert json.loads(top.attrs["class_code_names"])["7"] == "loam"

    with xr.open_dataset(written[2]) as uncertainty:
        assert uncertainty["dominance_ratio"].values[0, 0] == pytest.approx(0.4)
        assert uncertainty["total_area_km2"].values[0, 0] == pytest.approx(100.0)
        # The per-rank shares travel with the metrics they explain.
        assert uncertainty["area_fraction"].sel(rank=2).values[0, 0] == pytest.approx(0.3)


def test_write_suffix_keeps_tiles_apart(tmp_path, top_classes):
    written = top_classes.write(tmp_path, suffix="_tile_r000_c000")
    assert written[0].name == "intermediate_top3_classes_tile_r000_c000.nc"
    # Without a soil stack only the two class files are written.
    assert len(written) == 2


def test_mask_cells_keeps_class_codes_integer(top_classes):
    mask = xr.DataArray(
        np.array([[True, False], [False, True]]),
        dims=("lat", "lon"),
        coords={"lat": top_classes.classes["lat"], "lon": top_classes.classes["lon"]},
    )
    top_classes.mask_cells(mask)

    codes = top_classes.classes["class_code"]
    assert np.issubdtype(codes.dtype, np.integer)
    # Excluded cells hold no class rather than a NaN-promoted float.
    assert int(codes.sel(rank=1).values[0, 1]) == 0
    assert int(codes.sel(rank=1).values[0, 0]) == 7
    assert np.isnan(top_classes.properties["clay"].values[0, 0, 0, 1])


# --------------------------------------------------------------------------- #
# Per-rank CSV export
# --------------------------------------------------------------------------- #
def test_exports_one_csv_per_rank_with_metadata(tmp_path, config, soil_dataset, top_classes):
    cells = _cells(config, soil_dataset)
    written = TopSoilExporter(config, reference_path=None).export(
        top_classes, cells, tmp_path
    )

    assert [p.name for p in written] == ["soil_1.csv", "soil_2.csv", "soil_3.csv"]

    rank1 = pd.read_csv(written[0])
    rank2 = pd.read_csv(written[1])
    # Both cells have a dominant class; only one has a second.
    assert len(rank1) == 2
    assert len(rank2) == 1
    assert list(rank2["soil_class_id"]) == [10]

    # Metadata sits after the SIMPLACE block, and the profile differs per rank.
    assert list(rank1.columns[-len(METADATA_COLUMNS):]) == METADATA_COLUMNS
    assert "clay_1" in rank1.columns
    assert rank1["clay_1"].iloc[0] == pytest.approx(20.0)
    assert rank2["clay_1"].iloc[0] == pytest.approx(35.0)
    # Rank 1 is the dominant class, so its rows match the single-profile export.
    assert set(rank1["location"]) == set(cells["SimplaceID"])


def test_per_rank_csv_conforms_to_the_reference_schema(tmp_path, config, soil_dataset, top_classes):
    ref = tmp_path / "soil_ref.csv"
    ref.write_text("location,soiltype,clay_1,CaCO3_1\n49612,AP5,7.7,202\n")

    cells = _cells(config, soil_dataset)
    written = TopSoilExporter(config, reference_path=ref).export(
        top_classes, cells, tmp_path
    )
    frame = pd.read_csv(written[0])

    # Reference columns in reference order, then the metadata block.
    assert list(frame.columns) == [
        "location", "soiltype", "clay_1", "CaCO3_1", *METADATA_COLUMNS
    ]
    assert set(frame["soiltype"]) == {"AP5"}
    assert frame["clay_1"].iloc[0] == pytest.approx(20.0)


def test_ptf_hydraulics_apply_to_the_dominant_rank_only(
    tmp_path, config, soil_dataset, top_classes
):
    # The PTF runs on the dominant class' pixels, so ranks 2..n must not inherit
    # its water retention.
    no_water = top_classes.properties.drop_vars(["wv0010", "wv0033", "wv1500"])
    top_classes.properties = no_water
    hydraulic = xr.Dataset(
        {
            "theta_fc": (("depth", "lat", "lon"), np.full((6, 2, 2), 0.29, "float32")),
        },
        coords={
            "depth": soil_dataset["depth"],
            "lat": soil_dataset["lat"],
            "lon": soil_dataset["lon"],
        },
    )

    cells = _cells(config, soil_dataset)
    written = TopSoilExporter(config, reference_path=None).export(
        top_classes, cells, tmp_path, hydraulic=hydraulic
    )
    rank1 = pd.read_csv(written[0])
    rank2 = pd.read_csv(written[1])

    assert rank1["soilwater_fc_1"].iloc[0] == pytest.approx(0.29)
    assert rank2["soilwater_fc_1"].iloc[0] == config.missing_value
