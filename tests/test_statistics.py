"""Tests for the three-primary-class statistics stage."""

from __future__ import annotations

import json

import numpy as np
import pandas as pd
import pytest
import xarray as xr

from data4simplace.grid import TargetGrid
from data4simplace.soil.aggregate import (
    CELL_STATISTICS,
    bin_describe,
    cell_geomean,
    cell_median,
    reducer_for,
)
from data4simplace.soil.dominant import (
    dominant_class_per_cell,
    rank_classes_per_cell,
    usda_profile_class,
)
from data4simplace.soil.statistics import PrimaryClassStatistics


@pytest.fixture()
def two_cell_grid() -> TargetGrid:
    return TargetGrid(min_lon=0.0, max_lon=0.2, min_lat=0.0, max_lat=0.1, resolution_deg=0.1)


def _fine(values, lon, lat=(0.05,), dtype="float64"):
    return xr.DataArray(
        np.asarray(values, dtype=dtype).reshape(len(lat), len(lon)),
        dims=("lat", "lon"),
        coords={"lat": np.asarray(lat), "lon": np.asarray(lon)},
    )


# --------------------------------------------------------------------------- #
# bin_describe
# --------------------------------------------------------------------------- #
def test_bin_describe_matches_pandas(two_cell_grid):
    # Cell 0: 1,2,3,10 ; cell 1: 4,4,4,8
    lon = [0.01, 0.03, 0.05, 0.07, 0.11, 0.13, 0.15, 0.17]
    values = [1.0, 2.0, 3.0, 10.0, 4.0, 4.0, 4.0, 8.0]
    stats = bin_describe(_fine(values, lon), two_cell_grid)

    left, right = pd.Series(values[:4]), pd.Series(values[4:])
    assert stats["count"].values[0, 0] == 4
    assert stats["mean"].values[0, 0] == pytest.approx(left.mean())
    assert stats["median"].values[0, 0] == pytest.approx(left.median())
    assert stats["std"].values[0, 0] == pytest.approx(left.std())      # ddof=1
    assert stats["kurt"].values[0, 0] == pytest.approx(left.kurt())    # excess
    assert stats["mean"].values[0, 1] == pytest.approx(right.mean())
    assert stats["kurt"].values[0, 1] == pytest.approx(right.kurt())


def test_bin_describe_empty_cell_is_nan_with_zero_count(two_cell_grid):
    stats = bin_describe(_fine([1.0, 3.0], lon=[0.02, 0.05]), two_cell_grid)
    assert stats["count"].values[0, 1] == 0
    assert np.isnan(stats["mean"].values[0, 1])
    assert np.isnan(stats["std"].values[0, 1])


def test_bin_describe_ignores_masked_pixels(two_cell_grid):
    stats = bin_describe(_fine([1.0, np.nan, 3.0], lon=[0.02, 0.05, 0.08]), two_cell_grid)
    assert stats["count"].values[0, 0] == 2
    assert stats["mean"].values[0, 0] == pytest.approx(2.0)


def test_bin_describe_single_pixel_has_no_spread(two_cell_grid):
    stats = bin_describe(_fine([5.0], lon=[0.02]), two_cell_grid)
    assert stats["count"].values[0, 0] == 1
    assert stats["mean"].values[0, 0] == pytest.approx(5.0)
    assert np.isnan(stats["std"].values[0, 0])   # needs 2 pixels
    assert np.isnan(stats["kurt"].values[0, 0])  # needs 4 pixels


def test_bin_describe_rejects_unknown_statistic(two_cell_grid):
    with pytest.raises(ValueError, match="Unsupported statistics"):
        bin_describe(_fine([1.0], lon=[0.02]), two_cell_grid, statistics=("mode",))


def test_cell_statistics_are_the_documented_set():
    assert CELL_STATISTICS == ("mean", "median", "std", "kurt", "count")


# --------------------------------------------------------------------------- #
# export statistic
# --------------------------------------------------------------------------- #
def test_reducer_for_median_overrides_every_rule():
    assert reducer_for("clay", "median") is cell_median
    assert reducer_for("soc", "median") is cell_median
    assert reducer_for("phh2o", "median") is cell_median


def test_reducer_for_mean_keeps_variable_rules():
    assert reducer_for("soc") is cell_geomean
    assert reducer_for("clay", "mean")(np.array([1.0, 3.0])) == pytest.approx(2.0)


def test_reducer_for_rejects_other_statistics():
    with pytest.raises(ValueError, match="Unsupported export statistic"):
        reducer_for("clay", "mode")


# --------------------------------------------------------------------------- #
# rank_classes_per_cell
# --------------------------------------------------------------------------- #
def _classes(codes, lon, lat=(0.05,)):
    return _fine(codes, lon, lat, dtype="int16").astype("int16")


def test_rank_classes_counts_and_shares(two_cell_grid):
    # Cell 0: three class-4 pixels, one class-7 -> 75 % / 25 %.
    ranked = rank_classes_per_cell(
        _classes([4, 4, 4, 7], lon=[0.01, 0.03, 0.05, 0.07]), two_cell_grid, n_classes=3
    )
    assert list(ranked["rank"].values) == [1, 2, 3]
    np.testing.assert_array_equal(ranked["class_code"].values[:, 0, 0], [4, 7, 0])
    np.testing.assert_array_equal(ranked["pixels"].values[:, 0, 0], [3, 1, 0])
    np.testing.assert_allclose(ranked["share_percent"].values[:2, 0, 0], [75.0, 25.0])
    assert np.isnan(ranked["share_percent"].values[2, 0, 0])


def test_rank_one_equals_the_dominant_class(two_cell_grid):
    classes = _classes([1, 1, 3, 3, 3], lon=[0.01, 0.03, 0.12, 0.14, 0.16])
    ranked = rank_classes_per_cell(classes, two_cell_grid, n_classes=2)
    dominant = dominant_class_per_cell(classes, two_cell_grid)
    np.testing.assert_array_equal(
        ranked["class_code"].sel(rank=1).values, dominant.values
    )


def test_rank_classes_ignores_zero_and_empty_cells(two_cell_grid):
    ranked = rank_classes_per_cell(
        _classes([0, 0, 5], lon=[0.01, 0.03, 0.05]), two_cell_grid, n_classes=2
    )
    np.testing.assert_array_equal(ranked["class_code"].values[:, 0, 0], [5, 0])
    assert ranked["share_percent"].values[0, 0, 0] == pytest.approx(100.0)
    # Cell 1 has no pixels at all.
    np.testing.assert_array_equal(ranked["class_code"].values[:, 0, 1], [0, 0])
    assert ranked["pixels"].values[0, 0, 1] == 0


def test_rank_classes_breaks_ties_by_lower_code(two_cell_grid):
    ranked = rank_classes_per_cell(
        _classes([7, 4], lon=[0.01, 0.03]), two_cell_grid, n_classes=2
    )
    np.testing.assert_array_equal(ranked["class_code"].values[:, 0, 0], [4, 7])


def test_rank_classes_requires_at_least_one_rank(two_cell_grid):
    with pytest.raises(ValueError, match="n_classes"):
        rank_classes_per_cell(_classes([1], lon=[0.01]), two_cell_grid, n_classes=0)


# --------------------------------------------------------------------------- #
# PrimaryClassStatistics
# --------------------------------------------------------------------------- #
@pytest.fixture()
def statistics(two_cell_grid) -> PrimaryClassStatistics:
    # Cell 0: 3 px of loamy_sand/clay_loam (11/4), 1 px of loam/loam (7/7).
    top = _classes([11, 11, 11, 7], lon=[0.01, 0.03, 0.05, 0.07])
    root = _classes([4, 4, 4, 7], lon=[0.01, 0.03, 0.05, 0.07])
    codes = usda_profile_class(top, root)
    ranked = rank_classes_per_cell(codes, two_cell_grid, n_classes=2)

    values = _fine([10.0, 20.0, 30.0, 99.0], lon=[0.01, 0.03, 0.05, 0.07])
    per_rank = []
    for rank in ranked["rank"].values:
        mask = codes == ranked["class_code"].sel(rank=rank).values[0, 0]
        described = bin_describe(values.where(mask), two_cell_grid)
        per_rank.append(described.expand_dims(depth=["0-5cm"]))
    stats = xr.Dataset(
        {
            f"clay_{stat}": xr.concat([r[stat] for r in per_rank], dim="rank")
            .assign_coords(rank=ranked["rank"].values)
            for stat in CELL_STATISTICS
        }
    )
    return PrimaryClassStatistics(classes=ranked, stats=stats, mode="usda_profile")


def test_code_lookup_names_present_codes(statistics):
    lookup = statistics.code_lookup()
    assert lookup[124] == "loamy_sand/clay_loam"   # (11 - 1) * 12 + 4
    assert lookup[79] == "loam/loam"
    assert 0 not in lookup


def test_class_table_has_one_row_per_present_rank(statistics, two_cell_grid):
    table = statistics.class_table(two_cell_grid)
    # Cell 1 is empty, so only the two ranks of cell 0 survive.
    assert len(table) == 2
    assert list(table["rank"]) == [1, 2]
    assert list(table["class_name"]) == ["loamy_sand/clay_loam", "loam/loam"]
    np.testing.assert_allclose(table["share_percent"], [75.0, 25.0])
    assert list(table["pixels"]) == [3, 1]
    assert table["SimplaceID"].tolist() == [1, 1]


def test_write_produces_netcdf_and_share_table(statistics, two_cell_grid, tmp_path):
    written = statistics.write(tmp_path, two_cell_grid)
    assert [p.name for p in written] == [
        "soil_class_statistics.nc",
        "soil_class_shares.nc",
        "soil_class_shares.csv",
    ]
    assert all(p.is_file() for p in written)

    with xr.open_dataset(written[0]) as stats:
        assert set(stats.dims) >= {"rank", "depth", "lat", "lon"}
        assert "clay_median" in stats.data_vars and "clay_kurt" in stats.data_vars
        # Rank 1 = the three-pixel class: mean 20, median 20.
        assert float(stats["clay_mean"].isel(rank=0, depth=0)[0, 0]) == pytest.approx(20.0)
        assert float(stats["clay_median"].isel(rank=0, depth=0)[0, 0]) == pytest.approx(20.0)
        assert json.loads(stats.attrs["class_code_names"])["79"] == "loam/loam"

    with xr.open_dataset(written[1]) as shares:
        assert float(shares["share_percent"].isel(rank=0)[0, 0]) == pytest.approx(75.0)

    table = pd.read_csv(written[2])
    assert list(table.columns) == [
        "SimplaceID", "row", "col", "lat", "lon", "rank", "class_code",
        "class_name", "pixels", "share_percent",
    ]
