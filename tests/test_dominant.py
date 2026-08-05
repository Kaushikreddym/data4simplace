"""Tests for USDA texture classification and dominant-class selection."""

from __future__ import annotations

import numpy as np
import pytest
import xarray as xr

from data4simplace.grid import TargetGrid
from data4simplace.soil.classify import (
    depth_bounds_cm,
    dominant_class_per_cell,
    dominant_pixel_mask,
    profile_class_name,
    rootzone_mean,
    split_profile_class,
    usda_profile_class,
    usda_texture_class,
)


def _classify(sand, silt, clay):
    return int(
        usda_texture_class(
            xr.DataArray([float(sand)]),
            xr.DataArray([float(silt)]),
            xr.DataArray([float(clay)]),
        ).values[0]
    )


@pytest.mark.parametrize(
    "sand,silt,clay,expected",
    [
        (92, 5, 3, 12),    # sand
        (82, 12, 6, 11),   # loamy sand
        (80, 10, 10, 10),  # sandy loam (silt+2*clay = 30, on the boundary)
        (65, 15, 20, 6),   # sandy clay loam
        (40, 40, 20, 7),   # loam
        (20, 60, 20, 8),   # silt loam
        (5, 90, 5, 9),     # silt
        (20, 20, 60, 1),   # clay
        (5, 55, 45, 2),    # silty clay
        (50, 10, 40, 3),   # sandy clay
    ],
)
def test_usda_texture_classes(sand, silt, clay, expected):
    assert _classify(sand, silt, clay) == expected


def test_usda_nan_is_unclassified():
    cls = usda_texture_class(
        xr.DataArray([np.nan]), xr.DataArray([40.0]), xr.DataArray([20.0])
    )
    assert int(cls.values[0]) == 0


@pytest.fixture()
def two_cell_grid() -> TargetGrid:
    return TargetGrid(min_lon=0.0, max_lon=0.2, min_lat=0.0, max_lat=0.1, resolution_deg=0.1)


def _classes(codes, lon, lat=(0.05,)):
    return xr.DataArray(
        np.asarray(codes, dtype="float64").reshape(len(lat), len(lon)),
        dims=("lat", "lon"),
        coords={"lat": np.asarray(lat), "lon": np.asarray(lon)},
    ).astype("int16")


def test_dominant_class_is_pixel_majority(two_cell_grid):
    # Cell 0: two class-1 pixels; cell 1: two class-3 pixels.
    dom = dominant_class_per_cell(
        _classes([1, 1, 3, 3], lon=[0.02, 0.05, 0.12, 0.15]), two_cell_grid
    )
    np.testing.assert_array_equal(dom.values, [[1, 3]])


def test_dominant_class_breaks_by_count(two_cell_grid):
    # Cell 0 has two class-4 and one class-7 pixel -> class 4 wins.
    dom = dominant_class_per_cell(
        _classes([4, 4, 7], lon=[0.02, 0.05, 0.08]), two_cell_grid
    )
    assert int(dom.values[0, 0]) == 4


def test_dominant_class_ignores_zero(two_cell_grid):
    # Zeros (unclassified) must not win even when most frequent.
    dom = dominant_class_per_cell(
        _classes([0, 0, 5], lon=[0.02, 0.05, 0.08]), two_cell_grid
    )
    assert int(dom.values[0, 0]) == 5


def test_dominant_class_empty_cell_is_zero(two_cell_grid):
    dom = dominant_class_per_cell(_classes([2, 2], lon=[0.02, 0.05]), two_cell_grid)
    assert int(dom.values[0, 1]) == 0


def test_dominant_pixel_mask_keeps_only_dominant(two_cell_grid):
    classes = _classes([1, 1, 3, 3, 7], lon=[0.02, 0.05, 0.12, 0.15, 0.18])
    dom = dominant_class_per_cell(classes, two_cell_grid)
    mask = dominant_pixel_mask(classes, dom)
    # Cell 0 dominant=1 (both kept); cell 1 dominant=3 (two kept, the class-7 dropped).
    np.testing.assert_array_equal(mask.values.astype(int), [[1, 1, 1, 1, 0]])


# --------------------------------------------------------------------------- #
# usda_profile: composite (topsoil, rooting-zone) classes
# --------------------------------------------------------------------------- #
def test_depth_bounds_cm():
    assert depth_bounds_cm("0-5cm") == (0.0, 5.0)
    assert depth_bounds_cm("60-100cm") == (60.0, 100.0)
    with pytest.raises(ValueError):
        depth_bounds_cm("topsoil")


def _layers(values: dict[str, float]) -> dict[str, xr.DataArray]:
    return {depth: xr.DataArray([v]) for depth, v in values.items()}


def test_rootzone_mean_weights_by_thickness():
    # 5-15 (10 cm) and 15-30 (15 cm) inside a 5-30 cm window -> (10*20 + 15*40)/25.
    mean = rootzone_mean(_layers({"5-15cm": 20.0, "15-30cm": 40.0}), 5.0, 30.0)
    assert float(mean.values[0]) == pytest.approx((10 * 20 + 15 * 40) / 25)


def test_rootzone_mean_clips_partial_layers_and_skips_outside():
    # Window 5-50 cm: 0-5 contributes nothing, 30-60 contributes only 20 of 30 cm.
    mean = rootzone_mean(
        _layers({"0-5cm": 100.0, "5-15cm": 10.0, "15-30cm": 10.0, "30-60cm": 40.0}),
        5.0,
        50.0,
    )
    assert float(mean.values[0]) == pytest.approx((10 * 10 + 15 * 10 + 20 * 40) / 45)


def test_rootzone_mean_ignores_nan_layers():
    mean = rootzone_mean(_layers({"5-15cm": np.nan, "15-30cm": 30.0}), 5.0, 30.0)
    assert float(mean.values[0]) == pytest.approx(30.0)


def test_rootzone_mean_requires_overlap():
    with pytest.raises(ValueError):
        rootzone_mean(_layers({"100-200cm": 10.0}), 5.0, 100.0)


@pytest.mark.parametrize("topsoil,rootzone", [(1, 1), (10, 4), (12, 12), (7, 12)])
def test_profile_class_roundtrip(topsoil, rootzone):
    code = int(
        usda_profile_class(xr.DataArray([topsoil]), xr.DataArray([rootzone])).values[0]
    )
    assert 1 <= code <= 144
    assert split_profile_class(code) == (topsoil, rootzone)


def test_profile_class_is_unique_per_combination():
    tops = np.repeat(np.arange(1, 13), 12)
    roots = np.tile(np.arange(1, 13), 12)
    codes = usda_profile_class(xr.DataArray(tops), xr.DataArray(roots)).values
    assert np.unique(codes).size == 144


def test_profile_class_unclassified_when_either_part_missing():
    codes = usda_profile_class(
        xr.DataArray([0, 5, 0]), xr.DataArray([7, 0, 0])
    ).values
    np.testing.assert_array_equal(codes, [0, 0, 0])
    assert split_profile_class(0) == (0, 0)


def test_profile_class_name():
    # loamy sand over clay loam: topsoil 11, rooting zone 4.
    code = int(usda_profile_class(xr.DataArray([11]), xr.DataArray([4])).values[0])
    assert profile_class_name(code) == "loamy_sand/clay_loam"
    assert profile_class_name(0) == "unclassified"


def test_dominant_vote_separates_equal_topsoils(two_cell_grid):
    # Same topsoil class (11) but different subsoils: 3 pixels sand-over-clay_loam
    # (11/4) and 2 pixels sand-over-sand (11/11) -> the layered profile wins and
    # the uniform-sand pixels are dropped, which plain 'usda' could not do.
    top = _classes([11, 11, 11, 11, 11], lon=[0.01, 0.03, 0.05, 0.07, 0.09])
    root = _classes([4, 4, 4, 11, 11], lon=[0.01, 0.03, 0.05, 0.07, 0.09])
    composite = usda_profile_class(top, root)
    dom = dominant_class_per_cell(composite, two_cell_grid)
    assert split_profile_class(int(dom.values[0, 0])) == (11, 4)
    mask = dominant_pixel_mask(composite, dom)
    np.testing.assert_array_equal(mask.values.astype(int), [[1, 1, 1, 0, 0]])


# --------------------------------------------------------------------------- #
# class_composition: per-class areas and cell heterogeneity (Method B)
# --------------------------------------------------------------------------- #
def test_class_composition_ranks_and_shares(two_cell_grid):
    from data4simplace.soil.classify import class_composition

    # Cell 0: 3x class 7, 1x class 10.  Cell 1: 2x class 3.
    classes = _classes(
        [7, 7, 7, 10, 3, 3], lon=[0.01, 0.03, 0.05, 0.07, 0.12, 0.15]
    )
    ranked, uncertainty = class_composition(classes, two_cell_grid, n_classes=3)

    np.testing.assert_array_equal(ranked["class_code"].values[:, 0, 0], [7, 10, 0])
    np.testing.assert_array_equal(ranked["pixels"].values[:, 0, 0], [3, 1, 0])
    np.testing.assert_allclose(ranked["share_percent"].values[0, 0, 0], 75.0)
    # Rank 3 does not exist in either cell.
    assert np.isnan(ranked["share_percent"].values[2, 0, 0])
    # Without pixel areas the area columns are simply absent.
    assert "area_km2" not in ranked.data_vars

    np.testing.assert_array_equal(uncertainty["n_classes"].values[0], [2, 1])
    np.testing.assert_allclose(uncertainty["dominance_ratio"].values[0, 0], 0.25)
    # A single-class cell is perfectly homogeneous.
    assert uncertainty["dominance_ratio"].values[0, 1] == 0.0
    assert uncertainty["shannon_entropy"].values[0, 1] == 0.0


def test_class_composition_entropy_is_normalised(two_cell_grid):
    from data4simplace.soil.classify import class_composition

    # Cell 0: two classes, equally frequent -> maximal entropy for 2 classes.
    _, even = class_composition(
        _classes([1, 1, 2, 2], lon=[0.01, 0.03, 0.05, 0.07]), two_cell_grid
    )
    assert even["shannon_entropy"].values[0, 0] == pytest.approx(1.0)

    # Four equally frequent classes are also maximal: the normalisation makes
    # cells with different class counts comparable.
    _, four = class_composition(
        _classes([1, 2, 3, 4], lon=[0.01, 0.03, 0.05, 0.07]), two_cell_grid
    )
    assert four["shannon_entropy"].values[0, 0] == pytest.approx(1.0)

    # A lopsided split scores below both.
    _, skewed = class_composition(
        _classes([1, 1, 1, 2], lon=[0.01, 0.03, 0.05, 0.07]), two_cell_grid
    )
    assert 0.0 < skewed["shannon_entropy"].values[0, 0] < 1.0


def test_class_composition_empty_cell_has_no_metrics(two_cell_grid):
    from data4simplace.soil.classify import class_composition

    _, uncertainty = class_composition(
        _classes([0, 0], lon=[0.02, 0.05]), two_cell_grid
    )
    assert int(uncertainty["n_classes"].values[0, 0]) == 0
    assert np.isnan(uncertainty["dominance_ratio"].values[0, 0])
    assert np.isnan(uncertainty["shannon_entropy"].values[0, 0])


def test_class_composition_areas_are_latitude_weighted(two_cell_grid):
    from data4simplace.soil.classify import class_composition
    from data4simplace.spatial.area import pixel_area_km2

    classes = _classes([7, 7, 7, 10, 3, 3], lon=[0.01, 0.03, 0.05, 0.07, 0.12, 0.15])
    area = pixel_area_km2(classes, dlat_deg=0.02, dlon_deg=0.02)
    ranked, uncertainty = class_composition(
        classes, two_cell_grid, n_classes=3, pixel_area=area
    )

    per_pixel = float(area.values[0, 0])
    assert ranked["area_km2"].values[0, 0, 0] == pytest.approx(3 * per_pixel, rel=1e-5)
    assert ranked["area_fraction"].values[0, 0, 0] == pytest.approx(0.75)
    assert uncertainty["total_area_km2"].values[0, 0] == pytest.approx(
        4 * per_pixel, rel=1e-5
    )
    # A cell with no classified pixels has no area, not a zero one.
    assert np.isnan(uncertainty["total_area_km2"].values[0, 1]) or (
        uncertainty["n_classes"].values[0, 1] > 0
    )


def test_rank_one_matches_the_dominant_vote(two_cell_grid):
    from data4simplace.soil.classify import class_composition

    # A tie: both the majority vote and the ranking must pick the lower code.
    classes = _classes([7, 7, 4, 4], lon=[0.01, 0.03, 0.05, 0.07])
    ranked, _ = class_composition(classes, two_cell_grid)
    dominant = dominant_class_per_cell(classes, two_cell_grid)
    np.testing.assert_array_equal(
        ranked["class_code"].sel(rank=1).values, dominant.values
    )
