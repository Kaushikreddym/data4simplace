"""Tests for SoilGrids value handling (scale factors and the no-data sentinel)."""

from __future__ import annotations

import numpy as np
import pytest
import xarray as xr

from data4simplace.soil.classify import usda_texture_class
from data4simplace.soil.soilgrids import (
    SCALE_FACTORS,
    ZERO_IS_NODATA,
    SoilGridsHandler,
)


def _coverage(values) -> xr.DataArray:
    return xr.DataArray(
        np.asarray(values, dtype="float64").reshape(1, -1),
        dims=("lat", "lon"),
        coords={"lat": [52.05], "lon": np.linspace(11.21, 11.29, len(values))},
    )


def test_unscale_applies_the_official_factor():
    unscaled = SoilGridsHandler.unscale(_coverage([250.0, 400.0]), "clay")
    np.testing.assert_allclose(unscaled.values[0], [25.0, 40.0])
    assert SCALE_FACTORS["clay"] == 10.0


@pytest.mark.parametrize("layer", sorted(ZERO_IS_NODATA))
def test_zero_is_masked_where_it_cannot_be_a_measurement(layer):
    masked = SoilGridsHandler.mask_nodata(_coverage([0.0, 12.0]), layer)
    assert np.isnan(masked.values[0, 0])
    assert masked.values[0, 1] == 12.0


def test_zero_is_kept_for_coarse_fragments():
    # cfvo == 0 is a real value: a soil free of coarse fragments.
    kept = SoilGridsHandler.mask_nodata(_coverage([0.0, 5.0]), "cfvo")
    np.testing.assert_allclose(kept.values[0], [0.0, 5.0])


def test_unknown_layer_is_left_untouched():
    kept = SoilGridsHandler.mask_nodata(_coverage([0.0, 1.0]), "wrb")
    np.testing.assert_allclose(kept.values[0], [0.0, 1.0])


def test_masking_keeps_nan_and_negative_free_values():
    masked = SoilGridsHandler.mask_nodata(_coverage([np.nan, 7.5]), "soc")
    assert np.isnan(masked.values[0, 0])
    assert masked.values[0, 1] == 7.5


def test_nodata_pixels_no_longer_classify_as_sand():
    # A 0/0/0 texture pixel scores as USDA sand (12) unless it is masked first:
    # that is how water and built-up gaps used to join the majority vote.
    zeros = _coverage([0.0])
    assert int(usda_texture_class(zeros, zeros, zeros).values[0, 0]) == 12

    masked = [SoilGridsHandler.mask_nodata(zeros, t) for t in ("sand", "silt", "clay")]
    assert int(usda_texture_class(*masked).values[0, 0]) == 0  # unclassified


def _water_dataset(wv0010, wv0033, wv1500) -> xr.Dataset:
    return xr.Dataset(
        {
            "wv0010": _coverage(wv0010),
            "wv0033": _coverage(wv0033),
            "wv1500": _coverage(wv1500),
        }
    )


def test_water_retention_layers_are_unscaled_to_vol_percent():
    # Mapped 0.1 vol% -> vol%: a stored 310 is 31 vol% at 33 kPa.
    unscaled = SoilGridsHandler.unscale(_coverage([310.0]), "wv0033")
    assert unscaled.values[0, 0] == 31.0
    assert {"wv0010", "wv0033", "wv1500"} <= set(ZERO_IS_NODATA)


def test_harmonise_clips_inverted_water_retention():
    # Cell 0 is already ordered; cell 1 has more water at 1500 kPa than at
    # 33 kPa (independent SoilGrids models), which would give SIMPLACE a
    # negative plant-available water.
    harmonised = SoilGridsHandler.harmonise_water_retention(
        _water_dataset([42.0, 30.0], [31.0, 18.0], [14.0, 25.0])
    )
    np.testing.assert_allclose(harmonised["wv0010"].values[0], [42.0, 30.0])
    np.testing.assert_allclose(harmonised["wv0033"].values[0], [31.0, 18.0])
    # The drier suction is clipped down to the wetter one, never the reverse.
    np.testing.assert_allclose(harmonised["wv1500"].values[0], [14.0, 18.0])


def test_harmonise_keeps_missing_cells_missing():
    harmonised = SoilGridsHandler.harmonise_water_retention(
        _water_dataset([42.0, np.nan], [31.0, 18.0], [np.nan, 14.0])
    )
    assert np.isnan(harmonised["wv1500"].values[0, 0])
    assert np.isnan(harmonised["wv0033"].values[0, 1])


# --------------------------------------------------------------------------- #
# Method A / Method B aggregation (soil.aggregation_method)
# --------------------------------------------------------------------------- #
def _fine_textures():
    """Four 250 m pixels inside one target cell: three loam, one sand."""
    lat = np.array([52.28, 52.22])
    lon = np.array([11.22, 11.28])

    def field(values):
        return xr.DataArray(
            np.asarray(values, dtype="float32").reshape(2, 2),
            dims=("lat", "lon"),
            coords={"lat": lat, "lon": lon},
        )

    # Pixel (1, 1) is sand (92/5/3); the other three are loam (40/40/20).
    return {
        "sand": {"0-5cm": field([[40, 40], [40, 92]])},
        "silt": {"0-5cm": field([[40, 40], [40, 5]])},
        "clay": {"0-5cm": field([[20, 20], [20, 3]])},
    }


def _handler(config_dict, method):
    from data4simplace.config import PipelineConfig
    from data4simplace.soil.soilgrids import SoilGridsHandler

    config_dict["flags"]["run_soil_processing"] = True
    config_dict["soil"] = {
        "layers": ["sand", "silt", "clay"],
        "depths": ["0-5cm"],
        "dominant_mode": "usda",
        "aggregation_method": method,
    }
    config = PipelineConfig.model_validate(config_dict)
    handler = SoilGridsHandler(config)
    fine = _fine_textures()
    handler._load_fine = lambda: (fine, fine["sand"]["0-5cm"])  # type: ignore[method-assign]
    return handler


def test_dominant_method_produces_no_class_stack(config_dict):
    handler = _handler(config_dict, "dominant")
    soil, _ = handler.load_processed()

    assert handler.top_classes is None
    # The cell is majority loam, so it exports the loam texture (normalised).
    assert float(soil["clay"].sel(depth="0-5cm").values[0, 0]) == pytest.approx(20.0)


def test_top3_method_aggregates_each_class_over_its_own_pixels(config_dict):
    handler = _handler(config_dict, "top3")
    soil, _ = handler.load_processed()
    top = handler.top_classes

    assert top is not None
    # Rank 1 is the dominant class, so Method B never disagrees with Method A.
    xr.testing.assert_allclose(top.properties.sel(rank=1, drop=True), soil)

    # Rank 2 is the single sand pixel, aggregated over itself alone.
    assert float(top.properties["clay"].sel(rank=2).values[0, 0, 0]) == pytest.approx(3.0)
    np.testing.assert_array_equal(
        top.classes["class_code"].values[:, 0, 0], [7, 12, 0]  # loam, sand, none
    )
    np.testing.assert_allclose(
        top.classes["area_fraction"].values[:2, 0, 0], [0.75, 0.25], rtol=1e-3
    )
    # Areas are real surface areas, not pixel counts.
    assert float(top.classes["area_km2"].values[0, 0, 0]) > 0
    # Area-weighted, so the southern (larger) sand pixel counts for slightly
    # more than the 0.25 its pixel count alone would give.
    dominance = float(top.uncertainty["dominance_ratio"].values[0, 0])
    assert dominance == pytest.approx(0.25, rel=1e-3)
    assert dominance > 0.25
    assert 0.0 < float(top.uncertainty["shannon_entropy"].values[0, 0]) < 1.0
