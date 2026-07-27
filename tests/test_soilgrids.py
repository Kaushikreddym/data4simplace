"""Tests for SoilGrids value handling (scale factors and the no-data sentinel)."""

from __future__ import annotations

import numpy as np
import pytest
import xarray as xr

from data4simplace.soil.dominant import usda_texture_class
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
