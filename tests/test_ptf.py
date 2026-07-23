"""Tests for the Saxton-Rawls pedotransfer functions."""

from __future__ import annotations

import numpy as np
import xarray as xr

from data4simplace.soil.ptf import saxton_rawls


def test_saxton_rawls_ordering_and_ranges():
    sand = xr.DataArray([40.0, 70.0, 15.0], dims="cell")
    clay = xr.DataArray([20.0, 10.0, 45.0], dims="cell")
    out = saxton_rawls(sand, clay)

    for var in ("theta_wp", "theta_fc", "theta_sat", "theta_paw", "ksat"):
        assert var in out

    wp = out["theta_wp"].values
    fc = out["theta_fc"].values
    sat = out["theta_sat"].values

    # physical ordering: wilting point < field capacity < saturation
    assert np.all(wp < fc)
    assert np.all(fc < sat)
    # volumetric contents bounded in [0, 1]
    assert np.all((wp >= 0) & (sat <= 1))
    # plant-available water is positive
    assert np.all(out["theta_paw"].values > 0)


def test_saxton_rawls_accepts_organic_matter_field():
    sand = xr.DataArray([50.0], dims="cell")
    clay = xr.DataArray([25.0], dims="cell")
    om = xr.DataArray([3.0], dims="cell")
    out = saxton_rawls(sand, clay, om)
    assert np.isfinite(out["ksat"].values).all()
