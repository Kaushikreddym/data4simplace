"""Optional pedotransfer functions (PTFs) for soil hydraulic parameters.

Only invoked when ``flags.compute_ptf`` is true. Implements the widely used
Saxton & Rawls (2006) equations, deriving water-retention points and saturated
hydraulic conductivity from texture and organic matter. All inputs and outputs
are :class:`~xarray.DataArray` / :class:`~xarray.Dataset` objects so the
derivation stays lazy and grid-aligned.
"""

from __future__ import annotations

import numpy as np
import xarray as xr


def saxton_rawls(
    sand: xr.DataArray,
    clay: xr.DataArray,
    organic_matter: xr.DataArray | None = None,
) -> xr.Dataset:
    """Derive soil hydraulic parameters via Saxton & Rawls (2006).

    Parameters
    ----------
    sand, clay:
        Mass fractions in **percent** (0-100).
    organic_matter:
        Organic matter in **percent**. If ``None``, assumed 2.5 % (a typical
        agricultural topsoil value). Organic matter ~= 1.724 * organic carbon.

    Returns
    -------
    xarray.Dataset
        Variables:

        - ``theta_wp``   wilting point volumetric water content [m3 m-3]
        - ``theta_fc``   field capacity volumetric water content [m3 m-3]
        - ``theta_sat``  saturation volumetric water content [m3 m-3]
        - ``theta_paw``  plant-available water (fc - wp) [m3 m-3]
        - ``ksat``       saturated hydraulic conductivity [mm h-1]

    Notes
    -----
    Coefficients follow Saxton & Rawls (2006), *Soil Sci. Soc. Am. J.* 70:1569.
    Inputs are converted to the fractions (0-1) the equations expect.
    """
    s = (sand / 100.0).astype("float64")
    c = (clay / 100.0).astype("float64")
    om_pct: xr.DataArray | float = 2.5 if organic_matter is None else organic_matter
    om = om_pct / 100.0

    # --- 1500 kPa (wilting point) --------------------------------------------
    theta_1500t = (
        -0.024 * s + 0.487 * c + 0.006 * om
        + 0.005 * (s * om) - 0.013 * (c * om)
        + 0.068 * (s * c) + 0.031
    )
    theta_wp = theta_1500t + (0.14 * theta_1500t - 0.02)

    # --- 33 kPa (field capacity) ---------------------------------------------
    theta_33t = (
        -0.251 * s + 0.195 * c + 0.011 * om
        + 0.006 * (s * om) - 0.027 * (c * om)
        + 0.452 * (s * c) + 0.299
    )
    theta_fc = theta_33t + (1.283 * theta_33t**2 - 0.374 * theta_33t - 0.015)

    # --- Saturation - 33 kPa (structure) -------------------------------------
    theta_s33t = (
        0.278 * s + 0.034 * c + 0.022 * om
        - 0.018 * (s * om) - 0.027 * (c * om)
        - 0.584 * (s * c) + 0.078
    )
    theta_s33 = theta_s33t + (0.636 * theta_s33t - 0.107)

    # --- Saturation ----------------------------------------------------------
    theta_sat = theta_fc + theta_s33 - 0.097 * s + 0.043

    # --- Saturated hydraulic conductivity ------------------------------------
    lam = (np.log(theta_fc) - np.log(theta_wp)) / (np.log(1500.0) - np.log(33.0))
    ksat = 1930.0 * (theta_sat - theta_fc) ** (3.0 - lam)  # mm h-1

    paw = theta_fc - theta_wp

    return xr.Dataset(
        {
            "theta_wp": theta_wp.astype("float32"),
            "theta_fc": theta_fc.astype("float32"),
            "theta_sat": theta_sat.astype("float32"),
            "theta_paw": paw.astype("float32"),
            "ksat": ksat.astype("float32"),
        }
    )
