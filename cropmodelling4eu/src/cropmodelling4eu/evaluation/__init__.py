"""Reusable pieces for the TorchCrop evaluation notebooks.

Four notebooks hold the workflow and the narrative; everything they call lives
here. Two compare against **CyBench** at country level
(``yield_evaluation.ipynb``, ``phenology_evaluation.ipynb``) and two against
0.5° **gridded** references per grid cell (``gdhy_yield_evaluation.ipynb``,
``sage_calendar_evaluation.ipynb``).

=================  ==========================================================
:mod:`config`      Paths, country scope, evaluation constants, stage registries
:mod:`doy`         Circular day-of-year arithmetic
:mod:`torchcrop`   Loader for the combined TorchCrop run
:mod:`cybench`     Loaders for the CyBench yield, calendar and crop-mask CSVs
:mod:`gdhy`        Loader for the GDHY 0.5° gridded yield product
:mod:`sage`        Loader for the SAGE 0.5° crop calendar
:mod:`regions`     Country geometry and the cell to country assignment
:mod:`aggregate`   Weighted (and circular) aggregation to country level
:mod:`grid`        Binning the 10 km run onto a 0.5° reference grid
:mod:`metrics`     RMSE, MAE, Bias, MAPE, R², Pearson r — linear and circular
:mod:`plots`       Publication figures and styled tables
:mod:`style`       One palette, one set of mark specs, two modes
=================  ==========================================================
"""

from __future__ import annotations

from . import (
    aggregate,
    config,
    cybench,
    germany,
    doy,
    gdhy,
    grid,
    metrics,
    plots,
    regions,
    sage,
    style,
    torchcrop,
)

__all__ = [
    "aggregate",
    "config",
    "cybench",
    "doy",
    "gdhy",
    "grid",
    "metrics",
    "plots",
    "regions",
    "sage",
    "style",
    "torchcrop",
]

__version__ = "0.2.0"
