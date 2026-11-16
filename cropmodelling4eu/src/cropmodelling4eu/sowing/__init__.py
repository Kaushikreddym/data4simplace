"""Learned sowing dates: a causal hazard network trained through an emergence model.

The rule-based SIMPLACE sowing (``SowingRule`` in the SUSTAg solution) is
saturated: ``Avg7Tmin <= 12 C`` is true when most European windows open, so
72 % of cell-years sow on the opening day and the date carries no year-to-year
signal (production notebook §7). This package replaces the *date* — not the
crop model — with a table both models already read, ``(SimplaceID, year,
sowing_doy)``.

Pipeline, each step a ``cm4eu sowing`` subcommand:

=================  =========================================================
``split``          PEP725 station blocks -> train / val / test, drawn once
``build-cache``    per (cell, season) daily weather from DOY 193 of Y-1, the
                   medium-term state and the prior-years normals
``fit-tsumem``     one global emergence thermal sum from PEP725 sowing ->
                   emergence lags (train blocks only)
``train``          the hazard network, CLMS emergence + PEP725 sowing loss
``predict``        the sowing table (+ its per-cell climatology, arm C)
``override-crops`` a copy of a calibrated crop directory with ``tsumem`` set
=================  =========================================================

Nothing here changes a default anywhere else: the old method is simply the
old sowing table and the old crop directory, see ``SOWING.md``.
"""

from .config import SowingConfig, load_sowing_config
from .emergence import EmergenceParams, emergence_index, fit_tsumem

__all__ = [
    "EmergenceParams",
    "SowingConfig",
    "emergence_index",
    "fit_tsumem",
    "load_sowing_config",
]
