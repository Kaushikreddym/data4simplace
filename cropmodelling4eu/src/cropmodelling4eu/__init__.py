"""cropmodelling4eu — run and evaluate crop models over the European export.

Everything downstream of `data4simplace`: build a run from a finished export,
execute it with **SIMPLACE** (via its Singularity container) or **torchcrop**
(differentiable LINTUL-5), collect the results into one schema, and evaluate
them against CyBench, GDHY and the SAGE crop calendar.

The two models share this package's reader layer, so they differ only in how a
cell is simulated — not in which cells exist, where they are, what soil they
have or when the crop goes in. That is what makes their results comparable
rather than merely adjacent.
"""

from __future__ import annotations

from importlib.metadata import PackageNotFoundError, version

from cropmodelling4eu.config import RunConfig, load_config

try:
    __version__ = version("cropmodelling4eu")
except PackageNotFoundError:  # pragma: no cover - source checkout without install
    __version__ = "0.1.0"

__all__ = ["RunConfig", "load_config", "__version__"]
