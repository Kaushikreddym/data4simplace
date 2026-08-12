"""Run the differentiable LINTUL-5 model (`torchcrop`) over the export.

``run`` simulates one shard of the runnable cell set and writes a Parquet of
per-(cell, season) summaries; ``maps`` concatenates the shards, grids them to
NetCDF and renders the spatial maps. Both are importable libraries and
``python -m`` entry points, which is what the SLURM scripts in ``submit/``
drive.

``torch`` is an optional dependency of this package (``pip install -e
'.[torchcrop]'``) — the SIMPLACE side needs none of it — so importing this
subpackage does not import torch. Only the modules that use it do.
"""

__all__ = ["maps", "run"]
