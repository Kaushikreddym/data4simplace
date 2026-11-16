"""The wide daily trajectory layout both crop models write.

One row per (cell, season, day) of the crop phase: :data:`DAILY_KEYS`, then one
float32 column per variable. ``year`` is the harvest year and ``das`` counts
days from each row's own sowing date, so torchcrop's
``torchcrop_daily_shard_<n>.parquet`` and SIMPLACE's ``simplace_daily.parquet``
align on keys alone.
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq

__all__ = ["DAILY_KEYS", "DailySink"]

DAILY_KEYS = ("SimplaceID", "year", "date", "doy", "das")


class DailySink:
    """Streams wide daily frames into one Parquet file.

    A CMIP6 historical region shard is ~0.7 G cell-day-variable values; held
    until the shard ends, as the first daily path did, that is several times
    the job's memory. Rows go to ``<path>.part`` batch by batch and the file
    is renamed into place only on :meth:`close`, so a shard killed mid-way
    never leaves a daily file the skip check would take for finished.
    """

    def __init__(self, path: Path) -> None:
        self.path = Path(path)
        self.part = self.path.with_name(self.path.name + ".part")
        self.rows = 0
        self._writer = None

    def __call__(self, frame: pd.DataFrame) -> None:
        if frame.empty:
            return
        table = pa.Table.from_pandas(frame, preserve_index=False)
        if self._writer is None:
            self.part.parent.mkdir(parents=True, exist_ok=True)
            self._writer = pq.ParquetWriter(self.part, table.schema, compression="zstd")
        self._writer.write_table(table)
        self.rows += len(frame)

    def close(self) -> Path:
        if self._writer is None:
            raise RuntimeError(f"no daily rows were produced for {self.path}")
        self._writer.close()
        self.part.replace(self.path)
        return self.path
