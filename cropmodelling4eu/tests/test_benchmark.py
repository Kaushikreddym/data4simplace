"""SIMPLACE task-log timings, in both run layouts."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from cropmodelling4eu.benchmark import simplace_bench_rows, simplace_log_timings

_BUILD = {"run_name": "runs/winter_wheat/calibrated/mswx",
          "start_date": "1979-01-01", "end_date": "2024-12-31"}
_LOG = """==================================================
SIMPLACE runs/winter_wheat/calibrated/mswx  lines 1-500  ({workers} worker(s))
==================================================
singularity run ... -l=1-500 -loglevel=ERROR
OK   lines 1-500 in 34m40s
"""


def _write_log(directory: Path, name: str, workers: int) -> None:
    directory.mkdir(parents=True, exist_ok=True)
    (directory / name).write_text(_LOG.format(workers=workers))


def test_production_layout(tmp_path: Path) -> None:
    """``logs/`` at the run root, ``build.json`` under ``simplace_regions``."""
    region = tmp_path / "simplace_regions" / "region_01"
    region.mkdir(parents=True)
    (region / "build.json").write_text(json.dumps(_BUILD))
    (region / "log").mkdir()
    (region / "log" / "lines_1-500.out").write_text("SIMPLACE's own log\n")
    _write_log(tmp_path / "logs", "sp_861414_0.out", workers=8)

    timings = simplace_log_timings([tmp_path])

    assert len(timings) == 1
    row = timings.iloc[0]
    assert (row.cells, row.seasons, row.workers, row.seconds) == (500, 45, 8, 2080)
    assert row.cell_years_per_core_s == pytest.approx(500 * 45 / 2080 / 8)


def test_plain_layout(tmp_path: Path) -> None:
    """``build.json`` beside ``log/``, the pre-reorg ``<run>/simplace`` shape."""
    run = tmp_path / "simplace"
    run.mkdir()
    (run / "build.json").write_text(json.dumps(_BUILD))
    _write_log(run / "log", "sp_1_0.out", workers=1)

    rows = simplace_bench_rows(simplace_log_timings([run]))

    assert rows.loc[0, "label"] == tmp_path.name
    assert rows.loc[0, "cell_years_per_s"] == pytest.approx(500 * 45 / 2080)


def test_missing_build_is_skipped(tmp_path: Path) -> None:
    _write_log(tmp_path / "logs", "sp_1_0.out", workers=1)
    assert simplace_log_timings([tmp_path]).empty
