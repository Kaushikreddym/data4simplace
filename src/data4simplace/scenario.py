"""Complete a weather-only scenario export with a base export's static layers.

A CMIP6 experiment changes only the weather. Soil, the site table and the
fertilizer schedule are properties of the place, so a scenario run exports
weather alone (restricted to ``paths.cell_set_from``'s cells) and this step
borrows the rest from that base export, leaving a directory both crop models
read exactly as they read any other export.

What is borrowed and what is not:

======================  =====================================================
``soil/*``              linked, every entry
``site/site.csv``       linked
``site/co2.csv``        **written**: the scenario's own CO2 (``paths.co2_file``,
                        ``site.co2_hold_year``), not the base run's
``management/*``        linked, except ``*_long.csv``: the long schedule keys on
                        ``Year`` and holds only the base run's years, so a
                        solution reading it would silently skip every season
                        outside them
``validation/``         not linked: observations are year-matched to the real
                        weather, which a free-running GCM does not reproduce
======================  =====================================================

Links point at the **resolved** target. A base export may itself be derived (a
fixed ``soil_long.csv`` beside links into the original), and cm4eu binds only
the directories an export's own links resolve into, two levels deep -- a link
to a link would leave the second hop unbound inside the SIMPLACE container.
"""

from __future__ import annotations

import json
import logging
import os
from datetime import datetime, timezone
from pathlib import Path

from data4simplace.climate import MSWXHandler
from data4simplace.config import PipelineConfig
from data4simplace.site import load_co2_series, write_co2_series

logger = logging.getLogger(__name__)

__all__ = ["link_base_export"]

#: Base-export entries never carried into a scenario: shard and marker trees
#: describe how the base was built, not what it holds.
_SKIP_PREFIXES = ("_shards", ".")


def _link(target: Path, link: Path) -> Path:
    """``link -> target.resolve()``, replacing a stale link but never a file."""
    if link.is_symlink():
        link.unlink()
    elif link.exists():
        raise FileExistsError(f"{link} exists and is not a link; refusing to replace it")
    link.symlink_to(target.resolve())
    return link


def _link_entries(src: Path, dst: Path, skip=lambda p: False) -> list[Path]:
    """Link every entry of ``src`` into a real directory ``dst``."""
    if not src.is_dir():
        raise FileNotFoundError(f"base export has no {src.name}/: {src}")
    dst.mkdir(parents=True, exist_ok=True)
    return [
        _link(entry, dst / entry.name)
        for entry in sorted(src.iterdir())
        if not entry.name.startswith(_SKIP_PREFIXES) and not skip(entry)
    ]


def _simulated_years(config: PipelineConfig) -> list[int]:
    start, end = int(config.time.start[:4]), int(config.time.end[:4])
    return list(range(start - 1, end + 2))


def _weather_cells(out_dir: Path) -> int:
    return sum(1 for _ in (out_dir / "weather").glob("*/daily_mean_RES1_C*R*.csv.gz"))


def link_base_export(config: PipelineConfig, tiles_expected: int | None = None) -> dict:
    """Borrow the base export's static layers and write the scenario's CO2.

    Parameters
    ----------
    config:
        The scenario run's configuration: ``paths.output_dir`` is the scenario
        export, ``paths.cell_set_from`` the base.
    tiles_expected:
        The run's tile count. When given, the step refuses to run until every
        tile carries a ``.done`` marker, so a partially written scenario never
        looks complete to a crop-model run.

    Returns
    -------
    dict
        The provenance record, also written to ``scenario.json``.
    """
    base = config.paths.cell_set_from
    if base is None:
        raise ValueError("--link-base requires paths.cell_set_from (the base export)")
    base, out = Path(base), Path(config.paths.output_dir)

    if tiles_expected is not None:
        done = len(list((out / ".tiles").glob("*.done")))
        if done < tiles_expected:
            raise RuntimeError(
                f"{tiles_expected - done} of {tiles_expected} tiles of {out} are "
                "unfinished; resubmit them before linking the base layers"
            )

    linked = [
        *_link_entries(base / "soil", out / "soil"),
        *_link_entries(
            base / "management", out / "management", skip=lambda p: p.stem.endswith("_long")
        ),
    ]
    if skipped := sorted(p.name for p in (base / "management").glob("*_long.csv")):
        logger.warning(
            "Not linked (year-keyed to the base run's window): %s", ", ".join(skipped)
        )
    (out / "site").mkdir(parents=True, exist_ok=True)
    linked.append(_link(base / "site" / "site.csv", out / "site" / "site.csv"))

    series, source = load_co2_series(
        config.paths.co2_file, _simulated_years(config), config.site.co2_hold_year
    )
    co2_path = write_co2_series(series, source, out)

    record = {
        "created": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "climate_source": config.climate.source,
        "experiment": config.climate.experiment,
        "layout": config.climate.layout,
        # Configured, not verified: a non-required variable with no files is
        # skipped by the reader and its column stays at the sentinel.
        "configured_variables": sorted(config.climate.variables.values()),
        "required_variables": sorted(config.climate.required),
        # Resolved against the source now, not taken from the config: a fill
        # applies only where the source has no files for the variable.
        "filled_from_mswx_climatology": {
            "variables": MSWXHandler(config).filled(),
            "years": list(config.climate.climatology_years),
            "window_days": config.climate.climatology_window_days,
        },
        "time": {"start": config.time.start, "end": config.time.end},
        "base_export": str(base.resolve()),
        "co2": {
            "file": str(config.paths.co2_file) if config.paths.co2_file else None,
            "source": source,
            "hold_year": config.site.co2_hold_year,
            "first": [int(series.index[0]), float(series.iloc[0])],
            "last": [int(series.index[-1]), float(series.iloc[-1])],
        },
        "weather_cells": _weather_cells(out),
        "linked": [os.path.relpath(p, out) for p in linked],
    }
    (out / "scenario.json").write_text(json.dumps(record, indent=2) + "\n")
    logger.info(
        "Scenario %s: %d weather cells, %d entries linked from %s, CO2 %s -> %s",
        config.climate.experiment, record["weather_cells"], len(linked), base,
        source, co2_path,
    )
    return record
