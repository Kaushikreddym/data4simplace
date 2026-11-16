"""Copies of crop YAMLs with emergence constants replaced — never in place.

An arm with the PEP725 ``tsumem`` runs on a *copy* of the calibrated crop
directory, so the original crops (arm A0 and production) are untouched and
falling back is choosing the old directory. The edit is textual: only the
numeric ``<name>: <value>`` line changes, so a copy diffs against its source
in exactly those lines and keeps the ``source_file:`` provenance line the
production driver checks for.
"""

from __future__ import annotations

import json
import logging
import re
import shutil
from pathlib import Path

logger = logging.getLogger(__name__)

__all__ = ["override_crop_file", "override_crops"]

_NUMBER = r"[-+0-9.eE]+"


def override_crop_file(source: Path, target: Path, values: dict[str, float]) -> dict[str, float]:
    """Write ``source`` to ``target`` with each named scalar replaced; return the old values."""
    text = Path(source).read_text()
    old = {}
    for name, value in values.items():
        pattern = re.compile(rf"^(\s+{re.escape(name)}:\s*)({_NUMBER})\s*$", re.M)
        found = pattern.findall(text)
        if len(found) != 1:
            raise ValueError(f"{source}: expected one numeric '{name}:' line, found {len(found)}")
        old[name] = float(found[0][1])
        text = pattern.sub(lambda m: f"{m.group(1)}{float(value)!r}", text)
    Path(target).parent.mkdir(parents=True, exist_ok=True)
    Path(target).write_text(text)
    return old


def override_crops(crops_dir: Path, out_dir: Path, values: dict[str, float]) -> Path:
    """Copy every file of ``crops_dir`` to ``out_dir``, overriding ``crop_*.yaml``."""
    crops_dir, out_dir = Path(crops_dir), Path(out_dir)
    if crops_dir.resolve() == out_dir.resolve():
        raise ValueError("refusing to override crops in place")
    out_dir.mkdir(parents=True, exist_ok=True)
    changed = {}
    for path in sorted(crops_dir.iterdir()):
        if path.is_file() and path.name.startswith("crop") and path.suffix == ".yaml":
            changed[path.name] = override_crop_file(path, out_dir / path.name, values)
        elif path.is_file():
            shutil.copy2(path, out_dir / path.name)
    if not changed:
        raise FileNotFoundError(f"no crop*.yaml in {crops_dir}")
    (out_dir / "override.json").write_text(json.dumps(
        {"source": str(crops_dir), "values": values, "previous": changed}, indent=1))
    logger.info("%d crop file(s) -> %s with %s", len(changed), out_dir, values)
    return out_dir
