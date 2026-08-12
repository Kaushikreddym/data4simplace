"""Read the export's soil profile, in either layout, as one tidy table.

data4simplace writes the profile two ways — wide (``soil.csv``, one row per
location with ``<stem>_<N>`` columns) and long (``soil_long.csv``, one row per
location and depth). Both describe the same soil, so both are read into the
same frame here and every model-facing function works off that.

**The layer geometry comes from the file, not from a constant.** The previous
runner carried ``LAYER_BOTTOMS_M = [0.1, 0.3, 0.5, 0.7, 1.0, 2.0]`` as a module
constant and applied it to whatever it was handed; a long export carrying
SoilGrids' native horizons (0.05/0.15/0.30/0.60/1.00/2.00 m) would have been
integrated over the wrong depths with no error anywhere. Here the depths are
read from ``SoilLayerDepth_<N>``/``depth_<N>`` in a wide file or the depth
column of a long one, and only fall back to the SIMPLACE default when a file
declares none.
"""

from __future__ import annotations

import logging
import re
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

__all__ = ["SoilProfiles", "read_soil"]

#: Matches a wide per-layer column ``<stem>_<N>``.
_LAYER_COL = re.compile(r"^(?P<stem>.+)_(?P<n>\d+)$")

#: Wide columns that declare the layer bottoms, in metres.
_DEPTH_STEMS = ("SoilLayerDepth", "depth")

#: Long-layout depth column names, in the dialects data4simplace can write.
_LONG_DEPTH_COLUMNS = ("Depth", "depth")

#: Long-layout key columns, likewise.
_LONG_KEY_COLUMNS = ("location", "Location", "soiltype")

#: SIMPLACE layer bottoms [m], used only when a file declares none.
_DEFAULT_BOTTOMS_M = (0.1, 0.3, 0.5, 0.7, 1.0, 2.0)

#: Long dialect column -> the canonical stem the wide layout uses. Reading a
#: long file therefore yields the same names as reading a wide one, and every
#: consumer is layout-blind.
_LONG_ALIASES: dict[str, str] = {
    # EU SUSTAg
    "LL": "soilwater_wp",
    "DUL": "soilwater_fc",
    "SAT": "soilwater_sat",
    "BD": "bulkdensity",
    "OC": "carbon",
    "ph": "PH",
    # ERA5
    "soilwater_wp": "soilwater_wp",
    "soilwater_fc": "soilwater_fc",
    "soilwater_sat": "soilwater_sat",
    "bulkdensity_perc": "bulkdensity",
    "clay_perc": "clay",
    "sand_perc": "sand",
    "carbon_perc": "carbon",
    # Shared spellings
    "soilwater_init": "soilwater_init",
    "soilwater_res": "soilwater_res",
    "clay": "clay",
    "sand": "sand",
    "silt": "silt",
}

#: Canonical stem -> factor turning a long dialect's unit into the wide one's.
#: ``carbon`` is the only one that differs: both long dialects write g/100g
#: where the wide layout writes g/kg.
_LONG_UNIT_FACTORS: dict[str, float] = {"carbon": 10.0}


@dataclass
class SoilProfiles:
    """Per-cell soil profiles with an explicit layer geometry.

    Attributes
    ----------
    values:
        ``{stem: [n_cells, n_layers]}`` in the wide layout's canonical units
        (texture %, bulk density kg/dm3, carbon g/kg, water m3/m3, mineral N
        kg/ha).
    ids:
        ``SimplaceID`` per row of every array, ascending.
    tops_m, bottoms_m:
        Layer boundaries in metres, ``[n_layers]``.
    """

    values: dict[str, np.ndarray]
    ids: np.ndarray
    tops_m: np.ndarray
    bottoms_m: np.ndarray

    @property
    def n_layers(self) -> int:
        return int(self.bottoms_m.size)

    def __contains__(self, stem: str) -> bool:
        return stem in self.values

    def overlap_thickness(self, top_m: float, bottom_m: float) -> np.ndarray:
        """Thickness [m] of each layer inside the window ``[top_m, bottom_m]``."""
        return np.clip(
            np.minimum(self.bottoms_m, bottom_m) - np.maximum(self.tops_m, top_m),
            0.0,
            None,
        )

    def select(self, ids: np.ndarray) -> "SoilProfiles":
        """The profiles of ``ids``, in that order.

        Raises
        ------
        KeyError
            If any id is absent. A missing profile must not be filled or
            skipped silently: the cell would run on another cell's soil, or
            drop out of the results with no record.
        """
        lookup = pd.Index(self.ids)
        positions = lookup.get_indexer(np.asarray(ids, dtype=np.int64))
        if (positions < 0).any():
            missing = np.asarray(ids)[positions < 0]
            raise KeyError(
                f"{missing.size} cell(s) have no soil profile, e.g. "
                f"{missing[:5].tolist()}"
            )
        return SoilProfiles(
            values={stem: array[positions] for stem, array in self.values.items()},
            ids=np.asarray(ids, dtype=np.int64),
            tops_m=self.tops_m,
            bottoms_m=self.bottoms_m,
        )

    def depth_mean(self, stem: str, top_m: float, bottom_m: float) -> np.ndarray:
        """Thickness-weighted mean of a layered property over a depth window."""
        weights = self.overlap_thickness(top_m, bottom_m)
        total = weights.sum()
        if total <= 0:
            raise ValueError(
                f"Depth window {top_m}-{bottom_m} m overlaps none of the "
                f"profile's layers (bottoms {self.bottoms_m.tolist()} m)"
            )
        return self.values[stem] @ weights / total

    def depth_sum(self, stem: str, top_m: float, bottom_m: float) -> np.ndarray:
        """Depth-integrated stock, for values that are already per-layer totals.

        The per-layer mineral-N columns are stocks over their own layer, so a
        window covering half a layer takes half its stock -- hence the weights
        are overlap *fractions* rather than thicknesses.
        """
        thickness = self.bottoms_m - self.tops_m
        weights = self.overlap_thickness(top_m, bottom_m) / thickness
        return self.values[stem] @ weights


def read_soil(export_dir: str | Path, layout: str = "auto") -> SoilProfiles:
    """Read the export's soil profiles from whichever layout is present.

    Parameters
    ----------
    export_dir:
        A data4simplace export directory.
    layout:
        ``auto`` prefers ``soil.csv`` and falls back to ``soil_long.csv``;
        ``wide`` and ``long`` demand one specifically.

    Raises
    ------
    FileNotFoundError
        If the requested layout is not in the export.
    """
    soil_dir = Path(export_dir) / "soil"
    wide, long_ = soil_dir / "soil.csv", soil_dir / "soil_long.csv"

    if layout == "wide" or (layout == "auto" and wide.is_file()):
        if not wide.is_file():
            raise FileNotFoundError(f"No wide soil file: {wide}")
        return _read_wide(wide)
    if layout in ("long", "auto"):
        if not long_.is_file():
            raise FileNotFoundError(
                f"No soil file in the export: neither {wide.name} nor "
                f"{long_.name} under {soil_dir}"
            )
        return _read_long(long_)
    raise ValueError(f"layout must be auto, wide or long; got {layout!r}")


def _read_wide(path: Path) -> SoilProfiles:
    """Read ``soil.csv``: one row per location, ``<stem>_<N>`` columns."""
    frame = pd.read_csv(path).sort_values("location", kind="stable")
    ids = frame["location"].to_numpy(dtype=np.int64)

    stems: dict[str, list[int]] = {}
    for column in frame.columns:
        match = _LAYER_COL.match(str(column))
        if match:
            stems.setdefault(match.group("stem"), []).append(int(match.group("n")))

    n_layers = max((max(layers) for layers in stems.values()), default=0)
    if n_layers == 0:
        raise ValueError(f"{path} carries no per-layer <stem>_<N> columns")

    bottoms_m = _wide_layer_bottoms(frame, stems, n_layers, path)
    values: dict[str, np.ndarray] = {}
    for stem, layers in stems.items():
        if stem in _DEPTH_STEMS or sorted(layers) != list(range(1, n_layers + 1)):
            continue
        block = frame[[f"{stem}_{n}" for n in range(1, n_layers + 1)]]
        values[stem] = block.apply(pd.to_numeric, errors="coerce").to_numpy(dtype="float64")

    logger.info(
        "Soil (wide) %s: %d profiles, %d layers to %.2f m, %d properties",
        path.name, len(ids), n_layers, bottoms_m[-1], len(values),
    )
    return SoilProfiles(
        values=values,
        ids=ids,
        tops_m=np.concatenate([[0.0], bottoms_m[:-1]]),
        bottoms_m=bottoms_m,
    )


def _wide_layer_bottoms(
    frame: pd.DataFrame, stems: dict[str, list[int]], n_layers: int, path: Path
) -> np.ndarray:
    """Layer bottoms [m] declared by the file, else the SIMPLACE default."""
    for stem in _DEPTH_STEMS:
        if stem not in stems:
            continue
        columns = [f"{stem}_{n}" for n in range(1, n_layers + 1)]
        if not set(columns) <= set(frame.columns):
            continue
        row = pd.to_numeric(frame[columns].iloc[0], errors="coerce")
        if row.notna().all() and row.is_monotonic_increasing:
            return row.to_numpy(dtype="float64")
    logger.warning(
        "%s declares no layer depths (%s_<N>); assuming the SIMPLACE default "
        "%s m. Check this against the solution if the run looks shallow.",
        path.name, "/".join(_DEPTH_STEMS), list(_DEFAULT_BOTTOMS_M[:n_layers]),
    )
    return np.asarray(_DEFAULT_BOTTOMS_M[:n_layers], dtype="float64")


def _read_long(path: Path) -> SoilProfiles:
    """Read ``soil_long.csv``: one row per location and depth."""
    frame = pd.read_csv(path, sep=None, engine="python")
    key = _first_present(frame.columns, _LONG_KEY_COLUMNS)
    depth = _first_present(frame.columns, _LONG_DEPTH_COLUMNS)
    if key is None or depth is None:
        raise ValueError(
            f"{path} is not a recognised long soil file: expected one of "
            f"{_LONG_KEY_COLUMNS} as the key and one of {_LONG_DEPTH_COLUMNS} "
            f"as the depth; it has {list(frame.columns)[:8]}"
        )

    frame = frame.sort_values([key, depth], kind="stable")
    bottoms_m = np.asarray(
        frame.loc[frame[key] == frame[key].iloc[0], depth], dtype="float64"
    )
    n_layers = bottoms_m.size

    counts = frame.groupby(key).size()
    ragged = counts[counts != n_layers]
    if not ragged.empty:
        raise ValueError(
            f"{path}: {len(ragged)} location(s) do not have {n_layers} layers "
            f"(e.g. {ragged.index[0]} has {int(ragged.iloc[0])}). A long file "
            f"must be rectangular for SIMPLACE to assemble its arrays."
        )

    ids = frame[key].drop_duplicates().to_numpy()
    ids = ids.astype(np.int64) if np.issubdtype(ids.dtype, np.number) else ids

    values: dict[str, np.ndarray] = {}
    for column in frame.columns:
        stem = _LONG_ALIASES.get(str(column))
        if stem is None or stem in values:
            continue
        block = pd.to_numeric(frame[column], errors="coerce").to_numpy(dtype="float64")
        values[stem] = block.reshape(len(ids), n_layers) * _LONG_UNIT_FACTORS.get(stem, 1.0)

    logger.info(
        "Soil (long) %s: %d profiles, %d layers to %.2f m, %d properties "
        "(key %r, depth %r)",
        path.name, len(ids), n_layers, bottoms_m[-1], len(values), key, depth,
    )
    return SoilProfiles(
        values=values,
        ids=ids,
        tops_m=np.concatenate([[0.0], bottoms_m[:-1]]),
        bottoms_m=bottoms_m,
    )


def _first_present(columns, candidates: tuple[str, ...]) -> str | None:
    """The first of ``candidates`` present in ``columns``."""
    present = {str(c) for c in columns}
    return next((c for c in candidates if c in present), None)
