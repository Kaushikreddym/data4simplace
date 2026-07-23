"""Reference CSV parsing engine and the shared exporter base class.

The SIMPLACE reference files are the source of truth for output structure. Each
exporter inspects the matching reference file at runtime to recover its exact
delimiter, header line, column order, missing-value sentinel and (for soil)
depth horizons, then writes new files that conform to that structure — rather
than hard-coding schemas that may drift from the model's expectations.
"""

from __future__ import annotations

import csv
import gzip
import logging
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

import pandas as pd

from data4simplace.config import PipelineConfig
from data4simplace.grid import TargetGrid

logger = logging.getLogger(__name__)

# Candidate delimiters probed when sniffing fails.
_DELIMITER_CANDIDATES = [",", ";", "\t", "|"]
# Tokens commonly used by SIMPLACE / crop models for missing data.
_MISSING_TOKENS = ["-99.9", "-999.9", "-9999", "-999", "-99.0", "-99", "NA", "NaN", "nan", ""]
# Depth horizon such as ``0-5cm``/``30to60cm``. A trailing length unit is
# required so ordinary suffixed names (e.g. ``CaCO3_1``) are not misread.
_DEPTH_RE = re.compile(r"(\d+)\s*(?:-|_|to)\s*(\d+)\s*(cm|mm|m)\b", re.IGNORECASE)
# Numeric layer suffix such as ``clay_1`` … ``clay_6``.
_LAYER_SUFFIX_RE = re.compile(r"_(\d+)$")


def _open_text(path: Path):
    """Open a text file, transparently decompressing ``.gz`` inputs."""
    if path.suffix.lower() == ".gz":
        return gzip.open(path, "rt", encoding="utf-8", errors="replace")
    return path.open("r", encoding="utf-8", errors="replace")


@dataclass
class ReferenceSpec:
    """Structure recovered from a SIMPLACE reference CSV.

    Attributes
    ----------
    delimiter:
        Field delimiter character.
    columns:
        Column names in their exact reference order.
    dtypes:
        Best-effort per-column pandas dtype string inferred from the sample.
    missing_value:
        Missing-data sentinel detected in the file (falls back to config).
    depths:
        Ordered depth horizons parsed from column names, if any.
    header_row:
        Zero-based index of the header line within the file.
    n_preview_rows:
        Number of data rows read while sniffing.
    source:
        Path of the parsed reference file, if any.
    """

    delimiter: str = ","
    columns: list[str] = field(default_factory=list)
    dtypes: dict[str, str] = field(default_factory=dict)
    missing_value: str = "-99"
    depths: list[str] = field(default_factory=list)
    header_row: int = 0
    n_preview_rows: int = 0
    source: Optional[Path] = None

    def empty_frame(self) -> pd.DataFrame:
        """An empty DataFrame carrying the reference column order."""
        return pd.DataFrame(columns=self.columns)


def _sniff_delimiter(sample: str) -> str:
    """Detect the delimiter of a text sample, robust to sniffer failure."""
    try:
        dialect = csv.Sniffer().sniff(sample, delimiters="".join(_DELIMITER_CANDIDATES))
        return dialect.delimiter
    except csv.Error:
        # Fall back to the candidate that yields the most, most-consistent splits.
        first_line = sample.splitlines()[0] if sample.splitlines() else ""
        counts = {d: first_line.count(d) for d in _DELIMITER_CANDIDATES}
        best = max(counts, key=lambda d: counts[d])
        return best if counts[best] > 0 else ","


def _detect_missing(frame: pd.DataFrame, default: str) -> str:
    """Return the most frequent recognised missing-value token in a sample."""
    counts: dict[str, int] = {}
    for token in _MISSING_TOKENS:
        if token == "":
            continue
        counts[token] = int((frame.astype(str) == token).to_numpy().sum())
    if counts and max(counts.values()) > 0:
        return max(counts, key=lambda t: counts[t])
    return default


def _parse_depths(columns: list[str]) -> list[str]:
    """Extract ordered depth horizons from column names.

    Recognises two conventions: explicit ranges with a length unit
    (``clay_0-5cm``) and numeric layer suffixes (``clay_1`` … ``clay_6``). The
    range form is preferred when present; otherwise the distinct ``_N`` suffixes
    are returned as ``"1"`` … ``"N"``.
    """
    ranges: dict[str, tuple[int, int]] = {}
    for col in columns:
        match = _DEPTH_RE.search(str(col))
        if match:
            top, bottom = int(match.group(1)), int(match.group(2))
            ranges.setdefault(f"{top}-{bottom}{match.group(3).lower()}", (top, bottom))
    if ranges:
        return [k for k, _ in sorted(ranges.items(), key=lambda kv: kv[1])]

    suffixes: set[int] = set()
    for col in columns:
        match = _LAYER_SUFFIX_RE.search(str(col))
        if match:
            suffixes.add(int(match.group(1)))
    return [str(n) for n in sorted(suffixes)]


def parse_reference_csv(path: str | Path, default_missing: str = "-99") -> ReferenceSpec:
    """Inspect a SIMPLACE reference CSV and recover its structure.

    Parameters
    ----------
    path:
        Path to the reference CSV file.
    default_missing:
        Sentinel to assume when none is detected in the sample.

    Returns
    -------
    ReferenceSpec
        The recovered structure specification.

    Raises
    ------
    FileNotFoundError
        If ``path`` does not exist.
    """
    ref_path = Path(path)
    if not ref_path.is_file():
        raise FileNotFoundError(f"Reference file not found: {ref_path}")

    with _open_text(ref_path) as handle:
        sample = handle.read(8192)
    if not sample.strip():
        raise ValueError(f"Reference file is empty: {ref_path}")

    delimiter = _sniff_delimiter(sample)
    preview = pd.read_csv(ref_path, sep=delimiter, nrows=200, dtype=str, keep_default_na=False)
    columns = [str(c) for c in preview.columns]

    typed = pd.read_csv(ref_path, sep=delimiter, nrows=200)
    dtypes = {str(c): str(typed[c].dtype) for c in typed.columns}

    spec = ReferenceSpec(
        delimiter=delimiter,
        columns=columns,
        dtypes=dtypes,
        missing_value=_detect_missing(preview, default_missing),
        depths=_parse_depths(columns),
        header_row=0,
        n_preview_rows=len(preview),
        source=ref_path,
    )
    logger.info(
        "Parsed reference %s: %d columns, delimiter %r, missing %r, %d depths",
        ref_path.name,
        len(spec.columns),
        spec.delimiter,
        spec.missing_value,
        len(spec.depths),
    )
    return spec


class BaseExporter:
    """Base class for SIMPLACE exporters.

    Handles reference-spec discovery, ``SimplaceID`` assignment and conformant
    CSV writing. Subclasses implement :meth:`build_frame`.

    Parameters
    ----------
    config:
        The validated pipeline configuration.
    reference_path:
        Path (file or directory) to inspect for the reference structure. A
        directory is scanned for the first ``*.csv`` file.
    """

    #: Human-readable kind, used in log messages and default filenames.
    kind: str = "generic"

    def __init__(self, config: PipelineConfig, reference_path: str | Path | None) -> None:
        self._config = config
        self._grid = TargetGrid.from_config(config.grid)
        self._reference_path = Path(reference_path) if reference_path else None
        self._spec: ReferenceSpec | None = None

    # ------------------------------------------------------------------ #
    # Reference handling
    # ------------------------------------------------------------------ #
    def _resolve_reference_file(self) -> Optional[Path]:
        """Return a concrete reference CSV path from a file or directory."""
        ref = self._reference_path
        if ref is None:
            return None
        if ref.is_file():
            return ref
        if ref.is_dir():
            csvs = sorted(ref.glob("*.csv")) or sorted(ref.glob("*.csv.gz"))
            if csvs:
                return csvs[0]
            logger.warning("No CSV files in reference directory %s", ref)
        else:
            logger.warning("Reference path does not exist: %s", ref)
        return None

    @property
    def spec(self) -> ReferenceSpec:
        """The reference spec, parsed on first use with a safe fallback."""
        if self._spec is None:
            ref_file = self._resolve_reference_file()
            if ref_file is not None:
                self._spec = parse_reference_csv(
                    ref_file, default_missing=str(self._config.missing_value)
                )
            else:
                logger.warning(
                    "No %s reference available; using built-in fallback schema", self.kind
                )
                self._spec = self.fallback_spec()
        return self._spec

    def fallback_spec(self) -> ReferenceSpec:
        """Schema used when no reference file can be inspected.

        Subclasses override this with a documented default matching SIMPLACE
        conventions so the pipeline still produces usable output offline.
        """
        return ReferenceSpec(
            delimiter=",",
            columns=[],
            missing_value=str(self._config.missing_value),
        )

    # ------------------------------------------------------------------ #
    # Frame construction (subclass responsibility)
    # ------------------------------------------------------------------ #
    def build_frame(self, *args, **kwargs) -> pd.DataFrame:  # noqa: D401
        """Build the export DataFrame. Implemented by subclasses."""
        raise NotImplementedError

    # ------------------------------------------------------------------ #
    # Conformant writing
    # ------------------------------------------------------------------ #
    def conform(self, frame: pd.DataFrame) -> pd.DataFrame:
        """Reorder/fill a frame to match the reference column order.

        Columns present in the reference but missing from ``frame`` are added
        and filled with the missing-value sentinel. Extra columns are dropped
        so the output matches the reference exactly (when a reference exists).
        """
        spec = self.spec
        if not spec.columns:
            return frame
        out = frame.copy()
        for col in spec.columns:
            if col not in out.columns:
                out[col] = spec.missing_value
        out = out[spec.columns]
        return out.fillna(spec.missing_value)

    def write_csv(self, frame: pd.DataFrame, path: str | Path) -> Path:
        """Write a conformant CSV using the reference delimiter and sentinel."""
        out_path = Path(path)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        conformed = self.conform(frame)
        conformed.to_csv(
            out_path,
            sep=self.spec.delimiter,
            index=False,
            na_rep=self.spec.missing_value,
        )
        logger.info("Wrote %s (%d rows) -> %s", self.kind, len(conformed), out_path)
        return out_path
