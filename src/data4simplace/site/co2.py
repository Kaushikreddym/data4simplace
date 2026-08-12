"""Annual atmospheric CO2 concentration for the simulated years.

Both models take CO2 as a scalar per season — LINTUL-5 scales assimilation with
it and SIMPLACE reads it from its own CSV interface — and neither the MSWX nor
the SoilGrids inputs carry it. CO2 is well mixed on the timescale a crop model
integrates over, so this is one global series rather than a per-cell field: the
inter-hemispheric gradient is a few ppm against a 90 ppm rise over the export's
own 1979-2024 window.

The series is read from ``paths.co2_file`` (two columns, year and ppm, any
header spelling). Without one, a built-in table of global annual means is
interpolated and the written file records that it is a fallback, so a run made
before the real series arrived stays identifiable afterwards.
"""

from __future__ import annotations

import logging
from pathlib import Path

import pandas as pd

logger = logging.getLogger(__name__)

__all__ = ["FALLBACK_CO2_PPM", "load_co2_series", "write_co2_series"]

#: Global-mean CO2 [ppm] at decadal anchors, from NOAA/ESRL annual means. Used
#: only when no ``paths.co2_file`` is configured; values between the anchors are
#: linearly interpolated, which tracks the real series to within ~1 ppm.
FALLBACK_CO2_PPM: dict[int, float] = {
    1979: 336.8,
    1990: 354.4,
    2000: 369.6,
    2010: 389.9,
    2020: 414.2,
    2024: 422.7,
}

#: Accepted spellings of the two columns, lower-cased.
_YEAR_NAMES = ("year", "yr", "time")
_VALUE_NAMES = ("co2", "co2_ppm", "ppm", "value", "concentration", "mean")


def _pick_column(frame: pd.DataFrame, candidates: tuple[str, ...], position: int) -> str:
    """Column matching one of ``candidates`` by name, else the one at ``position``."""
    lowered = {str(c).strip().lower(): str(c) for c in frame.columns}
    for name in candidates:
        if name in lowered:
            return lowered[name]
    return str(frame.columns[position])


def load_co2_series(
    path: str | Path | None, years: range | list[int] | None = None
) -> tuple[pd.Series, str]:
    """Annual CO2 [ppm] indexed by year, plus the provenance of the series.

    Parameters
    ----------
    path:
        ``paths.co2_file``. ``None`` selects :data:`FALLBACK_CO2_PPM`.
    years:
        Years the series must cover. Gaps inside the input's own range are
        linearly interpolated; years outside it are carried from the nearest
        end and logged, because a crop run that silently drops a season is
        harder to notice than one that reuses a neighbouring year's CO2.

    Returns
    -------
    tuple
        ``(series, source)`` where ``source`` is ``"file"`` or ``"fallback"``.

    Raises
    ------
    FileNotFoundError
        If ``path`` is given but absent — an unreadable path is a
        configuration error, not a reason to fall back silently.
    ValueError
        If the file has fewer than two columns or no usable rows.
    """
    if path is None:
        series = pd.Series(FALLBACK_CO2_PPM, name="co2_ppm", dtype="float64")
        source = "fallback"
        logger.warning(
            "No paths.co2_file configured; using the built-in global-mean CO2 "
            "table (%d-%d). Set it once the real series is available.",
            int(series.index.min()), int(series.index.max()),
        )
    else:
        csv_path = Path(path)
        if not csv_path.is_file():
            raise FileNotFoundError(f"CO2 file not found: {csv_path}")
        frame = pd.read_csv(csv_path)
        if frame.shape[1] < 2:
            raise ValueError(
                f"{csv_path} needs a year column and a ppm column; it has "
                f"{list(frame.columns)}"
            )
        year_col = _pick_column(frame, _YEAR_NAMES, 0)
        value_col = _pick_column(frame, _VALUE_NAMES, 1)
        series = (
            frame[[year_col, value_col]]
            .apply(pd.to_numeric, errors="coerce")
            .dropna()
            .astype({year_col: int})
            .groupby(year_col)[value_col]
            .mean()
            .rename("co2_ppm")
        )
        if series.empty:
            raise ValueError(f"{csv_path} holds no numeric year/ppm rows")
        source = "file"
        logger.info(
            "CO2 series from %s: %d years (%d-%d), %s -> %s",
            csv_path.name, len(series), int(series.index.min()),
            int(series.index.max()), year_col, value_col,
        )

    series.index.name = "year"
    if years is None:
        return series.sort_index(), source

    wanted = [int(y) for y in years]
    full = range(min(min(wanted), int(series.index.min())),
                 max(max(wanted), int(series.index.max())) + 1)
    dense = series.reindex(full).interpolate(limit_direction="both")

    outside = [y for y in wanted if y < series.index.min() or y > series.index.max()]
    if outside:
        logger.warning(
            "%d requested year(s) fall outside the CO2 series %d-%d and were "
            "carried from its nearest end: %s",
            len(outside), int(series.index.min()), int(series.index.max()),
            ", ".join(str(y) for y in outside[:10]) + (" ..." if len(outside) > 10 else ""),
        )
    return dense.reindex(wanted).rename("co2_ppm"), source


def write_co2_series(
    series: pd.Series, source: str, output_dir: str | Path
) -> Path:
    """Write ``site/co2.csv`` as ``year,co2_ppm`` with a provenance comment.

    The comment line is a ``#`` header the CSV readers on both sides skip, and
    it is the only place a fallback series is distinguishable from a real one
    once the file has left this process.
    """
    out_path = Path(output_dir) / "site" / "co2.csv"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    # Rounded on the way out: interpolating between annual means leaves float
    # noise (368.08000000000004) that reads as a precision the series does not
    # have, and 0.01 ppm is already far below the inter-annual spread.
    frame = series.round(2).rename("co2_ppm").rename_axis("year").reset_index()
    with out_path.open("w", encoding="utf-8") as handle:
        handle.write(f"# atmospheric CO2 [ppm], source: {source}\n")
        frame.to_csv(handle, index=False)
    logger.info("Wrote CO2 series (%d years, %s) -> %s", len(frame), source, out_path)
    return out_path
