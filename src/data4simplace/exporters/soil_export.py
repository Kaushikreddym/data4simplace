"""SIMPLACE soil profile file generator.

The SIMPLACE soil reference (``soil.csv``) is a **wide** table: one row per
``location``, with per-layer soil properties encoded as ``<stem>_<N>`` columns
(e.g. ``clay_1`` … ``clay_6``) plus single-value profile attributes. This
exporter reproduces that structure from the regridded SoilGrids dataset:

* **Spatial properties** (``clay``, ``sand``, ``bulkdensity``, ``carbon``,
  ``PH``) are taken from SoilGrids, remapped from SoilGrids' standard depth
  intervals onto the SIMPLACE layer boundaries by overlap-weighted averaging.
* **Hydraulic properties** (``soilwater_fc``/``_wp``/``_sat``/``_init``) come
  from the SoilGrids volumetric water-content layers (``wv0033``/``wv1500``/
  ``wv0010``), which are measured-data predictions rather than a texture
  regression. The Saxton–Rawls pedotransfer output is only a fallback, used for
  whichever of those columns the water-content layers do not cover.
* **Initial mineral N** (``ammonium``, ``nitrate``) is derived from the
  SoilGrids total-N layer: total N and bulk density give a per-layer stock in
  kg N/ha, of which ``soil.mineral_n_fraction`` is taken as mineral and split by
  ``soil.ammonium_share``.
* **Non-derivable columns** (soiltype, CaCO3, P pools, initialisation
  constants …) are carried over from the reference template row so the file
  stays runnable; without a reference they fall back to the missing sentinel.

Keying: outputs use our own 10 km grid cell as ``location`` (each carries a full
lat/lon). The reference ``location`` IDs are not reused because the project's
``location.csv`` provides latitude only — sampling SoilGrids needs longitude too.

Two exporters share all of that machinery: :class:`SoilExporter` writes the one
``soil.csv`` of the dominant class, and :class:`TopSoilExporter` writes the same
schema once per primary soil class (``soil_1.csv`` … ``soil_n.csv``) with the
per-class metadata block appended.
"""

from __future__ import annotations

import logging
import re
from pathlib import Path

import numpy as np
import pandas as pd
import xarray as xr

from data4simplace.config import PipelineConfig
from data4simplace.exporters.base_exporter import (
    BaseExporter,
    ReferenceSpec,
    parse_reference_csv,
)
from data4simplace.exporters.layout import SOIL_DIALECTS, LongDialect, select_dialect
from data4simplace.soil.multiclass import METADATA_COLUMNS, TopClassAggregation

logger = logging.getLogger(__name__)

# Reference column stem  ->  source SoilGrids variable (units already unscaled:
# clay/sand in %, bulkdensity in kg/dm3, carbon in g/kg, PH on the pH scale).
_STEM_TO_SOILGRIDS = {
    "clay": "clay",
    "sand": "sand",
    # Neither silt nor nitrogen appears in the Brandenburg wide reference, so
    # `conform` drops them there. They are carried because the long dialects do
    # declare silt, and because the C:N ratio needs total N.
    "silt": "silt",
    "nitrogen": "nitrogen",
    "bulkdensity": "bdod",
    "carbon": "soc",
    "PH": "phh2o",
    # Water retention straight from the SoilGrids volumetric water contents:
    # 33 kPa is field capacity, 1500 kPa the wilting point, and 10 kPa - the
    # wettest suction SoilGrids publishes - stands in for saturation.
    "soilwater_fc": "wv0033",
    "soilwater_wp": "wv1500",
    "soilwater_sat": "wv0010",
    "soilwater_init": "wv0033",  # initial water content initialised at field capacity
}
# Reference column stem  ->  Saxton-Rawls PTF variable (volumetric, m3/m3).
# Only consulted for stems the SoilGrids layers above did not fill, so the PTF
# is a fallback for runs configured without the wv* layers.
_STEM_TO_PTF = {
    "soilwater_fc": "theta_fc",
    "soilwater_wp": "theta_wp",
    "soilwater_sat": "theta_sat",
    "soilwater_init": "theta_fc",
}
# Unit conversion applied after the depth remap, per reference column stem. The
# wv* layers are un-scaled to vol% (10^-2 cm3/cm3); SIMPLACE wants m3/m3.
_STEM_UNIT_FACTOR = {
    "soilwater_fc": 0.01,
    "soilwater_wp": 0.01,
    "soilwater_sat": 0.01,
    "soilwater_init": 0.01,
}
# SIMPLACE layer bottom depths (m) used when the reference does not specify them.
_DEFAULT_LAYER_BOTTOMS_M = (0.1, 0.3, 0.5, 0.7, 1.0, 2.0)
# Matches a per-layer column ``<stem>_<N>``.
_LAYER_COL_RE = re.compile(r"^(?P<stem>.+)_(?P<n>\d+)$")


def _parse_interval_cm(label: str) -> tuple[float, float]:
    """Parse a SoilGrids depth label such as ``"5-15cm"`` -> ``(5.0, 15.0)``."""
    token = label.lower().replace("cm", "").strip()
    top, bottom = token.split("-")
    return float(top), float(bottom)


def _bottoms_to_intervals_cm(bottoms_cm: list[float]) -> list[tuple[float, float]]:
    """Turn cumulative bottom depths into ``(top, bottom)`` intervals."""
    intervals: list[tuple[float, float]] = []
    top = 0.0
    for bottom in bottoms_cm:
        intervals.append((top, bottom))
        top = bottom
    return intervals


def remap_depth_weighted(
    values: np.ndarray,
    src_intervals: list[tuple[float, float]],
    dst_intervals: list[tuple[float, float]],
) -> np.ndarray:
    """Overlap-weighted remap of a depth-resolved cube onto new layers.

    Parameters
    ----------
    values:
        Array shaped ``(n_src_layers, ...)``.
    src_intervals, dst_intervals:
        Lists of ``(top, bottom)`` depth intervals (same unit) for the source
        and destination layering.

    Returns
    -------
    numpy.ndarray
        Array shaped ``(n_dst_layers, ...)``; destination layers with no source
        overlap are ``NaN``.
    """
    out = np.full((len(dst_intervals),) + values.shape[1:], np.nan, dtype="float64")
    for j, (d_top, d_bot) in enumerate(dst_intervals):
        acc = np.zeros(values.shape[1:], dtype="float64")
        wsum = 0.0
        for i, (s_top, s_bot) in enumerate(src_intervals):
            overlap = max(0.0, min(d_bot, s_bot) - max(d_top, s_top))
            if overlap > 0:
                acc = acc + np.nan_to_num(values[i]) * overlap
                wsum += overlap
        if wsum > 0:
            out[j] = acc / wsum
    return out


class SoilExporter(BaseExporter):
    """Export a SIMPLACE wide-format soil profile file from soil data."""

    kind = "soil"

    def __init__(self, config: PipelineConfig, reference_path: str | Path | None) -> None:
        super().__init__(config, reference_path)
        self._template: dict[str, object] | None = None

    def fallback_spec(self) -> ReferenceSpec:
        """Compact wide schema used when no reference file is available."""
        stems = ["clay", "sand", "bulkdensity", "carbon", "PH",
                 "soilwater_fc", "soilwater_wp", "soilwater_sat", "soilwater_init",
                 "ammonium", "nitrate"]
        columns = ["location"]
        columns += [f"{stem}_{n}" for n in range(1, 7) for stem in stems]
        columns += [f"SoilLayerDepth_{n}" for n in range(1, 7)]
        return ReferenceSpec(
            delimiter=",",
            columns=columns,
            missing_value=str(self._config.missing_value),
        )

    # ------------------------------------------------------------------ #
    # Reference template (constant / non-derivable columns)
    # ------------------------------------------------------------------ #
    def _reference_template(self) -> dict[str, object]:
        """First reference row as ``{column: value}`` for carry-over defaults."""
        if self._template is not None:
            return self._template
        ref_file = self._resolve_reference_file()
        if ref_file is None:
            self._template = {}
            return self._template
        row = pd.read_csv(ref_file, sep=self.spec.delimiter, nrows=1).iloc[0]
        self._template = {str(k): v for k, v in row.to_dict().items()}
        return self._template

    def _layer_bottoms_cm(self, n_layers: int) -> list[float]:
        """SIMPLACE layer bottom depths (cm), read from the reference if present."""
        template = self._reference_template()
        bottoms: list[float] = []
        for n in range(1, n_layers + 1):
            val = template.get(f"SoilLayerDepth_{n}", template.get(f"depth_{n}"))
            if val is None:
                bottoms = []
                break
            bottoms.append(float(val) * 100.0)  # metres -> cm
        if not bottoms:
            bottoms = [b * 100.0 for b in _DEFAULT_LAYER_BOTTOMS_M][:n_layers]
        return bottoms

    @staticmethod
    def _n_layers(columns: list[str]) -> int:
        """Highest per-layer suffix present in the reference columns."""
        suffixes = [int(m.group("n")) for c in columns if (m := _LAYER_COL_RE.match(c))]
        return max(suffixes) if suffixes else 6

    # ------------------------------------------------------------------ #
    # Initial mineral N
    # ------------------------------------------------------------------ #
    def _mineral_nitrogen(
        self,
        soil: xr.Dataset,
        src_intervals: list[tuple[float, float]],
        dst_intervals: list[tuple[float, float]],
    ) -> dict[str, np.ndarray]:
        """Per-layer initial ammonium/nitrate cubes (kg N/ha) from total N.

        SoilGrids ``nitrogen`` is *total* (largely organic) N in g/kg, while
        SIMPLACE initialises with *mineral* N per layer. Bulk density turns the
        concentration into an N density, which is thickness-weighted onto the
        SIMPLACE layers and integrated over their thickness::

            density[g N/dm3] = nitrogen[g/kg] * bdod[kg/dm3]
            stock[kg N/ha]   = density * 100 * thickness[cm]

        The density is formed **before** the depth remap: the stock is additive
        over depth, so remapping the product is exact where remapping the two
        factors separately would not be.

        ``soil.mineral_n_fraction`` of that stock is taken as mineral N and split
        by ``soil.ammonium_share``. Both are initialisation assumptions, not
        measurements.

        Returns an empty mapping when the fraction is 0 or either input layer is
        missing, leaving the columns to the reference constants.
        """
        settings = self._config.soil
        if settings.mineral_n_fraction <= 0.0:
            return {}
        if not {"nitrogen", "bdod"}.issubset(soil.data_vars):
            logger.info(
                "ammonium_*/nitrate_* left at the reference constants: "
                "soil.layers needs both 'nitrogen' and 'bdod'"
            )
            return {}

        density = np.asarray(soil["nitrogen"].values) * np.asarray(soil["bdod"].values)
        density = remap_depth_weighted(density, src_intervals, dst_intervals)
        thickness = np.array([bot - top for top, bot in dst_intervals], dtype="float64")
        thickness = thickness.reshape((-1,) + (1,) * (density.ndim - 1))

        mineral = density * 100.0 * thickness * settings.mineral_n_fraction
        return {
            "ammonium": mineral * settings.ammonium_share,
            "nitrate": mineral * (1.0 - settings.ammonium_share),
        }

    # ------------------------------------------------------------------ #
    # Property cubes (shared by both layouts)
    # ------------------------------------------------------------------ #
    def _property_cubes(
        self,
        soil: xr.Dataset,
        hydraulic: xr.Dataset | None,
        src_intervals: list[tuple[float, float]],
        dst_intervals: list[tuple[float, float]],
    ) -> dict[str, np.ndarray]:
        """Every derivable property as ``{stem: (n_layers, n_lat, n_lon)}``.

        This is the whole computation both the wide and the long serialisation
        rest on: the SoilGrids layers remapped onto ``dst_intervals``, the PTF
        as a fallback for water contents the ``wv*`` layers did not fill, and
        the initial mineral N. Values are in the pipeline's canonical units
        (texture %, bulk density kg/dm3, carbon g/kg, water m3/m3, mineral N
        kg/ha), which is what the layouts convert *from*.
        """
        remapped: dict[str, np.ndarray] = {}
        for stem, var in _STEM_TO_SOILGRIDS.items():
            if var in soil.data_vars:
                cube = remap_depth_weighted(
                    np.asarray(soil[var].values), src_intervals, dst_intervals
                )
                remapped[stem] = cube * _STEM_UNIT_FACTOR.get(stem, 1.0)
        if hydraulic is not None and "depth" in hydraulic.dims:
            # The PTF only covers the depths where sand and clay were both
            # available, so it carries its own source intervals.
            hydraulic = hydraulic.load()
            ptf_intervals = [_parse_interval_cm(str(d)) for d in hydraulic["depth"].values]
            for stem, var in _STEM_TO_PTF.items():
                # SoilGrids' measured-data water contents win where present.
                if stem in remapped or var not in hydraulic.data_vars:
                    continue
                remapped[stem] = remap_depth_weighted(
                    np.asarray(hydraulic[var].values), ptf_intervals, dst_intervals
                )

        remapped.update(self._mineral_nitrogen(soil, src_intervals, dst_intervals))
        return remapped

    # ------------------------------------------------------------------ #
    # Frame construction
    # ------------------------------------------------------------------ #
    def build_frame(
        self,
        soil: xr.Dataset,
        cell_table: pd.DataFrame,
        hydraulic: xr.Dataset | None = None,
    ) -> pd.DataFrame:
        """Build the wide-format soil table (one row per grid cell / location).

        Parameters
        ----------
        soil:
            Regridded soil dataset; variables carry a ``depth`` dimension.
        cell_table:
            Grid cell table (``SimplaceID``, row, col, lat, lon).
        hydraulic:
            Optional Saxton-Rawls dataset aligned to ``soil`` (with ``depth``).
        """
        if "depth" not in soil.dims:
            raise ValueError("Soil dataset must carry a 'depth' dimension")

        columns = self.spec.columns or self.fallback_spec().columns
        n_layers = self._n_layers(columns)
        src_intervals = [_parse_interval_cm(str(d)) for d in soil["depth"].values]
        dst_intervals = _bottoms_to_intervals_cm(self._layer_bottoms_cm(n_layers))
        template = self._reference_template()

        soil = soil.load()
        remapped = self._property_cubes(soil, hydraulic, src_intervals, dst_intervals)

        rows = np.asarray(cell_table["row"], dtype=int)
        cols = np.asarray(cell_table["col"], dtype=int)
        sentinel = self.spec.missing_value

        data: dict[str, object] = {}
        for col in columns:
            if col == "location":
                data[col] = np.asarray(cell_table["SimplaceID"], dtype=np.int64)
                continue
            match = _LAYER_COL_RE.match(col)
            if match and match.group("stem") in remapped:
                layer = int(match.group("n")) - 1
                cube = remapped[match.group("stem")]
                if layer < cube.shape[0]:
                    values = cube[layer, rows, cols]
                    data[col] = np.round(values, 5)
                    continue
            # Non-derivable: carry the reference constant, else the sentinel.
            data[col] = template.get(col, sentinel)

        frame = pd.DataFrame(data, columns=columns)

        # Drop cells with no valid derived soil data (e.g. masked-out cells).
        derived_cols = [
            c for c in columns
            if (m := _LAYER_COL_RE.match(c)) and m.group("stem") in remapped
        ]
        if derived_cols:
            valid = ~frame[derived_cols].apply(pd.to_numeric, errors="coerce").isna().all(axis=1)
            frame = frame[valid].reset_index(drop=True)
        return frame

    def export(
        self,
        soil: xr.Dataset,
        cell_table: pd.DataFrame,
        output_dir: str | Path,
        hydraulic: xr.Dataset | None = None,
    ) -> Path:
        """Write the wide-format soil file; return its path."""
        frame = self.build_frame(soil, cell_table, hydraulic)
        out_path = Path(output_dir) / "soil" / "soil.csv"
        return self.write_csv(frame, out_path)

    # ------------------------------------------------------------------ #
    # Tidy profile table (the long layout's input)
    # ------------------------------------------------------------------ #
    def build_profile_table(
        self,
        soil: xr.Dataset,
        cell_table: pd.DataFrame,
        hydraulic: xr.Dataset | None = None,
        depths: str = "native",
    ) -> pd.DataFrame:
        """One row per ``(location, depth)`` in the pipeline's canonical units.

        This is the same computation :meth:`build_frame` serialises wide, left
        un-pivoted. It is what the long layouts render, and it is deliberately
        *not* a reshape of the wide frame: the wide frame has already been
        remapped onto the reference's six SIMPLACE layers and rounded, whereas a
        long file can carry SoilGrids' own horizons.

        Parameters
        ----------
        depths:
            ``native`` keeps SoilGrids' horizons (no depth remap at all);
            ``simplace`` remaps onto the wide reference's layer bottoms, so the
            two files describe the same layering and can be compared directly.

        Returns
        -------
        pandas.DataFrame
            ``location``, ``depth_top_cm``, ``depth_bottom_cm``, every derivable
            canonical property, and the profile-level ``soilwater_fc_global`` /
            ``soilwater_sat_global`` repeated on each of a location's rows.
            Rows are ordered by location, then by depth.
        """
        if "depth" not in soil.dims:
            raise ValueError("Soil dataset must carry a 'depth' dimension")
        if depths not in ("native", "simplace"):
            raise ValueError(f"depths must be 'native' or 'simplace', got {depths!r}")

        src_intervals = [_parse_interval_cm(str(d)) for d in soil["depth"].values]
        if depths == "native":
            dst_intervals = list(src_intervals)
        else:
            columns = self.spec.columns or self.fallback_spec().columns
            dst_intervals = _bottoms_to_intervals_cm(
                self._layer_bottoms_cm(self._n_layers(columns))
            )

        soil = soil.load()
        cubes = self._property_cubes(soil, hydraulic, src_intervals, dst_intervals)

        rows = np.asarray(cell_table["row"], dtype=int)
        cols = np.asarray(cell_table["col"], dtype=int)
        ids = np.asarray(cell_table["SimplaceID"], dtype=np.int64)
        n_cells, n_layers = len(ids), len(dst_intervals)

        # Cell-major: a location's layers stay contiguous and in depth order,
        # which is what SIMPLACE's DOUBLEARRAY assembly assumes.
        frame = pd.DataFrame(
            {
                "location": np.repeat(ids, n_layers),
                "depth_top_cm": np.tile([t for t, _ in dst_intervals], n_cells),
                "depth_bottom_cm": np.tile([b for _, b in dst_intervals], n_cells),
            }
        )
        for stem, cube in cubes.items():
            frame[stem] = cube[:, rows, cols].T.ravel()

        # Per-layer properties SoilGrids has no source for. The wide reference
        # carries constants for them, and a long solution declares them too
        # (SUSTAg reads both a residual and a reduction-point water content),
        # so they come across the same way rather than as a sentinel.
        for stem in ("soilwater_res", "soilwater_red"):
            values = self._template_layer_values(stem, n_layers)
            if values is not None:
                frame[stem] = np.tile(values, n_cells)

        frame = self._add_profile_scalars(frame)
        return frame

    def _template_layer_values(self, stem: str, n_layers: int) -> np.ndarray | None:
        """Per-layer constants for a property the pipeline cannot derive.

        Some columns a long solution declares (``soilwater_res``) have no
        SoilGrids source; the wide export carries the reference's own constants
        for them, and so does this. ``None`` when the reference has no such
        column, leaving the dialect to write the sentinel.
        """
        template = self._reference_template()
        values = [template.get(f"{stem}_{n}") for n in range(1, n_layers + 1)]
        if any(v is None for v in values):
            return None
        return np.asarray(values, dtype="float64")

    @staticmethod
    def _add_profile_scalars(frame: pd.DataFrame) -> pd.DataFrame:
        """Attach the profile-level ``*_global`` water contents.

        Both are thickness-weighted means over the whole profile, repeated on
        every row of their location -- which is exactly the ``DOUBLE``-typed
        resource entry a long solution declares next to its ``DOUBLEARRAY``
        columns.
        """
        thickness = frame["depth_bottom_cm"] - frame["depth_top_cm"]
        for stem, name in (
            ("soilwater_fc", "soilwater_fc_global"),
            ("soilwater_sat", "soilwater_sat_global"),
        ):
            if stem not in frame:
                continue
            weighted = (frame[stem] * thickness).groupby(frame["location"]).sum()
            total = thickness.groupby(frame["location"]).sum()
            frame[name] = frame["location"].map(weighted / total)
        return frame


class LongSoilExporter(SoilExporter):
    """Export the soil profile in the row-per-depth layout.

    Same computation as :class:`SoilExporter`, different serialisation: instead
    of pivoting the layers into ``<stem>_<N>`` columns it writes one row per
    ``(location, depth)`` and lets SIMPLACE assemble the arrays, which is what a
    solution declaring ``datatype="DOUBLEARRAY"`` expects.

    The dialect (column names, units, delimiter) is selected from
    ``reference.soil_file_long`` when one is configured, and defaults to the EU
    SUSTAg spelling otherwise. See :mod:`data4simplace.exporters.layout`.
    """

    kind = "soil (long)"

    def __init__(self, config: PipelineConfig, reference_path: str | Path | None) -> None:
        # The long reference, not the wide one, defines this file's structure.
        super().__init__(config, config.reference.soil_file_long)
        # ... but the wide reference still supplies the layer bottoms and the
        # constants for columns SoilGrids cannot derive, so it is kept.
        self._wide = SoilExporter(config, reference_path)
        self._dialect: LongDialect | None = None

    def fallback_spec(self) -> ReferenceSpec:
        """The selected dialect's own columns, when no long reference is given."""
        return ReferenceSpec(
            delimiter=self.dialect.delimiter,
            columns=self.dialect.column_names,
            missing_value=str(self._config.missing_value),
        )

    @property
    def dialect(self) -> LongDialect:
        """The long dialect this run writes."""
        if self._dialect is None:
            reference = self._resolve_reference_file()
            columns = (
                parse_reference_csv(
                    reference, default_missing=str(self._config.missing_value)
                ).columns
                if reference is not None
                else []
            )
            self._dialect = select_dialect(columns, SOIL_DIALECTS, kind="soil")
        return self._dialect

    def build_frame(
        self,
        soil: xr.Dataset,
        cell_table: pd.DataFrame,
        hydraulic: xr.Dataset | None = None,
    ) -> pd.DataFrame:
        """Build the long soil table (one row per location and depth)."""
        profile = self._wide.build_profile_table(
            soil, cell_table, hydraulic, depths=self._config.soil.long_depths
        )
        self._warn_about_unmapped(profile)

        frame = self.dialect.build(profile)
        # Drop locations with nothing derivable, matching the wide exporter's
        # rule that a written profile is one aggregated from real pixels.
        derived = [
            column.name
            for column in self.dialect.columns
            if column.source in profile.columns and column.source != "location"
        ]
        if derived:
            valid = ~frame[derived].apply(pd.to_numeric, errors="coerce").isna().all(axis=1)
            frame = frame[valid]
        frame = frame.reset_index(drop=True)
        logger.info(
            "Long soil table: %d locations x %d depths = %d rows (dialect %r, "
            "depths %s)",
            frame[self.dialect.key_column].nunique() if not frame.empty else 0,
            profile.groupby("location").size().max() if not profile.empty else 0,
            len(frame), self.dialect.name, self._config.soil.long_depths,
        )
        return frame

    def conform(self, frame: pd.DataFrame) -> pd.DataFrame:
        """Reference column order, with non-derivable columns kept runnable.

        A real long reference declares more than SoilGrids can supply: the EU
        SUSTAg file adds ``alfa``, ``n``, ``ksat``, ``macroporevolume``,
        ``dampingdepth``, ``drainage_rate``, ``RootingDepth`` and ``Soiltype``,
        and the ERA5 one adds the van Genuchten pair, ``DZF`` and
        ``slim_alpha``. The base class would fill those with the missing
        sentinel, which SIMPLACE cannot run on -- so, exactly as the wide
        exporter does with ``_reference_template``, they are carried over from
        the reference's own first row.
        """
        spec = self.spec
        if not spec.columns:
            return frame
        # The long reference's own first row, then the configured constants:
        # config wins, since it is the more deliberate of the two.
        template = {**self._long_template(), **self._config.soil.long_constants}
        out = frame.copy()
        unfilled = []
        for column in spec.columns:
            if column in out.columns:
                continue
            if column in template:
                out[column] = template[column]
            else:
                out[column] = spec.missing_value
                unfilled.append(column)
        if unfilled:
            logger.warning(
                "Long soil columns %s have no derivable source, no reference "
                "row and no soil.long_constants entry; written as %s, which "
                "SIMPLACE cannot run on",
                unfilled, spec.missing_value,
            )
        return out[spec.columns].fillna(spec.missing_value)

    def _long_template(self) -> dict[str, object]:
        """First row of the long reference, for the columns we cannot derive."""
        reference = self._resolve_reference_file()
        if reference is None:
            return {}
        row = pd.read_csv(reference, sep=self.spec.delimiter, nrows=1)
        if row.empty:
            return {}
        return {str(k): v for k, v in row.iloc[0].to_dict().items()}

    def _warn_about_unmapped(self, profile: pd.DataFrame) -> None:
        """Name every derived property the dialect has no column for.

        Silence here would be the dangerous case: a solution that reads
        ``silt`` from a dialect that does not write it fails inside the
        container, and a property dropped without a word is invisible.
        """
        carried = self.dialect.sources()
        skip = {"depth_top_cm", "depth_bottom_cm", "location"}
        unmapped = sorted(set(profile.columns) - carried - skip)
        if unmapped:
            logger.warning(
                "Long dialect %r has no column for %s; %s dropped rather than "
                "written under a guessed name",
                self.dialect.name, ", ".join(unmapped),
                "they are" if len(unmapped) > 1 else "it is",
            )

    def export(
        self,
        soil: xr.Dataset,
        cell_table: pd.DataFrame,
        output_dir: str | Path,
        hydraulic: xr.Dataset | None = None,
    ) -> Path:
        """Write ``soil/soil_long.csv``; return its path."""
        frame = self.build_frame(soil, cell_table, hydraulic)
        return self.write_csv(frame, Path(output_dir) / "soil" / "soil_long.csv")


class TopSoilExporter(SoilExporter):
    """Export one SIMPLACE soil file per primary soil class."""

    kind = "soil (per class)"

    def conform(self, frame: pd.DataFrame) -> pd.DataFrame:
        """Conform the SIMPLACE columns, then re-append the metadata columns.

        :meth:`BaseExporter.conform` drops anything outside the reference schema,
        which is what keeps ``soil.csv`` loadable; here the metadata columns have
        to survive, so they are set aside and appended after the reference block.
        """
        present = [col for col in METADATA_COLUMNS if col in frame.columns]
        if not present:
            return super().conform(frame)
        metadata = frame[present].reset_index(drop=True)
        conformed = super().conform(frame.drop(columns=present)).reset_index(drop=True)
        return pd.concat([conformed, metadata], axis=1)

    def build_rank_frame(
        self,
        top_classes: TopClassAggregation,
        rank: int,
        cell_table: pd.DataFrame,
        hydraulic: xr.Dataset | None = None,
    ) -> pd.DataFrame:
        """Build one rank's table: SIMPLACE columns plus the class metadata.

        Rows whose cell has no class at this rank (``soil_class_id == 0``) are
        dropped, as are the rows :meth:`SoilExporter.build_frame` already drops
        for having no derivable soil value.
        """
        properties = top_classes.properties.sel(rank=rank, drop=True)
        frame = super().build_frame(properties, cell_table, hydraulic)
        if frame.empty:
            return frame

        metadata = top_classes.metadata_frame(rank, cell_table)
        # ``location`` is the SimplaceID (see SoilExporter), which is what keys
        # the metadata back onto the surviving rows.
        frame = frame.merge(
            metadata, how="left", left_on="location", right_on="SimplaceID"
        )
        return frame[frame["soil_class_id"] > 0].reset_index(drop=True)

    def export(
        self,
        top_classes: TopClassAggregation,
        cell_table: pd.DataFrame,
        output_dir: str | Path,
        hydraulic: xr.Dataset | None = None,
    ) -> list[Path]:
        """Write ``soil_<rank>.csv`` for every rank; return the paths written.

        Parameters
        ----------
        top_classes:
            The per-class aggregation from the soil stage.
        cell_table:
            The exported cell table (already restricted to the exported cells).
        output_dir:
            Pipeline output directory; files land in its ``soil/`` subdirectory.
        hydraulic:
            Optional PTF fallback output. It is derived from the **dominant**
            class' pixels, so it is only applied to rank 1; the other ranks leave
            those columns at the reference constant rather than borrow rank 1's
            hydraulics.
        """
        if hydraulic is not None and top_classes.ranks.size > 1:
            logger.info(
                "PTF hydraulics are dominant-class values; applying them to "
                "soil_1.csv only"
            )

        written: list[Path] = []
        for rank in top_classes.ranks:
            rank = int(rank)
            frame = self.build_rank_frame(
                top_classes, rank, cell_table, hydraulic if rank == 1 else None
            )
            if frame.empty:
                logger.warning("No cells have a rank-%d soil class; skipping", rank)
                continue
            written.append(
                self.write_csv(frame, Path(output_dir) / "soil" / f"soil_{rank}.csv")
            )
        return written
