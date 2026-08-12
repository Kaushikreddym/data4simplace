"""Configuration parsing and validation for the ``data4simplace`` pipeline.

The public entry point is :func:`load_config`, which reads a YAML file and
returns a validated :class:`PipelineConfig`. All downstream modules receive
this immutable, type-checked object rather than raw dictionaries.
"""

from __future__ import annotations

from pathlib import Path
from typing import Literal, Optional

import yaml
from pydantic import BaseModel, ConfigDict, Field, model_validator


class ExecutionFlags(BaseModel):
    """Boolean switches controlling which pipeline stages run."""

    model_config = ConfigDict(extra="forbid")

    run_climate_processing: bool = False
    run_soil_processing: bool = False
    compute_ptf: bool = False
    run_npk_processing: bool = False
    # Per-cell sowing calendar, altitude and the CO2 series -- the inputs both
    # SIMPLACE and torchcrop need that no other stage supplies. See the site block.
    run_site_processing: bool = False
    # Classify every target cell irrigated (1) / rainfed (0) from the crop's
    # irrigated and rainfed harvested area, and add the vIRR column to the
    # SIMPLACE management file. See the irrigation block.
    run_irrigation_classification: bool = False
    apply_agricultural_mask: bool = False
    # Intermediate NetCDF/CSV statistics for the n most frequent soil classes
    # per cell (see soil.n_primary_classes); the SIMPLACE CSVs are unaffected.
    write_soil_statistics: bool = False
    export_simplace_weather: bool = False
    export_simplace_soil: bool = False
    export_simplace_management: bool = False
    # site/site.csv (one row per cell: lat/lon, altitude, sowing calendar) and
    # site/co2.csv (the annual global CO2 series).
    export_simplace_site: bool = False
    # One SIMPLACE soil CSV per primary class (soil_1.csv .. soil_n.csv), each
    # aggregated over its own 250 m pixels and carrying the class' area and the
    # cell's heterogeneity metrics. Requires soil.aggregation_method: top3.
    export_top3_soil_csvs: bool = False


class GridConfig(BaseModel):
    """Definition of the unified target grid in geographic coordinates."""

    model_config = ConfigDict(extra="forbid")

    resolution_deg: float = Field(0.1, gt=0.0, description="Cell size in degrees (~10 km).")
    min_lon: float = Field(..., ge=-180.0, le=180.0)
    max_lon: float = Field(..., ge=-180.0, le=180.0)
    min_lat: float = Field(..., ge=-90.0, le=90.0)
    max_lat: float = Field(..., ge=-90.0, le=90.0)
    crs: str = "EPSG:4326"

    @model_validator(mode="after")
    def _check_bounds(self) -> "GridConfig":
        if self.min_lon >= self.max_lon:
            raise ValueError("grid.min_lon must be < grid.max_lon")
        if self.min_lat >= self.max_lat:
            raise ValueError("grid.min_lat must be < grid.max_lat")
        return self


class TimeConfig(BaseModel):
    """Temporal window for climate processing (ISO ``YYYY-MM-DD``)."""

    model_config = ConfigDict(extra="forbid")

    start: str
    end: str


class PathsConfig(BaseModel):
    """Input data locations and the output directory."""

    model_config = ConfigDict(extra="forbid")

    mswx_root: Path
    soilgrids_root: Optional[Path] = None
    npk_root: Optional[Path] = None
    # PROBA-V LC100 Crops-CoverFraction GeoTIFF: the pipeline's single cropland
    # definition. Filters SoilGrids 250 m pixels before the dominant-soil-type
    # aggregation, and selects the 10 km cells that get exported
    # (flags.apply_agricultural_mask). Unset -> no cropland filtering at all.
    cropland_weights_path: Optional[Path] = None
    # MIRCA-OS Monthly Growing Area Grids: either the year folder itself or a
    # parent holding one folder per year. Needed by the irrigation stage when
    # irrigation.source is ``mirca`` or ``merged``.
    mirca_root: Optional[Path] = None
    # ECIRA base directory, i.e. the parent of Crop_IR / Crop_A / Crop_RF.
    # Needed when irrigation.source is ``ecira`` or ``merged``.
    ecira_root: Optional[Path] = None
    # Directory of gridded crop calendars, read by the site stage. Which product
    # is expected there follows site.calendar_source: ``sage`` looks for
    # ``<crop>.crop.calendar.fill.nc``, ``ggcmi`` for the phase-3 files.
    calendar_root: Optional[Path] = None
    # Terrain DEM for the per-cell altitude, e.g.
    # ``GMTED2010_15n015_00625deg.nc``. Must be *terrain*: a geoid/EGM96
    # conversion grid is rejected by name (see site/elevation.py).
    dem_path: Optional[Path] = None
    # Annual global CO2 series as ``year,ppm``. Unset -> the built-in
    # global-mean table, and the written file records that it is a fallback.
    co2_file: Optional[Path] = None
    output_dir: Path = Path("./output")


class ReferenceConfig(BaseModel):
    """Locations of the SIMPLACE reference files (output structure source)."""

    model_config = ConfigDict(extra="forbid")

    weather_dir: Optional[Path] = None
    soil_dir: Optional[Path] = None
    management_file: Optional[Path] = None
    # Long-layout references (export.layout: long | both). The dialect -- column
    # names, units and delimiter -- is selected from these files' columns; unset
    # falls back to the built-in EU SUSTAg spelling.
    soil_file_long: Optional[Path] = None
    management_file_long: Optional[Path] = None


class ExportConfig(BaseModel):
    """Which layout(s) the soil and management files are written in.

    ``wide`` is the Brandenburg reference schema: one row per location, with the
    depth axis in the column names (``clay_1`` ... ``clay_6``). ``long`` is the
    row-per-depth schema the EU SUSTAg and ERA5 solutions read, where every
    property is declared ``datatype="DOUBLEARRAY"`` and SIMPLACE assembles the
    arrays from the rows sharing a key.

    ``both`` writes each file twice. They are two serialisations of one
    computation, so the second costs almost nothing and lets a wide-driven and a
    long-driven SIMPLACE run be compared on identical inputs.
    """

    model_config = ConfigDict(extra="forbid")

    layout: Literal["wide", "long", "both"] = "wide"
    # Per-file overrides. None -> whatever `layout` says.
    soil_layout: Optional[Literal["wide", "long", "both"]] = None
    management_layout: Optional[Literal["wide", "long", "both"]] = None

    def resolved(self, kind: Literal["soil", "management"]) -> str:
        """The layout in force for one product."""
        override = self.soil_layout if kind == "soil" else self.management_layout
        return override or self.layout

    def writes(self, kind: Literal["soil", "management"], layout: str) -> bool:
        """Whether ``kind`` is written in ``layout`` ("wide" or "long")."""
        resolved = self.resolved(kind)
        return resolved == layout or resolved == "both"


class ClimateConfig(BaseModel):
    """MSWX variable mapping and dask chunk sizes."""

    model_config = ConfigDict(extra="forbid")

    variables: dict[str, str] = Field(default_factory=dict)
    chunks: dict[str, int] = Field(default_factory=lambda: {"time": 30, "lat": 512, "lon": 512})
    # Worker processes for the per-file MSWX reads. Processes, not threads: the
    # HDF5 library serialises reads on a global lock, so a thread pool measures
    # *slower* than serial. None -> SLURM_CPUS_PER_TASK (capped at 16, past
    # which the reads stop scaling and only add open() contention).
    read_workers: int | None = None


class NPKConfig(BaseModel):
    """Fertilizer input source and the SIMPLACE schedule it is written into."""

    model_config = ConfigDict(extra="forbid")

    # ``npkgrids``  - one crop-specific NPKGRIDS v1.08 netCDF per crop, holding
    #                 N / P2O5 / K2O application rates in kg/ha at 0.05 deg.
    # ``rasters``   - the generic per-nutrient GeoTIFF/netCDF discovery, kept for
    #                 other gridded fertilizer products.
    source: Literal["npkgrids", "rasters"] = "npkgrids"

    # --- NPKGRIDS selection ---------------------------------------------------
    # Crop suffix of the NPKGRIDS filename, e.g. ``wheat`` for
    # ``NPKGRIDSv1.08_wheat.nc``. Matched on the stem, so the version is free.
    crop: str = "wheat"
    # NPKGRIDS marks ocean with -1 (always dropped) and land the crop is not
    # grown on with 0. Averaging those zeros into a 10 km cell would dilute the
    # rate of the pixels that do grow the crop, so they are excluded by default
    # and a cell with no positive pixel is left out of the export entirely.
    include_zero_rate: bool = False
    # Minimum NPKGRIDS quality score (1 = highest, 0 = lowest) a source pixel
    # needs to contribute. None -> no quality filtering.
    min_quality: Optional[float] = Field(None, ge=0.0, le=1.0)
    # Carry the per-nutrient quality scores through to the regridded dataset.
    keep_quality: bool = False

    # --- SIMPLACE schedule ----------------------------------------------------
    # Value written into the schedule's ``crop`` column. SIMPLACE's crop names
    # are finer than NPKGRIDS' (winter vs. spring wheat share one raster), so
    # this is deliberately independent of ``crop``.
    simplace_crop: str = "winter_wheat"
    # ``FertilizerScenario`` column value. None -> take it from the reference.
    fertilizer_scenario: Optional[int] = None
    # Carrier the cell's N rate is applied as. Its mineral-N content is read
    # from fertilizer_composition.xml, so the product amount follows the
    # carrier: 1 g N needs 3.70 g KAS (27 % N) but only 2.17 g Urea (46 %).
    n_fertilizer: str = "KAS"
    # The reference schedule applies a compound PK whose P:K ratio is fixed,
    # which cannot honour two independent NPKGRIDS rates. The compound event is
    # therefore replaced by these two straight carriers, applied at the same DVS.
    p_fertilizer: str = "P"
    k_fertilizer: str = "K"
    # Relative split of the cell's N rate across the reference's N events. None
    # -> derived from the reference amounts and their carriers' N contents, so
    # the reference's own 50/25/25 timing pattern is preserved.
    n_split: Optional[list[float]] = None
    # ``fertilizer_composition.xml``. None -> the copy next to the reference
    # management CSV.
    composition_file: Optional[Path] = None
    # Decimal places for the written ``Amount`` (g product / m^2).
    amount_decimals: int = Field(3, ge=0, le=6)
    # What ``Amount`` means in the long layout (export.layout: long | both).
    # The SUSTAg long dialect carries no ``vType``, so there is no carrier to
    # divide by and the amount is the **nutrient** in g/m^2. Writing product
    # grams into a nutrient field is a silent factor-of-3.7 error for KAS, so
    # ``product`` is refused on a dialect with no fertilizer-type column.
    long_amount_basis: Literal["nutrient", "product"] = "nutrient"

    @model_validator(mode="after")
    def _check_split(self) -> "NPKConfig":
        if self.n_split is not None:
            if not self.n_split:
                raise ValueError("npk.n_split must not be an empty list")
            if any(share < 0 for share in self.n_split):
                raise ValueError("npk.n_split entries must be >= 0")
            if sum(self.n_split) <= 0:
                raise ValueError("npk.n_split must sum to more than 0")
        return self


class IrrigationConfig(BaseModel):
    """Irrigated / rainfed classification of the target cells.

    A cell is irrigated when the irrigated share of its crop harvested area,
    ``A_ir / (A_ir + A_rf)``, exceeds :attr:`threshold`. The label is written to
    the management file as the ``vIRR`` column.
    """

    model_config = ConfigDict(extra="forbid")

    # ``mirca``  - MIRCA-OS v0.1 Monthly Growing Area Grids (5 arcmin, global)
    # ``ecira``  - ECIRA v2.0 Crop_IR / Crop_A (1 km, EU/EEA only)
    # ``merged`` - ECIRA where it classifies the cell, MIRCA-OS elsewhere.
    #              ECIRA leads because MIRCA-OS inherits national statistics that
    #              report zero irrigated cereals for whole countries; MIRCA-OS is
    #              the only source outside ECIRA's EU/EEA footprint.
    source: Literal["merged", "mirca", "ecira"] = "merged"
    # Irrigated share above which a cell counts as irrigated.
    threshold: float = Field(0.5, ge=0.0, le=1.0)
    # A cell holding less of the crop than this is left unclassified (and so is
    # written as 0), rather than labelled off a rounding difference.
    min_crop_area_ha: float = Field(10.0, ge=0.0)
    # Data year of both products. MIRCA-OS v0.1 offers 2000/2005/2010/2015;
    # ECIRA v2.0 covers 2010-2020.
    year: int = Field(2015, ge=1900, le=2100)
    # Crop group both products are aggregated to. None -> derived from
    # npk.simplace_crop, which maps winter_wheat to ``cereals`` because ECIRA
    # publishes no wheat class (its CERE is cereals excluding maize and rice).
    crop_group: Optional[Literal["maize", "wheat", "cereals"]] = None
    # Name of the column added to the management CSV.
    column: str = "vIRR"
    # Write the gridded classification next to the schedule as a NetCDF.
    write_netcdf: bool = True


class SiteConfig(BaseModel):
    """Per-cell sowing calendar, altitude and CO2 (the site stage).

    These three are the inputs both crop models need and no other stage
    supplies, so before this stage each runner substituted its own constant --
    one sowing day-of-year for the whole continent, an altitude of zero and a
    hard-coded CO2 table.
    """

    model_config = ConfigDict(extra="forbid")

    # ``sage``  - Sacks et al. (2010) crop calendar: a 0.5 degree climatology of
    #             planting/harvest dates with a start/end window, assembled from
    #             census and extension reports. About a third of European cells
    #             are extrapolated from a neighbouring reporting unit, which the
    #             product flags and the export carries as ``calendar_filled``.
    # ``ggcmi`` - GGCMI phase 3 calendar: planting and maturity day, built for
    #             gridded crop models. The alternative when the SAGE dates are
    #             also the reference an evaluation compares the run against.
    calendar_source: Literal["sage", "ggcmi"] = "sage"
    # Crop name in the calendar product (SAGE ``Wheat.Winter``, GGCMI ``wwh``).
    # None -> derived from npk.simplace_crop, which has no default for crops the
    # mapping does not know: guessing would silently sow the wrong calendar.
    calendar_crop: Optional[str] = None
    # Sowing day-of-year for cells no calendar date could be sampled or filled
    # for. Written with ``calendar_source: fallback`` so an assumed date stays
    # distinguishable from a sampled one in every downstream analysis.
    fallback_sowing_doy: int = Field(270, ge=1, le=366)
    # Copy the nearest covered cell's calendar into cells the product does not
    # cover (coastal and fringe cropland). Bounded by the exported cell mask, so
    # the search never carries a land calendar out to sea.
    fill_calendar_gaps: bool = True
    # Terrain variable of paths.dem_path. None -> the first of elevation /
    # surface_altitude / altitude / ... that the file carries.
    dem_variable: Optional[str] = None

    # --- The planting window written into the fertilizer schedule ------------
    # A rule-based solution sows on the first day inside [start, end] on which a
    # weather rule holds, so it reads a window rather than a date. Written onto
    # every event row of the cell, like vIRR, because the schedule is the only
    # per-cell table such a solution already reads daily.
    write_management_window: bool = True
    window_start_column: str = "vSowWindowStartDOY"
    window_end_column: str = "vSowWindowEndDOY"
    # Bounds on the window's length. A zero-length window turns a rule back into
    # a fixed date; one longer than max_days lets the deadline sit so far out
    # that the forced-sowing guard stops being a guard. The min doubles as the
    # half-width of the window centred on the sowing date where the calendar
    # product publishes no start/end pair.
    window_min_days: int = Field(7, ge=1, le=182)
    window_max_days: int = Field(120, ge=1, le=364)


class SoilConfig(BaseModel):
    """SoilGrids layer, depth, projection and WCS-fetch configuration."""

    model_config = ConfigDict(extra="forbid")

    # ``wv0010``/``wv0033``/``wv1500`` are the SoilGrids volumetric water
    # contents at 10/33/1500 kPa; they fill the SIMPLACE soilwater_* block
    # directly (see exporters/soil_export.py) instead of the Saxton-Rawls PTF.
    layers: list[str] = Field(
        default_factory=lambda: [
            "clay", "silt", "sand", "bdod", "soc", "phh2o", "nitrogen",
            "wv0010", "wv0033", "wv1500",
        ]
    )
    depths: list[str] = Field(
        default_factory=lambda: [
            "0-5cm",
            "5-15cm",
            "15-30cm",
            "30-60cm",
            "60-100cm",
            "100-200cm",
        ]
    )
    homolosine_crs: str = "EPSG:152160"
    target_crs: str = "EPSG:4326"

    # Dominant soil-type aggregation (see CLAUDE.md workflow). ``none`` keeps the
    # legacy plain-mean regrid; the others filter 250 m pixels to cropland, pick
    # the dominant class per target cell and aggregate only those pixels:
    #   usda          - 12 USDA texture classes of the topsoil layer
    #   usda_profile  - 12 x 12 composite (topsoil, rooting-zone) texture classes
    #   wrb           - SoilGrids WRB Reference Soil Group
    dominant_mode: Literal["usda", "usda_profile", "wrb", "none"] = "usda"
    # How many classes per cell are aggregated:
    #   dominant - Method A: one profile per cell, from the dominant class only
    #   top3     - Method B: the ``n_primary_classes`` most frequent classes, each
    #              aggregated over its own pixels, with per-class areas and
    #              cell-level heterogeneity metrics. Rank 1 is identical to
    #              Method A, so soil.csv is unchanged either way.
    aggregation_method: Literal["dominant", "top3"] = "dominant"
    # Bottom of the rooting zone (cm) whose thickness-weighted texture gives the
    # second key of the ``usda_profile`` composite class; its top is the bottom
    # of the topsoil layer (5 cm for the standard SoilGrids depths).
    rootzone_bottom_cm: float = Field(100.0, gt=0.0)
    # Minimum PROBA-V cropland cover fraction (0-1) for a 100/250 m pixel to be
    # kept, both when filtering the soil pixels and when deciding which 10 km
    # cells count as cropland.
    cropland_min_fraction: float = Field(0.8, ge=0.0, le=1.0)
    # How many qualifying native PROBA-V pixels a target cell needs to be
    # exported when ``flags.apply_agricultural_mask`` is set. At 100 m there are
    # ~10,000 pixels per 10 km cell, so 1 keeps any cell with a trace of
    # cropland; raise it to drop marginal cells.
    min_cropland_pixels: int = Field(1, ge=1)
    # How many classes per cell to describe when flags.write_soil_statistics is
    # set: rank 1 is the dominant (exported) class, the rest quantify what the
    # single-profile export leaves out.
    n_primary_classes: int = Field(3, ge=1, le=12)
    # Statistic the SIMPLACE CSVs carry: ``mean`` applies the variable-specific
    # mean rules (arithmetic / geometric / pH H+), ``median`` the plain median.
    export_statistic: Literal["mean", "median"] = "mean"
    # Depth axis of the long-layout soil file (export.layout: long | both).
    #   native   - SoilGrids' own horizons, with no depth remap at all. The long
    #              format puts depth in the rows, so it has no fixed layer count
    #              to force them onto; this is the whole point of the layout.
    #   simplace - remap onto the wide reference's layer bottoms, so the long and
    #              wide files describe the same layering and compare directly.
    long_depths: Literal["native", "simplace"] = "native"
    # Per-column constants for long-layout columns SoilGrids cannot derive
    # (van Genuchten ``alfa``/``n``, ``ksat``, ``macroporevolume``,
    # ``dampingdepth``, ``drainage_rate``, ``deltatheta``, ``maxRootingDepth``,
    # ``Soiltype`` ...). A long reference's own first row is used where one is
    # configured; this fills the rest. Left empty they are written as the
    # missing sentinel, which SIMPLACE cannot run on -- so a solution declaring
    # them needs an entry here.
    long_constants: dict[str, object] = Field(default_factory=dict)

    # --- Initial mineral N from the SoilGrids total-N layer -------------------
    # SoilGrids ``nitrogen`` is *total* (largely organic) N, while SIMPLACE's
    # ammonium_*/nitrate_* columns want initial *mineral* N. The exporter turns
    # total N into a per-layer stock (kg N/ha) and takes this fraction of it;
    # 1 % is the usual order of magnitude for arable topsoils. Set 0 to leave the
    # columns at the reference constants.
    mineral_n_fraction: float = Field(0.01, ge=0.0, le=1.0)
    # Share of that mineral N written as ammonium; the remainder becomes nitrate.
    ammonium_share: float = Field(0.3, ge=0.0, le=1.0)
    # WRB Reference Soil Group coverage id (SoilGrids ``wrb`` map) for ``wrb`` mode.
    wrb_layer: str = "MostProbable"
    # Fill target cells left empty (coastal/islands) from the nearest valid cell.
    # Off by default: a filled cell carries a *borrowed* profile, and because the
    # exported cell set is the cells that have soil, turning this on both
    # substitutes neighbours' values and widens the export to those cells.
    fill_missing: bool = False

    # WCS fallback: fetch coverages from the ISRIC service when a local tile is
    # not found under ``paths.soilgrids_root``.
    use_wcs: bool = True
    wcs_stat: str = "mean"
    wcs_crs: str = "EPSG:4326"  # request CRS; 4326 fetches directly in lon/lat
    wcs_cache_dir: Optional[Path] = None
    wcs_timeout: int = 180


class PipelineConfig(BaseModel):
    """Fully validated configuration for a pipeline run."""

    model_config = ConfigDict(extra="forbid")

    flags: ExecutionFlags
    grid: GridConfig
    time: TimeConfig
    paths: PathsConfig
    reference: ReferenceConfig = Field(default_factory=ReferenceConfig)
    export: ExportConfig = Field(default_factory=ExportConfig)
    climate: ClimateConfig = Field(default_factory=ClimateConfig)
    soil: SoilConfig = Field(default_factory=SoilConfig)
    site: SiteConfig = Field(default_factory=SiteConfig)
    npk: NPKConfig = Field(default_factory=NPKConfig)
    irrigation: IrrigationConfig = Field(default_factory=IrrigationConfig)
    missing_value: float = -99.0

    @model_validator(mode="after")
    def _check_multiclass(self) -> "PipelineConfig":
        """The per-class CSVs need the per-class aggregation to have run."""
        if self.flags.export_top3_soil_csvs:
            if self.soil.aggregation_method != "top3":
                raise ValueError(
                    "flags.export_top3_soil_csvs requires "
                    "soil.aggregation_method: top3 (the per-class profiles are "
                    "only aggregated in that mode)"
                )
            if not self.flags.run_soil_processing:
                raise ValueError(
                    "flags.export_top3_soil_csvs requires flags.run_soil_processing"
                )
        if self.soil.aggregation_method == "top3" and self.soil.dominant_mode == "none":
            raise ValueError(
                "soil.aggregation_method: top3 needs a soil classification; "
                "set soil.dominant_mode to usda, usda_profile or wrb"
            )
        return self

    @model_validator(mode="after")
    def _check_management(self) -> "PipelineConfig":
        """The fertilizer schedule needs its NPK stage and a data root."""
        if self.flags.export_simplace_management:
            if not self.flags.run_npk_processing:
                raise ValueError(
                    "flags.export_simplace_management requires "
                    "flags.run_npk_processing (the schedule amounts come from "
                    "the aligned NPK rates)"
                )
            if self.npk.source == "npkgrids" and self.paths.npk_root is None:
                raise ValueError(
                    "npk.source: npkgrids requires paths.npk_root to point at "
                    "the NPKGRIDS netCDF directory"
                )
        return self

    @model_validator(mode="after")
    def _check_site(self) -> "PipelineConfig":
        """The site stage needs both of its inputs; the export needs the stage."""
        if self.flags.export_simplace_site and not self.flags.run_site_processing:
            raise ValueError(
                "flags.export_simplace_site requires flags.run_site_processing "
                "(site.csv is written from that stage's calendar and altitude)"
            )
        if not self.flags.run_site_processing:
            return self

        missing = [
            name
            for name, value in (
                ("calendar_root", self.paths.calendar_root),
                ("dem_path", self.paths.dem_path),
            )
            if value is None
        ]
        if missing:
            raise ValueError(
                "flags.run_site_processing requires "
                + " and ".join(f"paths.{name}" for name in missing)
                + ". paths.co2_file stays optional -- without it the built-in "
                "global-mean CO2 table is written and labelled a fallback."
            )
        return self

    @model_validator(mode="after")
    def _check_irrigation(self) -> "PipelineConfig":
        """The irrigation stage needs the roots of whichever source it reads."""
        if not self.flags.run_irrigation_classification:
            return self

        needed = {
            "mirca": (("mirca_root", self.paths.mirca_root),),
            "ecira": (("ecira_root", self.paths.ecira_root),),
            "merged": (
                ("mirca_root", self.paths.mirca_root),
                ("ecira_root", self.paths.ecira_root),
            ),
        }[self.irrigation.source]
        missing = [name for name, value in needed if value is None]
        if missing:
            raise ValueError(
                f"irrigation.source: {self.irrigation.source} requires "
                + " and ".join(f"paths.{name}" for name in missing)
                + (
                    " (merged reads ECIRA inside its EU/EEA footprint and "
                    "MIRCA-OS outside it)"
                    if self.irrigation.source == "merged"
                    else ""
                )
            )
        return self


def load_config(path: str | Path) -> PipelineConfig:
    """Read and validate a YAML configuration file.

    Parameters
    ----------
    path:
        Path to the YAML configuration file.

    Returns
    -------
    PipelineConfig
        A validated, immutable configuration object.

    Raises
    ------
    FileNotFoundError
        If ``path`` does not exist.
    ValueError
        If the YAML is malformed or fails schema validation.
    """
    cfg_path = Path(path)
    if not cfg_path.is_file():
        raise FileNotFoundError(f"Configuration file not found: {cfg_path}")

    with cfg_path.open("r", encoding="utf-8") as handle:
        raw = yaml.safe_load(handle)

    if not isinstance(raw, dict):
        raise ValueError(f"Configuration root must be a mapping, got {type(raw).__name__}")

    return PipelineConfig.model_validate(raw)
