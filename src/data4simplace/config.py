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
    output_dir: Path = Path("./output")


class ReferenceConfig(BaseModel):
    """Locations of the SIMPLACE reference files (output structure source)."""

    model_config = ConfigDict(extra="forbid")

    weather_dir: Optional[Path] = None
    soil_dir: Optional[Path] = None
    management_file: Optional[Path] = None


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
    climate: ClimateConfig = Field(default_factory=ClimateConfig)
    soil: SoilConfig = Field(default_factory=SoilConfig)
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
