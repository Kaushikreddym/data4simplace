"""Settings for a calibration run, and the constants the plan fixes.

Everything here is a *decision* recorded as data — a number that came out of
``CALIBRATION.md`` and that a run must not silently change. The run
configuration proper (grid, export, seasons) stays
:class:`cropmodelling4eu.config.RunConfig`; this adds only what calibrating
needs on top of running.
"""

from __future__ import annotations

from pathlib import Path
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, model_validator

__all__ = [
    "CLMS_HARVEST_OFFSET_DAYS",
    "CalibrationConfig",
    "DataConfig",
    "LossConfig",
    "OptimConfig",
    "RegionConfig",
    "STAGE_PARAMETERS",
]

#: Days to **add** to a CLMS ``CPMCH`` harvest date to put it on the ground
#: scale. Measured against PEP725 (a person in a field) rather than against
#: CyBench ``eos``: identical on the 10 km and the NUTS support, unchanged
#: under 3x3 spatial smoothing, and constant year to year (-15.3 .. -23.0 d,
#: with the 2022 drought year at -16.5), so it is a product bias and not a
#: support artefact or a hot summer. Applied as a fixed correction so ``tsum2``
#: does not absorb it.
CLMS_HARVEST_OFFSET_DAYS: float = 18.7

#: Days to **subtract** from a CLMS ``CPMCE`` emergence date, same measurement.
#: An order of magnitude smaller than the harvest offset, which is why CLMS
#: emergence keeps its level term at full weight.
CLMS_EMERGENCE_OFFSET_DAYS: float = 6.0

#: A stage observed on or after this day-of-year belongs to the season harvested
#: the *following* calendar year. The same cut
#: :mod:`data4simplace.phenology.pep725` uses, and for the same reason: DOY 180
#: sits in the empty middle of the autumn stages' distribution.
AUTUMN_CUT_DOY: int = 180

#: Parameter-spec file shipped with each stage.
STAGE_PARAMETERS: dict[str, str] = {
    "phenology": "params_phenology.yaml",
    "yield": "params_yield.yaml",
    "lai": "params_lai.yaml",
}


class RegionConfig(BaseModel):
    """How the calibration regions are built.

    EnZ (13 zones) is the base rather than EnS (84 strata): 84 against the
    observation density is hopelessly over-parameterised. Each zone is then
    split on the crop calendar and the yield level, which is what separates a
    zone straddling a season break (ATC spans Ireland and SW France) and a zone
    mixing 3 t/ha and 9 t/ha districts — a region mixing those fits neither.
    """

    model_config = ConfigDict(extra="forbid")

    enz_shapefile: Path = Path("/data01/FDS/muduchuru/Data/Agri/EnSv8/data/enz_v8.shp")
    #: Target region count. The clustering picks ``k`` per zone by silhouette
    #: inside this band rather than being told a number.
    n_regions: tuple[int, int] = (15, 30)
    #: A region below either floor is merged back into its zone's largest.
    min_cells_per_region: int = Field(50, ge=1)
    min_observation_years_per_region: int = Field(3, ge=1)
    #: Maximum sub-clusters attempted inside one EnZ zone. 3 rather than 4:
    #: at 4 the silhouette picks enough of them that Europe comes out at 33
    #: regions, above the target band, and each extra region is another
    #: parameter set fitted on fewer observations.
    max_clusters_per_zone: int = Field(3, ge=1)
    seed: int = 0
    #: ``(source, target)`` region names folded together after clustering, as a
    #: deliberate override of the silhouette's split. Both names must belong to
    #: the **same** EnZ zone — a region is a subdivision of one, and a pair
    #: spanning two would break that.
    #:
    #: ``PAN-1`` into ``PAN-3``: the silhouette splits Pannonian three ways, but
    #: 1 and 3 differ by 0.7 t/ha and less than two days of calendar (sos 44.3 vs
    #: 44.8, eos 213.9 vs 212.1), which is inside what the yield-level layer is
    #: meant to separate; PAN-2 sits 1.1 t/ha above both. Merging them fits one
    #: parameter set on 60 units instead of two on 33 and 27.
    #:
    #: The merged region is **renamed** for its parts (``PAN-1+3``) so no
    #: surviving name silently changes meaning.
    merge_regions: tuple[tuple[str, str], ...] = (("PAN-1", "PAN-3"),)

    @model_validator(mode="after")
    def _check_merges(self) -> "RegionConfig":
        crossing = [
            (source, target)
            for source, target in self.merge_regions
            if source.rsplit("-", 1)[0] != target.rsplit("-", 1)[0]
        ]
        if crossing:
            raise ValueError(
                f"merge_regions pairs must share an EnZ zone; {crossing} do not"
            )
        return self


class DataConfig(BaseModel):
    """Which observations enter, and how the seasons are cached."""

    model_config = ConfigDict(extra="forbid")

    #: Days of weather kept before the **region's anchor** sowing day. The
    #: window is cut to anchor - ``pre_days`` .. + ``window_days`` rather than
    #: the production run's 600+, which is the cheapest of the three memory
    #: mitigations and the only one that is free.
    pre_days: int = Field(15, ge=0)
    #: Measured, not the plan's estimate. CALIBRATION.md sized this at ~330 d
    #: ("sowing - 15 to maturity + 30"); on the harmonised SUSTAg WW crop at
    #: `idsl = 2` that is far too short — anthesis alone lands ~295 d after
    #: sowing and DVS 2 is **never reached** inside 345 days, so a 330-day
    #: window silently trains on emergence and nothing else. It also has to
    #: cover the gap between the region's anchor sowing day and its latest
    #: sower, which is why the number is not simply the season length.
    window_days: int = Field(450, ge=60)
    #: ``(SimplaceID, year, sowing_doy)`` CSV to sow on instead of the export's
    #: site calendar — the ``sowing_from_simplace.csv`` a **finished** SIMPLACE
    #: run writes. It is a static file, read once; nothing here re-runs
    #: SIMPLACE.
    #:
    #: This is not a refinement, it is the sowing convention the production run
    #: and every number in CALIBRATION.md are on. The two differ by a median of
    #: **22 days** (SIMPLACE's rule-based date is earlier), which moves the
    #: emergence bias by the same amount. And the site calendar publishes *one*
    #: date per cell for every season, so simulated emergence would carry no
    #: interannual variation at all — which is precisely the anomaly stage 1
    #: exists to fit.
    #:
    #: A (cell, season) the file does not cover is **dropped**, not fallen back
    #: on, so one pool never mixes two sowing conventions.
    sowing_file: Path | None = None
    #: Memory-mapped season cache. ``None`` puts it under the run's output
    #: directory. Size it deliberately: 8 000 cells x 25 years x 330 d x 8 vars
    #: x 4 B is 210 GB, so the pool must be restricted to cell-years that carry
    #: an observation and the cache built per pool, not per domain.
    cache_dir: Path | None = None
    #: Cap on cells simulated per observation unit. A NUTS-2 unit can hold
    #: hundreds; the unit mean converges long before that and the cost is
    #: linear.
    max_cells_per_unit: int = Field(12, ge=1)
    #: Fraction of the pool drawn per epoch, stratified by region and year.
    fraction: float = Field(0.20, gt=0.0, le=1.0)
    #: Cell-years per forward pass. Not 2048: ``ModelOutput`` retains every
    #: per-day state and there is no gradient checkpointing upstream.
    batch_size: int = Field(192, ge=1)
    #: Segment length for the checkpointed forward; 0 disables checkpointing.
    #: ``sqrt(pre_days + window_days)`` is the memory optimum, so ~21 at the
    #: default window.
    checkpoint_segment: int = Field(21, ge=0)
    seed: int = 0


class LossConfig(BaseModel):
    """Term weights, set by what each reference can actually measure.

    The defaults are not uniform and are not a style choice: CLMS emergence
    scatters 1.7x more than the ground observation and correlates with it at
    r 0.11, so an emergence-anomaly term against CLMS fits noise.
    """

    model_config = ConfigDict(extra="forbid")

    emergence_level: float = 1.0
    emergence_anomaly_clms: float = Field(0.05, ge=0.0)
    emergence_anomaly_pep: float = 1.0
    harvest_level: float = 1.0
    harvest_anomaly: float = 1.0
    anthesis_level: float = 1.0
    #: Penalty on maturity falling *after* the observed harvest. Asymmetric:
    #: a crop cannot be harvested before it matures, but maturing well before
    #: harvest is ordinary drydown.
    maturity_after_harvest: float = 2.0
    #: Days of maturity-before-harvest tolerated before the mild side bites.
    drydown_days: float = 14.0
    yield_level: float = 1.0
    yield_anomaly: float = 1.0
    #: Huber transition for the yield anomaly term [t/ha].
    yield_huber_delta: float = 1.0
    canopy_level: float = 1.0
    canopy_senescence: float = 1.0
    #: Retention weights when a stage carries an earlier stage's loss forward.
    #: Stage 3 lowers LAI, which lowers yield, so ``lambda_yield`` is sized for
    #: a canopy being *cut* rather than raised.
    lambda_phenology: float = 0.5
    lambda_yield: float = 1.0
    #: L2 pull toward the physical prior, in units of the prior's own spread.
    prior_l2: float = Field(0.05, ge=0.0)
    #: L2 shrinkage of a region's deviation toward the global latent.
    pooling_l2: float = Field(0.1, ge=0.0)


class OptimConfig(BaseModel):
    """Adam, upstream's settings.

    ``betas=(0.8, 0.95)`` is documented upstream as what stops ``tsum1`` and
    ``tsum2`` making compensating excursions, since both move maturity. It is
    the same conditioning problem the identifiability rule meets with a prior,
    and both instruments are used.
    """

    model_config = ConfigDict(extra="forbid")

    lr: float = Field(0.12, gt=0.0)
    betas: tuple[float, float] = (0.8, 0.95)
    max_epochs: int = Field(80, ge=1)
    #: Epochs without a validation improvement before the stage stops.
    patience: int = Field(12, ge=1)
    grad_clip: float = Field(1.0, ge=0.0)
    pooling: Literal["none", "hierarchical"] = "hierarchical"


class CalibrationConfig(BaseModel):
    """A whole calibration run."""

    model_config = ConfigDict(extra="forbid")

    #: Stages in the order they run. They **accumulate** terms rather than
    #: replacing the objective, so stage 3 cannot silently undo stage 2.
    #:
    #: ``lai`` is switched OFF. Its fAPAR senescence term is broken on both
    #: sides: ``sigmoid_fall_day`` at k=1 returns ~T/2 regardless of timing (a
    #: +38 d season shift moved it -5 d), and ``AUTUMN_CUT_DOY = 180`` groups
    #: the June-July decline into the *next* season, so observed half-fall
    #: dates land in Nov-Jan. Run 829246 showed the cost: stage 3 dragged test
    #: yield bias from +0.01 to -3.24 t/ha. Restore
    #: ``("phenology", "yield", "lai")`` once both are fixed.
    stage_order: tuple[str, ...] = ("phenology", "yield")
    #: Joint fine-tune at a small learning rate after the staged fits. Its
    #: reason was stage 3 removing leaf area stage 2 fits yield through, so it
    #: is off while ``lai`` is; turn both back on together.
    joint_finetune: bool = False
    joint_lr_scale: float = Field(0.1, gt=0.0)

    regions: RegionConfig = Field(default_factory=RegionConfig)
    data: DataConfig = Field(default_factory=DataConfig)
    loss: LossConfig = Field(default_factory=LossConfig)
    optim: OptimConfig = Field(default_factory=OptimConfig)

    #: Inclusive harvest-year range the phenology term can use. CLMS covers
    #: 2017-2024 only, so its temporal split is leave-one-year-out, not the
    #: 70/15/15 the yield term gets over 25 years.
    phenology_years: tuple[int, int] = (2017, 2024)
    #: Where checkpoints, logs and the calibrated crop file are written.
    #: ``None`` puts them under ``<output_dir>/<run_name>/calibration``.
    out_dir: Path | None = None

    @model_validator(mode="after")
    def _check_stages(self) -> "CalibrationConfig":
        unknown = set(self.stage_order) - set(STAGE_PARAMETERS)
        if unknown:
            raise ValueError(
                f"unknown stage(s) {sorted(unknown)}; known: {sorted(STAGE_PARAMETERS)}"
            )
        return self
