"""Settings of the sowing pipeline, as data.

Every number here is a decision from the planning thread of 2026-11-12, recorded
where a run cannot change it silently:

* the cache opens on **DOY 193** of the year before harvest: the earliest SAGE
  ``plant.start`` in the export is DOY 213, and 20 days before it covers every
  rolling indicator the short-term branch needs. CyBench's ``sos`` cannot
  anchor it — for winter wheat it is spring green-up (DOY 29-86).
* the short-term branch sees at most ``short_receptive_days`` of weather, so it
  cannot memorise a calendar: any skill it shows is a response to weather.
* normals are drawn from autumns ``<= Y-2`` for harvest year Y. The autumn of
  Y-1 *is* the season being decided.
"""

from __future__ import annotations

from pathlib import Path

import yaml
from pydantic import BaseModel, ConfigDict, Field, model_validator

__all__ = ["LossWeights", "ModelConfig", "SowingConfig", "TrainConfig", "load_sowing_config"]

_DATA = Path("/data01/FDS/muduchuru/Data/SIMPLACE")


class ModelConfig(BaseModel):
    """Hazard-network sizes."""

    model_config = ConfigDict(extra="forbid")

    short_channels: int = Field(32, ge=4)
    #: Dilations of the causal TCN (kernel 3). Receptive field
    #: ``1 + 2 * sum(dilations)``; the default reaches 21 days.
    short_dilations: tuple[int, ...] = (1, 2, 4, 3)
    medium_channels: int = Field(16, ge=4)
    long_channels: int = Field(32, ge=4)
    dropout: float = Field(0.1, ge=0.0, lt=1.0)

    @property
    def short_receptive_days(self) -> int:
        return 1 + 2 * sum(self.short_dilations)


class LossWeights(BaseModel):
    """Term weights. CLMS emergence carries level, PEP725 carries the anomaly."""

    model_config = ConfigDict(extra="forbid")

    clms_emergence: float = Field(1.0, ge=0.0)
    pep_sowing: float = Field(1.0, ge=0.0)
    pep_emergence: float = Field(0.5, ge=0.0)
    #: Laplace scale [d] of the PEP725 sowing label: a farmer's date, filed by
    #: a volunteer, is not exact to the day.
    pep_sowing_scale_days: float = Field(3.0, gt=0.0)
    #: Initial Laplace scales [d]; both are learned.
    clms_scale_init_days: float = Field(10.0, gt=0.0)
    pep_emergence_scale_init_days: float = Field(5.0, gt=0.0)


class TrainConfig(BaseModel):
    """Optimiser and sampling."""

    model_config = ConfigDict(extra="forbid")

    batch_size: int = Field(512, ge=8)
    lr: float = Field(2e-3, gt=0.0)
    weight_decay: float = Field(1e-4, ge=0.0)
    max_epochs: int = Field(60, ge=1)
    patience: int = Field(8, ge=1)
    #: Samples per epoch, drawn with replacement balanced over the two sources.
    samples_per_epoch: int = Field(200_000, ge=1000)
    #: Share of each epoch drawn from PEP725 rows (they are ~10x fewer).
    pep_share: float = Field(0.3, ge=0.0, le=1.0)
    seed: int = 0
    device: str = "cpu"
    threads: int = Field(16, ge=1)


class SowingConfig(BaseModel):
    """A whole sowing run."""

    model_config = ConfigDict(extra="forbid")

    #: Export the weather, site and soil come from.
    export_dir: Path = _DATA / "cropmodelling4eu_inputs/exports/mswx/EU_val_nfix"
    #: Run config whose ``grid`` block defines SimplaceID.
    run_config: Path = _DATA / "cropmodelling4eu/runs/winter_wheat/calibrated/mswx/config.yaml"
    #: Calibration directory: ``cell_to_adm.parquet`` and ``phenology_split.parquet``.
    calibration_dir: Path = _DATA / "cropmodelling4eu/winter_wheat_2000_2024_euval/calibration_dm"
    #: Any one calibrated crop YAML, read for ``teffmx``. ``None`` takes region
    #: 01 of ``calibration_dir``'s yield stage.
    crop_file: Path | None = None
    pep725_root: Path = Path("/data01/FDS/muduchuru/Data/Agri/PEP725")
    #: Everything this pipeline writes.
    out_dir: Path = _DATA / "cropmodelling4eu/sowing_network"

    #: Harvest years cached and predicted. 1990 is the first with >= 10
    #: prior autumns in a 1979-2024 record.
    years: tuple[int, int] = (1990, 2024)
    cache_start_doy: int = Field(193, ge=1, le=300)
    cache_days: int = Field(240, ge=120, le=400)
    #: Last allowed sowing day-of-year (of Y-1); the hazard is forced to 1 on
    #: it, as ``vSowForceAtWindowEnd`` does. PEP725 sowing reaches DOY 358.
    last_sowing_doy: int = Field(365, ge=200, le=366)

    normals_years: int = Field(30, ge=5)
    normals_min_years: int = Field(10, ge=3)
    #: Topsoil depth [m] the bucket and the soil statics describe.
    topsoil_m: float = Field(0.3, gt=0.0)

    clms_years: tuple[int, int] = (2017, 2024)
    #: Subtracted from CLMS ``CPMCE`` (PEP725-measured product bias).
    clms_emergence_offset_days: float = 6.0
    #: PEP725 station blocks: block edge [deg] and the train/val/test shares.
    pep_block_deg: float = Field(1.0, gt=0.0)
    pep_split: tuple[float, float, float] = (0.7, 0.15, 0.15)

    #: Emergence thermal sum [C d]. ``None`` reads ``tsumem.json`` written by
    #: ``fit-tsumem``; training refuses to start without one or the other.
    tsumem: float | None = None
    tbasem: float = 0.0

    model: ModelConfig = Field(default_factory=ModelConfig)
    loss: LossWeights = Field(default_factory=LossWeights)
    train: TrainConfig = Field(default_factory=TrainConfig)

    workers: int = Field(32, ge=1)

    @model_validator(mode="after")
    def _check(self) -> "SowingConfig":
        if abs(sum(self.pep_split) - 1.0) > 1e-6:
            raise ValueError(f"pep_split must sum to 1, got {self.pep_split}")
        if self.sowing_days >= self.cache_days:
            raise ValueError("cache_days must extend past last_sowing_doy to date emergence")
        if self.model.short_receptive_days > 31:
            raise ValueError("the short-term branch must not see more than a month")
        return self

    @property
    def sowing_days(self) -> int:
        """Candidate sowing days, cache index ``0 .. sowing_days - 1``."""
        return self.last_sowing_doy - self.cache_start_doy + 1

    @property
    def crop_path(self) -> Path:
        return self.crop_file or self.calibration_dir / "yield/crops/crop_region_01.yaml"

    @property
    def cache_dir(self) -> Path:
        return self.out_dir / "cache"

    @property
    def tsumem_file(self) -> Path:
        return self.out_dir / "tsumem.json"

    @property
    def pep_split_file(self) -> Path:
        return self.out_dir / "pep725_split.parquet"

    @property
    def model_dir(self) -> Path:
        return self.out_dir / "model"


def load_sowing_config(path: str | Path | None = None, **overrides) -> SowingConfig:
    """Defaults, then the YAML at ``path``, then keyword overrides."""
    raw = yaml.safe_load(Path(path).read_text()) if path else {}
    return SowingConfig(**{**(raw or {}), **overrides})
