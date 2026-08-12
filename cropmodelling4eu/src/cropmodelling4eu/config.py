"""Run configuration for ``cropmodelling4eu``.

One config drives both models, because the parts they share — which export to
read, which cells, which seasons — are the parts that decide whether their
results are comparable. Model-specific settings live in their own blocks.

The grid block is the piece that most needs to be configuration rather than
constants: ``SimplaceID`` is row-major over the export's own grid, so decoding
an id into a lon/lat needs the same bounds and resolution the export was written
with. :meth:`RunConfig.from_export` reads them back out of the export's frozen
``_work/config_run.yaml`` so a run cannot silently decode ids against the wrong
grid.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Literal, Optional

import yaml
from pydantic import BaseModel, ConfigDict, Field, model_validator

logger = logging.getLogger(__name__)

__all__ = [
    "GridConfig",
    "PathsConfig",
    "RunConfig",
    "SeasonConfig",
    "SimplaceConfig",
    "TorchcropConfig",
    "load_config",
]


class GridConfig(BaseModel):
    """The export's target grid, which defines ``SimplaceID``.

    Ids are 1-based and row-major from the **north-west** corner, so
    ``row = (id - 1) // n_lon`` and ``col = (id - 1) % n_lon``. Every one of
    those numbers depends on the grid, which is why it is configuration and not
    a module constant.
    """

    model_config = ConfigDict(extra="forbid")

    resolution_deg: float = Field(0.1, gt=0.0)
    min_lon: float = Field(-17.0, ge=-180.0, le=180.0)
    max_lon: float = Field(52.0, ge=-180.0, le=180.0)
    min_lat: float = Field(34.0, ge=-90.0, le=90.0)
    max_lat: float = Field(72.0, ge=-90.0, le=90.0)

    @model_validator(mode="after")
    def _check_bounds(self) -> "GridConfig":
        if self.min_lon >= self.max_lon:
            raise ValueError("grid.min_lon must be < grid.max_lon")
        if self.min_lat >= self.max_lat:
            raise ValueError("grid.min_lat must be < grid.max_lat")
        return self

    @property
    def n_lon(self) -> int:
        """Columns in the grid — the stride of the ``SimplaceID`` numbering."""
        return int(round((self.max_lon - self.min_lon) / self.resolution_deg))

    @property
    def n_lat(self) -> int:
        """Rows in the grid."""
        return int(round((self.max_lat - self.min_lat) / self.resolution_deg))


class PathsConfig(BaseModel):
    """Where the export is and where results go."""

    model_config = ConfigDict(extra="forbid")

    #: A finished data4simplace export: weather/, soil/, site/, management/.
    export_dir: Path
    #: Root for this package's own outputs; each run gets a sub-directory.
    output_dir: Path = Path("./runs")
    #: ``fertilizer_composition.xml``. Not part of the export -- it lives next
    #: to the SIMPLACE reference the exporter was built from.
    composition_file: Optional[Path] = None


class SeasonConfig(BaseModel):
    """Which harvest years to simulate.

    A harvest year names the season, not the calendar year the crop grows in: a
    winter crop sown in September of Y-1 is harvested in Y. The earliest
    runnable year is therefore one past the weather record's first year.
    """

    model_config = ConfigDict(extra="forbid")

    start_year: int = Field(2000, ge=1900, le=2200)
    end_year: int = Field(2024, ge=1900, le=2200)
    crop: str = "wheat"

    @model_validator(mode="after")
    def _check_range(self) -> "SeasonConfig":
        if self.start_year > self.end_year:
            raise ValueError("season.start_year must be <= season.end_year")
        return self

    @property
    def years(self) -> list[int]:
        return list(range(self.start_year, self.end_year + 1))


class TorchcropConfig(BaseModel):
    """LINTUL-5 run settings."""

    model_config = ConfigDict(extra="forbid")

    # 1 potential | 2 water-limited | 3 water+N | 4 water+NPK. 3 is the default
    # because no European soil P or K layer exists, so 4 would run on
    # torchcrop's built-in defaults rather than on data.
    iopt: Literal[1, 2, 3, 4] = 3
    #: Cells per model call. LINTUL-5 loops over days in Python, so per-cell
    #: cost collapses as the batch grows; the ceiling is memory, since
    #: ``ModelOutput`` retains every per-day state.
    batch_size: int = Field(2048, ge=1)
    #: Weather reads are gzip-bound and release the GIL, so threads scale.
    io_workers: int = Field(16, ge=1)
    #: The model works on ``[B]``-shaped tensors, too small to repay many
    #: intra-op threads; the cores are better spent on ``io_workers``.
    torch_threads: int = Field(4, ge=0)
    device: str = "cpu"
    rootzone_m: float = Field(1.0, gt=0.0)
    profile_bottom_m: float = Field(2.0, gt=0.0)
    #: Months of crop-free spin-up before sowing, to let the water balance relax
    #: out of the initial profile.
    spinup_months: int = Field(6, ge=0)
    #: Days of weather a window must still hold after sowing to be run.
    min_days_after_sowing: int = Field(365, ge=1)


class SimplaceConfig(BaseModel):
    """SIMPLACE (Singularity) run settings."""

    model_config = ConfigDict(extra="forbid")

    #: Directory holding the solution/, project/ and data/ of a working SIMPLACE
    #: project. Its XML is rewritten into the run workspace, never edited.
    template_dir: Optional[Path] = None
    #: Solution and project files inside ``template_dir``. None -> the only
    #: ``*.sol.xml`` / ``*.proj.xml`` found there.
    solution_file: Optional[Path] = None
    project_file: Optional[Path] = None
    #: The container. The cluster ships several versions side by side.
    singularity_image: Path = Path(
        "/beegfs/common/singularity/simplace/simplace_5.0-3897.sif"
    )
    #: Simulation window written into the solution's startdate/enddate.
    start_date: str = "1979-01-01"
    end_date: str = "2024-12-31"
    #: Value of the solution's ``vIOPT``.
    iopt: Literal[1, 2, 3, 4] = 3
    #: Lines of the project CSV per array task. 500 is measured, not guessed:
    #: a cell costs ~72 s over a 1979-2024 window, so 500 is a ~10 h task
    #: against a 12 h wall clock. Oversizing this is the one failure the run
    #: scripts handle badly -- a task killed by the wall clock leaves no
    #: completion stamp, so a retry restarts that range from its first line and
    #: the range never finishes. Scale it with the simulated window.
    lines_per_task: int = Field(500, ge=1)
    #: Threads used to convert the weather when ``weather_contract`` is set.
    #: Gzip and CSV parsing both release the GIL, and a continental run is
    #: otherwise an hour of wall time in the build step alone.
    io_workers: int = Field(16, ge=1)
    #: Which weather contract the solution reads: ``brandenburg`` (6 positional
    #: columns, radiation kJ/m2/d) or ``sustag_v2`` (comma separated, MJ/m2/d,
    #: wind km/day). None leaves the export alone, which is only right if its
    #: units already match the solution -- and for radiation they do not.
    weather_contract: Optional[Literal["brandenburg", "sustag_v2"]] = None
    #: How that contract is honoured. ``transform`` scales the column inside
    #: the solution's own SQL transformer and keeps the weather symlinked;
    #: ``files`` writes converted copies to a cache. Prefer ``transform``: it
    #: costs one token of SQL where ``files`` costs a second copy of the
    #: weather (~15 GB for Europe) and ~1.5 h to write. ``files`` remains for a
    #: solution with no transform to patch, or one whose contract also changes
    #: the file's shape (``sustag_v2`` reorders and adds columns).
    weather_conversion: Literal["transform", "files"] = "transform"
    #: The transform whose statement ``weather_conversion: transform`` patches.
    weather_transform_id: str = "weather_transform"
    #: Project-file columns the export has no opinion about, written verbatim.
    #: A solution keys its crop parameters, treatment and climate scenario off
    #: these (``vCrop``, ``vTrt``, ``vClimPerCO2_ID``, ``vENZ`` ...), so a wrong
    #: value selects the wrong parameter set rather than failing — which is why
    #: they are configuration rather than a guess.
    project_constants: dict[str, object] = Field(default_factory=dict)


class RunConfig(BaseModel):
    """A validated run configuration."""

    model_config = ConfigDict(extra="forbid")

    run_name: str = "winter_wheat"
    paths: PathsConfig
    grid: GridConfig = Field(default_factory=GridConfig)
    season: SeasonConfig = Field(default_factory=SeasonConfig)
    torchcrop: TorchcropConfig = Field(default_factory=TorchcropConfig)
    simplace: SimplaceConfig = Field(default_factory=SimplaceConfig)

    @property
    def run_dir(self) -> Path:
        """Where this run's outputs land."""
        return Path(self.paths.output_dir) / self.run_name

    @classmethod
    def from_export(cls, export_dir: str | Path, **overrides) -> "RunConfig":
        """Build a config whose grid is read back from the export itself.

        A tiled data4simplace run freezes its configuration to
        ``<export>/_work/config_run.yaml``. Reading the grid from there rather
        than from a default means a run cannot decode ``SimplaceID``s against a
        grid the export was not written with -- which would place every cell at
        the wrong coordinates and read the wrong weather file, silently.
        """
        export_dir = Path(export_dir)
        grid = _grid_from_export(export_dir)
        payload: dict = {"paths": {"export_dir": str(export_dir)}, **overrides}
        if grid is not None and "grid" not in payload:
            payload["grid"] = grid
        return cls.model_validate(payload)


def _grid_from_export(export_dir: Path) -> Optional[dict]:
    """The ``grid:`` block of an export's frozen run config, if it has one."""
    frozen = Path(export_dir) / "_work" / "config_run.yaml"
    if not frozen.is_file():
        logger.warning(
            "No frozen config at %s; falling back to the default Europe grid. "
            "Check that it matches the export, since SimplaceID decoding "
            "depends on it.", frozen,
        )
        return None
    with frozen.open("r", encoding="utf-8") as handle:
        raw = yaml.safe_load(handle) or {}
    grid = raw.get("grid")
    if not isinstance(grid, dict):
        return None
    # The export's grid block carries a crs the run config has no use for.
    keep = {"resolution_deg", "min_lon", "max_lon", "min_lat", "max_lat"}
    resolved = {k: v for k, v in grid.items() if k in keep}
    logger.info("Grid read from %s: %s", frozen, resolved)
    return resolved


def load_config(path: str | Path) -> RunConfig:
    """Read and validate a YAML run configuration.

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
        raise ValueError(
            f"Configuration root must be a mapping, got {type(raw).__name__}"
        )
    return RunConfig.model_validate(raw)
