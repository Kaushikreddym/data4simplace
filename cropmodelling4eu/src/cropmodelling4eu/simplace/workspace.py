"""Materialise a SIMPLACE run workspace from a template and an export.

SIMPLACE runs inside a Singularity container against a directory tree it calls
``SIMPLACE_WORK``. This builds one per run:

    <run_dir>/workspace/
    ├── solution/<name>.sol.xml     from the template, variables overridden
    ├── project/<name>.proj.xml     from the template, filename repointed
    ├── project/project_<crop>.csv  generated from the export's cells
    └── data/                       template XML + the export's CSVs
    <run_dir>/weather/<row>/…       the weather tree the solution expects

Three decisions shape it:

**The template is never edited.** Its XML is parsed, adjusted in memory and
written into the workspace, so a run cannot corrupt the project it was built
from and two runs can differ only in their workspace.

**Nothing is copied — every data file is a symlink.** The workspace holds only
what this package *generates*: the two rewritten XML files, the project CSV and
``location.csv``. The export's tables, the weather and the template's static
model parameters (crop, SLIM, soil C/N/P, ``management.xml``,
``fertilizer_composition.xml``) are all linked, so a built workspace is a few
hundred kilobytes and can be inspected, deleted and rebuilt freely. The cost is
that editing the template changes every workspace built from it; ``build.json``
records the template's path and the build time so a surprise is at least
traceable.

**The weather tree is linked by row.** The solution reads
``${_DATADIR_}/${vRow}/daily_mean_RES1_C${vColumn}R${vRow}.csv.gz`` — nested one
directory per grid row. data4simplace now writes that shape directly, so the
links mirror the export rather than reshaping it; an export written before that
change is flat in ``weather/`` and the same code still nests it. Either way only
the project's own cells are linked, which costs inodes but no bytes.

**A unit mismatch is reconciled in the solution, not in the data.** The
export's radiation is in W m-2 where the solution declares kJ m-2 d-1, and the
default ``weather_conversion: transform`` scales the column inside the
solution's own SQL transformer, so the weather stays symlinked. Only
``weather_conversion: files`` writes converted copies, to a shared cache
outside the workspace that the workspace then links into — so even then it
holds no copies of its own.
"""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

import numpy as np

from cropmodelling4eu.config import GridConfig, RunConfig, SimplaceConfig
from cropmodelling4eu.export import ExportBundle
from cropmodelling4eu.export.cells import id_to_rowcol, weather_path
from cropmodelling4eu.simplace.project import (
    build_location_frame,
    build_project_frame,
    project_columns,
    write_project_csv,
)
from cropmodelling4eu.simplace.weather import (
    BRANDENBURG,
    SUSTAG_V2,
    WeatherConversion,
    build_weather_cache,
    cached_weather_path,
    declared_factors,
)
from cropmodelling4eu.simplace.solution import (
    SolutionDocument,
    read_solution,
    validate_resources,
)

logger = logging.getLogger(__name__)

__all__ = ["Workspace", "build_workspace", "WEATHER_CONTRACTS"]

#: Sub-directories of the template's ``data/`` linked verbatim: the model
#: parameters the export has no opinion about.
_LINKED_DATA_DIRS = ("crop", "slim", "soilcnp", "soil_cn")

#: Files of ``data/management/`` taken from the template rather than the export.
_LINKED_MANAGEMENT = ("management.xml", "fertilizer_composition.xml")

#: ``simplace.weather_contract`` -> the conversion it names.
WEATHER_CONTRACTS: dict[str, WeatherConversion] = {
    "brandenburg": BRANDENBURG,
    "sustag_v2": SUSTAG_V2,
}

#: Container-internal mount points, matching ``sbatch_simplace.sh``.
CONTAINER_WORKDIR = "/simplace/SIMPLACE_WORK"
CONTAINER_DATADIR = "/data"
CONTAINER_OUTDIR = "/outputs"
CONTAINER_PROJECTDIR = "/projects"


@dataclass(slots=True)
class Workspace:
    """A built run workspace and everything a job needs to run it.

    Attributes
    ----------
    root:
        The run directory.
    work_dir:
        The ``SIMPLACE_WORK`` tree, bound to ``/simplace/SIMPLACE_WORK``.
    data_dir:
        The weather tree, bound to ``/data``.
    out_dir:
        Where SIMPLACE writes, bound to ``/outputs``.
    solution, project:
        Paths **inside** ``work_dir``, as ``simplace run -s=/-p=`` wants them.
    n_lines:
        Lines in the project CSV — the run's total work.
    project_csv:
        The generated per-cell table. Its lines are what ``-l=START-END``
        selects, so a retry script reads it to map a line range back to cells.
    weather_cache:
        Where the converted weather lives when a contract is in force, and
        ``None`` when the export is linked unchanged. It must be bound, since
        the workspace only links into it.
    """

    root: Path
    work_dir: Path
    data_dir: Path
    out_dir: Path
    solution: Path
    project: Path
    n_lines: int
    export_dir: Path
    project_csv: Optional[Path] = None
    weather_cache: Optional[Path] = None
    template_dir: Optional[Path] = None

    @property
    def container_solution(self) -> str:
        """``-s=`` argument: the solution's path inside the container."""
        return f"{CONTAINER_WORKDIR}/{self.solution.relative_to(self.work_dir)}"

    @property
    def container_project(self) -> str:
        """``-p=`` argument: the project's path inside the container."""
        return f"{CONTAINER_WORKDIR}/{self.project.relative_to(self.work_dir)}"

    def binds(self, log_dir: Path | None = None) -> str:
        """The ``singularity -B`` bind list for this workspace.

        The **export is bound read-only at its own absolute path**, which is
        not decoration. The workspace links to the export rather than copying
        70 000 weather files and a 70 000-row ``soil.csv``, and a symlink is
        resolved *inside* the container: without the export visible at the path
        its targets name, every link dangles and SIMPLACE fails with a
        ``NullPointerException`` on a file whose path it just printed.
        """
        pairs = [
            f"{self.work_dir}:{CONTAINER_WORKDIR}",
            f"{self.out_dir}:{CONTAINER_OUTDIR}",
            f"{self.data_dir}:{CONTAINER_DATADIR}",
            f"{self.root / 'projects'}:{CONTAINER_PROJECTDIR}",
            f"{self.export_dir}:{self.export_dir}:ro",
        ]
        if self.template_dir is not None:
            # The template's crop and SLIM parameters are linked too, so its
            # tree has to resolve inside the container for the same reason.
            pairs.append(f"{self.template_dir}:{self.template_dir}:ro")
        if self.weather_cache is not None:
            pairs.append(f"{self.weather_cache}:{self.weather_cache}:ro")
        if log_dir is not None:
            pairs.append(f"{log_dir}:/simplace/log")
        return ",".join(pairs)


def _resolve_template(config: RunConfig) -> tuple[Path, Path, Path]:
    """``(template_dir, solution, project)`` from the config or by discovery."""
    settings = config.simplace
    if settings.template_dir is None:
        raise ValueError(
            "simplace.template_dir must point at a SIMPLACE project holding "
            "solution/, project/ and data/"
        )
    template = Path(settings.template_dir)
    if not template.is_dir():
        raise FileNotFoundError(f"simplace.template_dir is not a directory: {template}")

    solution = _pick(template, settings.solution_file, "solution", "*.sol.xml")
    project = _pick(template, settings.project_file, "project", "*.proj.xml")
    return template, solution, project


def _pick(template: Path, configured: Optional[Path], sub: str, pattern: str) -> Path:
    """A configured file, or the single match under ``template/<sub>``."""
    if configured is not None:
        path = Path(configured)
        if not path.is_absolute():
            path = template / path
        if not path.is_file():
            raise FileNotFoundError(f"Configured {sub} file not found: {path}")
        return path

    matches = sorted(
        p for p in (template / sub).glob(pattern) if not p.name.startswith(".")
    )
    if not matches:
        raise FileNotFoundError(f"No {pattern} under {template / sub}")
    if len(matches) > 1:
        # A project folder often keeps a "copy" alongside the live file; picking
        # silently would run the wrong science.
        raise ValueError(
            f"{template / sub} holds several {pattern} files "
            f"({', '.join(p.name for p in matches)}); set simplace.{sub}_file"
        )
    return matches[0]


def _link_weather(
    export_dir: Path,
    data_dir: Path,
    ids: np.ndarray,
    grid: GridConfig,
    conversion: WeatherConversion | None = None,
    cache_dir: Path | None = None,
) -> int:
    """Recreate the ``<row>/<file>`` weather tree the solution reads.

    Sources from the export, or from the converted cache when one is in force —
    the tree's shape is the same either way, so the solution's filename pattern
    does not depend on which. Only the project's own cells are linked; a capped
    smoke test would otherwise create ~70 000 symlinks to run four.
    """
    cells = np.asarray(ids)
    rows, _ = id_to_rowcol(cells, grid)
    sources = (
        weather_path(export_dir, int(cell), grid)
        if cache_dir is None or conversion is None
        else cached_weather_path(cache_dir, int(cell), grid, conversion)
        for cell in cells
    )
    linked = 0
    for source, row in zip(sources, rows):
        if source.is_file():
            _link(source, data_dir / str(int(row)) / source.name)
            linked += 1
    logger.info("Weather: %d symlinks under %s (nested by grid row)", linked, data_dir)
    return linked


def _populate_data(
    template: Path, bundle: ExportBundle, work_data: Path, crop: str
) -> None:
    """Link the template's static inputs and the export's tables.

    Everything here is a symlink, including the template's own XML: the
    workspace is a *view* onto inputs that live elsewhere, not a copy of them.
    """
    template_data = template / "data"
    management = work_data / "management"
    management.mkdir(parents=True, exist_ok=True)

    for name in _LINKED_DATA_DIRS:
        if (source := template_data / name).is_dir():
            _link_tree(source, work_data / name)
    for name in _LINKED_MANAGEMENT:
        if (source := template_data / "management" / name).is_file():
            _link(source, management / name)

    _link(bundle.export_dir / "soil" / "soil.csv", work_data / "soil" / "soil.csv")
    if (site := bundle.export_dir / "site" / "site.csv").is_file():
        _link(site, work_data / "site" / "site.csv")

    schedule = next(
        (
            p
            for p in sorted((bundle.export_dir / "management").glob(f"fertilizer_*{crop}*.csv"))
            if not p.stem.endswith("_long")
        ),
        None,
    )
    if schedule is None:
        logger.warning(
            "No fertilizer schedule for %r in the export; the solution's "
            "management interface will not resolve", crop,
        )
    else:
        _link(schedule, management / schedule.name)


def _link_tree(source_dir: Path, target_dir: Path) -> int:
    """Mirror ``source_dir``'s structure with one symlink per file.

    Per file rather than one link to the directory: a solution reads individual
    files out of ``data/crop/``, and a linked *directory* would silently pick up
    anything added to the template afterwards.
    """
    files = [p for p in sorted(source_dir.rglob("*")) if p.is_file()]
    for source in files:
        _link(source, target_dir / source.relative_to(source_dir))
    return len(files)


def _link(source: Path, target: Path) -> None:
    """Symlink ``source`` to ``target``, replacing an existing link."""
    if not source.is_file():
        logger.warning("Input file missing, not linked: %s", source)
        return
    target.parent.mkdir(parents=True, exist_ok=True)
    if target.is_symlink() or target.exists():
        target.unlink()
    target.symlink_to(source.resolve())


def build_workspace(
    config: RunConfig,
    bundle: ExportBundle,
    max_cells: int = 0,
    validate: bool = True,
    strict: bool = False,
    cells: np.ndarray | None = None,
    root: Path | None = None,
    weather_cache: Path | None = None,
) -> Workspace:
    """Build the run workspace and the project CSV; validate before returning.

    Parameters
    ----------
    root:
        Where the tree is created. ``None`` uses ``<output_dir>/<run_name>/
        simplace``. Passing it explicitly is what ``cm4eu simplace build
        --out-dir`` does, so a workspace can be built somewhere inspectable
        without moving the run's results.
    weather_cache:
        Where converted weather is kept when ``simplace.weather_contract`` is
        set. ``None`` uses ``<output_dir>/weather_cache/<contract>`` — outside
        the run directory on purpose, so runs and retries share it.
    max_cells:
        Cap the project file, for a smoke test. An **even subsample**, not the
        head: ``SimplaceID`` runs north-to-south, so truncating would give the
        test a narrow Scandinavian band.
    cells:
        Run exactly these cells instead. What makes a model comparison
        like-for-like: both models must get the same cell set, not two
        samples of one.
    validate:
        Check every resource against the files the workspace now holds. On by
        default, because the alternative is a Java stack trace minutes into a
        job.
    strict:
        Treat a declared-but-absent column as fatal too. Off by default —
        SIMPLACE tolerates some, and the Brandenburg solution's own working
        data has two such mismatches.

    Raises
    ------
    ValidationError
        If a resource's file is missing, or (in strict mode) if any declared
        column is absent.
    """
    template, solution_src, project_src = _resolve_template(config)
    settings = config.simplace

    root = Path(root) if root is not None else config.run_dir / "simplace"
    work_dir = root / "workspace"
    data_dir = root / "weather"
    out_dir = root / "out"
    # `state/` holds one stamp per line range that exited zero, which is what
    # the generated submit.sh derives its retry set from.
    for directory in (
        work_dir, data_dir, out_dir,
        root / "projects", root / "log", root / "state",
    ):
        directory.mkdir(parents=True, exist_ok=True)

    ids = bundle.ids
    if cells is not None:
        ids = np.intersect1d(np.asarray(cells, dtype=np.int64), ids)
        if ids.size == 0:
            raise ValueError("none of the requested cells is runnable")
        logger.info("Project restricted to %d requested cells", ids.size)
    if max_cells and ids.size > max_cells:
        ids = ids[:: int(np.ceil(ids.size / max_cells))][:max_cells]
        logger.info("Project capped to %d cells (even subsample)", ids.size)

    _populate_data(template, bundle, work_dir / "data", config.season.crop)

    conversion = WEATHER_CONTRACTS.get(settings.weather_contract or "")
    convert_files = conversion is not None and settings.weather_conversion == "files"
    if convert_files:
        weather_cache = Path(
            weather_cache
            or Path(config.paths.output_dir) / "weather_cache" / conversion.name
        )
        build_weather_cache(
            bundle.export_dir, weather_cache, ids, config.grid, conversion,
            workers=settings.io_workers,
        )
    else:
        weather_cache = None
    _link_weather(
        bundle.export_dir, data_dir, ids, config.grid,
        conversion if convert_files else None, weather_cache,
    )

    # --- project CSV ------------------------------------------------------
    project_doc = read_solution(project_src)
    columns = project_columns(project_doc)
    frame = build_project_frame(
        ids,
        config.grid,
        columns,
        altitude=bundle.site.altitude_m(ids),
        sowing_doy=bundle.site.sowing_doy(ids),
        sowing_window=bundle.site.sowing_window(ids),
        irrigated=bundle.irrigated,
        constants=settings.project_constants,
    )
    csv_name = f"project_{config.season.crop}.csv"
    project_csv = write_project_csv(frame, work_dir / "project" / csv_name)

    # location.csv is generated, never copied: it carries the per-cell latitude
    # the solution reads daily, and the template's copy holds Brandenburg's.
    write_project_csv(
        build_location_frame(ids, config.grid, bundle.site.altitude_m(ids)),
        work_dir / "data" / "management" / "location.csv",
    )

    # --- XML --------------------------------------------------------------
    project_out = _write_project_xml(project_doc, work_dir, csv_name)
    solution_out = _write_solution_xml(
        read_solution(solution_src), work_dir, config,
        None if convert_files else conversion,
    )

    workspace = Workspace(
        root=root,
        work_dir=work_dir,
        data_dir=data_dir,
        out_dir=out_dir,
        solution=solution_out,
        project=project_out,
        n_lines=len(frame),
        export_dir=bundle.export_dir,
        project_csv=project_csv,
        weather_cache=weather_cache,
        template_dir=template,
    )

    if validate:
        # Resources are validated against the workspace on *this* filesystem,
        # so _WORKDIR_ expands to the host path rather than the container's.
        _, warnings = validate_resources(
            read_solution(solution_out),
            {"_WORKDIR_": str(work_dir), "_DATADIR_": str(data_dir)},
            strict=strict,
        )
        logger.info(
            "Solution validated against the workspace: every file present, "
            "%d column mismatch(es)", len(warnings),
        )

    _write_manifest(workspace, config, conversion)
    logger.info(
        "Workspace built: %s (%d lines, solution %s)",
        work_dir, workspace.n_lines, solution_out.name,
    )
    return workspace


def _write_manifest(
    workspace: Workspace, config: RunConfig, conversion: WeatherConversion | None
) -> Path:
    """Record what this tree was built from, as ``<root>/build.json``.

    Every data file in the workspace is a link, so the tree alone does not say
    which template, export or contract produced it — and a link silently
    follows its target if that changes.
    """
    settings = config.simplace
    payload = {
        "built": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "run_name": config.run_name,
        "crop": config.season.crop,
        "export_dir": str(workspace.export_dir),
        "template_dir": str(workspace.template_dir),
        "solution": str(workspace.solution),
        "project": str(workspace.project),
        "project_csv": str(workspace.project_csv),
        "n_lines": workspace.n_lines,
        "lines_per_task": settings.lines_per_task,
        "singularity_image": str(settings.singularity_image),
        "start_date": settings.start_date,
        "end_date": settings.end_date,
        "iopt": settings.iopt,
        "weather_contract": conversion.name if conversion else None,
        "weather_conversion": settings.weather_conversion if conversion else None,
        "weather_cache": str(workspace.weather_cache or "") or None,
    }
    path = workspace.root / "build.json"
    path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    return path


def _write_project_xml(
    document: SolutionDocument, work_dir: Path, csv_name: str
) -> Path:
    """Repoint the project's data interface at the generated CSV."""
    for interface in document.root.findall(".//interfaces/interface"):
        filename = interface.find("filename")
        if filename is None or not (filename.text or "").endswith(".csv"):
            continue
        filename.text = f"${{_WORKDIR_}}/project/{csv_name}"
    return document.write(work_dir / "project" / document.path.name)


def _write_solution_xml(
    document: SolutionDocument,
    work_dir: Path,
    config: RunConfig,
    conversion: WeatherConversion | None = None,
) -> Path:
    """Override the run's window and mode, then write into the workspace."""
    settings = config.simplace
    overrides = {
        "startdate": _to_simplace_date(settings.start_date),
        "enddate": _to_simplace_date(settings.end_date),
        "vIOPT": settings.iopt,
    }
    changed = document.set_variables(overrides)
    logger.info(
        "Solution variables overridden: %s", ", ".join(changed) if changed else "(none)"
    )
    if conversion is not None:
        _scale_weather_transform(document, settings, conversion)
    return document.write(work_dir / "solution" / document.path.name)


def _scale_weather_transform(
    document: SolutionDocument, settings: SimplaceConfig, conversion: WeatherConversion
) -> None:
    """Apply the contract's unit factors inside the solution's SQL transform."""
    weather = next((r for r in document.resources if r.id == "weather"), None)
    if weather is None:
        raise ValueError(
            f"{document.path.name} declares no 'weather' resource, so "
            f"weather_conversion: transform has nothing to scale"
        )
    if not (factors := declared_factors(list(weather.columns), conversion)):
        logger.info("Contract %r needs no unit change", conversion.name)
        return
    document.scale_transform(settings.weather_transform_id, factors)
    logger.info(
        "Weather units scaled inside %s: %s (the export stays symlinked)",
        settings.weather_transform_id,
        ", ".join(f"{k} x {v:g}" for k, v in factors.items()),
    )


def _to_simplace_date(value: str) -> str:
    """``YYYY-MM-DD`` -> SIMPLACE's ``DD.MM.YYYY``.

    A date written in the wrong order is not rejected by SIMPLACE — it is
    parsed into a different day — so the conversion is explicit rather than
    left to whatever the config happens to hold.
    """
    text = str(value).strip()
    if "." in text:
        return text  # already SIMPLACE's order
    year, month, day = text.split("-")
    return f"{int(day):02d}.{int(month):02d}.{year}"
