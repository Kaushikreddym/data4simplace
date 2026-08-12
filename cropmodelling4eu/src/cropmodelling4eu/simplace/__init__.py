"""Run SIMPLACE over the export, in its Singularity container.

SIMPLACE is a Java model shipped as a container, so a run is: build a workspace
it can read, generate the project CSV whose lines are the unit of work, invoke
the container over a line range, and collect the per-cell CSVs it writes.

    cm4eu simplace build   --config run.yaml [--out-dir DIR] [--cells 4]
    cm4eu simplace run     --config run.yaml --lines 1-4
    cm4eu simplace collect --config run.yaml

The build step validates the solution's ``<resource>`` headers against the
files the workspace actually holds, so a missing column fails in a second with
the column named rather than minutes into a job with a Java stack trace.

It also writes the run's **own** ``submit.sh`` and ``run_task.sh`` into the
built directory (see :mod:`.submit`), so the tree can be inspected by hand and
re-run — in whole or in part — without this package.

This subpackage imports no torch — the SIMPLACE side needs none of it.
"""

from cropmodelling4eu.simplace.collect import collect_run, read_yearly, to_run_schema
from cropmodelling4eu.simplace.project import (
    build_project_frame,
    line_ranges,
    write_project_csv,
)
from cropmodelling4eu.simplace.run import build_command, run_lines
from cropmodelling4eu.simplace.solution import (
    Resource,
    SolutionDocument,
    ValidationError,
    read_solution,
    validate_resources,
)
from cropmodelling4eu.simplace.submit import (
    SlurmResources,
    id_column_index,
    write_run_scripts,
)
from cropmodelling4eu.simplace.workspace import Workspace, build_workspace

__all__ = [
    "Resource",
    "SlurmResources",
    "SolutionDocument",
    "ValidationError",
    "Workspace",
    "build_command",
    "build_project_frame",
    "build_workspace",
    "collect_run",
    "id_column_index",
    "line_ranges",
    "read_solution",
    "read_yearly",
    "run_lines",
    "to_run_schema",
    "validate_resources",
    "write_project_csv",
    "write_run_scripts",
]
