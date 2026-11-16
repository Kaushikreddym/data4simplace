"""Invoke SIMPLACE in its Singularity container over a range of project lines.

SIMPLACE is distributed as a container, not a Python package, so a run is a
subprocess:

    singularity run -B <binds> <image> /simplace/simplace run \\
        -s=<solution> -p=<project> -o=/outputs -t=CLUSTER -l=START-END

``-l`` is the parallelism axis: each task takes a contiguous range of the
project CSV's lines. That is the one structural difference from the torchcrop
side, where cells are dealt round-robin — SIMPLACE selects by range, so a
round-robin split is not expressible.
"""

from __future__ import annotations

import logging
import os
import shlex
import subprocess
import time
from pathlib import Path

from cropmodelling4eu.config import RunConfig, SimplaceConfig
from cropmodelling4eu.simplace.workspace import (
    CONTAINER_DATADIR,
    CONTAINER_OUTDIR,
    CONTAINER_PROJECTDIR,
    CONTAINER_WORKDIR,
    Workspace,
)

logger = logging.getLogger(__name__)

__all__ = ["build_command", "run_lines", "runtime_env", "runtime_image"]


def runtime_image(settings: SimplaceConfig) -> Path:
    """The container a run launches: the glibc Java image when one is set."""
    return Path(settings.java_image or settings.singularity_image)


def runtime_env(settings: SimplaceConfig) -> dict[str, str]:
    """Environment the glibc Java image needs to run the shipped SIMPLACE.

    The launcher honours INSTALL4J_JAVA_HOME_OVERRIDE; the rest is the shipped
    image's own environment, which the Java image does not carry.
    MALLOC_ARENA_MAX caps glibc's per-thread arenas, which musl never made.
    Empty for the shipped image, which needs nothing.
    """
    if settings.java_image is None:
        return {}
    return {
        "INSTALL4J_JAVA_HOME_OVERRIDE": "/opt/java/openjdk",
        "SIMPLACE_DIR": "/simplace",
        "SIMPLACE_WORKDIR": CONTAINER_WORKDIR,
        "OUTDIR": CONTAINER_OUTDIR,
        "DATADIR": CONTAINER_DATADIR,
        "PROJECTDIR": CONTAINER_PROJECTDIR,
        "LANG": "C.UTF-8",
        "LC_ALL": "C.UTF-8",
        "MALLOC_ARENA_MAX": "2",
    }


def build_command(
    workspace: Workspace,
    config: RunConfig,
    first_line: int,
    last_line: int,
    debug: bool = False,
) -> list[str]:
    """The ``singularity run`` argv for one line range.

    Returned as a list rather than a string so it is run without a shell —
    nothing here should be re-parsed, and the paths can contain anything.
    """
    settings = config.simplace
    binds = workspace.binds(log_dir=workspace.root / "log")
    if settings.simplace_dir is not None:
        binds = f"{settings.simplace_dir}:/simplace,{binds}"
    return [
        "singularity", "run",
        "-B", binds,
        str(runtime_image(settings)),
        "/simplace/simplace", "run",
        f"-s={workspace.container_solution}",
        f"-p={workspace.container_project}",
        f"-o={CONTAINER_OUTDIR}",
        "-t=CLUSTER",
        f"-l={first_line}-{last_line}",
        *([] if debug else ["-loglevel=ERROR"]),
    ]


def run_lines(
    workspace: Workspace,
    config: RunConfig,
    first_line: int,
    last_line: int,
    debug: bool = False,
    dry_run: bool = False,
) -> int:
    """Run one line range to completion; return the process exit code.

    Raises
    ------
    FileNotFoundError
        If the container image is absent. Checked before the call because
        ``singularity`` reports a missing image the same way it reports a
        missing binary, which sends people looking in the wrong place.
    """
    image = runtime_image(config.simplace)
    if not image.is_file():
        raise FileNotFoundError(
            f"Singularity image not found: {image}. Set simplace.java_image or "
            f"simplace.singularity_image to one of the images on this cluster."
        )

    command = build_command(workspace, config, first_line, last_line, debug)
    logger.info("lines %d-%d: %s", first_line, last_line, shlex.join(command))
    if dry_run:
        return 0

    started = time.time()
    result = subprocess.run(command, check=False, env={**os.environ, **runtime_env(config.simplace)})
    elapsed = (time.time() - started) / 60.0
    if result.returncode == 0:
        logger.info(
            "lines %d-%d finished in %.1f min; outputs under %s",
            first_line, last_line, elapsed, workspace.out_dir,
        )
    else:
        # No output is written on failure, so the range stays in the retry set.
        logger.error(
            "lines %d-%d exited %d after %.1f min",
            first_line, last_line, result.returncode, elapsed,
        )
    return result.returncode
