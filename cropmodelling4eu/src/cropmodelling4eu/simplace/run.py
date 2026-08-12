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
import shlex
import subprocess
import time
from pathlib import Path

from cropmodelling4eu.config import RunConfig
from cropmodelling4eu.simplace.workspace import (
    CONTAINER_OUTDIR,
    Workspace,
)

logger = logging.getLogger(__name__)

__all__ = ["build_command", "run_lines"]


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
    image = Path(config.simplace.singularity_image)
    return [
        "singularity", "run",
        "-B", workspace.binds(log_dir=workspace.root / "log"),
        str(image),
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
    image = Path(config.simplace.singularity_image)
    if not image.is_file():
        raise FileNotFoundError(
            f"Singularity image not found: {image}. Set "
            f"simplace.singularity_image to one of the images on this cluster."
        )

    command = build_command(workspace, config, first_line, last_line, debug)
    logger.info("lines %d-%d: %s", first_line, last_line, shlex.join(command))
    if dry_run:
        return 0

    started = time.time()
    result = subprocess.run(command, check=False)
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
