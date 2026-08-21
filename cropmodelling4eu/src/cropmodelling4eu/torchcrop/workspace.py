"""A working directory for a torchcrop run: what it will use, on disk, first.

SIMPLACE has had one all along — ``cm4eu simplace build`` writes a tree whose
XML, project CSV and linked inputs *are* the run, so a question about what it
did is answered by opening a file. torchcrop had nothing equivalent: its crop
parameters lived inside the installed ``torchcrop`` package, its sowing dates
inside the export, and a finished run left no record of either. "Which crop did
this run use?" could only be answered by re-deriving it.

This module writes that record **before** the array is submitted, on the login
node, so a mistake in it costs seconds rather than showing up in every task:

===========================  ==================================================
``crop_<crop>.yaml``         The crop the run will use, as a complete torchcrop
                             preset. Loaded back with
                             ``CropParameters(config_file=...)``, so it is the
                             input and not a description of one — edit it and
                             the next run uses the edit.
``crop_parameter_audit.csv`` Every parameter beside SIMPLACE's, with a verdict,
                             when a SIMPLACE crop file is available to compare
                             against.
``config_run.yaml``          The resolved ``RunConfig``, so the grid, seasons
                             and export the run decoded against are recorded
                             next to its outputs.
===========================  ==================================================

The crop file is written whichever source it comes from. With
``crop_source="torchcrop"`` — the default, and what the published Europe runs
use — it is the bundled preset written out unchanged, which changes no result
and makes the parameters checkable by opening them. With
``crop_source="simplace"`` it is built from the solution's own ``crop.xml``
(plus the ``NRF``/``PRF``/``KRF`` recovery fractions of ``management.xml``),
which is what makes a like-for-like comparison against SIMPLACE a comparison of
the models rather than of two crop files.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from pathlib import Path

import yaml

from cropmodelling4eu.config import RunConfig
from cropmodelling4eu.torchcrop import params as crop_params

logger = logging.getLogger(__name__)

__all__ = ["TorchcropWorkspace", "load_crop_parameters", "prepare_workspace"]

#: Sub-directory of the run's output directory. Named for what it is: the run's
#: inputs, not its results, which land beside it.
WORKSPACE_DIR = "workspace"

AUDIT_FILE = "crop_parameter_audit.csv"
CONFIG_FILE = "config_run.yaml"


@dataclass(frozen=True, slots=True)
class TorchcropWorkspace:
    """A prepared working directory and the files a run reads out of it."""

    root: Path
    crop_file: Path
    config_file: Path
    audit_file: Path | None
    crop_source: str

    def summarise(self) -> str:
        lines = [
            f"Workspace : {self.root}",
            f"  crop    : {self.crop_file.name}  (from {self.crop_source})",
            f"  config  : {self.config_file.name}",
        ]
        if self.audit_file is not None:
            lines.append(f"  audit   : {self.audit_file.name}")
        return "\n".join(lines)


def crop_file_name(crop: str) -> str:
    """``crop_<crop>.yaml`` — named for the crop, not for where it came from.

    A run has one crop, and the file is *it*; the source is recorded inside the
    file (``source_file``, ``provenance``) rather than in its name, so swapping
    the source does not change what a script has to open.
    """
    return f"crop_{crop}.yaml"


def prepare_workspace(
    config: RunConfig,
    out_dir: Path,
    crop_source: str = "torchcrop",
    crop_xml: Path | None = None,
    seeds_xml: Path | None = None,
    management_xml: Path | None = None,
    overwrite: bool = True,
) -> TorchcropWorkspace:
    """Write the run's inputs to ``<out_dir>/workspace/`` and return their paths.

    ``crop_source="simplace"`` requires ``crop_xml``; the SIMPLACE template's
    own paths are the sensible values, and
    :func:`cropmodelling4eu.torchcrop.params.compare_crop_parameters` is run
    whenever ``crop_xml`` is given, whichever source is selected — the audit is
    worth having precisely when the run is *not* using SIMPLACE's crop.
    """
    if crop_source not in ("torchcrop", "simplace"):
        raise ValueError(f"crop_source must be 'torchcrop' or 'simplace', not {crop_source!r}")
    if crop_source == "simplace" and crop_xml is None:
        raise ValueError("crop_source='simplace' needs crop_xml")

    root = Path(out_dir) / WORKSPACE_DIR
    root.mkdir(parents=True, exist_ok=True)
    crop = config.season.crop
    crop_path = root / crop_file_name(crop)

    if crop_path.exists() and not overwrite:
        logger.info("keeping the existing %s", crop_path)
    else:
        crop_params.write_crop_yaml(
            crop_path,
            simplace_crop_xml=crop_xml if crop_source == "simplace" else None,
            seeds_xml=seeds_xml if crop_source == "simplace" else None,
            crop_name=crop,
            management_xml=management_xml if crop_source == "simplace" else None,
        )

    audit_path: Path | None = None
    if crop_xml is not None and Path(crop_xml).is_file():
        comparison = crop_params.compare_crop_parameters(
            crop_xml, crop_name=crop, management_xml=management_xml
        )
        audit_path = root / AUDIT_FILE
        comparison.to_csv(audit_path, index=False)
        differing = int((comparison["status"] != "same").sum())
        if crop_source == "torchcrop" and differing:
            logger.warning(
                "this run uses torchcrop's own crop parameters, %d of which "
                "differ from SIMPLACE's -- see %s. That is a valid choice, but "
                "a SIMPLACE comparison then differs by the crop as well as the "
                "model", differing, audit_path,
            )

    config_path = root / CONFIG_FILE
    with config_path.open("w") as handle:
        yaml.safe_dump(config.model_dump(mode="json"), handle, sort_keys=False)

    workspace = TorchcropWorkspace(
        root=root, crop_file=crop_path, config_file=config_path,
        audit_file=audit_path, crop_source=crop_source,
    )
    logger.info("%s", workspace.summarise())
    return workspace


def load_crop_parameters(crop_file: Path | None, crop: str):
    """The crop a run should use: the workspace file, or the bundled preset.

    A missing file is an error rather than a silent fallback — a run told to
    use a crop file and given a path that is not there has almost certainly
    been pointed at the wrong workspace.
    """
    from torchcrop.parameters.crop_params import CropParameters

    if crop_file is None:
        return CropParameters(crop_name=crop)
    crop_file = Path(crop_file)
    if not crop_file.is_file():
        raise FileNotFoundError(
            f"crop file {crop_file} does not exist; prepare the workspace first "
            f"(cropmodelling4eu.torchcrop.workspace.prepare_workspace)"
        )
    logger.info("crop parameters from %s", crop_file)
    return CropParameters(config_file=str(crop_file))
