"""Thin CLI over ``cropmodelling4eu.torchcrop.workspace.prepare_workspace``.

Every submit script that runs torchcrop calls this before queuing anything —
the production array (``submit_torchcrop.sh``), its smoke test, and the
chained smoke test (``submit_cropmodelling.sh --smoke``) — so the workspace
layout has one implementation rather than a bash heredoc repeated three times.
"""

from __future__ import annotations

import argparse
import logging
from pathlib import Path

from cropmodelling4eu.config import load_config
from cropmodelling4eu.torchcrop.workspace import prepare_workspace


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--out-dir", type=Path, required=True,
                        help="workspace/ is created under this directory")
    parser.add_argument("--crop-source", choices=("torchcrop", "simplace"), default="torchcrop")
    parser.add_argument("--crop-xml", type=Path,
                        help="SIMPLACE crop.xml; needed for --crop-source simplace "
                             "and for the audit either way")
    parser.add_argument("--seeds-xml", type=Path)
    parser.add_argument("--management-xml", type=Path)
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(levelname)-7s %(message)s")
    config = load_config(args.config)
    workspace = prepare_workspace(
        config, args.out_dir, args.crop_source,
        args.crop_xml, args.seeds_xml, args.management_xml,
    )
    print(workspace.crop_file)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
