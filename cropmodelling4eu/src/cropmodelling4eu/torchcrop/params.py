"""Compare the crop parameters SIMPLACE runs with against torchcrop's preset.

The two models are only comparable to the extent they are given the same crop.
Both parameter sets descend from the same document — SIMPLACE's Brandenburg
``data/crop/crop.xml`` and torchcrop's ``wheat.yaml`` both say *"CROP DATA FILE
for use with LINTUL5 (NPK lim.) for wheat, August 2011, based on WOFOST WHEAT,
WINTER 102"* — which makes any difference between them a divergence in one of
the two copies rather than two defensible parameterisations. It also makes the
differences easy to miss: 95 % of the values agree, so a spot check passes.

This module does the comparison exhaustively and by name, so a run can state
which crop it actually ran rather than assuming.

The mapping is explicit and one-way (SIMPLACE → torchcrop). It has to be:
SIMPLACE stores a lookup table as two parallel ``<parameter>`` lists
(``SLATableDVS`` / ``SLATableSLA``) where torchcrop stores one list of pairs
(``slatb``), and the two spell most scalars differently in case alone but not
all of them (``RGRLAI`` → ``rgrl``). Guessing by lowercasing would silently
skip exactly the parameters that have been renamed.
"""

from __future__ import annotations

import logging
import xml.etree.ElementTree as ET
from pathlib import Path

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

__all__ = [
    "SCALARS",
    "TABLES",
    "compare_crop_parameters",
    "CROP_KEY_IDS",
    "crop_block_key",
    "simplace_table",
    "crop_parameters_from_simplace",
    "load_simplace_crop",
    "summarise",
]

#: SIMPLACE ``crop.xml`` id -> torchcrop preset scalar name.
SCALARS: dict[str, str] = {
    "TBASEM": "tbasem", "TEFFMX": "teffmx", "TSUMEM": "tsumem",
    "IDSL": "idsl", "TSUM1": "tsum1", "TSUM2": "tsum2", "DVSI": "dvsi",
    # Vernalisation. Absent from Brandenburg's crop.xml, present in the EU
    # SUSTAg one -- and inert unless carried across: torchcrop returns
    # vernfac = 1 whenever versat == vbase, so idsl = 2 with these unmapped is
    # a no-op that looks like a working configuration.
    "VERSAT": "versat", "VBASE": "vbase",
    "TDWI": "tdwi", "RGRLAI": "rgrl", "LAICR": "laicr", "TBASE": "tbase",
    "SPA": "spa", "NSLA": "nsla", "NLAI": "nlai", "NLUE": "nlue",
    "RDRNS": "rdrns", "RDRL": "rdrl", "RDRSHM": "rdrshm", "DVSDLT": "dvsdlt",
    "DVSDR": "dvsdr", "RDI": "rdi", "RRI": "rri", "RDMCR": "rdmcr",
    "IAIRDU": "iairdu", "CFET": "cfet", "DEPNR": "depnr", "NPART": "npart",
    "FNTRT": "fntrt", "DVSNT": "dvsnt", "DVSNLT": "dvsnlt", "NFIXF": "nfixf",
    "TCNT": "tcnt", "TCPT": "tcpt", "TCKT": "tckt",
    "FRNX": "frnx", "FRPX": "frpx", "FRKX": "frkx",
    "LRNR": "lrnr", "LSNR": "lsnr", "LRPR": "lrpr", "LSPR": "lspr",
    "LRKR": "lrkr", "LSKR": "lskr",
    "RNFLV": "rnflv", "RNFST": "rnfst", "RNFRT": "rnfrt",
    "RPFLV": "rpflv", "RPFST": "rpfst", "RPFRT": "rpfrt",
    "RKFLV": "rkflv", "RKFST": "rkfst", "RKFRT": "rkfrt",
    "NMAXSO": "nmaxso", "PMAXSO": "pmaxso", "KMAXSO": "kmaxso",
}

#: SIMPLACE ``management.xml`` id -> torchcrop scalar. The nutrient **recovery
#: fractions** are crop parameters in torchcrop and management parameters in
#: SIMPLACE, so they live in a different file on one side and are easy to miss.
#: They agree today (0.7 / 0.2 / 0.6 on both sides, from the same WOFOST
#: lineage) — but by coincidence, not by construction: edit the solution's
#: management.xml and nothing would carry the change across. Mapping them makes
#: the agreement a fact about the run rather than about two defaults.
MANAGEMENT_SCALARS: dict[str, str] = {"NRF": "nrf", "PRF": "prf", "KRF": "krf"}

#: Section the recovery fractions are written into, since torchcrop's bundled
#: presets do not carry them at all.
RECOVERY_SECTION = "Nutrient recovery (SIMPLACE management.xml)"

#: torchcrop table name -> the SIMPLACE (x, y) parameter pair it is stored as.
TABLES: dict[str, tuple[str, str]] = {
    "dtsmtb": ("TsumIncrementTableMeanTemp", "TsumIncrementTableRate"),
    "phottb": ("PhotoperiodTableHour", "PhotoperiodTableFactor"),
    "vernrt": ("VernalisationTableMeanTemp", "VernalisationTableRate"),
    "slatb": ("SLATableDVS", "SLATableSLA"),
    "ssatb": ("SSATableDVS", "SSATableSSA"),
    "kdiftb": ("KDIFTableDVS", "KDIFTableK"),
    "ruetb": ("RUETableDVS", "RUETableRUE"),
    "fltb": ("LeavesPartitioningTableDVS", "LeavesPartitioningTableFraction"),
    "fstb": ("StemsPartitioningTableDVS", "StemsPartitioningTableFraction"),
    "fotb": ("StorageOrgansPartitioningTableDVS",
             "StorageOrgansPartitioningTableFraction"),
    "frtb": ("RootsPartitioningTableDVS", "RootsPartitioningTableFraction"),
    "rdrltb": ("RDRLeavesTableMeanTemp", "RDRLeavesTableRelativeRate"),
    "rdrrtb": ("RDRRootsTableDVS", "RDRRootsTableRelativeRate"),
    "rdrstb": ("RDRStemsTableDVS", "RDRStemsTableRelativeRate"),
    "nmxlv": ("NMaxTableDVS", "NMaxTableConcentration"),
    "pmxlv": ("PMaxTableDVS", "PMaxTableConcentration"),
    "kmxlv": ("KMaxTableDVS", "KMaxTableConcentration"),
    "tmnftb": ("TMNFTableMinTemperature", "TMNFTableFactor"),
    "tmpftb": ("TMPFTableMeanTemperature", "TMPFTableFactor"),
    "cotb": ("COTableCo2", "COTableFactor"),
}

#: Present in SIMPLACE's crop file but not a LINTUL-5 growth parameter: they
#: drive its harvest, fresh-matter and Kcb evapotranspiration modules, which
#: torchcrop does not implement. Absence here is not a divergence.
SIMPLACE_ONLY = frozenset({
    "CropName", "DVSEND", "CropHt", "KcbIni", "KcbMid", "KcbEnd",
    "HarvestLeaves", "HarvestStems", "RemoveRatioStorageOrgan",
    "RemoveRatioStraw", "FreshratioStorageOrgan", "FreshratioStraw",
    "YieldAdjustRatio", "FRTDM", "MaximalSeminalRootLengthPerDay",
    # management.xml: soil initial conditions, not crop parameters. Their
    # torchcrop counterparts are on SoilParameters and this pipeline takes them
    # from the export's own soil.csv per cell rather than from one constant --
    # except rtnmins, which torchcrop.run lists in ASSUMPTIONS at SIMPLACE's
    # own 0.025.
    "location", "NMINS", "PMINS", "KMINS", "RTNMINS", "RTPMINS", "RTKMINS",
})


#: The id a ``<crop>`` block is keyed by. Both spellings occur in files this
#: project reads: Brandenburg's ``crop.xml`` uses ``CropName`` with a long name
#: (``winter_wheat``), the EU SUSTAg ``LINTUL5_crop.xml`` uses ``Crop`` with a
#: four-letter code (``WW``). Neither file says which convention it follows, so
#: both are tried.
CROP_KEY_IDS: tuple[str, ...] = ("CropName", "Crop")


def crop_block_key(block: ET.Element) -> str:
    """The crop identifier a ``<crop>`` block declares, or ``""``."""
    for key in CROP_KEY_IDS:
        value = block.findtext(f"parameter[@id='{key}']")
        if value and value.strip():
            return value.strip()
    return ""


def load_simplace_crop(path: Path, crop_name: str | None = None) -> dict:
    """Parse a SIMPLACE parameter XML into ``{parameter id: float | list}``.

    Handles the two shapes these files take: one ``<crop>`` block per crop
    (``crop.xml``, ``LINTUL5_crop.xml``, ``seeds.xml``), selected by
    ``crop_name``, and a single unnamed block (``management.xml``). The element
    is named differently in each, so the first block holding ``<parameter>``
    children is taken when there is no ``<crop>``.

    **A multi-crop file without a ``crop_name`` is an error, not a default.**
    Falling back to the first block reads ``LINTUL5_crop.xml`` as *maize* and
    then runs a wheat simulation on it — a silent, plausible-looking wrong
    answer. A single-block file still needs no name.

    Raises:
        ValueError: If ``crop_name`` matches no block, or if the file holds
            several blocks and no ``crop_name`` was given.
    """
    root = ET.parse(Path(path)).getroot()
    crops = root.findall("crop")
    if crops and crop_name is not None:
        matched = [c for c in crops if crop_block_key(c) == crop_name]
        if not matched:
            available = [crop_block_key(c) or "?" for c in crops]
            raise ValueError(
                f"no crop {crop_name!r} in {path}; it declares {available}"
            )
        crops = matched
    elif len(crops) > 1:
        raise ValueError(
            f"{path} holds {len(crops)} crop blocks "
            f"({[crop_block_key(c) or '?' for c in crops]}) and no crop was "
            "named. Pass simplace_crop -- defaulting to the first block would "
            "parameterise this run from whichever crop happens to be listed "
            "first."
        )
    if not crops:
        crops = [block for block in root if block.find("parameter") is not None]
    if not crops:
        raise ValueError(f"no <parameter> blocks in {path}")

    out: dict[str, float | list[float] | str] = {}
    for parameter in crops[0].findall("parameter"):
        values = [float(v.text) for v in parameter.findall("value")]
        if values:
            out[parameter.get("id")] = values
        else:
            text = (parameter.text or "").strip()
            try:
                out[parameter.get("id")] = float(text)
            except ValueError:
                out[parameter.get("id")] = text
    return out


def simplace_table(simplace: dict, tc_name: str) -> list[list[float]] | None:
    """A torchcrop table read from whichever shape the SIMPLACE file uses.

    Two conventions are in play and neither file declares which it follows:

    * **Two parameters**, x and y in separate ids — ``SLATableDVS`` +
      ``SLATableSLA``. Brandenburg's ``crop.xml``, and what :data:`TABLES` maps.
    * **One parameter**, x and y interleaved in a single ``<value>`` list —
      ``SLATB``. The EU SUSTAg ``LINTUL5_crop.xml``, whose ids are the
      torchcrop table names uppercased.

    Reading only the first shape leaves every table in a SUSTAg-sourced file
    silently on torchcrop's bundled preset, which is the quiet half of a
    harmonisation that looks like it worked.

    Returns:
        ``[[x, y], ...]``, or ``None`` when neither shape is present.
    """
    x_name, y_name = TABLES.get(tc_name, (None, None))
    if x_name in simplace and y_name in simplace:
        return [[float(x), float(y)]
                for x, y in zip(simplace[x_name], simplace[y_name])]

    flat = simplace.get(tc_name.upper())
    if isinstance(flat, list) and flat:
        if len(flat) % 2:
            raise ValueError(
                f"{tc_name.upper()} holds {len(flat)} values; an interleaved "
                "(x, y) table must have an even count"
            )
        return [[float(flat[i]), float(flat[i + 1])]
                for i in range(0, len(flat), 2)]
    return None


def _torchcrop_preset(crop_name: str) -> tuple[dict, dict]:
    """The bundled preset, flattened to ``(scalars, tables)``."""
    import yaml
    from torchcrop.parameters.crop_params import _builtin_crop_path

    preset = yaml.safe_load(Path(_builtin_crop_path(crop_name)).read_text())
    scalars: dict[str, float] = {}
    tables: dict[str, list] = {}
    for section in preset["sections"].values():
        scalars |= section.get("scalars", {})
        tables |= section.get("tables", {})
    return scalars, tables


def compare_crop_parameters(
    simplace_crop_xml: Path,
    crop_name: str = "wheat",
    simplace_crop: str | None = None,
    rtol: float = 1e-6,
    management_xml: Path | None = None,
) -> pd.DataFrame:
    """Every crop parameter, side by side, with a verdict per row.

    Returns
    -------
    DataFrame
        ``parameter, kind, simplace, torchcrop, status``, where ``status`` is
        ``same``, ``differs``, ``simplace_only`` or ``torchcrop_only``.
        A table row compares the full ``(x, y)`` curve, not a summary of it:
        two curves that share their end points can still differ everywhere in
        between, which is exactly what a partitioning table does.
    """
    simplace = load_simplace_crop(simplace_crop_xml, simplace_crop)
    tc_scalars, tc_tables = _torchcrop_preset(crop_name)

    scalars = dict(SCALARS)
    if management_xml is not None and Path(management_xml).is_file():
        # The recovery fractions are in no preset, so their torchcrop side is
        # the dataclass default rather than a preset entry.
        from torchcrop.parameters.crop_params import CropParameters

        defaults = CropParameters(crop_name=crop_name)
        simplace |= load_simplace_crop(management_xml)
        tc_scalars |= {
            tc: float(getattr(defaults, tc))
            for tc in MANAGEMENT_SCALARS.values() if hasattr(defaults, tc)
        }
        scalars |= MANAGEMENT_SCALARS

    rows: list[dict] = []
    for sp_name, tc_name in scalars.items():
        if sp_name not in simplace and tc_name not in tc_scalars:
            continue
        sp_value = simplace.get(sp_name)
        tc_value = tc_scalars.get(tc_name)
        if sp_value is None or tc_value is None:
            status = "simplace_only" if tc_value is None else "torchcrop_only"
        else:
            status = "same" if np.isclose(sp_value, tc_value, rtol=rtol) else "differs"
        rows.append({"parameter": f"{sp_name} / {tc_name}", "kind": "scalar",
                     "simplace": sp_value, "torchcrop": tc_value, "status": status})

    for tc_name in TABLES:
        sp_pairs = simplace_table(simplace, tc_name)
        tc_pairs = [list(pair) for pair in tc_tables[tc_name]] if tc_name in tc_tables else None
        if sp_pairs is None and tc_pairs is None:
            continue
        if sp_pairs is None or tc_pairs is None:
            status = "simplace_only" if tc_pairs is None else "torchcrop_only"
        elif len(sp_pairs) != len(tc_pairs):
            status = "differs"
        else:
            status = (
                "same"
                if np.allclose(np.array(sp_pairs), np.array(tc_pairs), rtol=rtol)
                else "differs"
            )
        rows.append({"parameter": tc_name, "kind": "table",
                     "simplace": sp_pairs, "torchcrop": tc_pairs, "status": status})

    # Both table spellings count as mapped: the two-parameter ids of
    # TABLES, and the single interleaved id (the torchcrop name uppercased).
    unmapped = set(simplace) - set(scalars) - SIMPLACE_ONLY - {
        name for pair in TABLES.values() for name in pair
    } - {tc_name.upper() for tc_name in TABLES}
    if unmapped:
        logger.warning(
            "SIMPLACE crop file holds %d parameter(s) this comparison does not "
            "map: %s -- add them to SCALARS/TABLES or SIMPLACE_ONLY",
            len(unmapped), sorted(unmapped),
        )
    return pd.DataFrame(rows)


def write_crop_yaml(
    path: Path,
    simplace_crop_xml: Path | None = None,
    seeds_xml: Path | None = None,
    crop_name: str = "wheat",
    simplace_crop: str | None = None,
    management_xml: Path | None = None,
    seeds_crop_name: str | None = None,
) -> Path:
    """Write SIMPLACE's crop as a torchcrop preset YAML, and return the path.

    torchcrop takes a crop as a file — ``CropParameters(config_file=...)`` —
    so a run that is meant to use SIMPLACE's crop should be given SIMPLACE's
    crop as such a file, rather than having a parameter object mutated after
    construction. The file is then the thing that ran: diffable against the
    bundled ``wheat.yaml``, re-loadable on its own, and inspectable next to
    the outputs it produced.

    The preset written is **complete**, not a patch. It keeps the bundled
    preset's own section layout and fills every entry, taking SIMPLACE's value
    where the mapping has one and the bundled value where it does not, so
    nothing silently falls back to the generic ``default.yaml`` at load time.
    Each parameter's origin is recorded under ``provenance``.

    ``seeds.xml`` is read but **not mapped**. It parameterises SIMPLACE's
    SeedsToSprouts module — a seed weight distributed to roots and leaves over
    the days after sowing — and torchcrop has no counterpart: its only
    establishment parameter is ``tdwi``, the initial crop dry weight, which is
    a different quantity (15 vs 5 g/m² for this crop). Mapping one onto the
    other would be an invention, so the values are recorded in the file under
    ``simplace_seeds`` and named as an unmodelled difference.
    """
    import yaml
    from torchcrop.parameters.crop_params import _builtin_crop_path

    # Without a SIMPLACE crop the bundled preset is written out unchanged. The
    # file is then still the crop that ran, which is the point of having one:
    # a run whose parameters live only inside an installed package cannot be
    # checked by opening anything.
    simplace = (
        load_simplace_crop(simplace_crop_xml, simplace_crop)
        if simplace_crop_xml is not None else {}
    )
    preset = yaml.safe_load(Path(_builtin_crop_path(crop_name)).read_text())
    to_simplace_scalar = {tc: sp for sp, tc in SCALARS.items()}

    provenance: dict[str, str] = {}
    for section in preset["sections"].values():
        for tc_name in list(section.get("scalars", {})):
            sp_name = to_simplace_scalar.get(tc_name)
            if sp_name is not None and sp_name in simplace:
                section["scalars"][tc_name] = float(simplace[sp_name])
                provenance[tc_name] = f"simplace:{sp_name}"
            else:
                provenance[tc_name] = f"torchcrop:{crop_name}"
        for tc_name in list(section.get("tables", {})):
            values = simplace_table(simplace, tc_name)
            if values is not None:
                section["tables"][tc_name] = values
                provenance[tc_name] = f"simplace:{tc_name.upper()}"
            else:
                provenance[tc_name] = f"torchcrop:{crop_name}"

    # Parameters SIMPLACE supplies, torchcrop has a field for, and the bundled
    # preset has no slot for. Filling only the preset's own slots drops them
    # silently -- and for the vernalisation block that is not a small loss: the
    # SUSTAg crop runs IDSL = 2, and torchcrop returns vernfac = 1 whenever
    # versat == vbase, so idsl = 2 with versat/vbase/vernrt missing is a no-op
    # sitting next to a TSUM1 that was calibrated *with* vernalisation. The
    # crop would run far too fast, and nothing in the file would say why.
    from torchcrop.parameters.crop_params import CropParameters

    fields = set(CropParameters(crop_name=crop_name).__dataclass_fields__)
    written = {
        name
        for section in preset["sections"].values()
        for group in ("scalars", "tables")
        for name in section.get(group, {})
    }
    extra_scalars = {
        tc: float(simplace[sp])
        for sp, tc in SCALARS.items()
        if tc not in written and tc in fields and sp in simplace
    }
    extra_tables = {}
    for tc_name in TABLES:
        if tc_name in written or tc_name not in fields:
            continue
        values = simplace_table(simplace, tc_name)
        if values is not None:
            extra_tables[tc_name] = values
    if extra_scalars or extra_tables:
        preset["sections"]["SIMPLACE (absent from the torchcrop preset)"] = {
            "scalars": extra_scalars, "tables": extra_tables,
        }
        for tc_name in (*extra_scalars, *extra_tables):
            provenance[tc_name] = f"simplace:{tc_name.upper()}"
        logger.info(
            "carried %d parameter(s) the bundled preset has no slot for: %s",
            len(extra_scalars) + len(extra_tables),
            sorted([*extra_scalars, *extra_tables]),
        )

    # The recovery fractions come from a different SIMPLACE file and are in no
    # bundled preset, so they are added as their own section rather than
    # overwritten into an existing one.
    if management_xml is not None and Path(management_xml).is_file():
        management = load_simplace_crop(management_xml)
        recovery = {
            tc_name: float(management[sp_name])
            for sp_name, tc_name in MANAGEMENT_SCALARS.items()
            if sp_name in management
        }
        if recovery:
            preset["sections"][RECOVERY_SECTION] = {"scalars": recovery}
            provenance |= {
                tc: f"simplace:{sp}" for sp, tc in MANAGEMENT_SCALARS.items()
                if tc in recovery
            }

    unused = sorted(
        tc_name for tc_name in {**{v: k for k, v in SCALARS.items()}, **TABLES}
        if tc_name not in provenance
    )
    preset |= {
        "crop_name": f"{crop_name}_simplace" if simplace else crop_name,
        "source_file": str(simplace_crop_xml) if simplace else preset.get("source_file"),
        "source_crop_name": simplace.get("CropName", crop_name),
        "description": (
            (f"{crop_name} as parameterised by SIMPLACE's "
             f"{Path(simplace_crop_xml).name}, on the layout of torchcrop's "
             f"bundled {crop_name}.yaml"
             if simplace else
             f"torchcrop's bundled {crop_name} preset, written out unchanged")
            + ". Written by cropmodelling4eu.torchcrop.params.write_crop_yaml"
        ),
        "provenance": provenance,
        "unmapped_simplace_parameters": unused,
    }

    if seeds_xml is not None and Path(seeds_xml).is_file():
        # seeds.xml keys its blocks differently from the crop file -- long
        # names (`winter_wheat`) where LINTUL5_crop.xml uses codes (`WW`) --
        # so the crop block's own key cannot simply be reused. These values are
        # recorded and never mapped, so failing to find the block must not stop
        # a run: the note is provenance, not a parameter.
        seeds_crop = simplace.get("CropName") or seeds_crop_name
        try:
            seeds = load_simplace_crop(seeds_xml, seeds_crop)
        except ValueError as error:
            logger.warning(
                "not recording %s: %s", Path(seeds_xml).name, error
            )
            seeds = None
    else:
        seeds = None

    if seeds is not None:
        preset["simplace_seeds"] = {
            "source_file": str(seeds_xml),
            "values": {k: v for k, v in seeds.items() if k != "CropName"},
            "note": (
                "Read for the record and deliberately not mapped: these drive "
                "SIMPLACE's SeedsToSprouts establishment module, which torchcrop "
                "does not implement. Its only establishment parameter is tdwi "
                "(initial crop dry weight), a different quantity from SeedWeight. "
                "This is a structural difference between the two models, not a "
                "parameter one."
            ),
        }

    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w") as handle:
        yaml.safe_dump(preset, handle, sort_keys=False, default_flow_style=False)
    taken = sum(1 for origin in provenance.values() if origin.startswith("simplace"))
    logger.info(
        "wrote %s: %d of %d parameters from %s, %d kept from the bundled %s preset",
        path, taken, len(provenance),
        Path(simplace_crop_xml).name if simplace else "(none)",
        len(provenance) - taken, crop_name,
    )
    return path


def crop_parameters_from_simplace(
    simplace_crop_xml: Path,
    crop_name: str = "wheat",
    simplace_crop: str | None = None,
    seeds_xml: Path | None = None,
    yaml_path: Path | None = None,
    management_xml: Path | None = None,
    output_dir: Path | None = None,
):
    """A torchcrop ``CropParameters`` carrying SIMPLACE's values.

    Goes through :func:`write_crop_yaml` and torchcrop's own
    ``config_file=`` loader rather than assigning attributes, so the object in
    memory and the file on disk cannot disagree. Pass ``yaml_path`` to keep the
    file beside a run's outputs; without it the preset is written to a scratch
    file under ``output_dir/tmp`` (a run-scoped, cluster-visible directory
    rather than the node-local system temp dir) and discarded. With neither
    given, it falls back to the system temp directory.

    This is what makes a two-model comparison a comparison **of the models**:
    run with the presets as shipped and 21 of 72 parameters differ, including
    ``TSUM1`` (1623 vs 1050) and the leaf death rates (4x), so the two would be
    growing different crops before the first line of physics runs.
    """
    import tempfile

    from torchcrop.parameters.crop_params import CropParameters

    if yaml_path is not None:
        path = write_crop_yaml(
            yaml_path, simplace_crop_xml, seeds_xml, crop_name, simplace_crop,
            management_xml,
        )
        return CropParameters(config_file=str(path))

    scratch_dir = None
    if output_dir is not None:
        scratch_dir = Path(output_dir) / "tmp"
        scratch_dir.mkdir(parents=True, exist_ok=True)

    with tempfile.TemporaryDirectory(dir=scratch_dir) as tmp:
        path = write_crop_yaml(
            Path(tmp) / "crop.yaml", simplace_crop_xml, seeds_xml,
            crop_name, simplace_crop, management_xml,
        )
        return CropParameters(config_file=str(path))


def summarise(comparison: pd.DataFrame) -> str:
    """One-paragraph verdict plus the differing rows, for a run log."""
    counts = comparison["status"].value_counts().to_dict()
    lines = [
        "Crop parameters, SIMPLACE crop.xml vs torchcrop preset: "
        + ", ".join(f"{n} {status}" for status, n in sorted(counts.items()))
    ]
    differing = comparison[comparison["status"] != "same"]
    for _, row in differing.iterrows():
        lines.append(
            f"  {row['status']:14s} {row['parameter']:24s} "
            f"simplace={row['simplace']}  torchcrop={row['torchcrop']}"
        )
    return "\n".join(lines)
