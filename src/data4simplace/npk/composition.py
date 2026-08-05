"""SIMPLACE ``fertilizer_composition.xml`` parser.

The nutrient content of every fertilizer type SIMPLACE knows is declared in the
reference ``fertilizer_composition.xml`` that sits next to the management CSV.
The management exporter needs those contents to invert a nutrient rate into a
product amount, so they are read from the reference file rather than hard-coded:
the same rate becomes a different ``Amount`` depending on which carrier the
schedule uses.

Amounts in a SIMPLACE fertilizer schedule are grams of *product* per square
metre; the contents here are grams of *element* per gram of product. Hence

    Amount [g product / m^2] = nutrient [g element / m^2] / content [g/g]
"""

from __future__ import annotations

import logging
import xml.etree.ElementTree as ET
from dataclasses import dataclass
from pathlib import Path

logger = logging.getLogger(__name__)

#: Molar-mass ratios converting the oxide forms fertilizer statistics are
#: reported in (NPKGRIDS included) to the elemental forms SIMPLACE's composition
#: file declares. P2O5 -> 2 P / (2 P + 5 O); K2O -> 2 K / (2 K + O).
P2O5_TO_P = 0.436421
K2O_TO_K = 0.830147


@dataclass(frozen=True)
class FertilizerComposition:
    """Elemental content of one fertilizer type, in g element / g product.

    Attributes
    ----------
    name:
        The ``vType`` value used in the schedule CSV (e.g. ``KAS``, ``P``).
    mineral_n:
        Nitrate plus ammonium — the ``NitrateAndAmmonium`` entry, which is what
        a mineral N rate has to be divided by.
    nitrate, ammonium, phosphorus, potassium, organic_c, organic_n:
        The remaining declared contents.
    """

    name: str
    mineral_n: float = 0.0
    nitrate: float = 0.0
    ammonium: float = 0.0
    phosphorus: float = 0.0
    potassium: float = 0.0
    organic_c: float = 0.0
    organic_n: float = 0.0

    @property
    def carries_n(self) -> bool:
        """Whether the type supplies mineral or organic nitrogen."""
        return self.mineral_n > 0.0 or self.organic_n > 0.0

    @property
    def carries_p(self) -> bool:
        """Whether the type supplies phosphorus."""
        return self.phosphorus > 0.0

    @property
    def carries_k(self) -> bool:
        """Whether the type supplies potassium."""
        return self.potassium > 0.0


#: XML ``parameter id`` -> :class:`FertilizerComposition` field.
_FIELDS = {
    "NitrateAndAmmonium": "mineral_n",
    "Nitrate": "nitrate",
    "Ammonium": "ammonium",
    "Phosphorus": "phosphorus",
    "Potassium": "potassium",
    "OrganicC": "organic_c",
    "OrganicN": "organic_n",
}


def _to_float(text: str | None) -> float:
    """Parse a composition value, tolerating the file's padded whitespace."""
    if text is None or not text.strip():
        return 0.0
    return float(text.strip())


def parse_fertilizer_composition(path: str | Path) -> dict[str, FertilizerComposition]:
    """Read ``fertilizer_composition.xml`` into a ``vType`` -> composition map.

    Parameters
    ----------
    path:
        Path to the SIMPLACE ``fertilizer_composition.xml``.

    Returns
    -------
    dict[str, FertilizerComposition]
        One entry per ``<fertilizer>`` block, keyed by its ``Fertilizertype``.

    Raises
    ------
    FileNotFoundError
        If ``path`` does not exist.
    ValueError
        If the XML is malformed or declares no usable fertilizer type.
    """
    xml_path = Path(path)
    if not xml_path.is_file():
        raise FileNotFoundError(f"Fertilizer composition file not found: {xml_path}")

    try:
        root = ET.parse(xml_path).getroot()
    except ET.ParseError as exc:  # pragma: no cover - malformed reference file
        raise ValueError(f"Malformed fertilizer composition XML {xml_path}: {exc}") from exc

    compositions: dict[str, FertilizerComposition] = {}
    for block in root.findall("fertilizer"):
        params = {p.get("id"): p.text for p in block.findall("parameter")}
        name = (params.get("Fertilizertype") or "").strip()
        if not name:
            continue
        values = {field: _to_float(params.get(key)) for key, field in _FIELDS.items()}
        # Some blocks declare only the split forms; recover the mineral N total.
        if values["mineral_n"] <= 0.0:
            values["mineral_n"] = values["nitrate"] + values["ammonium"]
        compositions[name] = FertilizerComposition(name=name, **values)

    if not compositions:
        raise ValueError(f"No fertilizer types declared in {xml_path}")

    logger.info("Parsed %d fertilizer types from %s", len(compositions), xml_path.name)
    return compositions


def default_composition_path(management_reference: str | Path | None) -> Path | None:
    """Locate ``fertilizer_composition.xml`` next to the management reference.

    SIMPLACE keeps the composition file in the same ``data/management`` folder as
    the schedule CSV, so the reference path already points at its directory.
    Returns ``None`` when there is no reference or no such file.
    """
    if management_reference is None:
        return None
    ref = Path(management_reference)
    folder = ref if ref.is_dir() else ref.parent
    candidate = folder / "fertilizer_composition.xml"
    return candidate if candidate.is_file() else None
