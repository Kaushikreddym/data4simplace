"""Read, validate and rewrite a SIMPLACE solution and its project XML.

A solution declares the files it reads as ``<interface>`` elements and the
columns it expects from each as a ``<resource><header>`` of ``<res>`` entries.
Those declarations are checkable against the files an export actually wrote,
and checking them is the whole point of this module: a missing column currently
surfaces as a Java stack trace in a container log, minutes into a job, naming
an internal class rather than the column.

**A ``datatype`` does not fix the file layout.** ``DOUBLEARRAY`` means the value
is an array; SIMPLACE fills one either from ``<id>_1`` … ``<id>_N`` columns in a
single row (the wide Brandenburg ``soil.csv``) or from repeated rows sharing the
resource key (the long EU SUSTAg file). Both satisfy the same declaration, so
:func:`validate_resources` accepts either and reports only the case where
neither is present.
"""

from __future__ import annotations

import logging
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

import pandas as pd
from lxml import etree

logger = logging.getLogger(__name__)

__all__ = [
    "Resource",
    "SolutionDocument",
    "ValidationError",
    "read_solution",
    "validate_resources",
]

#: ``${_WORKDIR_}``-style placeholders SIMPLACE expands at run time.
_PLACEHOLDER = re.compile(r"\$\{(?P<name>[A-Za-z0-9_.]+)\}")

#: Array-valued datatypes, which a wide file encodes as ``<id>_<N>`` columns.
_ARRAY_TYPES = {"DOUBLEARRAY", "INTARRAY", "CHARARRAY", "BOOLEANARRAY"}


class ValidationError(RuntimeError):
    """A solution asks for something the export does not provide."""


@dataclass(slots=True)
class Resource:
    """One ``<resource>``: the columns a solution reads from one interface.

    Attributes
    ----------
    id:
        The resource id.
    interface:
        The interface it reads through.
    filename:
        The interface's raw ``<filename>``, placeholders unexpanded.
    divider:
        The interface's field separator; ``None`` when the element is empty,
        which SIMPLACE reads as whitespace/tab.
    columns:
        ``{res id: datatype}`` in declaration order.
    """

    id: str
    interface: str
    filename: str
    divider: Optional[str]
    columns: dict[str, str] = field(default_factory=dict)

    @property
    def array_columns(self) -> list[str]:
        return [name for name, kind in self.columns.items() if kind in _ARRAY_TYPES]

    def missing_from(self, header: list[str]) -> list[str]:
        """Declared columns that ``header`` satisfies in neither encoding.

        A scalar column must be present by name. An array column is satisfied
        by its bare name (long) **or** by at least one ``<name>_<N>`` (wide).
        """
        present = {str(c) for c in header}
        suffixed = {
            match.group(1)
            for column in present
            if (match := re.match(r"^(.+)_(\d+)$", column))
        }
        return [
            name
            for name, kind in self.columns.items()
            if name not in present and not (kind in _ARRAY_TYPES and name in suffixed)
        ]


@dataclass(slots=True)
class SolutionDocument:
    """A parsed solution or project XML, with its resources resolved."""

    path: Path
    tree: etree._ElementTree
    resources: list[Resource]

    @property
    def root(self) -> etree._Element:
        return self.tree.getroot()

    def variables(self) -> dict[str, str]:
        """The solution's ``<var>`` defaults, by id."""
        return {
            var.get("id"): (var.text or "").strip()
            for var in self.root.findall(".//variables/var")
            if var.get("id")
        }

    def set_variables(self, values: dict[str, object]) -> list[str]:
        """Override ``<var>`` defaults in place; return the ids that changed.

        Only existing variables are touched. Adding one silently would produce
        a solution that runs but ignores it, since a component reads a variable
        by an ``<input source=...>`` the solution author wrote.
        """
        changed = []
        for var in self.root.findall(".//variables/var"):
            name = var.get("id")
            if name in values:
                new = str(values[name])
                if (var.text or "").strip() != new:
                    var.text = new
                    changed.append(name)
        unknown = sorted(set(values) - {v.get("id") for v in self.root.findall(".//variables/var")})
        if unknown:
            logger.warning(
                "Solution %s declares no variable(s) %s; they were not added, "
                "since a variable no component reads has no effect",
                self.path.name, unknown,
            )
        return changed

    def set_input_source(self, input_id: str, source: str) -> int:
        """Repoint every ``<input id=input_id>``'s ``source`` in place.

        Unlike :meth:`set_variables`, this rewires a component's wiring
        rather than a value — e.g. switching ``iTRANRF`` from
        ``LintulWaterStress.TRANRF`` (computed) to a constant ``<var>`` such
        as ``vTRANRF`` for a potential-production run, matching the
        already-present but disabled alternative some solutions ship
        (``<!--input id="iTRANRF" source="vTRANRF" /-->``). Returns the
        number of ``<input>`` elements changed.
        """
        elements = self.root.findall(f".//input[@id='{input_id}']")
        if not elements:
            logger.warning(
                "Solution %s has no <input id=%r> to repoint at %r",
                self.path.name, input_id, source,
            )
        changed = 0
        for element in elements:
            if element.get("source") != source:
                element.set("source", source)
                changed += 1
        return changed

    def scale_transform(self, transform_id: str, factors: dict[str, float]) -> str:
        """Multiply columns by a constant inside a transform's SQL statement.

        A unit fix belongs here rather than in the data: a
        ``DefaultSQLStatementTransformer`` already reads the resource and
        rewrites it, so scaling a column costs one token of SQL where
        converting the files costs a second copy of the weather.

        Substitution is word-boundary on the **resource-declared** id, which is
        what the SQL sees — SIMPLACE binds CSV columns positionally, so the
        file's own header names never appear here. Returns the new statement.
        """
        transform = self.root.find(f".//transform[@id='{transform_id}']")
        if transform is None:
            raise KeyError(f"solution {self.path.name} declares no transform {transform_id!r}")
        statement = transform.find("input[@id='statement']")
        if statement is None or not statement.text:
            raise KeyError(f"transform {transform_id!r} has no <input id='statement'>")

        text = statement.text
        for column, factor in factors.items():
            pattern = rf"\b{re.escape(column)}\b"
            if not re.search(pattern, text):
                tokens = sorted(set(re.findall(r"\b[A-Za-z_]\w*\b", text)))
                raise KeyError(
                    f"transform {transform_id!r} does not reference {column!r}; "
                    f"its statement mentions {tokens}"
                )
            text = re.sub(pattern, f"({column}*{factor:g})", text)
        statement.text = text
        return text

    def write(self, path: str | Path) -> Path:
        """Write the (possibly modified) document, keeping its DOCTYPE."""
        out = Path(path)
        out.parent.mkdir(parents=True, exist_ok=True)
        self.tree.write(
            str(out), encoding="UTF-8", xml_declaration=True, pretty_print=False
        )
        return out


def _interfaces(root: etree._Element) -> dict[str, tuple[str, Optional[str]]]:
    """``{interface id: (filename, divider)}``."""
    return {
        name: (
            interface.findtext("filename", default="").strip(),
            divider.text if (divider := interface.find("divider")) is not None
            and divider.text else None,
        )
        for interface in root.findall(".//interfaces/interface")
        if (name := interface.get("id"))
    }


def read_solution(path: str | Path) -> SolutionDocument:
    """Parse a solution or project XML and resolve its resources.

    The DTD is declared but not fetched: SIMPLACE's DTD lives on a public URL,
    and a compute node has no reason to reach it. ``resolve_entities=False``
    keeps the parse local and side-effect free.
    """
    path = Path(path)
    if not path.is_file():
        raise FileNotFoundError(f"SIMPLACE XML not found: {path}")

    parser = etree.XMLParser(
        remove_blank_text=False, resolve_entities=False, load_dtd=False, no_network=True
    )
    tree = etree.parse(str(path), parser)
    root = tree.getroot()
    interfaces = _interfaces(root)

    resources: list[Resource] = []
    # A project XML spells them <var>, a solution <res>; both live under
    # <resource><header>.
    for element in root.findall(".//resource"):
        interface = element.get("interface")
        if interface is None:
            continue
        filename, divider = interfaces.get(interface, ("", None))
        columns = {
            entry.get("id"): (entry.get("datatype") or "CHAR").upper()
            for entry in element.findall("header/*")
            if entry.get("id")
        }
        resources.append(
            Resource(
                id=element.get("id") or interface,
                interface=interface,
                filename=filename,
                divider=divider,
                columns=columns,
            )
        )

    logger.info(
        "Parsed %s: %d interfaces, %d resources", path.name, len(interfaces), len(resources)
    )
    return SolutionDocument(path=path, tree=tree, resources=resources)


def expand(filename: str, substitutions: dict[str, str]) -> str:
    """Expand ``${...}`` placeholders, leaving unknown ones in place.

    Unknown placeholders are left rather than blanked because a per-cell
    interface legitimately carries them (``${vRow}`` in the weather filename);
    blanking would turn a template into a wrong concrete path.
    """
    return _PLACEHOLDER.sub(
        lambda m: substitutions.get(m.group("name"), m.group(0)), filename
    )


def validate_resources(
    document: SolutionDocument,
    substitutions: dict[str, str],
    strict: bool = False,
) -> tuple[list[str], list[str]]:
    """Check every CSV resource against the file it points at.

    Problems are split by how certain they are to be fatal, because a
    validator that over-claims gets switched off:

    **Missing files are errors.** A resource pointing at a path that does not
    exist cannot work, and is raised whatever ``strict`` says.

    **Missing columns are warnings.** SIMPLACE turns out to tolerate them: the
    Brandenburg solution declares ``Type`` where its own working
    ``fertilizer_winter_wheat.csv`` has ``vType``, and
    ``LowerBoundaryPConcentration`` where its ``soil.csv`` has
    ``LowerBoundaryConcentration`` — and that pair is a run that has produced
    output. So a name mismatch is reported, loudly and with both sides named,
    but it does not stop a build unless asked to.

    Parameters
    ----------
    document:
        A parsed solution.
    substitutions:
        Placeholder values, e.g. ``{"_WORKDIR_": "/path/to/workspace"}``.
    strict:
        Also raise on missing columns. Off by default, for the reason above.

    Returns
    -------
    tuple
        ``(errors, warnings)``.

    Raises
    ------
    ValidationError
        On any missing file, or on any problem at all in strict mode. Every
        problem is listed, not the first, so one build cycle fixes one export.
    """
    errors: list[str] = []
    warnings: list[str] = []

    for resource in document.resources:
        path_text = expand(resource.filename, substitutions)
        if _PLACEHOLDER.search(path_text):
            # A per-cell interface (the weather file carries ${vRow}); its
            # existence is checked by the workspace builder, not here.
            logger.debug("resource %s is per-cell (%s); skipped", resource.id, path_text)
            continue
        path = Path(path_text)
        if path.suffix.lower() not in (".csv", ".gz"):
            continue  # XML inputs carry no header to check
        if not path.exists():
            errors.append(
                f"resource {resource.id!r} reads {path}, which does not exist"
            )
            continue

        try:
            header = list(
                pd.read_csv(path, sep=resource.divider or None, engine="python", nrows=0).columns
            )
        except Exception as exc:  # noqa: BLE001 - report, do not abort the sweep
            errors.append(f"resource {resource.id!r} could not read {path.name}: {exc}")
            continue

        missing = resource.missing_from(header)
        if missing:
            warnings.append(
                f"resource {resource.id!r} declares column(s) {missing} that "
                f"{path.name} does not have (it has {header[:8]}"
                + (" ...)" if len(header) > 8 else ")")
            )
        else:
            logger.info(
                "resource %-16s OK  (%d columns from %s)",
                resource.id, len(resource.columns), path.name,
            )

    for message in warnings:
        logger.warning("%s", message)

    fatal = errors + (warnings if strict else [])
    if fatal:
        raise ValidationError(
            "The solution asks for data the workspace does not provide:\n  - "
            + "\n  - ".join(fatal)
        )
    return errors, warnings
