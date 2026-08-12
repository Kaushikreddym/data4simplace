"""The sustag_v2 soil dialect must satisfy the v2 solution's declaration."""

from __future__ import annotations

from pathlib import Path

import pytest

from cropmodelling4eu.simplace.solution import read_solution

pytest.importorskip("data4simplace")
from data4simplace.exporters.layout import SOIL_DIALECTS  # noqa: E402

SOLUTION = (
    Path(__file__).resolve().parents[1]
    / "templates/sustag/solution/EU_SUSTAg_macsurClimateRO_initialize_v2.sol.xml"
)

#: Columns the solution declares that SoilGrids cannot derive. They come from
#: `soil.long_constants` (or a long reference's own first row), never guessed.
NON_DERIVABLE = {
    "alfa", "n", "ksat", "macroporevolume", "dampingdepth",
    "drainage_rate", "deltatheta", "maxRootingDepth", "Soiltype",
}


def _soil_resource():
    return next(r for r in read_solution(SOLUTION).resources if r.id == "soil")


def test_solution_template_is_present_and_parses():
    assert SOLUTION.is_file()
    document = read_solution(SOLUTION)
    assert {r.id for r in document.resources} >= {"soil", "fertilizer", "weather"}


def test_v2_dialect_writes_every_derivable_column():
    """No declared, derivable column may be missing from the dialect."""
    soil = _soil_resource()
    dialect = SOIL_DIALECTS["sustag_v2"]

    missing = set(soil.missing_from(dialect.column_names))
    assert missing == NON_DERIVABLE, f"unexpectedly unwritten: {missing - NON_DERIVABLE}"


def test_v2_dialect_writes_nothing_the_solution_does_not_read():
    soil = _soil_resource()
    dialect = SOIL_DIALECTS["sustag_v2"]
    extra = [c for c in dialect.column_names if c not in soil.columns]
    assert extra == []


def test_v2_dialect_is_selected_from_the_solutions_own_names():
    """Given the solution's column list, the right dialect must win."""
    from data4simplace.exporters.layout import select_dialect

    soil = _soil_resource()
    chosen = select_dialect(list(soil.columns), SOIL_DIALECTS, kind="soil")
    assert chosen.name == "sustag_v2"


def test_solution_takes_the_sowing_window_from_the_project_file():
    """The v2 solution is rule-based; its window start is a per-cell input."""
    document = read_solution(SOLUTION)
    variables = document.variables()
    assert "vSowWindowStartDOY" in variables
    assert "vSowWindowLengthDays" in variables
    # vIDPL survives as a legacy default but no longer drives phenology.
    text = SOLUTION.read_text()
    assert 'source="SowingDate.SowingDOY"' in text
