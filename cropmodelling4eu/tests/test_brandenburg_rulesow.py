"""The ported Brandenburg solution must sow by rule, and differ by nothing else.

The whole value of this template is that it is the stock solution plus the
sowing rules: if anything else drifts, a smoke test against it stops being
attributable to the sowing change. So the diff itself is the assertion.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from cropmodelling4eu.simplace.project import project_columns
from cropmodelling4eu.simplace.solution import read_solution

TEMPLATES = Path(__file__).resolve().parents[1] / "templates/brandenburg"
SOLUTION = TEMPLATES / "solution/brandenburg_rulesow.sol.xml"
PROJECT = TEMPLATES / "project/brandenburg_rulesow.proj.xml"

#: What this template was derived from. Absent on a machine that is not the
#: cluster, which is why every test touching it skips rather than fails.
STOCK = Path(
    "/data01/FDS/muduchuru/codes/SIMPLACE/Brandenburg_1KM_winter_wheat"
    "/solution/solution.sol.xml"
)

#: The only lines the port may remove from the stock solution.
REMOVED = {
    "=== Purpose === Runs the cefit CKA solution ===",
    '<var id="vIDPL" datatype="INT" unit="-">250</var>',
    '<action rule="${CURRENT.DOY}==${vIDPL}">',
    '<input id="cIDPL" source="vIDPL" />',
    '<out id="PlantingDOY" rule="vIDPL" datatype="INT" unit="-" />',
    '<description title="USL" author="Sabine, Andi E." qualitylevel="I">WIKI_START',
}


def test_templates_parse():
    assert {r.id for r in read_solution(SOLUTION).resources} >= {"weather", "soil"}


def test_sowing_is_rule_based_not_a_fixed_day():
    text = SOLUTION.read_text()
    assert '<action rule="${SowingRule.DoSow}">' in text
    # The prose in <description> names the old trigger, so match the element.
    assert '<action rule="${CURRENT.DOY}==${vIDPL}">' not in text
    for component in ("SowingWindowClimate", "SowingRule", "SowingDate"):
        assert f'simcomponent id="{component}"' in text


def test_the_reported_sowing_date_is_the_realized_one():
    """`PlantingDOY` must be a result, not `vIDPL` echoed back."""
    text = SOLUTION.read_text()
    assert '<out id="PlantingDOY" rule="SowingDate.SowingDOY"' in text
    assert '<input id="cIDPL" source="SowingDate.SowingDOY" />' in text
    assert 'rule="vIDPL"' not in text
    # Without this a deadline-forced date is indistinguishable from a rule-driven one.
    assert '<out id="ForcedSow" rule="SowingRule.SeasonSowForced"' in text


def test_the_rules_read_this_solutions_own_inputs():
    """A rule pointing at the SUSTAg names would silently never fire."""
    text = SOLUTION.read_text()
    assert "${weather.AirTemperatureMin}" in text and "${weather.Rain}" in text
    assert "${weather.tmin}" not in text and "${weather.rain}" not in text
    assert "${CropManagement.WithCrop}" not in text
    assert "${DefaultManagement.WithCrop}" in text


def test_the_window_is_a_per_cell_project_column():
    """The project file is the fallback when the schedule carries no window."""
    columns = project_columns(read_solution(PROJECT))
    assert {"vSowWindowStartDOY", "vSowWindowLengthDays"} <= set(columns)


def test_the_window_is_read_from_the_fertilizer_schedule():
    """SAGE's start/end pair reaches the rule through the management file."""
    document = read_solution(SOLUTION)
    resource = next(r for r in document.resources if r.id == "sowwindow")
    # Positional binding: every column of the file up to the pair, in order.
    assert list(resource.columns) == [
        "location", "FertilizerScenario", "crop", "Event", "vType", "DVS",
        "Amount", "vIRR", "vSowWindowStartDOY", "vSowWindowEndDOY",
    ]
    interfaces = {
        i.get("id"): i.findtext("filename")
        for i in document.root.findall(".//interfaces/interface")
    }
    # The same file the schedule is read from, not a second copy to keep in step.
    assert interfaces["sowwindowfile"] == interfaces["managementfile2"]


def test_the_schedules_window_wins_and_the_project_file_is_the_fallback():
    text = SOLUTION.read_text()
    assert (
        'rule="if(check:isNull(${sowwindow.vSowWindowStartDOY})) '
        "{ ${vSowWindowStartDOY} } else { ${sowwindow.vSowWindowStartDOY} }\"" in text
    )
    assert (
        'rule="if(check:isNull(${sowwindow.vSowWindowEndDOY})) '
        "{ ${vSowWindowStartDOY} + ${vSowWindowLengthDays} } else "
        '{ ${sowwindow.vSowWindowEndDOY} }"' in text
    )


def test_every_rule_tests_the_resolved_window_not_the_raw_variable():
    """A rule left on ${vSowWindowStartDOY} would ignore the cell's own window."""
    text = SOLUTION.read_text()
    for stale in (
        "${CURRENT.DOY} == ${vSowWindowStartDOY}",
        "${CURRENT.DOY} ge ${vSowWindowStartDOY}",
        "${CURRENT.DOY} le (${vSowWindowStartDOY} + ${vSowWindowLengthDays})",
        "${CURRENT.DOY} == (${vSowWindowStartDOY} + ${vSowWindowLengthDays})",
    ):
        assert stale not in text, stale
    assert "${SowingWindowDates.WindowStart}" in text
    assert "${SowingWindowDates.WindowEnd}" in text


def test_the_window_actually_used_is_reported():
    """With two possible sources, a sowing date is unreadable without them."""
    text = SOLUTION.read_text()
    assert '<out id="SowWindowStart" rule="SowingWindowDates.WindowStart"' in text
    assert '<out id="SowWindowEnd" rule="SowingWindowDates.WindowEnd"' in text


def test_the_window_cannot_wrap_the_year_end():
    """The solution tests `DOY <= start + length`, which has no wrap."""
    variables = read_solution(SOLUTION).variables()
    start = int(variables["vSowWindowStartDOY"])
    length = int(variables["vSowWindowLengthDays"])
    assert start + length <= 365


@pytest.mark.skipif(not STOCK.is_file(), reason="the stock template is not on this host")
def test_nothing_but_sowing_was_changed():
    """Every removed line must be one of the five the port is allowed to remove."""
    stock = [line.strip() for line in STOCK.read_text().splitlines()]
    ported = {line.strip() for line in SOLUTION.read_text().splitlines()}
    removed = {line for line in stock if line and line not in ported}
    assert removed == REMOVED, f"unexpected removals: {removed - REMOVED}"
