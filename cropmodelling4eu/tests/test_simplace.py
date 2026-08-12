"""Tests for the SIMPLACE runner.

SIMPLACE itself runs in a container, so what is tested here is everything
around it: the project and location tables, the solution parse and validation,
the line-range split, the container command, and the collector's mapping from
SIMPLACE's output columns into the shared run schema.
"""

from __future__ import annotations

import subprocess

import numpy as np
import pandas as pd
import pytest

from cropmodelling4eu.config import GridConfig, RunConfig
from cropmodelling4eu.simplace.collect import read_yearly, to_run_schema
from cropmodelling4eu.simplace.project import (
    ADMIN_PLACEHOLDER,
    build_location_frame,
    build_project_frame,
    line_ranges,
)
from cropmodelling4eu.simplace.run import build_command
from cropmodelling4eu.simplace.submit import id_column_index, write_run_scripts
from cropmodelling4eu.simplace.weather import BRANDENBURG, declared_factors
from cropmodelling4eu.simplace.solution import (
    Resource,
    ValidationError,
    read_solution,
    validate_resources,
)
from cropmodelling4eu.simplace.workspace import Workspace

from .conftest import TEST_CELLS, TEST_GRID

PROJECT_COLUMNS = [
    "projectid", "simulationid", "vColumn", "vRow", "vLocationID",
    "vlat", "vlon", "vNUTSID", "vSTATE_ID", "vSTATE_NAME",
]


# --------------------------------------------------------------------------- #
# Project and location tables
# --------------------------------------------------------------------------- #


def test_project_frame_carries_the_grid_position(grid):
    frame = build_project_frame(np.array(TEST_CELLS), grid, PROJECT_COLUMNS)

    assert list(frame.columns) == PROJECT_COLUMNS
    assert len(frame) == len(TEST_CELLS)
    assert frame["vLocationID"].tolist() == TEST_CELLS
    # vColumn/vRow build the weather filename, so they must be the grid's.
    assert frame.loc[0, "vRow"] == 0 and frame.loc[0, "vColumn"] == 0
    assert frame.loc[2, "vRow"] == 1  # id 12 on a 10-wide grid
    assert frame["vlat"].between(TEST_GRID["min_lat"], TEST_GRID["max_lat"]).all()


def test_administrative_columns_are_an_explicit_placeholder(grid):
    """A fabricated region code would travel into SIMPLACE's own output."""
    frame = build_project_frame(np.array(TEST_CELLS), grid, PROJECT_COLUMNS)
    for column in ("vNUTSID", "vSTATE_ID", "vSTATE_NAME"):
        assert (frame[column] == ADMIN_PLACEHOLDER).all()


def test_unknown_non_administrative_column_is_reported(grid, caplog):
    build_project_frame(np.array(TEST_CELLS), grid, [*PROJECT_COLUMNS, "vMystery"])
    assert "vMystery" in caplog.text
    assert "not administrative" in caplog.text


def test_project_frame_takes_site_attributes(grid):
    frame = build_project_frame(
        np.array(TEST_CELLS),
        grid,
        ["vLocationID", "vAltitude", "vSowingDOY"],
        altitude=np.array([10.0, 20.0, 300.0, 1200.0]),
        sowing_doy=np.array([280, 280, 295, 295]),
    )
    assert frame["vAltitude"].tolist() == [10.0, 20.0, 300.0, 1200.0]
    assert frame["vSowingDOY"].tolist() == [280, 280, 295, 295]


def test_rule_based_solution_gets_a_per_cell_sowing_window(grid, run_config):
    """The v2 SUSTAg solution sows on rules inside a per-cell window.

    It reads `vSowWindowStartDOY` from the project file, which is what turns
    the SAGE planting window into an input instead of a constant.
    """
    from cropmodelling4eu.export import resolve_export

    bundle = resolve_export(run_config)
    ids = bundle.ids
    frame = build_project_frame(
        ids,
        grid,
        ["vLocationId", "vSowWindowStartDOY", "vSowWindowLengthDays"],
        sowing_window=bundle.site.sowing_window(ids),
    )
    # The fixture's window is 270..300, so every cell starts at 270 for 30 days.
    assert (frame["vSowWindowStartDOY"] == 270).all()
    assert (frame["vSowWindowLengthDays"] == 30).all()
    # The window must not run past the year end, which the solution's
    # `DOY <= start + length` test cannot express.
    assert (frame["vSowWindowStartDOY"] + frame["vSowWindowLengthDays"] <= 365).all()


def test_sowing_window_falls_back_to_the_date_itself(run_config, export_dir):
    """Without a window, one is centred on the sowing date rather than dropped."""
    from cropmodelling4eu.export.site import read_site

    path = export_dir / "site" / "site.csv"
    frame = pd.read_csv(path)
    frame = frame.drop(columns=["sowing_start_doy", "sowing_end_doy"])
    frame.to_csv(path, index=False)

    site = read_site(export_dir)
    start, length = site.sowing_window(np.array(TEST_CELLS))
    # Sown 280/295 -> a symmetric default window around each.
    assert start.tolist() == [273, 273, 288, 288]
    assert (length == 14).all()


def test_project_constants_are_written_verbatim(grid):
    """Scenario keys the export has no opinion about come from config."""
    frame = build_project_frame(
        np.array(TEST_CELLS), grid,
        ["vLocationId", "vCrop", "vTrt", "vENZ"],
        constants={"vCrop": "WW", "vTrt": "basic", "vENZ": 4},
    )
    assert (frame["vCrop"] == "WW").all()
    assert (frame["vTrt"] == "basic").all()
    assert (frame["vENZ"] == 4).all()


def test_location_frame_is_generated_not_copied(grid):
    """Per-cell latitude drives day length; a copied file would flatten it."""
    frame = build_location_frame(
        np.array(TEST_CELLS), grid, altitude=np.array([10.0, 20.0, 300.0, 1200.0])
    )
    assert list(frame.columns) == ["location", "Latitude", "SunInclination", "Altitude"]
    assert frame["Latitude"].nunique() > 1
    assert (frame["SunInclination"] == -4.0).all()
    assert frame["Altitude"].tolist() == [10.0, 20.0, 300.0, 1200.0]


# --------------------------------------------------------------------------- #
# Line ranges
# --------------------------------------------------------------------------- #


def test_line_ranges_cover_every_line_once():
    ranges = line_ranges(10, 4)
    assert ranges == [(1, 4), (5, 8), (9, 10)]
    covered = [line for first, last in ranges for line in range(first, last + 1)]
    assert covered == list(range(1, 11))


def test_line_ranges_edge_cases():
    assert line_ranges(0, 5) == []
    assert line_ranges(3, 10) == [(1, 3)]
    with pytest.raises(ValueError):
        line_ranges(10, 0)


# --------------------------------------------------------------------------- #
# Solution parsing and validation
# --------------------------------------------------------------------------- #


def _write_solution(tmp_path, divider: str = ",") -> "tuple":
    """A minimal solution declaring one CSV resource, plus the CSV."""
    csv = tmp_path / "soil.csv"
    csv.write_text("location,clay_1,clay_2,depth_1,depth_2\n1,20,20,0.1,0.3\n")
    xml = tmp_path / "test.sol.xml"
    xml.write_text(
        "<?xml version='1.0' encoding='UTF-8'?>\n"
        "<solution>\n"
        "  <variables>\n"
        "    <var id='vIOPT' datatype='INT'>4</var>\n"
        "    <var id='startdate' datatype='DATE'>01.01.2044</var>\n"
        "  </variables>\n"
        "  <interfaces>\n"
        "    <interface id='soilfile' type='CSV'>\n"
        f"      <divider>{divider}</divider>\n"
        "      <filename>${_WORKDIR_}/soil.csv</filename>\n"
        "    </interface>\n"
        "  </interfaces>\n"
        "  <resource id='soil' interface='soilfile'>\n"
        "    <header>\n"
        "      <res id='location' datatype='INT'/>\n"
        "      <res id='clay' datatype='DOUBLEARRAY'/>\n"
        "      <res id='depth' datatype='DOUBLEARRAY'/>\n"
        "    </header>\n"
        "  </resource>\n"
        "</solution>\n"
    )
    return xml, csv


def test_array_resource_is_satisfied_by_either_encoding():
    """DOUBLEARRAY says the value is an array, not how the file encodes it."""
    resource = Resource(
        id="soil", interface="soilfile", filename="", divider=",",
        columns={"location": "INT", "clay": "DOUBLEARRAY"},
    )
    # Wide: clay_1..clay_6 satisfies `clay`.
    assert resource.missing_from(["location", "clay_1", "clay_2"]) == []
    # Long: a bare `clay` column satisfies it too.
    assert resource.missing_from(["location", "clay"]) == []
    # Neither is a genuine miss.
    assert resource.missing_from(["location", "sand_1"]) == ["clay"]


def test_validation_accepts_a_wide_file_for_an_array_resource(tmp_path):
    xml, _ = _write_solution(tmp_path)
    errors, warnings = validate_resources(
        read_solution(xml), {"_WORKDIR_": str(tmp_path)}
    )
    assert errors == [] and warnings == []


def test_missing_file_is_an_error_whatever_strict_says(tmp_path):
    xml, csv = _write_solution(tmp_path)
    csv.unlink()
    with pytest.raises(ValidationError, match="does not exist"):
        validate_resources(read_solution(xml), {"_WORKDIR_": str(tmp_path)})


def test_missing_column_warns_but_does_not_block(tmp_path, caplog):
    xml, csv = _write_solution(tmp_path)
    csv.write_text("location,sand_1\n1,40\n")

    errors, warnings = validate_resources(
        read_solution(xml), {"_WORKDIR_": str(tmp_path)}
    )
    assert errors == []
    assert any("clay" in w for w in warnings)
    assert "declares column(s)" in caplog.text

    # ... unless strict is asked for.
    with pytest.raises(ValidationError, match="clay"):
        validate_resources(read_solution(xml), {"_WORKDIR_": str(tmp_path)}, strict=True)


def test_per_cell_interfaces_are_not_validated_as_files(tmp_path):
    """A ${vRow} filename is a template, not a path to check."""
    xml = tmp_path / "wx.sol.xml"
    xml.write_text(
        "<?xml version='1.0'?><solution><interfaces>"
        "<interface id='weatherfile' type='CSV'><divider/>"
        "<filename>${_DATADIR_}/${vRow}/x_C${vColumn}R${vRow}.csv.gz</filename>"
        "</interface></interfaces>"
        "<resource id='w' interface='weatherfile'><header>"
        "<res id='Date' datatype='DATE'/></header></resource></solution>"
    )
    errors, warnings = validate_resources(read_solution(xml), {"_DATADIR_": "/data"})
    assert errors == [] and warnings == []


def test_solution_variables_are_overridden_in_place(tmp_path):
    xml, _ = _write_solution(tmp_path)
    document = read_solution(xml)
    assert document.variables()["vIOPT"] == "4"

    changed = document.set_variables({"vIOPT": 3, "startdate": "01.01.1999"})
    assert sorted(changed) == ["startdate", "vIOPT"]
    assert document.variables()["vIOPT"] == "3"


def test_unknown_variable_is_not_added_silently(tmp_path, caplog):
    xml, _ = _write_solution(tmp_path)
    document = read_solution(xml)
    assert document.set_variables({"vNotDeclared": 1}) == []
    assert "declares no variable" in caplog.text


# --------------------------------------------------------------------------- #
# The container command
# --------------------------------------------------------------------------- #


def _workspace(tmp_path) -> Workspace:
    work = tmp_path / "workspace"
    (work / "solution").mkdir(parents=True)
    (work / "project").mkdir(parents=True)
    return Workspace(
        root=tmp_path,
        work_dir=work,
        data_dir=tmp_path / "weather",
        out_dir=tmp_path / "out",
        solution=work / "solution" / "s.sol.xml",
        project=work / "project" / "p.proj.xml",
        n_lines=4,
        export_dir=tmp_path / "export",
    )


def test_command_uses_container_paths(tmp_path, run_config):
    workspace = _workspace(tmp_path)
    command = build_command(workspace, run_config, 1, 4)

    assert command[0] == "singularity"
    assert "-s=/simplace/SIMPLACE_WORK/solution/s.sol.xml" in command
    assert "-p=/simplace/SIMPLACE_WORK/project/p.proj.xml" in command
    assert "-l=1-4" in command
    assert "-loglevel=ERROR" in command
    # --debug keeps SIMPLACE's own logging.
    assert "-loglevel=ERROR" not in build_command(workspace, run_config, 1, 4, debug=True)


def test_binds_include_the_export_read_only(tmp_path, run_config):
    """Without it, every symlink into the export dangles inside the container."""
    workspace = _workspace(tmp_path)
    binds = workspace.binds()
    assert f"{workspace.export_dir}:{workspace.export_dir}:ro" in binds
    assert f"{workspace.work_dir}:/simplace/SIMPLACE_WORK" in binds
    assert f"{workspace.data_dir}:/data" in binds


# --------------------------------------------------------------------------- #
# Collection
# --------------------------------------------------------------------------- #


def _yearly(tmp_path, simplace_id: int) -> None:
    """A SIMPLACE yearly output: semicolon-separated, no location column."""
    directory = tmp_path / "yearly"
    directory.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(
        {
            "projectid": [simplace_id, simplace_id],
            "Year": [2000, 2001],
            "DevStage": [2.02, 2.00],
            "PlantingDOY": [250, 250],
            "MaturityDOY": [228, 220],
            "Yield_t_ha": [6.84, 7.50],
            "AGBiomass_t_ha": [17.15, 19.16],
            "maxLAI": [3.60, 3.92],
            "TRANRF": [0.53, 0.88],
            "NNI": [0.95, 0.91],
            "inputChemN_kg_ha": [180.0, 180.0],
        }
    ).to_csv(directory / f"{simplace_id}_yearly.csv", sep=";", index=False)


def test_read_yearly_sniffs_the_separator(tmp_path):
    for simplace_id in TEST_CELLS[:2]:
        _yearly(tmp_path, simplace_id)
    frame = read_yearly(tmp_path)
    assert len(frame) == 4
    assert set(frame["SimplaceID"]) == set(TEST_CELLS[:2])
    assert "Yield_t_ha" in frame.columns


def test_read_yearly_reports_an_empty_run(tmp_path):
    with pytest.raises(FileNotFoundError, match="No .*_yearly.csv"):
        read_yearly(tmp_path)


def test_collect_maps_into_the_shared_schema(tmp_path, grid):
    _yearly(tmp_path, TEST_CELLS[0])
    frame = to_run_schema(read_yearly(tmp_path), grid)

    # The torchcrop schema, so one evaluation serves both models.
    for column in ("SimplaceID", "year", "yield_g_m2", "biomass_g_m2", "max_lai",
                   "lon", "lat", "days_to_maturity", "model"):
        assert column in frame.columns
    assert (frame["model"] == "simplace").all()
    # t/ha -> g/m2 is a factor of 100.
    assert frame["yield_g_m2"].iloc[0] == pytest.approx(684.0)
    assert frame["biomass_g_m2"].iloc[0] == pytest.approx(1715.0)
    # kg/ha -> g/m2 is a factor of 10.
    assert frame["n_applied_g_m2"].iloc[0] == pytest.approx(18.0)


def test_days_to_maturity_wraps_a_winter_season(tmp_path, grid):
    _yearly(tmp_path, TEST_CELLS[0])
    frame = to_run_schema(read_yearly(tmp_path), grid)
    # Sown DOY 250, mature DOY 228 the next year -> 343 days, not -22.
    assert frame["days_to_maturity"].iloc[0] == pytest.approx(343.0)
    assert (frame["days_to_maturity"] > 0).all()


def test_absent_output_columns_are_reported(tmp_path, grid, caplog):
    directory = tmp_path / "yearly"
    directory.mkdir(parents=True)
    pd.DataFrame({"Year": [2000], "Yield_t_ha": [6.0]}).to_csv(
        directory / f"{TEST_CELLS[0]}_yearly.csv", sep=";", index=False
    )
    to_run_schema(read_yearly(tmp_path), grid)
    assert "carries no" in caplog.text
    assert "max_lai" in caplog.text


# --------------------------------------------------------------------------- #
# The built directory and its run scripts
# --------------------------------------------------------------------------- #


def _built(tmp_path, project_csv: str = "project_wheat.csv") -> Workspace:
    """A workspace whose project CSV exists, which is all the scripts need."""
    root = tmp_path / "run"
    work = root / "workspace"
    (work / "project").mkdir(parents=True)
    csv_path = work / "project" / project_csv
    build_project_frame(
        np.array(TEST_CELLS), GridConfig(**TEST_GRID), PROJECT_COLUMNS
    ).to_csv(csv_path, index=False)
    return Workspace(
        root=root,
        work_dir=work,
        data_dir=root / "weather",
        out_dir=root / "out",
        solution=work / "solution" / "s.sol.xml",
        project=work / "project" / "p.proj.xml",
        n_lines=len(TEST_CELLS),
        export_dir=tmp_path / "export",
        project_csv=csv_path,
        template_dir=tmp_path / "template",
    )


def test_run_scripts_land_in_the_built_directory(tmp_path, run_config):
    """Everything needed to run is in the output directory, not in the package."""
    paths = write_run_scripts(_built(tmp_path), run_config)

    assert set(paths) == {"env", "task", "submit"}
    for path in paths.values():
        assert path.parent == tmp_path / "run"
        assert path.is_file()
    # Both scripts must be executable, or `./submit.sh` fails for the one
    # reason a user would not think to check.
    for key in ("task", "submit"):
        assert paths[key].stat().st_mode & 0o111


def test_generated_scripts_are_valid_bash(tmp_path, run_config):
    paths = write_run_scripts(_built(tmp_path), run_config)
    for path in paths.values():
        result = subprocess.run(["bash", "-n", str(path)], capture_output=True)
        assert result.returncode == 0, f"{path.name}: {result.stderr.decode()}"


def test_binds_are_relative_to_the_root_so_the_folder_can_be_moved(
    tmp_path, run_config
):
    """A copied run directory must still bind its own workspace, not the original."""
    paths = write_run_scripts(_built(tmp_path), run_config)
    env = paths["env"].read_text()
    assert "${SP_WORK_DIR}:/simplace/SIMPLACE_WORK" in env
    assert "${SP_ROOT}/weather:/data" in env
    # The export is external, so it stays absolute and read-only.
    assert f"{tmp_path / 'export'}:{tmp_path / 'export'}:ro" in env


def test_id_column_is_resolved_not_assumed(tmp_path):
    workspace = _built(tmp_path)
    # vLocationID is the 5th column of PROJECT_COLUMNS, and awk counts from 1.
    assert id_column_index(workspace.project_csv) == 5


def test_id_column_absence_is_an_error(tmp_path):
    """Better than defaulting to column 1 and checking the wrong field."""
    path = tmp_path / "project.csv"
    pd.DataFrame({"a": [1], "b": [2]}).to_csv(path, index=False)
    with pytest.raises(KeyError, match="no cell-id column"):
        id_column_index(path)


def test_scripts_refuse_a_workspace_without_a_project(tmp_path, run_config):
    workspace = _built(tmp_path)
    workspace.project_csv = None
    with pytest.raises(ValueError, match="project_csv"):
        write_run_scripts(workspace, run_config)


def _source(env_path, snippet: str) -> str:
    """Run a bash snippet with the generated run.env sourced."""
    result = subprocess.run(
        ["bash", "-c", f'set -uo pipefail; source "{env_path}"; {snippet}'],
        capture_output=True, text=True,
    )
    assert result.returncode == 0, result.stderr
    return result.stdout.strip()


def test_task_ranges_agree_with_the_python_split(tmp_path, run_config):
    """The shell and `line_ranges` must divide the work identically.

    They are two implementations of one rule, and a disagreement would leave a
    silent gap between what an array runs and what a retry believes it ran.
    """
    config = run_config.model_copy(
        update={"simplace": run_config.simplace.model_copy(
            update={"lines_per_task": 3})}
    )
    workspace = _built(tmp_path)
    workspace.n_lines = 10
    paths = write_run_scripts(workspace, config)

    expected = line_ranges(10, 3)
    for task, (first, last) in enumerate(expected):
        assert _source(paths["env"], f"sp_task_range {task}") == f"{first} {last}"
    assert _source(paths["env"], "echo ${SP_N_TASKS}") == str(len(expected))


def test_retry_set_is_derived_from_the_stamps(tmp_path, run_config):
    """A task is outstanding until it has exited zero — not until every cell
    has an output file. A cell that never reaches maturity produces none, and
    would otherwise be resubmitted for ever."""
    config = run_config.model_copy(
        update={"simplace": run_config.simplace.model_copy(
            update={"lines_per_task": 2})}
    )
    paths = write_run_scripts(_built(tmp_path), config)  # 4 cells -> 2 tasks

    # Nothing run: both tasks outstanding, no cell has output.
    state = _source(paths["env"], "sp_task_state")
    assert [line.split() for line in state.splitlines()] == [
        ["0", "1", "2", "2", "2", "0"],
        ["1", "3", "4", "2", "2", "0"],
    ]

    # Task 0 exits zero having written only one of its two cells.
    (tmp_path / "run" / "state").mkdir(parents=True)
    (tmp_path / "run" / "state" / "task_1-2.done").write_text("now")
    yearly = tmp_path / "run" / "out" / "yearly"
    yearly.mkdir(parents=True)
    (yearly / f"{TEST_CELLS[0]}_yearly.csv").write_text("Year;Yield_t_ha\n2000;6\n")

    state = _source(paths["env"], "sp_task_state")
    rows = [line.split() for line in state.splitlines()]
    # Task 0: one cell missing, but done -> not retried.
    assert rows[0] == ["0", "1", "2", "1", "2", "1"]
    assert rows[1][-1] == "0"


def test_a_stamp_is_keyed_on_the_line_range_not_the_task_index(tmp_path, run_config):
    """Re-splitting the work must invalidate the stamps rather than reuse them
    for a different set of lines."""
    workspace = _built(tmp_path)
    (tmp_path / "run" / "state").mkdir(parents=True)
    (tmp_path / "run" / "state" / "task_1-2.done").write_text("now")

    two = write_run_scripts(workspace, run_config.model_copy(
        update={"simplace": run_config.simplace.model_copy(
            update={"lines_per_task": 2})}))
    assert _source(two["env"], 'sp_task_done 0 && echo yes || echo no') == "yes"

    # Same task index, different lines: the stamp must not count.
    four = write_run_scripts(workspace, run_config.model_copy(
        update={"simplace": run_config.simplace.model_copy(
            update={"lines_per_task": 4})}))
    assert _source(four["env"], 'sp_task_done 0 && echo yes || echo no') == "no"


# --------------------------------------------------------------------------- #
# Unit reconciliation inside the solution's own SQL transform
# --------------------------------------------------------------------------- #

_TRANSFORM_SOLUTION = """<?xml version="1.0"?>
<solution>
  <interfaces><interface id="weatherfile" type="CSV"><filename>w.csv</filename></interface></interfaces>
  <resources>
    <resource id="weather" interface="weatherfile" frequence="DAILY"><header>
      <res id="CURRENTDATE" datatype="DATE"/>
      <res id="Rain" datatype="DOUBLE"/>
      <res id="AirTemperatureMin" datatype="DOUBLE"/>
      <res id="AirTemperatureMean" datatype="DOUBLE"/>
      <res id="AirTemperatureMax" datatype="DOUBLE"/>
      <res id="Irradiation" datatype="DOUBLE"/>
    </header></resource>
  </resources>
  <transformers>
    <transform id="weather_transform" frequence="DAILY">
      <input id="statement">SELECT CURRENTDATE, Rain AS RainAndIrr,
        Irradiation/1000 as SradiationMJ, AirTemperatureMax FROM weather</input>
    </transform>
  </transformers>
</solution>
"""


@pytest.fixture()
def transform_solution(tmp_path):
    path = tmp_path / "t.sol.xml"
    path.write_text(_TRANSFORM_SOLUTION)
    return read_solution(path)


def test_contract_factors_map_onto_the_declared_ids(transform_solution):
    """The export calls it `Radiation`; the SQL sees `Irradiation`.

    Only position relates the two — which is exactly how SIMPLACE binds a CSV
    resource, so the zip both translates the key and asserts the alignment.
    """
    weather = next(r for r in transform_solution.resources if r.id == "weather")
    assert declared_factors(list(weather.columns), BRANDENBURG) == {"Irradiation": 86.4}


def test_a_contract_wider_than_the_resource_is_refused():
    with pytest.raises(ValueError, match="declares only 2"):
        declared_factors(["CURRENTDATE", "Rain"], BRANDENBURG)


def test_scaling_rewrites_the_statement_in_place(transform_solution):
    transform_solution.scale_transform("weather_transform", {"Irradiation": 86.4})
    statement = transform_solution.root.find(
        ".//transform[@id='weather_transform']/input[@id='statement']"
    ).text
    # W/m2 x 86.4 = kJ/m2/d, and the solution's own /1000 then gives MJ/m2/d.
    assert "(Irradiation*86.4)/1000 as SradiationMJ" in statement
    # Untouched columns stay untouched, and the alias is not a column reference.
    assert "Rain AS RainAndIrr" in statement
    assert "SradiationMJ*" not in statement


def test_scaling_an_absent_column_is_an_error(transform_solution):
    """Silently scaling nothing would leave the units wrong and the run plausible."""
    with pytest.raises(KeyError, match="does not reference 'Windspeed'"):
        transform_solution.scale_transform("weather_transform", {"Windspeed": 86.4})


def test_scaling_an_absent_transform_is_an_error(transform_solution):
    with pytest.raises(KeyError, match="no transform 'nope'"):
        transform_solution.scale_transform("nope", {"Irradiation": 86.4})
