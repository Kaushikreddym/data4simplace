"""Tests for the reference parser and exporters."""

from __future__ import annotations

import pandas as pd

from data4simplace.exporters.base_exporter import parse_reference_csv
from data4simplace.exporters.soil_export import SoilExporter
from data4simplace.grid import TargetGrid


def test_parse_reference_detects_semicolon_and_missing(tmp_path):
    ref = tmp_path / "ref.csv"
    ref.write_text(
        "SimplaceID;DEPTH_TOP;DEPTH_BOTTOM;CLAY_0-5cm;CLAY_5-15cm\n"
        "1;0;5;20.0;-99\n"
        "1;5;15;-99;22.0\n"
    )
    spec = parse_reference_csv(ref)
    assert spec.delimiter == ";"
    assert spec.missing_value == "-99"
    assert spec.columns[0] == "SimplaceID"
    assert spec.depths == ["0-5cm", "5-15cm"]


def test_soil_exporter_writes_conformant_file(tmp_path, config, soil_dataset):
    grid = TargetGrid.from_config(config.grid)
    cells = grid.cell_table()
    # restrict to the two cells present in the fixture dataset
    cells = cells.iloc[:2].copy()
    cells["lat"] = soil_dataset["lat"].values
    cells["lon"] = soil_dataset["lon"].values

    exporter = SoilExporter(config, reference_path=None)  # uses fallback schema
    out = exporter.export(soil_dataset, cells, tmp_path)
    assert out.is_file()

    written = pd.read_csv(out)
    for col in ("SimplaceID", "DEPTH_TOP", "DEPTH_BOTTOM", "CLAY", "SAND"):
        assert col in written.columns
    # 2 cells x 2 depths
    assert len(written) == 4
    assert set(written["DEPTH_TOP"]) == {0, 5}


def test_soil_exporter_respects_reference_columns(tmp_path, config, soil_dataset):
    ref = tmp_path / "soil_ref.csv"
    ref.write_text("SimplaceID,DEPTH_TOP,DEPTH_BOTTOM,CLAY\n1,0,5,20\n")

    grid = TargetGrid.from_config(config.grid)
    cells = grid.cell_table().iloc[:2].copy()
    cells["lat"] = soil_dataset["lat"].values
    cells["lon"] = soil_dataset["lon"].values

    exporter = SoilExporter(config, reference_path=ref)
    out = exporter.export(soil_dataset, cells, tmp_path)
    written = pd.read_csv(out)
    # exactly the reference columns, in order — SAND/SILT dropped
    assert list(written.columns) == ["SimplaceID", "DEPTH_TOP", "DEPTH_BOTTOM", "CLAY"]
