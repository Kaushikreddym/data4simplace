"""Tests for the reference parser and exporters."""

from __future__ import annotations

import numpy as np
import pandas as pd

from data4simplace.exporters.base_exporter import parse_reference_csv
from data4simplace.exporters.soil_export import SoilExporter, remap_depth_weighted
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


def test_remap_depth_weighted_overlap():
    # Two source layers 0-10 / 10-20 with values 10 and 20; a destination layer
    # 0-20 must be the thickness-weighted mean (15), and 5-15 also 15.
    values = np.array([[10.0], [20.0]])  # (n_src, 1)
    src = [(0.0, 10.0), (10.0, 20.0)]
    out = remap_depth_weighted(values, src, [(0.0, 20.0), (5.0, 15.0)])
    assert np.allclose(out[:, 0], [15.0, 15.0])
    # A destination layer beyond all source coverage is NaN.
    out2 = remap_depth_weighted(values, src, [(20.0, 30.0)])
    assert np.isnan(out2[0, 0])


def _cells_over_fixture(config, soil_dataset):
    grid = TargetGrid.from_config(config.grid)
    cells = grid.cell_table().iloc[:2].copy()
    cells["row"] = [0, 1]
    cells["col"] = [0, 1]
    cells["lat"] = soil_dataset["lat"].values
    cells["lon"] = soil_dataset["lon"].values
    return cells


def test_soil_exporter_writes_wide_file(tmp_path, config, soil_dataset):
    cells = _cells_over_fixture(config, soil_dataset)

    exporter = SoilExporter(config, reference_path=None)  # fallback wide schema
    out = exporter.export(soil_dataset, cells, tmp_path)
    assert out.is_file() and out.name == "soil.csv"

    written = pd.read_csv(out)
    # One row per cell, wide format with per-layer columns.
    assert len(written) == 2
    assert "location" in written.columns
    for col in ("clay_1", "sand_1", "bulkdensity_1", "carbon_1", "PH_1"):
        assert col in written.columns
    # Uniform fixture -> clay stays 20 % after depth remap.
    assert np.allclose(written["clay_1"], 20.0)
    assert set(written["location"]) == set(cells["SimplaceID"])


def test_soil_exporter_conforms_and_carries_constants(tmp_path, config, soil_dataset):
    # Reference with a derivable column (clay_1), a per-layer non-derivable
    # column (nitrate_1) and a single-value constant (soiltype).
    ref = tmp_path / "soil_ref.csv"
    ref.write_text("location,soiltype,clay_1,nitrate_1\n49612,AP5,7.7,202\n")

    cells = _cells_over_fixture(config, soil_dataset)
    exporter = SoilExporter(config, reference_path=ref)
    out = exporter.export(soil_dataset, cells, tmp_path)
    written = pd.read_csv(out)

    # Exactly the reference columns, in order.
    assert list(written.columns) == ["location", "soiltype", "clay_1", "nitrate_1"]
    # clay_1 overwritten from SoilGrids (20), constants carried from the template.
    assert np.allclose(written["clay_1"], 20.0)
    assert set(written["soiltype"]) == {"AP5"}
    assert set(written["nitrate_1"]) == {202}
