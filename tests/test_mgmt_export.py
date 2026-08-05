"""Tests for the SIMPLACE fertilizer schedule exporter."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest
import xarray as xr

from data4simplace.config import PipelineConfig
from data4simplace.exporters.mgmt_export import ManagementExporter
from data4simplace.grid import TargetGrid
from data4simplace.npk.composition import K2O_TO_K, P2O5_TO_P
from tests.test_npk import COMPOSITION_XML

# The Brandenburg winter-wheat scenario: one compound PK, then three KAS top
# dressings splitting the N 50/25/25.
REFERENCE_CSV = (
    "location,FertilizerScenario,crop,Event,vType,DVS,Amount\n"
    "49612,2,winter_wheat,1,PK,0.001,40\n"
    "49612,2,winter_wheat,2,KAS,0.25,32\n"
    "49612,2,winter_wheat,3,KAS,0.4,16\n"
    "49612,2,winter_wheat,4,KAS,0.9,16\n"
    "49613,2,winter_wheat,1,PK,0.001,40\n"
    "49613,2,winter_wheat,2,KAS,0.25,32\n"
    "49613,2,winter_wheat,3,KAS,0.4,16\n"
    "49613,2,winter_wheat,4,KAS,0.9,16\n"
)


@pytest.fixture()
def reference_dir(tmp_path):
    """A SIMPLACE ``data/management`` folder with the schedule and compositions."""
    folder = tmp_path / "management"
    folder.mkdir()
    (folder / "fertilizer_winter_wheat.csv").write_text(REFERENCE_CSV)
    (folder / "fertilizer_composition.xml").write_text(COMPOSITION_XML)
    return folder


@pytest.fixture()
def mgmt_config(tmp_path):
    """A config on a 2x2 target grid with the management export enabled."""

    def _build(**npk):
        return PipelineConfig.model_validate(
            {
                "flags": {
                    "run_npk_processing": True,
                    "export_simplace_management": True,
                },
                "grid": {
                    "resolution_deg": 0.1,
                    "min_lon": 11.2,
                    "max_lon": 11.4,
                    "min_lat": 52.1,
                    "max_lat": 52.3,
                },
                "time": {"start": "1979-01-01", "end": "1979-01-03"},
                "paths": {
                    "mswx_root": str(tmp_path),
                    "npk_root": str(tmp_path),
                    "output_dir": str(tmp_path / "out"),
                },
                "npk": npk,
            }
        )

    return _build


@pytest.fixture()
def npk_dataset(mgmt_config):
    """N / P2O5 / K2O rates (kg/ha) on the 2x2 target grid, one cell missing."""
    grid = TargetGrid.from_config(mgmt_config().grid)
    coords = {"lat": grid.lat_centers, "lon": grid.lon_centers}
    return xr.Dataset(
        {
            "N": (("lat", "lon"), np.array([[200.0, 100.0], [150.0, np.nan]])),
            "P2O5": (("lat", "lon"), np.array([[40.0, 20.0], [30.0, np.nan]])),
            "K2O": (("lat", "lon"), np.array([[60.0, 30.0], [45.0, np.nan]])),
        },
        coords=coords,
    )


def _cells(config):
    return TargetGrid.from_config(config.grid).cell_table()


# --------------------------------------------------------------------------- #
# Template
# --------------------------------------------------------------------------- #
def test_template_splits_the_compound_pk_into_straight_carriers(
    mgmt_config, reference_dir
):
    exporter = ManagementExporter(
        mgmt_config(), reference_dir / "fertilizer_winter_wheat.csv"
    )
    template = exporter.template()

    assert [(e.vtype, e.dvs, e.nutrient) for e in template.events] == [
        ("P", 0.001, "P"),
        ("K", 0.001, "K"),
        ("KAS", 0.25, "N"),
        ("KAS", 0.4, "N"),
        ("KAS", 0.9, "N"),
    ]
    # The reference's 32/16/16 g of KAS is a 50/25/25 split of the N.
    assert [e.share for e in template.events if e.nutrient == "N"] == [0.5, 0.25, 0.25]
    assert template.scenario == 2


def test_template_n_split_weights_by_delivered_nutrient_not_amount(
    mgmt_config, reference_dir
):
    # 20 g of Urea (46 % N) delivers as much N as 34.07 g of KAS (27 %), so a
    # split on raw amounts would be wrong; weighting by nutrient gives 50/50.
    (reference_dir / "fertilizer_winter_wheat.csv").write_text(
        "location,FertilizerScenario,crop,Event,vType,DVS,Amount\n"
        "1,2,winter_wheat,1,Urea,0.25,20\n"
        "1,2,winter_wheat,2,KAS,0.5,34.074\n"
    )
    exporter = ManagementExporter(
        mgmt_config(), reference_dir / "fertilizer_winter_wheat.csv"
    )
    shares = [e.share for e in exporter.template().events if e.nutrient == "N"]
    assert shares == pytest.approx([0.5, 0.5], abs=1e-4)


def test_template_honours_a_configured_split_and_carrier(mgmt_config, reference_dir):
    config = mgmt_config(n_fertilizer="Urea", n_split=[0.6, 0.2, 0.2])
    events = ManagementExporter(
        config, reference_dir / "fertilizer_winter_wheat.csv"
    ).template().events
    n_events = [e for e in events if e.nutrient == "N"]
    assert [e.vtype for e in n_events] == ["Urea"] * 3
    assert [e.share for e in n_events] == pytest.approx([0.6, 0.2, 0.2])
    assert all(e.content == pytest.approx(0.46) for e in n_events)


def test_template_rejects_a_mismatched_split(mgmt_config, reference_dir):
    config = mgmt_config(n_split=[0.5, 0.5])  # the reference has three N events
    exporter = ManagementExporter(config, reference_dir / "fertilizer_winter_wheat.csv")
    with pytest.raises(ValueError, match="npk.n_split has 2 entries"):
        exporter.template()


def test_template_rejects_a_carrier_without_the_nutrient(mgmt_config, reference_dir):
    config = mgmt_config(n_fertilizer="P")  # a phosphate carries no N
    exporter = ManagementExporter(config, reference_dir / "fertilizer_winter_wheat.csv")
    with pytest.raises(ValueError, match="carries no mineral N"):
        exporter.template()


def test_template_rejects_an_undeclared_reference_type(mgmt_config, reference_dir):
    (reference_dir / "fertilizer_winter_wheat.csv").write_text(
        "location,FertilizerScenario,crop,Event,vType,DVS,Amount\n"
        "1,2,winter_wheat,1,Unobtanium,0.25,32\n"
    )
    exporter = ManagementExporter(
        mgmt_config(), reference_dir / "fertilizer_winter_wheat.csv"
    )
    with pytest.raises(ValueError, match="Unobtanium"):
        exporter.template()


# --------------------------------------------------------------------------- #
# Frame construction
# --------------------------------------------------------------------------- #
def test_amounts_deliver_exactly_the_cell_rates(mgmt_config, reference_dir, npk_dataset):
    config = mgmt_config()
    exporter = ManagementExporter(config, reference_dir / "fertilizer_winter_wheat.csv")
    frame = exporter.build_frame(npk_dataset, _cells(config))

    cell = frame[frame["location"] == 1]  # the (52.25, 11.25) cell: 200/40/60
    # N: sum of product x content, back to g N / m^2 (200 kg/ha = 20 g/m^2).
    n_delivered = (cell.loc[cell["vType"] == "KAS", "Amount"] * 0.27).sum()
    assert n_delivered == pytest.approx(20.0, rel=1e-3)
    # P and K land on the elemental equivalent of the oxide rate.
    p_delivered = (cell.loc[cell["vType"] == "P", "Amount"] * 0.4364).sum()
    assert p_delivered == pytest.approx(40.0 * 0.1 * P2O5_TO_P, rel=1e-3)
    k_delivered = (cell.loc[cell["vType"] == "K", "Amount"] * 0.8302).sum()
    assert k_delivered == pytest.approx(60.0 * 0.1 * K2O_TO_K, rel=1e-3)


def test_schedule_shape_and_columns(mgmt_config, reference_dir, npk_dataset):
    config = mgmt_config()
    exporter = ManagementExporter(config, reference_dir / "fertilizer_winter_wheat.csv")
    frame = exporter.build_frame(npk_dataset, _cells(config))

    assert list(frame.columns) == [
        "location", "FertilizerScenario", "crop", "Event", "vType", "DVS", "Amount",
    ]
    # Three cells carry rates; the fourth is NaN in the NPK data and is dropped.
    assert sorted(frame["location"].unique()) == [1, 2, 3]
    assert len(frame) == 3 * 5
    # Events are numbered 1..5 per location, in ascending DVS.
    for _, rows in frame.groupby("location"):
        assert list(rows["Event"]) == [1, 2, 3, 4, 5]
        assert list(rows["DVS"]) == sorted(rows["DVS"])
    assert (frame["crop"] == "winter_wheat").all()


def test_higher_n_rate_gives_proportionally_more_product(
    mgmt_config, reference_dir, npk_dataset
):
    config = mgmt_config()
    exporter = ManagementExporter(config, reference_dir / "fertilizer_winter_wheat.csv")
    frame = exporter.build_frame(npk_dataset, _cells(config))
    kas = frame[frame["vType"] == "KAS"].groupby("location")["Amount"].sum()
    # Cell 1 is at 200 kg N/ha and cell 2 at 100 -> twice the product, to within
    # the per-event rounding (npk.amount_decimals over three dressings).
    assert kas[1] == pytest.approx(2.0 * kas[2], abs=3 * 1e-3)


def test_cells_without_rates_are_dropped_not_defaulted(
    mgmt_config, reference_dir, npk_dataset
):
    config = mgmt_config()
    exporter = ManagementExporter(config, reference_dir / "fertilizer_winter_wheat.csv")
    empty = xr.full_like(npk_dataset, np.nan)
    assert exporter.build_frame(empty, _cells(config)).empty


def test_negative_rates_are_treated_as_absent(mgmt_config, reference_dir, npk_dataset):
    config = mgmt_config()
    exporter = ManagementExporter(config, reference_dir / "fertilizer_winter_wheat.csv")
    # An unmasked NPKGRIDS ocean sentinel must never become an application.
    dirty = npk_dataset.copy()
    dirty["N"] = dirty["N"].where(dirty["lon"] > 11.3, -1.0)
    frame = exporter.build_frame(dirty, _cells(config))
    assert (frame["Amount"] > 0).all()
    assert not ((frame["location"] == 1) & (frame["vType"] == "KAS")).any()
    # The cell keeps its P and K events, renumbered without a gap.
    cell = frame[frame["location"] == 1]
    assert list(cell["Event"]) == [1, 2]


# --------------------------------------------------------------------------- #
# Writing
# --------------------------------------------------------------------------- #
def test_export_writes_a_reference_shaped_file(
    tmp_path, mgmt_config, reference_dir, npk_dataset
):
    config = mgmt_config(simplace_crop="winter_wheat")
    exporter = ManagementExporter(config, reference_dir / "fertilizer_winter_wheat.csv")
    out = exporter.export(npk_dataset, _cells(config), tmp_path)

    assert out.name == "fertilizer_winter_wheat.csv"
    written = pd.read_csv(out)
    reference = pd.read_csv(reference_dir / "fertilizer_winter_wheat.csv")
    assert list(written.columns) == list(reference.columns)
    assert written["FertilizerScenario"].eq(2).all()


def test_export_without_a_reference_uses_the_fallback_scenario(
    tmp_path, mgmt_config, npk_dataset
):
    config = mgmt_config()
    exporter = ManagementExporter(config, reference_path=None)
    frame = pd.read_csv(exporter.export(npk_dataset, _cells(config), tmp_path))
    assert sorted(frame["vType"].unique()) == ["K", "KAS", "P"]
    assert len(frame) == 3 * 5


# --------------------------------------------------------------------------- #
# Irrigation column
# --------------------------------------------------------------------------- #
def _classification(config, virr):
    """A classification covering the fixture's 2x2 grid."""
    from data4simplace.management.irrigation import SOURCE_MIRCA, IrrigationClassification

    virr = np.asarray(virr, dtype="int8")
    return IrrigationClassification(
        virr=virr,
        fraction=np.where(virr == 1, 0.9, 0.1).astype("float32"),
        source_id=np.full(virr.shape, SOURCE_MIRCA, dtype="int8"),
        crop_group="cereals",
        source="merged",
        threshold=config.irrigation.threshold,
        min_crop_area_ha=10.0,
        year=2015,
    )


def test_virr_column_repeats_the_cell_label_across_its_events(
    mgmt_config, reference_dir, npk_dataset
):
    config = mgmt_config()
    exporter = ManagementExporter(config, reference_dir / "fertilizer_winter_wheat.csv")
    irrigation = _classification(config, [[1, 0], [0, 1]])

    frame = exporter.build_frame(npk_dataset, _cells(config), irrigation=irrigation)

    assert list(frame.columns)[-1] == "vIRR"
    # Every event row of a location carries that cell's single label.
    per_cell = frame.groupby("location")["vIRR"].nunique()
    assert (per_cell == 1).all()
    # SimplaceID is row-major, so cells 1..4 map to [[1, 0], [0, 1]]; cell 4 has
    # no NPK rate in the fixture and is dropped from the schedule entirely.
    assert frame.groupby("location")["vIRR"].first().to_dict() == {1: 1, 2: 0, 3: 0}


def test_without_a_classification_the_schedule_keeps_the_reference_columns(
    mgmt_config, reference_dir, npk_dataset
):
    config = mgmt_config()
    exporter = ManagementExporter(config, reference_dir / "fertilizer_winter_wheat.csv")
    frame = exporter.build_frame(npk_dataset, _cells(config))
    assert "vIRR" not in frame.columns


def test_virr_column_name_is_configurable(mgmt_config, reference_dir, npk_dataset):
    config = mgmt_config()
    config.irrigation.column = "irrigated"
    exporter = ManagementExporter(config, reference_dir / "fertilizer_winter_wheat.csv")
    frame = exporter.build_frame(
        npk_dataset, _cells(config), irrigation=_classification(config, [[1, 1], [1, 1]])
    )
    assert "irrigated" in frame.columns and "vIRR" not in frame.columns


def test_an_empty_schedule_still_declares_the_virr_column(mgmt_config, reference_dir):
    config = mgmt_config()
    exporter = ManagementExporter(config, reference_dir / "fertilizer_winter_wheat.csv")
    empty = xr.Dataset()
    frame = exporter.build_frame(
        empty, _cells(config), irrigation=_classification(config, [[1, 1], [1, 1]])
    )
    assert frame.empty and "vIRR" in frame.columns


def test_export_writes_the_virr_column(tmp_path, mgmt_config, reference_dir, npk_dataset):
    config = mgmt_config(simplace_crop="winter_wheat")
    reference_columns = list(pd.read_csv(reference_dir / "fertilizer_winter_wheat.csv").columns)
    exporter = ManagementExporter(config, reference_dir / "fertilizer_winter_wheat.csv")
    # A separate output root: export() writes to <output_dir>/management/, which
    # is the reference folder itself when tmp_path is passed.
    out = exporter.export(
        npk_dataset,
        _cells(config),
        tmp_path / "out",
        irrigation=_classification(config, [[1, 0], [0, 1]]),
    )
    written = pd.read_csv(out)
    # The reference's own columns are preserved, in order, with vIRR appended.
    assert list(written.columns) == reference_columns + ["vIRR"]
    assert set(written["vIRR"].unique()) <= {0, 1}
