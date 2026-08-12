"""Tests for the torchcrop runner.

The model itself is torchcrop's to test; what is tested here is the wiring —
that the export's soil, site, schedule and weather reach LINTUL-5 as the right
parameters, and that per-cell sowing dates are grouped rather than averaged.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

torch = pytest.importorskip("torch")
pytest.importorskip("torchcrop")

from cropmodelling4eu.export import resolve_export  # noqa: E402
from cropmodelling4eu.torchcrop.run import (  # noqa: E402
    build_site_params,
    build_soil_params,
    fertilizer_from_dvs,
    group_by_sowing,
    run_shard,
)

from .conftest import TEST_CELLS  # noqa: E402


# --------------------------------------------------------------------------- #
# Sowing groups
# --------------------------------------------------------------------------- #


def test_group_by_sowing_splits_the_shard():
    ids = np.array([1, 2, 3, 4, 5])
    doys = np.array([280, 295, 280, 295, 280])
    groups = group_by_sowing(ids, doys)

    assert sorted(groups) == [280, 295]
    assert groups[280].tolist() == [1, 3, 5]
    assert groups[295].tolist() == [2, 4]
    # Every cell lands in exactly one group.
    assert sum(g.size for g in groups.values()) == ids.size


def test_group_by_sowing_keeps_a_single_date_in_one_group():
    """The old constant-DOY behaviour is the one-group case, unchanged."""
    ids = np.arange(1, 11)
    groups = group_by_sowing(ids, np.full(10, 270))
    assert list(groups) == [270]
    assert groups[270].tolist() == ids.tolist()


def test_shard_groups_match_the_site_table(run_config):
    """The fixture's two sowing dates must survive into two groups."""
    bundle = resolve_export(run_config)
    ids = bundle.ids
    groups = group_by_sowing(ids, bundle.site.sowing_doy(ids))
    assert sorted(groups) == [280, 295]


# --------------------------------------------------------------------------- #
# Parameter construction
# --------------------------------------------------------------------------- #


def test_soil_params_integrate_over_the_files_own_layers(run_config):
    bundle = resolve_export(run_config)
    soil = bundle.soil.select(bundle.ids)
    params = build_soil_params(soil, rootzone_m=1.0, profile_bottom_m=2.0)

    n = bundle.ids.size
    assert params.wcfc.shape == (n,)
    # The fixture profile is uniform, so the rooting-zone mean is the layer value.
    assert params.wcfc.numpy() == pytest.approx(0.31)
    assert params.wcwp.numpy() == pytest.approx(0.14)
    assert params.wcst.numpy() == pytest.approx(0.42)
    # Air-dry is clamped below the wilting point.
    assert (params.wcad.numpy() < params.wcwp.numpy()).all()
    # Mineral N: (12 + 28) kg/ha in each of the five rooting-zone layers,
    # kg/ha -> g/m2 is a factor of 10.
    assert params.nminti.numpy() == pytest.approx((12.0 + 28.0) * 5 / 10.0)


def test_soil_params_follow_a_different_layer_geometry(run_config, export_dir):
    """Halving the profile depth must change the depth-integrated stock."""
    bundle = resolve_export(run_config)
    deep = build_soil_params(bundle.soil, rootzone_m=1.0, profile_bottom_m=2.0)
    shallow = build_soil_params(bundle.soil, rootzone_m=0.3, profile_bottom_m=2.0)
    # A 0.3 m rooting zone holds two layers of stock, not five.
    assert shallow.nminti.numpy() == pytest.approx((12.0 + 28.0) * 2 / 10.0)
    assert (shallow.nminti.numpy() < deep.nminti.numpy()).all()


def test_site_params_come_from_the_export(run_config):
    bundle = resolve_export(run_config)
    ids = np.array(TEST_CELLS)
    params = build_site_params(ids, 2000, 280, bundle)

    # Altitude is per cell, from site.csv -- not a constant zero.
    assert params.altitude.numpy().tolist() == [10.0, 20.0, 300.0, 1200.0]
    assert params.altitude.numpy().std() > 0
    # CO2 is the export's series for that year.
    assert params.co2.numpy() == pytest.approx(369.6)
    # idpl is the group's sowing day, which the window was built from.
    assert params.idpl.numpy() == pytest.approx(280.0)
    # Latitude is decoded from the grid.
    assert params.latitude.numpy().max() <= run_config.grid.max_lat


def test_site_params_use_the_configured_grid(run_config):
    """A different grid must move the cells, not silently reuse Europe's."""
    bundle = resolve_export(run_config)
    ids = np.array([TEST_CELLS[0]])
    here = build_site_params(ids, 2000, 280, bundle).latitude.numpy()[0]
    assert run_config.grid.min_lat <= here <= run_config.grid.max_lat


# --------------------------------------------------------------------------- #
# Fertilizer placement
# --------------------------------------------------------------------------- #


def test_fertilizer_is_placed_at_the_first_day_past_each_stage():
    ids = np.array([1])
    # A monotone DVS trajectory over 10 days, with the leading pre-sowing entry.
    dvs = torch.tensor([[0.0, *np.linspace(0.0, 1.0, 10)]])
    plans = {1: np.array([[0.25, 5.0, 0.0, 0.0], [0.9, 3.0, 0.0, 0.0]])}

    applied = fertilizer_from_dvs(ids, dvs, plans)
    assert applied.shape == (1, 10, 3)
    # Both doses land, on exactly one day each.
    assert applied[0, :, 0].sum().item() == pytest.approx(8.0)
    assert int((applied[0, :, 0] > 0).sum()) == 2
    # The DVS 0.25 dose comes first.
    days = torch.nonzero(applied[0, :, 0]).flatten().tolist()
    assert days[0] < days[1]


def test_cells_without_a_plan_run_unfertilised():
    ids = np.array([1, 999])
    dvs = torch.tensor([[0.0, *np.linspace(0.0, 1.0, 5)]] * 2)
    plans = {1: np.array([[0.25, 5.0, 0.0, 0.0]])}

    applied = fertilizer_from_dvs(ids, dvs, plans)
    assert applied[0].sum().item() == pytest.approx(5.0)
    assert applied[1].sum().item() == 0.0


def test_stages_never_reached_are_not_applied():
    """A dose keyed past the trajectory's end must not be dumped on day 0."""
    ids = np.array([1])
    dvs = torch.tensor([[0.0, *np.linspace(0.0, 0.5, 5)]])
    plans = {1: np.array([[0.9, 4.0, 0.0, 0.0]])}
    assert fertilizer_from_dvs(ids, dvs, plans).sum().item() == 0.0


# --------------------------------------------------------------------------- #
# End to end
# --------------------------------------------------------------------------- #


@pytest.mark.slow
def test_run_shard_writes_a_parquet(tmp_path, run_config):
    """One shard over the fixture export, end to end through LINTUL-5."""
    out = run_shard(run_config, shard=0, n_shards=1, out_dir=tmp_path)
    assert out.is_file()

    frame = pd.read_parquet(out)
    assert set(frame["SimplaceID"]) <= set(TEST_CELLS)
    for column in ("yield_g_m2", "biomass_g_m2", "max_lai", "sowing_doy", "irri"):
        assert column in frame.columns
    # Both sowing groups reach the output, so grouping did not drop a batch.
    assert sorted(frame["sowing_doy"].unique()) == [280, 295]
    # Yields are finite and not absurd for winter wheat [g/m2].
    assert frame["yield_g_m2"].notna().all()
    assert frame["yield_g_m2"].between(0, 3000).all()


@pytest.mark.slow
def test_shard_is_idempotent(tmp_path, run_config):
    first = run_shard(run_config, shard=0, n_shards=1, out_dir=tmp_path)
    mtime = first.stat().st_mtime_ns
    again = run_shard(run_config, shard=0, n_shards=1, out_dir=tmp_path)
    assert again == first and again.stat().st_mtime_ns == mtime
