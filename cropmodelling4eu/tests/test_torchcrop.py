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
    run_cells,
    run_shard,
    sowing_plan,
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


def test_sowing_table_groups_by_year_not_by_doy():
    """A simulated-sowing table must not fragment a year by per-cell date --
    that fragmentation (one batch per (year, doy)) is exactly the slowdown
    this grouping is meant to avoid."""
    ids = np.array([1, 2, 3, 4])
    sowing = pd.DataFrame(
        {
            "SimplaceID": [1, 2, 3, 4, 1, 2, 3, 4],
            "year": [2000, 2000, 2000, 2000, 2001, 2001, 2001, 2001],
            # Every cell disagrees with every other -- four groups under the
            # old (year, doy) grouping, one under the new (year) grouping.
            "sowing_doy": [270, 275, 280, 285, 268, 271, 279, 290],
        }
    )
    plan = sowing_plan(ids, [2000, 2001], np.full(4, 270), sowing)

    assert len(plan) == 2
    for years, group, doy in plan:
        assert len(years) == 1
        assert sorted(group.tolist()) == [1, 2, 3, 4]
        assert doy.size == 4


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


@pytest.mark.slow
def test_run_shard_daily_writes_a_matching_trajectory_file(tmp_path, run_config):
    """--daily writes a second Parquet beside the summary, same cells and
    seasons, one row per (cell, day, variable) for exactly the requested set."""
    out = run_shard(run_config, shard=0, n_shards=1, out_dir=tmp_path, daily=True)
    daily_path = tmp_path / "torchcrop_daily_shard_000.parquet"
    assert daily_path.is_file()

    summary = pd.read_parquet(out)
    daily = pd.read_parquet(daily_path)
    assert set(daily["variable"].unique()) == {"LAI", "AGB", "NNI", "TRANRF"}
    assert set(daily["SimplaceID"]) <= set(summary["SimplaceID"])
    assert set(daily["year"]) <= set(summary["year"])
    assert daily["value"].notna().all()


@pytest.mark.slow
def test_run_shard_daily_is_idempotent_with_the_summary(tmp_path, run_config):
    """A shard already done (both files present) is not re-run even with
    --daily; a summary-only shard from an earlier run is topped up rather
    than silently treated as complete."""
    run_shard(run_config, shard=0, n_shards=1, out_dir=tmp_path)
    daily_path = tmp_path / "torchcrop_daily_shard_000.parquet"
    assert not daily_path.exists()

    run_shard(run_config, shard=0, n_shards=1, out_dir=tmp_path, daily=True)
    assert daily_path.is_file()


@pytest.mark.slow
def test_run_cells_both_matches_separate_summary_and_daily_calls(run_config):
    """mode="both" shares one _simulate call per batch (see run_shard's own
    daily=True path) -- it must still reproduce exactly what mode="summary"
    and mode="daily" give when run separately, the two-pass smoke test used
    before run_cells_torchcrop.py --daily-out existed."""
    ids = np.array(TEST_CELLS)

    summary, daily = run_cells(run_config, ids, [2000], mode="both")
    summary_only = run_cells(run_config, ids, [2000], mode="summary")
    daily_only = run_cells(run_config, ids, [2000], mode="daily")

    pd.testing.assert_frame_equal(
        summary.sort_values("SimplaceID").reset_index(drop=True),
        summary_only.sort_values("SimplaceID").reset_index(drop=True),
    )
    pd.testing.assert_frame_equal(
        daily.sort_values(["SimplaceID", "date", "variable"]).reset_index(drop=True),
        daily_only.sort_values(["SimplaceID", "date", "variable"]).reset_index(drop=True),
    )


@pytest.mark.slow
def test_merged_sowing_batch_matches_singleton_runs(run_config):
    """Cells sowing on different simulated dates in the same year now share a
    batch (see sowing_plan): each must still latch on its *own* idpl inside
    the shared, earliest-anchored window and reproduce exactly what running
    it alone -- one cell, its own window -- would give. This is the
    correctness check for widening batches beyond a single sowing date."""
    ids = np.array(TEST_CELLS)
    sowing = pd.DataFrame(
        {"SimplaceID": ids, "year": 2000, "sowing_doy": [270, 278, 288, 296]}
    )

    merged = run_cells(run_config, ids, [2000], sowing=sowing)
    singles = pd.concat(
        [
            run_cells(run_config, np.array([sid]), [2000], sowing=sowing)
            for sid in ids
        ],
        ignore_index=True,
    )

    merged = merged.sort_values("SimplaceID").reset_index(drop=True)
    singles = singles.sort_values("SimplaceID").reset_index(drop=True)
    assert merged["SimplaceID"].tolist() == singles["SimplaceID"].tolist()
    # sowing_doy/days_to_maturity/max_lai are exact -- integers or a single
    # forward-Euler max, with no cross-cell reduction to reorder. The
    # continuous stress means (tranrf_mean, nni_mean) accumulate over ~200
    # daily steps, so a batch-of-4 vs. batch-of-1 op can round differently in
    # float32; a real windowing/idpl bug would show up as a wrong sowing day
    # or a several-percent yield/stress shift, not float32 noise, so a loose
    # tolerance still catches it.
    for column in ("sowing_doy", "days_to_maturity", "max_lai"):
        np.testing.assert_allclose(
            merged[column].to_numpy(dtype=float),
            singles[column].to_numpy(dtype=float),
            rtol=1e-5, atol=1e-6, err_msg=column,
        )
    for column in ("yield_g_m2", "tranrf_mean", "nni_mean"):
        np.testing.assert_allclose(
            merged[column].to_numpy(dtype=float),
            singles[column].to_numpy(dtype=float),
            rtol=1e-3, atol=1e-4, err_msg=column,
        )
