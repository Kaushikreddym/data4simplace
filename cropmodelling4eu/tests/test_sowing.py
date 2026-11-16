"""The learned-sowing pipeline: torchcrop parity, causality, leakage, fallbacks."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest
import torch

from cropmodelling4eu.sowing.cache import doy_to_index, index_to_doy
from cropmodelling4eu.sowing.config import ModelConfig, SowingConfig
from cropmodelling4eu.sowing.crops import override_crop_file, override_crops
from cropmodelling4eu.sowing.emergence import EmergenceParams, emergence_index, fit_tsumem
from cropmodelling4eu.sowing.features import LONG_FEATURES, cell_features
from cropmodelling4eu.sowing.model import SowingHazardNet, first_event_log_probs, laplace_mixture_nll
from cropmodelling4eu.torchcrop.run import crossing_day


def _torchcrop_emergence(davtmp: np.ndarray, sow: int, p: EmergenceParams) -> float:
    """torchcrop's own path: latch on the sowing day, accumulate, crossing_day on states[1:]."""
    rate = np.clip(davtmp - p.tbasem, 0.0, p.teffmx - p.tbasem) * (np.arange(davtmp.size) >= sow)
    tsump = torch.tensor(np.cumsum(rate))[None]
    return float(crossing_day(tsump, p.tsumem))


@pytest.mark.parametrize("sow", [0, 1, 17, 60])
def test_emergence_matches_torchcrop(sow):
    rng = np.random.default_rng(sow)
    davtmp = rng.uniform(-4.0, 22.0, 120)
    p = EmergenceParams(150.0, 0.0, 30.0)
    ours = float(emergence_index(torch.tensor(davtmp)[None], torch.tensor([[sow]]), p))
    assert ours == pytest.approx(_torchcrop_emergence(davtmp, sow, p), abs=1e-6)


def test_emergence_never_reached_is_nan():
    out = emergence_index(torch.zeros(1, 30), torch.tensor([[0, 5]]), EmergenceParams(50.0))
    assert torch.isnan(out).all()


def test_fit_tsumem_recovers_truth():
    rng = np.random.default_rng(0)
    temp = rng.uniform(5.0, 15.0, (400, 90))
    sow = rng.integers(0, 30, 400)
    truth = EmergenceParams(140.0)
    obs = emergence_index(torch.tensor(temp), torch.tensor(sow)[:, None], truth).squeeze(1).numpy()
    result = fit_tsumem(temp, sow, obs)
    assert result["tsumem"] == pytest.approx(140.0, abs=1.0)
    assert result["mae_days"] < 0.1


def test_first_event_distribution_sums_to_one_and_forces_deadline():
    logits = torch.full((3, 40), -20.0)
    p = first_event_log_probs(logits).exp()
    assert torch.allclose(p.sum(dim=1), torch.ones(3), atol=1e-5)
    assert (p[:, -1] > 0.99).all()


def test_mixture_nll_prefers_the_right_day():
    log_p = torch.log(torch.tensor([[0.98, 0.01, 0.01]]))
    centres = torch.tensor([[10.0, 20.0, 30.0]])
    near, far = (laplace_mixture_nll(log_p, centres, torch.tensor([x]), torch.tensor(2.0)) for x in (10.0, 30.0))
    assert near < far


def test_network_is_causal():
    torch.manual_seed(0)
    net = SowingHazardNet(7, 6, 19, 50, ModelConfig(dropout=0.0)).eval()
    short, medium, long = torch.randn(2, 80, 7), torch.randn(2, 80, 6), torch.randn(2, 19)
    start = torch.tensor([10.0, 20.0])
    base = net(short, medium, long, start)
    cut = 30
    short2, medium2 = short.clone(), medium.clone()
    short2[:, cut + 1:] += 5.0
    medium2[:, cut + 1:] -= 5.0
    after = net(short2, medium2, long, start)
    assert torch.allclose(base[:, :cut + 1], after[:, :cut + 1])
    assert not torch.allclose(base[:, cut + 1:], after[:, cut + 1:])


def test_short_branch_receptive_field():
    torch.manual_seed(0)
    cfg = ModelConfig(dropout=0.0)
    net = SowingHazardNet(7, 6, 19, 60, cfg).eval()
    short = torch.randn(1, 80, 7)
    medium, long, start = torch.zeros(1, 80, 6), torch.zeros(1, 19), torch.tensor([10.0])
    bumped = short.clone()
    bumped[:, 5] += 10.0
    diff = (net(bumped, medium, long, start) - net(short, medium, long, start)).abs()[0]
    reach = int(torch.nonzero(diff > 1e-6).max())
    assert reach - 5 < cfg.short_receptive_days


def test_doy_index_roundtrip():
    years = np.array([2018, 2018, 2021, 2024])
    doy = np.array([193, 300, 365, 20])
    idx = doy_to_index(doy, years, 193)
    assert idx[0] == 0 and idx[1] == 107
    assert (index_to_doy(idx.astype(int), years, 193) == doy).all()


def _synthetic_record(seed: int = 0):
    dates = pd.date_range("1979-01-01", "2010-12-31", freq="D")
    rng = np.random.default_rng(seed)
    doy = dates.dayofyear.to_numpy()
    temp = 10 + 10 * np.sin(2 * np.pi * (doy - 110) / 365) + rng.normal(0, 2, doy.size)
    ch = np.stack([doy, temp, temp - 5, temp + 5, rng.uniform(2, 25, doy.size),
                   rng.exponential(2.0, doy.size), np.full(doy.size, 1.2), np.full(doy.size, 2.0)], axis=1)
    return dates, ch.astype(np.float32)


def test_normals_do_not_see_the_current_season():
    dates, ch = _synthetic_record()
    kw = dict(capacity_mm=40.0, cache_start_doy=193, cache_days=240, normals_years=30, normals_min_years=10)
    base = cell_features(dates, ch, [2000], **kw)
    hot = ch.copy()
    autumn = (dates.year == 1999) & (dates.month >= 8)
    hot[autumn, 1:4] += 15.0
    changed = cell_features(dates, hot, [2000], **kw)
    assert base.valid[0] and changed.valid[0]
    np.testing.assert_array_equal(base.long, changed.long)
    assert not np.array_equal(base.short, changed.short)
    assert base.long.shape == (1, len(LONG_FEATURES))


def test_too_few_prior_autumns_is_invalid():
    dates, ch = _synthetic_record()
    out = cell_features(dates, ch, [1985, 1995], capacity_mm=40.0, cache_start_doy=193,
                        cache_days=240, normals_years=30, normals_min_years=10)
    assert out.valid.tolist() == [False, True]


_CROP = """source_file: /x/LINTUL5_crop.xml
crop:
  params:
      tbasem: 0.0
      teffmx: 30.0
      tsumem: 198.26783752441406
mapping:
  tsumem: simplace:TSUMEM
"""


def test_override_crop_file_changes_one_line(tmp_path):
    src, dst = tmp_path / "crop_region_01.yaml", tmp_path / "out" / "crop_region_01.yaml"
    src.write_text(_CROP)
    old = override_crop_file(src, dst, {"tsumem": 151.0})
    assert old == {"tsumem": pytest.approx(198.26783752441406)}
    changed = [a for a, b in zip(src.read_text().splitlines(), dst.read_text().splitlines()) if a != b]
    assert changed == ["      tsumem: 198.26783752441406"]
    assert src.read_text() == _CROP


def test_override_crops_refuses_in_place(tmp_path):
    (tmp_path / "crop_region_01.yaml").write_text(_CROP)
    with pytest.raises(ValueError):
        override_crops(tmp_path, tmp_path, {"tsumem": 1.0})


def test_fixed_parameters_default_is_the_old_calibration(tmp_path):
    from cropmodelling4eu.calibration.config import STAGE_PARAMETERS, CalibrationConfig
    from cropmodelling4eu.calibration.problem import load_stage_specs
    from pathlib import Path

    import cropmodelling4eu.calibration as cal

    assert CalibrationConfig().fixed_parameters == {}
    path = Path(cal.__file__).parent / STAGE_PARAMETERS["phenology"]
    free = {s.name for s in load_stage_specs(path).specs}
    fixed = {s.name for s in load_stage_specs(path, fixed=("crop.tsumem",)).specs}
    assert "crop.tsumem" in free and free - fixed == {"crop.tsumem"}
    with pytest.raises(ValueError):
        CalibrationConfig(fixed_parameters={"crop.vernrt@3.0": 0.5})


def test_sowing_config_guards():
    assert SowingConfig().sowing_days == 173
    with pytest.raises(ValueError):
        SowingConfig(cache_days=150)


def test_shift_table_clips(tmp_path):
    import pandas as pd
    from cropmodelling4eu.sowing.oracle import shift_table

    src = tmp_path / "s.csv"
    pd.DataFrame({"SimplaceID": [1, 2], "year": [2000, 2000], "sowing_doy": [210, 360]}).to_csv(src, index=False)
    out = pd.read_csv(shift_table(src, 14, tmp_path / "p14.csv"))
    assert out["sowing_doy"].tolist() == [224, 365]
    assert pd.read_csv(shift_table(src, -14, tmp_path / "m14.csv"))["sowing_doy"].tolist() == [200, 346]


def test_oracle_picks_max_and_breaks_ties_towards_zero(tmp_path):
    import json
    import pandas as pd
    from cropmodelling4eu.sowing.oracle import assemble_oracle

    def run(offset, yields):
        path = tmp_path / f"{offset}.parquet"
        pd.DataFrame({"SimplaceID": [1, 2, 3], "year": [2000] * 3, "sowing_doy": [280 + offset] * 3,
                      "yield_t_ha": yields}).to_parquet(path)
        return path

    runs = {0: run(0, [5.0, 6.0, float("nan")]), -14: run(-14, [7.0, 6.0, 4.0]), 14: run(14, [4.0, 6.0, 3.0])}
    best = assemble_oracle(runs, tmp_path / "D").set_index("SimplaceID")
    assert best["sowing_offset"].tolist() == [-14, 0, -14]
    assert best["yield_t_ha"].tolist() == [7.0, 6.0, 4.0]
    summary = json.loads((tmp_path / "D" / "oracle_summary.json").read_text())
    assert summary["cell_seasons"] == 3 and summary["median_gain_vs_offset0"] == 1.0
    assert (tmp_path / "D" / "torchcrop" / "torchcrop_europe.parquet").is_file()
