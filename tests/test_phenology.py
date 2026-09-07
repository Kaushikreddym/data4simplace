"""Tests for the CLMS phenology (validation) stage.

The first two are the load-bearing ones. Everything this stage reports is a
quantile taken from a histogram rather than from the values themselves, and
every administrative unit that straddles a tile boundary is reconstructed by
adding histograms together. If either property fails, the numbers are quietly
wrong rather than visibly broken.
"""

from __future__ import annotations

from datetime import date

import numpy as np
import pytest

from data4simplace.phenology.decode import (
    N_BINS,
    anchor_for,
    days_to_doy,
    decode_yydoy,
)
from data4simplace.phenology.reduce import PERCENTILES, quantiles_from_counts
from data4simplace.phenology.seasons import (
    SPLIT_CROPS,
    antimode_cut,
    crop_label,
    crop_labels,
    crop_lookup,
    crop_slots,
    default_cut_day,
)


def _histogram(values: np.ndarray) -> np.ndarray:
    return np.bincount(values.astype(np.int64), minlength=N_BINS)[:N_BINS]


@pytest.mark.parametrize("seed", range(8))
def test_quantiles_match_numpy_exactly(seed: int) -> None:
    """A histogram quantile is the sample's quantile, not an approximation."""
    rng = np.random.default_rng(seed)
    values = rng.integers(0, N_BINS, size=rng.integers(1, 5000))
    got = quantiles_from_counts(_histogram(values), PERCENTILES)
    expected = np.quantile(values.astype(float), PERCENTILES, method="linear")
    assert np.allclose(got, expected, atol=1e-9), (got, expected)


def test_quantiles_of_empty_histogram_are_nan() -> None:
    assert np.isnan(quantiles_from_counts(np.zeros(N_BINS, dtype=np.int64))).all()


def test_quantiles_of_single_value() -> None:
    counts = np.zeros(N_BINS, dtype=np.int64)
    counts[137] = 9
    assert np.allclose(quantiles_from_counts(counts), 137.0)


@pytest.mark.parametrize("seed", range(8))
def test_histograms_merge_across_tiles(seed: int) -> None:
    """Summing per-tile histograms reproduces the whole-unit answer.

    This is why the stage accumulates counts instead of medians. A unit split
    across three tiles must give the same quantiles as the same pixels read in
    one pass -- taking a median per tile and averaging would not.
    """
    rng = np.random.default_rng(seed)
    parts = [rng.integers(0, N_BINS, size=rng.integers(1, 2000)) for _ in range(3)]
    merged = sum(_histogram(p) for p in parts)

    whole = np.concatenate(parts)
    assert np.allclose(
        quantiles_from_counts(merged, PERCENTILES),
        np.quantile(whole.astype(float), PERCENTILES, method="linear"),
        atol=1e-9,
    )


def test_axis_reaches_the_end_of_the_season_year() -> None:
    """The histogram must hold every date the product can encode.

    512 bins was six short: it clipped genuine 28 December harvests. The axis
    has to span the anchor to 31 December of the season year, which is 517 days
    and 518 across a leap year.
    """
    for season_year in (2020, 2021, 2022, 2023, 2024):
        span = (date(season_year, 12, 31) - anchor_for(season_year)).days
        assert span < N_BINS, f"{season_year}: needs {span + 1} bins, have {N_BINS}"


def test_decode_yydoy_round_trips() -> None:
    """A YYDOY value decodes to the day-of-year it encodes."""
    season = 2022
    # 15 October 2021 (autumn, the winter mode) and 22 March 2022 (spring).
    raw = np.array([[21_288, 22_081]], dtype=np.int32)
    days = decode_yydoy(raw, season)
    assert days_to_doy(days[0, 0], season) == 288
    assert days_to_doy(days[0, 1], season) == 81


def test_decode_yydoy_masks_flags() -> None:
    """Flags are NaN, never a date. Averaging one in is the classic wrong answer."""
    raw = np.array([[0, 65526, 65531, 65535, 22_081]], dtype=np.int32)
    days = decode_yydoy(raw, 2022)
    assert np.isnan(days[0, :4]).all()
    assert np.isfinite(days[0, 4])


def test_antimode_finds_the_trough_of_a_bimodal_histogram() -> None:
    """A clean autumn/spring pair yields a measured cut between the modes."""
    season = 2022
    counts = np.zeros(N_BINS, dtype=np.int64)
    autumn = (date(season - 1, 10, 15) - anchor_for(season)).days
    spring = (date(season, 3, 20) - anchor_for(season)).days
    counts[autumn - 20 : autumn + 20] = 1000
    counts[spring - 20 : spring + 20] = 400

    cut = antimode_cut(counts, season)
    assert cut.source == "antimode"
    assert autumn < cut.day < spring


def test_antimode_falls_back_when_one_mode_is_missing() -> None:
    """A region growing only winter wheat has no boundary; inventing one is worse.

    German oilseed rape is the real case: almost entirely autumn-sown, so the
    spring mode is a rounding error and the cut must fall back rather than split
    a single mode down the middle.
    """
    season = 2022
    counts = np.zeros(N_BINS, dtype=np.int64)
    autumn = (date(season - 1, 10, 15) - anchor_for(season)).days
    counts[autumn - 20 : autumn + 20] = 1000

    cut = antimode_cut(counts, season)
    assert cut.source == "fallback"
    assert cut.day == default_cut_day(season)
    assert cut.reason


def test_default_cut_is_new_year() -> None:
    for season in (2020, 2022, 2024):
        assert days_to_doy(default_cut_day(season), season) == 1


def test_split_crops_are_named_with_their_season() -> None:
    assert crop_label(1110, "winter") == "wheat_winter"
    assert crop_label(1110, "spring") == "wheat_spring"
    assert crop_label(1130) == "maize"
    assert "sugar_beet" in crop_labels()


def test_naming_refuses_a_mismatched_season() -> None:
    """A season on a single-season crop, or none on a split one, is a bug."""
    with pytest.raises(ValueError):
        crop_label(1130, "winter")
    with pytest.raises(ValueError):
        crop_label(1110)


def test_spring_slot_follows_winter() -> None:
    """zonal.py adds 1 to reach spring; that only works if they are adjacent."""
    slots = crop_slots()
    by_index = {s.index: s for s in slots}
    for slot in slots:
        if slot.season == "winter":
            nxt = by_index[slot.index + 1]
            assert nxt.code == slot.code and nxt.season == "spring"


def test_crop_lookup_agrees_with_the_slots() -> None:
    slot_of, splits = crop_lookup()
    for slot in crop_slots():
        if slot.season in (None, "winter"):
            assert slot_of[slot.code] == slot.index
    for code in SPLIT_CROPS:
        assert splits[code]
    assert slot_of[0] == -1        # "no cropland" is not a crop
    assert slot_of[65535] == -1    # nor is "outside area"
