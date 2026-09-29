"""Honest ranges: synthetic checks that the ranges hold what they claim, and for the right reasons."""

import math

import numpy as np
import pandas as pd
import pytest

from yieldpred import ranges
from yieldpred.appdata import range_record, range_summary


def test_bounds_use_the_finite_sample_rank_not_the_plain_percentile():
    errors = np.arange(1, 40)                    # 39 misses: 1..39
    low, high = ranges.conformal_bounds(errors, 0.8)
    k = math.ceil(40 * 0.9)                      # 36th smallest, not the 90th percentile (35.2)
    assert high == 36 and low == 39 - k + 1


def test_too_few_misses_give_no_range():
    assert ranges.conformal_bounds(np.arange(ranges.MIN_ERRORS - 1)) is None


def test_bounds_hold_about_the_level_on_new_draws():
    rng = np.random.default_rng(0)
    low, high = ranges.conformal_bounds(rng.normal(0, 10, 2000), 0.8)
    fresh = rng.normal(0, 10, 20000)
    assert 0.77 < np.mean((fresh >= low) & (fresh <= high)) < 0.83


def test_a_fold_never_uses_its_own_misses():
    errors = np.r_[np.zeros(50), np.full(50, 100.0)]    # fold 1's misses are all huge
    folds = np.r_[np.zeros(50), np.ones(50)].astype(int)
    low, high = ranges.cross_conformal(errors, folds)
    assert np.all(high[folds == 1] == 0)                # built only from fold 0's zeros
    assert np.all(low[folds == 0] == 100)


def irrigation_data(n=3000, seed=1):
    """Dryland misses three times the size of irrigated ones."""
    rng = np.random.default_rng(seed)
    share = rng.uniform(0, 1, n)
    errors = rng.normal(0, 30 - 20 * share)
    return share, errors, rng.integers(0, 5, n)


def test_one_width_undercovers_dryland_and_scaling_fixes_it():
    share, errors, folds = irrigation_data()
    frame = pd.DataFrame({"error": errors, "group": ranges.irrigation_group(share).to_numpy()})

    low, high = ranges.cross_conformal(errors, folds)
    plain = ranges.coverage(frame.assign(low=low, high=high), "group")["held_pct"]
    assert plain["dryland"] < 72 and plain["irrigated"] > 90

    low, high = ranges.cross_conformal(errors, folds, share)
    scaled = ranges.coverage(frame.assign(low=low, high=high), "group")
    assert scaled["held_pct"].between(75, 85).all()
    assert scaled.loc["dryland", "width_bu"] > scaled.loc["irrigated", "width_bu"]


def test_expected_miss_never_drops_below_the_floor():
    scale = ranges.expected_miss([0, 1], [10, 0], [5.0])    # extrapolates far below zero
    assert scale[0] == ranges.MIN_SCALE


def seasons(biases, n=40, seed=2):
    rng = np.random.default_rng(seed)
    return pd.DataFrame([{"year": 2000 + i, "error": b + e}
                         for i, b in enumerate(biases) for e in rng.normal(0, 5, n)])


def test_next_season_ranges_widen_after_seasons_that_miss():
    frame = seasons([0] * 6 + [25] * 4)                # four shock seasons in a row
    result = ranges.forward_years(frame, step=0.05).set_index("year")
    assert result.loc[2006, "held_pct"] < 20
    assert result["level_used"].is_monotonic_increasing
    assert result.loc[2009, "width_bu"] > result.loc[2006, "width_bu"]


def test_season_share_separates_shared_from_independent_misses():
    assert ranges.season_share(seasons([0] * 10)) < 0.05
    assert ranges.season_share(seasons([-20, 20] * 5)) > 0.9


def test_range_record_and_summary_for_display():
    frame = pd.DataFrame({"fips": ["1", "1", "1", "2"], "inside": [True, False, True, True]})
    assert range_record(frame, "1") == {"held": 2, "years": 3}

    coverage = pd.DataFrame({
        "ranges": ["one width"] * 2 + ["scaled"] * 2, "group": ["all", "dryland"] * 2,
        "rows": [10, 4] * 2, "held_pct": [80, 70, 80, 79], "width_bu": [40, 40, 40, 50]})
    table = range_summary(coverage)
    assert list(table["group"]) == ["all", "dryland"]
    assert table.loc[1, "scaled: held %"] == pytest.approx(79)
    assert table.loc[1, "county-years"] == 4


def test_the_shown_range_is_the_one_whose_worst_group_is_closest_to_80():
    table = pd.DataFrame({"ranges": ["one width"] * 3 + ["scaled"] * 3,
                          "group": ["all", "dryland", "irrigated"] * 2,
                          "held_pct": [78.0, 68.0, 86.0, 78.0, 81.0, 78.5]})
    assert ranges.pick_variant(table) == ("scaled", 2.0)
    soy_like = table.assign(held_pct=[77.0, 72.0, 75.0, 76.0, 67.8, 76.0])
    assert ranges.pick_variant(soy_like)[0] == "one width"      # scaling made the worst group worse
