"""The leak machinery has one job it must not get wrong: leave the test set alone.

Every claim in the leak control rests on the two runs predicting identical rows.
If the filter touched the test folds, the before-and-after would be comparing two
different questions and the difference would mean nothing. Most of what follows
checks that property from several directions, because it is the one that would
fail silently.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest
from sklearn.model_selection import GroupKFold

from yieldpred.leak import (LeakFreeGroupKFold, ThinnedGroupKFold, leak_report,
                            power_cells)


def weather(fips_to_series: dict) -> pd.DataFrame:
    """Build a weather table where the caller controls which counties match."""
    rows = []
    for fips, offset in fips_to_series.items():
        for year in range(2000, 2005):
            rows.append({"fips": fips, "year": year,
                         "gdd": 3000.0 + offset + year,
                         "precip_mm": 400.0 + offset,
                         "tmax_jul_c": 30.0 + offset / 100})
    return pd.DataFrame(rows)


# --------------------------------------------------------------- power_cells

def test_identical_weather_means_one_cell():
    w = weather({"19001": 0.0, "19003": 0.0, "19005": 7.0})
    cells = power_cells(w)
    assert cells["19001"] == cells["19003"], "identical series must share a cell id"
    assert cells["19005"] != cells["19001"]
    assert cells.nunique() == 2


def test_cell_ids_are_stable_across_calls():
    """Hashing must not depend on run-to-run randomness, or the ids change."""
    w = weather({"19001": 0.0, "19003": 5.0})
    assert power_cells(w).to_dict() == power_cells(w).to_dict()


def test_row_order_does_not_change_the_answer():
    w = weather({"19001": 0.0, "19003": 0.0, "19005": 3.0})
    shuffled = w.sample(frac=1.0, random_state=7)
    assert power_cells(w).to_dict() == power_cells(shuffled).to_dict()


def test_missing_column_is_an_error_not_a_wrong_answer():
    w = weather({"19001": 0.0}).drop(columns="tmax_jul_c")
    with pytest.raises(KeyError, match="tmax_jul_c"):
        power_cells(w)


# ---------------------------------------------------------------- leak_report

def test_leak_report_counts_only_cells_that_cross_a_district():
    cells = pd.Series({"a": "c1", "b": "c1", "c": "c2", "d": "c2", "e": "c3"})
    districts = pd.Series({"a": "NORTH", "b": "SOUTH",     # crosses: counted
                           "c": "EAST", "d": "EAST",       # shared, same district
                           "e": "WEST"})                   # alone
    r = leak_report(cells, districts)
    assert r["counties"] == 5
    assert r["cells"] == 3
    assert r["cells_crossing_districts"] == 1
    assert r["counties_leaking"] == 2
    assert r["share_leaking"] == pytest.approx(0.4)


def test_no_leak_is_reported_as_zero_not_as_an_error():
    cells = pd.Series({"a": "c1", "b": "c2"})
    districts = pd.Series({"a": "NORTH", "b": "SOUTH"})
    assert leak_report(cells, districts)["counties_leaking"] == 0


# -------------------------------------------------------- LeakFreeGroupKFold

def toy(n_groups: int = 5, per_group: int = 20):
    """Rows laid out the way the real data is: mostly group-local cells, a few crossing.

    Getting this fixture wrong is instructive. A first version gave every row one
    of seven cyclic cell ids, so every cell appeared in every group - and the
    filter correctly emptied every training fold. Real counties are not like that:
    a POWER cell serves two or three adjacent counties, and only sometimes do they
    straddle a district line. The fixture has to reproduce *that*, or it tests a
    situation the guard exists to reject.
    """
    n = n_groups * per_group
    X = pd.DataFrame({"x": np.arange(n, dtype=float)})
    y = pd.Series(np.arange(n, dtype=float))
    groups = np.repeat(np.arange(n_groups), per_group)

    cells = np.array([f"c{g}" for g in groups], dtype=object)
    # Each shared cell spans exactly two adjacent groups, a few rows on each side.
    for g in range(n_groups - 1):
        for offset in range(min(3, per_group // 2)):
            cells[g * per_group + offset] = f"s{g}"
            cells[(g + 1) * per_group + per_group // 2 + offset] = f"s{g}"
    return X, y, groups, cells


def test_test_folds_are_identical_to_plain_groupkfold():
    """The property everything else depends on."""
    X, y, groups, cells = toy()
    plain = [list(te) for _, te in GroupKFold(5).split(X, y, groups)]
    leakfree = [list(te) for _, te in LeakFreeGroupKFold(5, cells).split(X, y, groups)]
    assert plain == leakfree


def test_no_training_row_shares_a_cell_with_its_test_fold():
    X, y, groups, cells = toy()
    for train, test in LeakFreeGroupKFold(5, cells).split(X, y, groups):
        assert not set(cells[train]) & set(cells[test])


def test_training_rows_are_only_ever_removed_never_added():
    X, y, groups, cells = toy()
    plain = [set(tr) for tr, _ in GroupKFold(5).split(X, y, groups)]
    for (train, _), original in zip(
            LeakFreeGroupKFold(5, cells).split(X, y, groups), plain):
        assert set(train) <= original


def test_training_cost_reports_what_was_dropped():
    X, y, groups, cells = toy()
    splitter = LeakFreeGroupKFold(5, cells)
    plain_sizes = [len(tr) for tr, _ in GroupKFold(5).split(X, y, groups)]
    kept_sizes = [len(tr) for tr, _ in splitter.split(X, y, groups)]
    cost = splitter.training_cost()
    assert cost["folds"] == 5
    assert cost["rows_dropped_per_fold"] == [p - k for p, k in zip(plain_sizes, kept_sizes)]
    assert 0 < cost["share_of_training_lost"] < 1


def test_a_disjoint_cell_layout_drops_nothing():
    """When cells never cross groups there is no leak, so nothing should change."""
    X, y, groups, _ = toy()
    cells = np.array([f"cell{g}" for g in groups])       # one cell per group
    splitter = LeakFreeGroupKFold(5, cells)
    plain = [(list(tr), list(te)) for tr, te in GroupKFold(5).split(X, y, groups)]
    got = [(list(tr), list(te)) for tr, te in splitter.split(X, y, groups)]
    assert plain == got
    assert splitter.training_cost()["rows_dropped_total"] == 0


def test_per_county_cells_are_rejected():
    """A per-county Series is the obvious wrong thing to pass; it must not slide by."""
    X, y, groups, _ = toy()
    with pytest.raises(ValueError, match="per-row"):
        list(LeakFreeGroupKFold(5, np.array(["a", "b", "c"])).split(X, y, groups))


def test_a_fold_that_loses_everything_raises_rather_than_scoring_nonsense():
    X, y, groups, _ = toy(n_groups=5, per_group=4)
    cells = np.full(len(X), "one-cell-everywhere")   # every row leaks into every fold
    with pytest.raises(ValueError, match="entire training set"):
        list(LeakFreeGroupKFold(5, cells).split(X, y, groups))


# --------------------------------------------------------- ThinnedGroupKFold

def test_the_control_drops_the_same_counts_from_the_same_folds():
    X, y, groups, cells = toy()
    splitter = LeakFreeGroupKFold(5, cells)
    kept = [len(tr) for tr, _ in splitter.split(X, y, groups)]
    plain = [len(tr) for tr, _ in GroupKFold(5).split(X, y, groups)]

    control = splitter.matched_control(seed=0)
    control_sizes = [len(tr) for tr, _ in control.split(X, y, groups)]
    assert control_sizes == kept
    assert [p - c for p, c in zip(plain, control_sizes)] == splitter.training_cost()[
        "rows_dropped_per_fold"]


def test_the_control_also_leaves_the_test_folds_alone():
    X, y, groups, cells = toy()
    splitter = LeakFreeGroupKFold(5, cells)
    list(splitter.split(X, y, groups))
    plain = [list(te) for _, te in GroupKFold(5).split(X, y, groups)]
    got = [list(te) for _, te in splitter.matched_control().split(X, y, groups)]
    assert plain == got


def test_different_seeds_drop_different_rows():
    """Otherwise averaging over seeds would be averaging one number with itself."""
    X, y, groups, cells = toy()
    splitter = LeakFreeGroupKFold(5, cells)
    list(splitter.split(X, y, groups))
    a = [set(tr) for tr, _ in splitter.matched_control(seed=0).split(X, y, groups)]
    b = [set(tr) for tr, _ in splitter.matched_control(seed=1).split(X, y, groups)]
    assert a != b


def test_the_control_cannot_be_built_before_the_filter_has_run():
    _, _, _, cells = toy()
    with pytest.raises(RuntimeError, match="run split"):
        LeakFreeGroupKFold(5, cells).matched_control()


def test_drop_counts_must_match_the_fold_count():
    with pytest.raises(ValueError, match="one drop count per fold"):
        ThinnedGroupKFold(5, [1, 2, 3])
