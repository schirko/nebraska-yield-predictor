"""Offline tests for the app's data-shaping helpers."""

import pandas as pd

from yieldpred.appdata import (county_choices, county_history, feature_correlations,
                               irrigation_gap, missing_data_message, scores_by_scheme,
                               state_yield_history, worst_years)


def yields_frame() -> pd.DataFrame:
    rows = []
    for year in (2015, 2016):
        for fips, name in [("31109", "Lancaster"), ("31047", "Dawson")]:
            rows += [
                {"fips": fips, "county_name": name, "year": year, "practice": "all",
                 "yield_bu_acre": 180.0},
                {"fips": fips, "county_name": name, "year": year, "practice": "irrigated",
                 "yield_bu_acre": 210.0},
                {"fips": fips, "county_name": name, "year": year,
                 "practice": "non_irrigated", "yield_bu_acre": 150.0},
            ]
    return pd.DataFrame(rows)


def model_frame() -> pd.DataFrame:
    return pd.DataFrame({
        "fips": ["31109"] * 3 + ["31047"] * 3,
        "county_name": ["Lancaster"] * 3 + ["Dawson"] * 3,
        "year": [2015, 2016, 2017] * 2,
        "yield_bu_acre": [170.0, 180.0, 190.0, 160.0, 175.0, 185.0],
        "gdd": [3300.0, 3400, 3500, 3200, 3350, 3450],
        "heat_days_32": [30.0, 25, 20, 35, 28, 22],
        "irrigation_share": [0.3, 0.3, 0.3, 0.8, 0.8, 0.8],
    })


def test_state_yield_history_averages_by_practice():
    out = state_yield_history(yields_frame())
    assert list(out.columns) == ["all", "irrigated", "non_irrigated"]
    assert out.loc[2015, "irrigated"] == 210.0


def test_irrigation_gap():
    gap = irrigation_gap(yields_frame())
    assert gap["gap"].tolist() == [60.0, 60.0]
    assert gap["counties"].tolist() == [2, 2]


def test_irrigation_gap_handles_missing_practice_data():
    only_all = yields_frame().query("practice == 'all'")
    assert irrigation_gap(only_all).empty


def test_county_history_is_newest_first():
    history = county_history(model_frame(), "31109")
    assert history["year"].tolist() == [2017, 2016, 2015]
    assert "gdd" in history.columns


def test_county_choices_sorted_by_name():
    choices = county_choices(model_frame())
    assert choices["county_name"].tolist() == ["Dawson", "Lancaster"]


def test_feature_correlations_sorted_by_strength():
    corr = feature_correlations(model_frame())
    assert corr["correlation"].abs().is_monotonic_decreasing
    assert set(corr["label"]) <= {"Year (trend)", "Growing degree days",
                                  "Days above 32°C", "Irrigation share"}


def test_scores_by_scheme_renames_columns():
    scores = pd.DataFrame({"model": ["Ridge"], "random_RMSE": [24.7], "random_R2": [0.4],
                           "spatial_RMSE": [27.7], "spatial_R2": [0.25]})
    out = scores_by_scheme(scores, "spatial")
    assert list(out.columns) == ["RMSE", "R2"]
    assert out.loc["Ridge", "R2"] == 0.25


def test_worst_years_ranks_by_absolute_error():
    errors = pd.DataFrame({"year": [2019, 2020, 2021], "fips": ["31109"] * 3,
                           "error": [9.9, -0.6, -9.4]})
    out = worst_years(errors)
    assert out["year"].tolist() == [2019, 2021, 2020]


def test_missing_data_message_names_the_script():
    assert missing_data_message({"yields": True, "model": True}) is None
    message = missing_data_message({"yields": False, "model": False})
    assert "fetch_nass_yields.py" in message and "train_baseline.py" in message
