"""Offline tests for the app's data-shaping helpers."""

import pandas as pd

import numpy as np
import pytest

from yieldpred.appdata import (_stem, county_choices, county_history, county_skill,
                               county_values, feature_correlations, irrigation_gap,
                               map_frame, missing_data_message, scores_by_scheme,
                               state_yield_history, transfer_by_year,
                               variance_decomposition, worst_years)


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


def test_map_frame_joins_features_when_available():
    errors = pd.DataFrame({"fips": ["31109"], "county_name": ["Lancaster"],
                           "year": [2015], "yield_bu_acre": [170.0],
                           "predicted": [175.0], "error": [5.0]})
    table = model_frame().assign(soil_water_cm=20.0, elevation_m=400.0)
    out = map_frame(errors, table)
    assert {"irrigation_share", "soil_water_cm", "elevation_m"} <= set(out.columns)
    assert out["irrigation_share"].item() == 0.3


def test_map_frame_without_model_table():
    errors = pd.DataFrame({"fips": ["31109"], "county_name": ["Lancaster"],
                           "year": [2015], "error": [5.0]})
    assert list(map_frame(errors, None).columns) == list(errors.columns)


def test_county_values_single_year_and_average():
    frame = pd.DataFrame({
        "fips": ["31109", "31109", "31047"], "county_name": ["Lancaster"] * 2 + ["Dawson"],
        "year": [2015, 2016, 2015], "yield_bu_acre": [170.0, 190.0, 160.0],
        "predicted": [175.0, 185.0, 165.0], "error": [5.0, -5.0, 5.0]})

    one_year = county_values(frame, "error", 2015)
    assert len(one_year) == 2
    assert set(one_year.columns) >= {"fips", "county_name", "error", "yield_bu_acre"}

    averaged = county_values(frame, "error", None)
    lancaster = averaged.loc[averaged["fips"] == "31109"]
    assert lancaster["error"].item() == 0.0          # +5 and -5 average out
    assert lancaster["yield_bu_acre"].item() == 180.0


# ----------------------------------------------- state-suffixed file naming

def test_stem_leaves_nebraska_unsuffixed():
    assert _stem("model_table", "ne") == "model_table.parquet"
    assert _stem("model_table", "ia") == "model_table_ia.parquet"


# ------------------------------------------------- variance decomposition

def panel(values: dict[str, list[float]], years=(2015, 2016)) -> pd.DataFrame:
    rows = [{"fips": fips, "year": year, "yield_bu_acre": value}
            for fips, series in values.items()
            for year, value in zip(years, series)]
    return pd.DataFrame(rows)


def test_variance_is_all_temporal_when_counties_move_together():
    shares = variance_decomposition(panel({"A": [100.0, 120.0], "B": [100.0, 120.0]}))
    assert shares["year"] == pytest.approx(1.0)
    assert shares["county"] == pytest.approx(0.0)


def test_variance_is_all_spatial_when_counties_never_change():
    shares = variance_decomposition(panel({"A": [100.0, 100.0], "B": [120.0, 120.0]}))
    assert shares["county"] == pytest.approx(1.0)
    assert shares["year"] == pytest.approx(0.0)


def test_variance_shares_are_a_partition():
    frame = panel({"A": [100.0, 135.0], "B": [118.0, 121.0], "C": [90.0, 160.0]})
    shares = variance_decomposition(frame)
    assert sum(shares.values()) == pytest.approx(1.0)
    assert all(value >= 0 for value in shares.values())


def test_variance_decomposition_survives_a_constant_column():
    shares = variance_decomposition(panel({"A": [150.0, 150.0], "B": [150.0, 150.0]}))
    assert shares == {"year": 0.0, "county": 0.0, "residual": 0.0}


# ------------------------------------------------------- transfer scoring

def transfer_frame() -> pd.DataFrame:
    """Two years: 2015 ranks counties perfectly, 2016 ranks them backwards."""
    return pd.DataFrame({
        "fips": ["19001", "19003", "19005"] * 2,
        "county_name": ["Adair", "Adams", "Allamakee"] * 2,
        "year": [2015] * 3 + [2016] * 3,
        "yield_bu_acre": [150.0, 170.0, 190.0, 150.0, 170.0, 190.0],
        "predicted": [155.0, 175.0, 195.0, 195.0, 175.0, 155.0],
    }).assign(error=lambda d: d["predicted"] - d["yield_bu_acre"])


def test_transfer_by_year_separates_level_error_from_ranking_skill():
    out = transfer_by_year(transfer_frame()).set_index("year")

    # Both years are +5 bu/acre high on average - identical level error...
    assert out.loc[2015, "mean_error"] == 5.0
    assert out.loc[2016, "mean_error"] == 5.0

    # ...but one ranks the counties perfectly and the other gets it exactly backwards.
    assert out.loc[2015, "county_skill"] == pytest.approx(1.0)
    assert out.loc[2016, "county_skill"] == pytest.approx(-1.0)
    assert out.loc[2015, "counties"] == 3


def test_county_skill_is_the_median_across_years():
    assert county_skill(transfer_frame()) == pytest.approx(0.0)


def test_county_skill_ignores_years_it_cannot_score():
    """A year with one county has no within-year correlation; it shouldn't poison the median."""
    frame = pd.concat([
        transfer_frame(),
        pd.DataFrame({"fips": ["19001"], "county_name": ["Adair"], "year": [2017],
                      "yield_bu_acre": [180.0], "predicted": [180.0], "error": [0.0]}),
    ], ignore_index=True)
    assert not np.isnan(county_skill(frame))
