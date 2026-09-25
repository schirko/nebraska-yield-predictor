"""Offline tests for the corn-soybean rotation features."""

import pandas as pd
import pytest

from yieldpred.rotation import (fill_gaps, harvested_ratio, rotation_features,
                                tidy_acres)


def acres(fips: str, years, values) -> pd.DataFrame:
    return pd.DataFrame({"fips": fips, "year": list(years), "acres": list(values)})


# ------------------------------------------------------------------ cleaning

def raw_rows(**overrides) -> pd.DataFrame:
    base = {"state_fips_code": ["31"], "county_code": ["109"], "year": ["2020"],
            "Value": ["120,000"], "prodn_practice_desc": ["ALL PRODUCTION PRACTICES"]}
    base.update(overrides)
    return pd.DataFrame(base)


def test_tidy_builds_fips_and_parses_thousands_separators():
    out = tidy_acres(raw_rows())
    assert out["fips"].item() == "31109"
    assert out["acres"].item() == 120000.0


def test_combined_counties_are_dropped():
    assert tidy_acres(raw_rows(county_code=["998"])).empty


def test_only_all_practice_rows_survive():
    """Keeping irrigated rows too would double-count a county's acres."""
    raw = pd.DataFrame({
        "state_fips_code": ["31"] * 3, "county_code": ["109"] * 3,
        "year": ["2020"] * 3, "Value": ["100,000", "60,000", "40,000"],
        "prodn_practice_desc": ["ALL PRODUCTION PRACTICES", "IRRIGATED",
                                "NON-IRRIGATED"]})
    out = tidy_acres(raw)
    assert len(out) == 1 and out["acres"].item() == 100000.0


def test_empty_input_returns_the_expected_columns():
    out = tidy_acres(pd.DataFrame())
    assert list(out.columns) == ["fips", "year", "acres"]


# ------------------------------------------------------------------ rotation

def test_corn_share_is_the_acreage_split():
    corn = acres("31109", [2020], [60_000])
    soy = acres("31109", [2020], [40_000])
    out = rotation_features(corn, soy)
    assert out["corn_share"].item() == pytest.approx(0.6)
    assert out["rotation_acres"].item() == 100_000


def test_a_clean_rotation_sits_near_a_half():
    corn = acres("31109", [2020], [50_000])
    soy = acres("31109", [2020], [50_000])
    assert rotation_features(corn, soy)["corn_share"].item() == pytest.approx(0.5)


def test_previous_year_soybean_share_is_last_years_complement():
    corn = acres("31109", [2019, 2020], [30_000, 70_000])
    soy = acres("31109", [2019, 2020], [70_000, 30_000])
    out = rotation_features(corn, soy).set_index("year")

    assert pd.isna(out.loc[2019, "soy_share_prev"])       # nothing before 2019
    assert out.loc[2020, "soy_share_prev"] == pytest.approx(0.7)


def test_the_lag_never_crosses_a_county_boundary():
    """A plain shift would hand one county's value to the next one in the table."""
    corn = pd.concat([acres("31109", [2019, 2020], [30_000, 70_000]),
                      acres("31047", [2020], [50_000])])
    soy = pd.concat([acres("31109", [2019, 2020], [70_000, 30_000]),
                     acres("31047", [2020], [50_000])])
    out = rotation_features(corn, soy)

    dawson = out[(out["fips"] == "31047") & (out["year"] == 2020)]
    assert pd.isna(dawson["soy_share_prev"].item())


def test_a_gap_in_the_record_does_not_count_as_last_year():
    corn = acres("31109", [2018, 2020], [30_000, 70_000])
    soy = acres("31109", [2018, 2020], [70_000, 30_000])
    out = rotation_features(corn, soy).set_index("year")
    assert pd.isna(out.loc[2020, "soy_share_prev"])       # 2018 is not "last year"


def test_counties_with_no_acres_are_dropped_rather_than_dividing_by_zero():
    corn = acres("31109", [2020], [0])
    soy = acres("31109", [2020], [0])
    assert rotation_features(corn, soy).empty


def test_missing_soybeans_yields_no_rotation():
    """A state that grows no soybeans has no rotation to measure."""
    out = rotation_features(acres("31109", [2020], [60_000]),
                            pd.DataFrame(columns=["fips", "year", "acres"]))
    assert out.empty
    assert "corn_share" in out.columns


def test_a_county_that_never_grows_soybeans_is_continuous_corn():
    """Absence is the answer here, not missing data.

    Nine Nebraska counties report corn and never soybeans - Sandhills ground where
    soybeans aren't grown. Their corn is 100% continuous, which is exactly what the
    feature exists to detect, so dropping them would delete the clearest cases.
    """
    corn = pd.concat([acres("31109", [2020], [60_000]),    # grows both
                      acres("31005", [2020], [9_000])])    # Arthur: corn only
    soy = acres("31109", [2020], [40_000])

    out = rotation_features(corn, soy).set_index("fips")
    assert out.loc["31109", "corn_share"] == pytest.approx(0.6)
    assert out.loc["31005", "corn_share"] == pytest.approx(1.0)
    assert bool(out.loc["31109", "rotation_observed"]) is True
    assert bool(out.loc["31005", "rotation_observed"]) is False


def test_one_suppressed_year_stays_missing():
    """A county that usually reports soybeans and skips a year is unknown, not zero."""
    corn = acres("31109", [2019, 2020], [60_000, 60_000])
    soy = acres("31109", [2019], [40_000])          # 2020 suppressed

    out = rotation_features(corn, soy)
    assert out["year"].tolist() == [2019]           # 2020 dropped, not set to 1.0


# -------------------------------------------------------- abandonment signal

def test_harvested_ratio_measures_abandonment():
    planted = acres("19153", [2019], [100_000])
    harvested = acres("19153", [2019], [82_000])
    out = harvested_ratio(planted, harvested)
    assert out["harvested_ratio"].item() == pytest.approx(0.82)


def test_harvested_ratio_is_not_a_model_feature():
    """Guard the distinction: abandonment is an outcome, not an input.

    If someone later adds it to ROTATION_FEATURES this test says why not.
    """
    from yieldpred.dataset import ROTATION_FEATURES
    assert "harvested_ratio" not in ROTATION_FEATURES


# ------------------------------------------------- silage correction

def test_silage_inflates_apparent_abandonment_when_ignored():
    """The bug this correction exists for: silage is subtracted from every county."""
    planted = acres("31109", [2020], [100_000])
    grain = acres("31109", [2020], [88_000])
    silage = acres("31109", [2020], [12_000])

    uncorrected = harvested_ratio(planted, grain)
    assert uncorrected["harvested_ratio"].item() == pytest.approx(0.88)
    assert not uncorrected["silage_corrected"].item()

    corrected = harvested_ratio(planted, grain, silage)
    assert corrected["harvested_ratio"].item() == pytest.approx(1.0)
    assert corrected["silage_share"].item() == pytest.approx(0.12)
    assert corrected["silage_corrected"].item()


def test_a_county_with_no_silage_row_is_treated_as_no_silage():
    planted = acres("31109", [2020], [100_000])
    grain = acres("31109", [2020], [95_000])
    silage = acres("31047", [2020], [5_000])          # a different county
    out = harvested_ratio(planted, grain, silage)
    assert out["acres_silage"].item() == 0.0
    assert out["harvested_ratio"].item() == pytest.approx(0.95)


def test_silage_exceeding_planted_is_dropped_rather_than_inverted():
    planted = acres("31009", [2020], [4_000])
    grain = acres("31009", [2020], [1_000])
    silage = acres("31009", [2020], [4_500])          # bought standing, or rounding
    assert harvested_ratio(planted, grain, silage).empty


# ------------------------------------------------------- gap filling

def observed_frame(years, shares) -> pd.DataFrame:
    return pd.DataFrame({"fips": "31109", "year": list(years),
                         "corn_share": list(shares),
                         "soy_share_prev": [0.5] * len(list(years)),
                         "rotation_acres": [100_000] * len(list(years)),
                         "rotation_observed": [True] * len(list(years))})


def test_gaps_are_carried_forward_and_flagged():
    out = fill_gaps(observed_frame([2000, 2002], [0.6, 0.7]), range(2000, 2003))
    assert out["year"].tolist() == [2000, 2001, 2002]
    assert out.loc[out["year"] == 2001, "corn_share"].item() == pytest.approx(0.6)
    assert out["rotation_observed"].tolist() == [True, False, True]


def test_carry_forward_never_reaches_backwards():
    """A temporal holdout must not be able to borrow from its own future."""
    out = fill_gaps(observed_frame([2002], [0.7]), range(2000, 2003))
    assert out["year"].tolist() == [2002]        # 2000 and 2001 stay missing


def test_filling_an_already_complete_series_changes_nothing():
    frame = observed_frame([2000, 2001], [0.6, 0.65])
    out = fill_gaps(frame, range(2000, 2002))
    assert out["rotation_observed"].all()
    assert out["corn_share"].tolist() == [0.6, 0.65]
