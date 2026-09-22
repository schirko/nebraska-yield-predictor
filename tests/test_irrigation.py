"""Offline tests for the irrigation-share feature."""

import pandas as pd

from yieldpred.irrigation import (build_share_series, combine_sources,
                                  observed_share, tidy_acres)


def acre_row(county_code, practice, value, year="2017"):
    return {"state_fips_code": "31", "county_code": county_code, "county_name": "X",
            "year": year, "prodn_practice_desc": practice, "Value": value}


def test_tidy_acres_drops_combined_counties_and_parses_values():
    raw = pd.DataFrame([
        acre_row("109", "ALL PRODUCTION PRACTICES", "120,000"),
        acre_row("109", "IRRIGATED", "72,000"),
        acre_row("998", "IRRIGATED", "5,000"),
        acre_row("047", "ALL PRODUCTION PRACTICES", "(D)"),
    ])
    out = tidy_acres(raw)
    assert set(out["fips"]) == {"31109"}
    assert out["acres"].max() == 120000


def test_observed_share():
    acres = pd.DataFrame({
        "fips": ["31109", "31109", "31047"],
        "year": [2017, 2017, 2017],
        "practice": ["all", "irrigated", "all"],
        "acres": [120000.0, 72000.0, 90000.0],
        "source": "SURVEY",
    })
    share = observed_share(acres)
    assert len(share) == 1  # 31047 has no irrigated row
    assert share["irrigation_share"].item() == 0.6


def test_build_share_series_interpolates_and_flags():
    observed = pd.DataFrame({"fips": ["31109", "31109"], "year": [2000, 2004],
                             "irrigation_share": [0.40, 0.60]})
    series = build_share_series(observed, range(2000, 2007))

    assert len(series) == 7
    mid = series.loc[series["year"] == 2002, "irrigation_share"].item()
    assert mid == 0.50                                     # linear interpolation
    assert series.loc[series["year"] == 2006, "irrigation_share"].item() == 0.60  # carried forward
    assert series["share_observed"].sum() == 2             # only the two real values


def test_combine_sources_prefers_survey():
    survey = pd.DataFrame({"fips": ["31109"], "year": [2012], "irrigation_share": [0.55]})
    census = pd.DataFrame({"fips": ["31109", "31109"], "year": [2012, 2022],
                           "irrigation_share": [0.99, 0.61]})
    out = combine_sources(survey, census)
    assert len(out) == 2
    assert out.loc[out["year"] == 2012, "irrigation_share"].item() == 0.55
