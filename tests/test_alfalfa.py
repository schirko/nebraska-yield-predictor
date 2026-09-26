"""Offline tests for the alfalfa data: NASS cleaning, weather features, and the coverage checks."""

import numpy as np
import pandas as pd
import pytest

from yieldpred.alfalfa import compare_with_state, coverage, model_table, season_features
from yieldpred.nass import tidy_county_alfalfa, tidy_state_alfalfa


def _row(county_code, stat, unit, value, practice="ALL PRODUCTION PRACTICES", year="2015", name="CUSTER"):
    return {
        "state_fips_code": "31", "state_alpha": "NE", "county_code": county_code, "county_name": name,
        "asd_code": "50", "asd_desc": "CENTRAL", "year": year, "prodn_practice_desc": practice,
        "statisticcat_desc": stat, "unit_desc": unit, "Value": value,
    }


def test_county_rows_become_one_row_per_county_year_and_practice():
    raw = pd.DataFrame([
        _row("041", "YIELD", "TONS / ACRE", "3.80"),
        _row("041", "AREA HARVESTED", "ACRES", "52,000"),
        _row("041", "PRODUCTION", "TONS", "197,600"),
        _row("041", "YIELD", "TONS / ACRE", "5.10", practice="IRRIGATED"),
        _row("041", "YIELD", "TONS / ACRE", "(D)", practice="NON-IRRIGATED"),        # withheld: no row
        _row("041", "PRICE RECEIVED", "$ / TON", "150"),                            # not one of ours
        _row("998", "YIELD", "TONS / ACRE", "3.1", name="OTHER (COMBINED) COUNTIES"),  # not a county
    ])
    df = tidy_county_alfalfa(raw)
    assert list(df["practice"]) == ["all", "irrigated"]
    row = df[df["practice"] == "all"].iloc[0]
    assert (row["fips"], row["county_name"], row["year"]) == ("31041", "Custer", 2015)
    assert (row["yield_tons_acre"], row["acres_harvested"], row["production_tons"]) == (3.8, 52000, 197600)
    assert np.isnan(df.loc[df["practice"] == "irrigated", "acres_harvested"].item())


def test_state_rows_and_empty_input():
    raw = pd.DataFrame([{**_row("", "YIELD", "TONS / ACRE", "3.9"), "county_code": None},
                        {**_row("", "AREA HARVESTED", "ACRES", "800,000"), "county_code": None}])
    st = tidy_state_alfalfa(raw)
    assert st[["year", "yield_tons_acre", "acres_harvested"]].values.tolist() == [[2015, 3.9, 800000]]
    assert tidy_county_alfalfa(pd.DataFrame()).empty and tidy_state_alfalfa(pd.DataFrame()).empty


def _daily(start="2014-01-01", end="2015-12-31", tmax=20.0, tmin=10.0, precip=1.0):
    dates = pd.date_range(start, end, freq="D")
    return pd.DataFrame({"date": dates, "tmax_c": tmax, "tmin_c": tmin, "precip_mm": precip})


def test_season_features_known_answers():
    daily = _daily()
    # Make the winter before 2015 bite: three nights below -18 C, the worst -25 C.
    for day, t in [("2014-12-20", -20.0), ("2015-01-10", -25.0), ("2015-02-02", -19.0)]:
        daily.loc[daily["date"] == day, "tmin_c"] = t
    f = season_features(daily).set_index("year")

    # (20 + 10) / 2 - 5 = 10 GDD a day; April-September has 183 days.
    assert f.loc[2015, "gdd_season"] == pytest.approx(1830.0)
    assert f.loc[2015, "precip_season_mm"] == pytest.approx(183.0)
    assert f.loc[2015, "precip_early_mm"] == pytest.approx(61.0)    # May 31 + June 30
    assert f.loc[2015, "precip_late_mm"] == pytest.approx(62.0)     # July 31 + August 31
    assert f.loc[2015, "heat_days"] == 0
    assert f.loc[2015, "winter_min_c"] == -25.0 and f.loc[2015, "freeze_days"] == 3
    # 2014 has no December before it in the file, so no winter rather than half a winter.
    assert np.isnan(f.loc[2014, "winter_min_c"])


def test_gdd_is_capped_and_floored():
    hot = season_features(_daily(tmax=40.0, tmin=0.0)).set_index("year")
    # High capped at 30, low floored at 5: (30 + 5) / 2 - 5 = 12.5 a day.
    assert hot.loc[2015, "gdd_season"] == pytest.approx(183 * 12.5)
    assert hot.loc[2015, "heat_days"] == 62


def _county(rows):
    return pd.DataFrame([{"fips": f, "county_name": f, "asd_desc": "X", "year": y, "practice": "all",
                          "yield_tons_acre": yld, "acres_harvested": ac} for f, y, yld, ac in rows])


def test_coverage_and_the_state_check():
    county = _county([("31001", 2010, 4.0, 30_000), ("31003", 2010, 2.0, 10_000),
                      ("31005", 2010, np.nan, 10_000)])            # yield withheld, acres known
    state = pd.DataFrame({"year": [2010], "practice": ["all"], "yield_tons_acre": [3.6],
                          "acres_harvested": [100_000]})
    cov = coverage(county, state).iloc[0]
    assert (cov["counties_with_yield"], cov["counties_listed"]) == (2, 3)
    assert cov["share_of_state_acres"] == pytest.approx(0.5)
    check = compare_with_state(county, state).iloc[0]
    # (4.0 * 30,000 + 2.0 * 10,000) / 40,000 = 3.5; the withheld county can't be in the average.
    assert check["county_weighted_yield"] == pytest.approx(3.5)
    assert check["difference"] == pytest.approx(-0.1)


def test_model_table_keeps_only_published_yields_with_weather():
    county = _county([("31001", 2010, 4.0, 30_000), ("31001", 2011, np.nan, 30_000), ("31003", 2010, 2.0, 1)])
    weather = pd.DataFrame({"fips": ["31001", "31001"], "year": [2010, 2011], "gdd_season": [2000.0, 2100.0]})
    table = model_table(county, weather)
    assert table[["fips", "year"]].values.tolist() == [["31001", 2010]]
    assert table["gdd_season"].item() == 2000.0
