"""Offline tests for the NASS cleaning logic (no API key or internet needed)."""

import pandas as pd

from yieldpred.nass import parse_value, tidy_county_yields


def _row(county_code, practice, value, year="2023", name="LANCASTER"):
    return {
        "state_fips_code": "31", "state_alpha": "NE", "county_code": county_code,
        "county_name": name, "asd_code": "80", "asd_desc": "SOUTHEAST",
        "year": year, "prodn_practice_desc": practice, "Value": value,
    }


def test_parse_value_handles_commas_and_suppression():
    out = parse_value(pd.Series(["1,234.5", " 180 ", "(D)", "(NA)"]))
    assert out.iloc[0] == 1234.5
    assert out.iloc[1] == 180
    assert out.iloc[2:].isna().all()


def test_tidy_builds_fips_maps_practice_and_drops_junk():
    raw = pd.DataFrame([
        _row("109", "IRRIGATED", "215.3"),
        _row("109", "NON-IRRIGATED", "160.1"),
        _row("109", "ALL PRODUCTION PRACTICES", "(D)"),        # suppressed -> dropped
        _row("998", "IRRIGATED", "200", name="OTHER (COMBINED) COUNTIES"),  # not a county
        _row("1", "IRRIGATED", "205", name="ADAMS"),
    ])
    df = tidy_county_yields(raw)

    assert set(df["fips"]) == {"31109", "31001"}
    assert set(df["practice"]) == {"irrigated", "non_irrigated"}
    assert len(df) == 3
    assert df.loc[df["fips"] == "31001", "county_name"].item() == "Adams"


def test_tidy_empty_input():
    assert tidy_county_yields(pd.DataFrame()).empty
