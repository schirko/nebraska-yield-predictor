"""Offline tests for the soil-rating helpers (no network)."""

import pandas as pd

from yieldpred.soils import counties_in_area, to_counties


def test_counties_in_area_single():
    assert counties_in_area("Adams County, Nebraska") == ["Adams"]


def test_counties_in_area_multiple():
    assert counties_in_area("Boyd and Keya Paha Counties, Nebraska") == ["Boyd", "Keya Paha"]


def test_counties_in_area_ignores_trailing_qualifiers():
    assert counties_in_area("Cherry County, Nebraska, Northern Part") == ["Cherry"]


def test_to_counties_expands_multi_county_areas():
    areas = pd.DataFrame([
        {"areasymbol": "NE001", "areaname": "Adams County, Nebraska",
         "nccpi_corn": 0.72, "rated_acres": 100000.0},
        {"areasymbol": "NE015", "areaname": "Boyd and Keya Paha Counties, Nebraska",
         "nccpi_corn": 0.31, "rated_acres": 200000.0},
    ])
    lookup = pd.DataFrame({"fips": ["31001", "31015", "31103"],
                           "county_name": ["Adams", "Boyd", "Keya Paha"]})

    out = to_counties(areas, lookup)
    assert set(out["county_name"]) == {"Adams", "Boyd", "Keya Paha"}
    assert out.loc[out["county_name"] == "Adams", "nccpi_corn"].item() == 0.72
    # Both counties of the combined survey area inherit its rating
    assert out.loc[out["county_name"] == "Boyd", "nccpi_corn"].item() == 0.31


def test_to_counties_acre_weights_overlapping_areas():
    areas = pd.DataFrame([
        {"areasymbol": "NE601", "areaname": "Cherry County, Nebraska, Northern Part",
         "nccpi_corn": 0.20, "rated_acres": 300000.0},
        {"areasymbol": "NE602", "areaname": "Cherry County, Nebraska, Southern Part",
         "nccpi_corn": 0.40, "rated_acres": 100000.0},
    ])
    lookup = pd.DataFrame({"fips": ["31031"], "county_name": ["Cherry"]})

    out = to_counties(areas, lookup)
    assert len(out) == 1
    # (0.20*300000 + 0.40*100000) / 400000 = 0.25
    assert out["nccpi_corn"].item() == 0.25
    assert out["survey_areas"].item() == 2


def test_to_counties_drops_areas_with_no_matching_county():
    areas = pd.DataFrame([{"areasymbol": "IA001", "areaname": "Adair County, Iowa",
                           "nccpi_corn": 0.8, "rated_acres": 1000.0}])
    lookup = pd.DataFrame({"fips": ["31001"], "county_name": ["Adams"]})
    assert to_counties(areas, lookup).empty
