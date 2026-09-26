"""The farm-account card: trend yield and its 1-in-10 low, checked against known answers."""

import csv

import pytest

from yieldpred import suite_card


def test_straight_line_is_recovered_exactly():
    history = {y: 100 + 2.0 * (y - 2000) for y in range(2000, 2020)}
    card = suite_card.county_card(history, "Test County", "NE", 2020)
    assert card["value"] == pytest.approx(140.0)
    assert card["low"] == pytest.approx(140.0)  # no year off the line, so no downside
    assert "rise about 2.0" in card["detail"]


def test_low_end_is_below_trend_but_above_the_worst_year():
    # A flat 200 with one bad year (140): the 1-in-10 low sits between the worst year and the trend.
    history = {2000 + i: 200.0 for i in range(10)}
    history[2004] = 140.0
    card = suite_card.county_card(history, "Test County", "NE", 2010)
    assert 140 < card["low"] < card["value"]


def test_quantile_matches_numpy_linear():
    assert suite_card.quantile([1, 2, 3, 4], 0.5) == 2.5
    assert suite_card.quantile([10, 20, 30, 40, 50], 0.1) == 14.0


def _write(path, rows):
    with open(path, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=["fips", "state_alpha", "county_name", "asd_code", "asd_desc", "year",
                                          "practice", "yield_bu_acre"])
        w.writeheader()
        w.writerows(rows)


def test_build_skips_short_and_stale_counties(tmp_path):
    rows = []
    for y in range(2000, 2026):  # a full county
        rows.append(dict(fips="31079", state_alpha="NE", county_name="HALL", asd_code=50, asd_desc="C", year=y,
                         practice="all", yield_bu_acre=150 + y - 2000))
        rows.append(dict(fips="31079", state_alpha="NE", county_name="HALL", asd_code=50, asd_desc="C", year=y,
                         practice="irrigated", yield_bu_acre=999))  # other practices are ignored
    for y in range(2000, 2015):  # stopped reporting in 2014
        rows.append(dict(fips="31001", state_alpha="NE", county_name="ADAMS", asd_code=80, asd_desc="S", year=y,
                         practice="all", yield_bu_acre=150))
    for y in range(2020, 2026):  # only six years
        rows.append(dict(fips="31003", state_alpha="NE", county_name="ANTELOPE", asd_code=30, asd_desc="N", year=y,
                         practice="all", yield_bu_acre=150))
    path = tmp_path / "ne_corn_yield_county.csv"
    _write(path, rows)
    data = suite_card.build([path])
    assert data["season"] == 2026 and set(data["cards"]) == {"31079"}
    assert data["cards"]["31079"]["value"] == pytest.approx(176.0)
    assert data["cards"]["31079"]["detail"].startswith("Hall County, NE")
    assert "since 2014" in data["skipped"]["31001"] and "6 years" in data["skipped"]["31003"]


def test_real_hall_county_matches_the_equipment_planner_notebook():
    """Equipment Planner notebook m7: Hall County corn trend +1.68 bu/yr, 2026 trend 212.7."""
    from pathlib import Path
    path = Path(__file__).resolve().parents[1] / "data" / "processed" / "ne_corn_yield_county.csv"
    if not path.exists():
        pytest.skip("processed yields not present")
    card = suite_card.build([path])["cards"]["31079"]
    assert card["value"] == pytest.approx(212.7, abs=0.05)
    assert "1.7 bu/acre a year" in card["detail"]
