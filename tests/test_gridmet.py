"""The gridMET path, tested without touching the network.

Every fetch here takes a `client` injection point, so the tests supply fake
responses shaped like pygridmet's and check the parts that are ours: the unit
conversion, the daily contract, and the spatial averaging.

The most important test in this file is `test_both_sources_feed_the_same_feature_code`.
The entire gridMET-vs-POWER comparison rests on the two products being summarised
by identical code — if the feature functions ever diverge, a score difference stops
meaning "the weather product changed" and starts meaning something unattributable.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from yieldpred import weather
from yieldpred.gridmet import (GRIDMET_VARS, KELVIN_OFFSET, _spatial_mean, _tidy,
                               compare_sources, county_mean_daily,
                               county_point_daily)


def fake_point_response(days: int = 400, kelvin: float = 300.0) -> pd.DataFrame:
    """What pygridmet's get_bycoords returns: a date-indexed frame."""
    dates = pd.date_range("2000-01-01", periods=days, freq="D")
    return pd.DataFrame(
        {"tmmx": kelvin, "tmmn": kelvin - 10.0, "pr": 2.0},
        index=pd.Index(dates, name="date"))


# ------------------------------------------------------------------- units

def test_kelvin_becomes_celsius():
    tidy = _tidy(fake_point_response(kelvin=300.0))
    assert tidy["tmax_c"].iloc[0] == pytest.approx(300.0 - KELVIN_OFFSET)
    assert tidy["tmin_c"].iloc[0] == pytest.approx(290.0 - KELVIN_OFFSET)


def test_precipitation_is_left_alone():
    """gridMET's pr is already mm, like POWER's PRECTOTCORR. Don't 'fix' it."""
    assert _tidy(fake_point_response())["precip_mm"].iloc[0] == pytest.approx(2.0)


def test_forgetting_the_conversion_would_be_caught_by_the_features():
    """Why the unit matters: raw Kelvin silently produces zero heat days.

    A unit bug here wouldn't raise - it would return a plausible-looking frame
    whose every threshold comparison is wrong in the same direction. This pins
    the failure mode so nobody 'simplifies' the conversion away.
    """
    correct = _tidy(fake_point_response(kelvin=310.0))     # 36.85 C - a hot day
    hot = weather.growing_season_features(correct)
    assert hot["heat_days_35"].iloc[0] > 0, "36.85 C must count as a 35 C day"

    # The failure that would actually ship: converting twice, or converting a
    # value that was already Celsius. Either way the frame looks fine and every
    # threshold silently reads cold, so no exception is ever raised.
    double_converted = correct.assign(tmax_c=correct["tmax_c"] - KELVIN_OFFSET)
    assert weather.growing_season_features(double_converted)["heat_days_35"].iloc[0] == 0


# -------------------------------------------------------------- the contract

def test_daily_frame_matches_what_power_returns():
    """Same columns, same names, same order - that is the whole design."""
    tidy = _tidy(fake_point_response())
    assert list(tidy.columns) == ["date", "tmax_c", "tmin_c", "precip_mm"]
    assert pd.api.types.is_datetime64_any_dtype(tidy["date"])


def test_missing_variable_is_an_error_with_a_useful_message():
    bad = fake_point_response().drop(columns="pr")
    with pytest.raises(KeyError, match="pr"):
        _tidy(bad)


def test_dates_come_back_sorted():
    shuffled = fake_point_response(days=50).sample(frac=1.0, random_state=0)
    tidy = _tidy(shuffled)
    assert tidy["date"].is_monotonic_increasing


# --------------------------------- regressions from the first real request

def test_unit_suffixed_column_names_are_accepted():
    """get_bycoords returns 'tmmx (K)', not 'tmmx'. Found on the first real county.

    The polygon path returns bare names and the point path returns suffixed ones,
    so matching the exact string passed one and failed the other. Both must work.
    """
    dates = pd.date_range("2000-01-01", periods=10, freq="D")
    suffixed = pd.DataFrame(
        {"tmmx (K)": 300.0, "tmmn (K)": 290.0, "pr (mm)": 2.0},
        index=pd.Index(dates, name="date"))
    tidy = _tidy(suffixed)
    assert list(tidy.columns) == ["date", "tmax_c", "tmin_c", "precip_mm"]
    assert tidy["tmax_c"].iloc[0] == pytest.approx(300.0 - KELVIN_OFFSET)


def test_bare_and_suffixed_names_give_identical_results():
    dates = pd.date_range("2000-01-01", periods=5, freq="D")
    values = {"tmmx": 301.5, "tmmn": 288.25, "pr": 3.5}
    bare = pd.DataFrame(values, index=pd.Index(dates, name="date"))
    suffixed = pd.DataFrame({f"{k} (K)" if k != "pr" else "pr (mm)": v
                             for k, v in values.items()},
                            index=pd.Index(dates, name="date"))
    pd.testing.assert_frame_equal(_tidy(bare), _tidy(suffixed))


def test_a_genuinely_missing_variable_still_raises():
    """Loosening the match must not make a real absence silent."""
    dates = pd.date_range("2000-01-01", periods=5, freq="D")
    frame = pd.DataFrame({"tmmx (K)": 300.0, "tmmn (K)": 290.0},
                         index=pd.Index(dates, name="date"))
    with pytest.raises(KeyError, match="pr"):
        _tidy(frame)


def test_an_unrelated_column_is_not_mistaken_for_a_wanted_one():
    dates = pd.date_range("2000-01-01", periods=5, freq="D")
    frame = pd.DataFrame(
        {"tmmx (K)": 300.0, "tmmn (K)": 290.0, "pr (mm)": 2.0,
         "prcp_other": 99.0, "srad (W/m2)": 210.0},
        index=pd.Index(dates, name="date"))
    tidy = _tidy(frame)
    assert tidy["precip_mm"].iloc[0] == pytest.approx(2.0), "took the wrong column"


def test_the_feature_summariser_is_called_exactly_once_per_source():
    """Regression: summarize() called spring_features on a raw frame -> KeyError.

    `growing_season_features` already calls `spring_features` internally, on a frame
    it prepared with year/month/gdd. Calling it again on the raw daily rows died with
    KeyError: 'month'. This pins the contract: the raw daily frame goes to
    growing_season_features and nothing else.
    """
    dates = pd.date_range("2000-01-01", periods=400, freq="D")
    daily = _tidy(pd.DataFrame(
        {"tmmx (K)": 300.0, "tmmn (K)": 288.0, "pr (mm)": 2.0},
        index=pd.Index(dates, name="date")))

    # The raw frame has no month column - that is the point.
    assert "month" not in daily.columns
    with pytest.raises(KeyError):
        weather.spring_features(daily)

    # But the supported entry point copes, and returns spring columns anyway.
    out = weather.growing_season_features(daily)
    for column in ("gdd", "precip_mm", "workable_days", "last_frost_doy"):
        assert column in out.columns, f"{column} missing from the summary"


# ------------------------------------------------------------ point fetching

def test_point_fetch_sends_lon_lat_in_that_order():
    """pygridmet takes (lon, lat). Swapping them silently returns another place."""
    seen = {}

    def fake(coords, dates, variables=None, crs=None):
        seen["coords"], seen["dates"], seen["variables"] = coords, dates, variables
        return fake_point_response()

    county_point_daily(41.5, -99.5, 2000, 2000, client=fake)
    assert seen["coords"] == (-99.5, 41.5), "longitude must come first"
    assert seen["dates"] == ("2000-01-01", "2000-12-31")
    assert seen["variables"] == GRIDMET_VARS


def test_years_before_the_dataset_starts_are_refused():
    with pytest.raises(ValueError, match="1979"):
        county_point_daily(41.5, -99.5, 1950, 2000, client=lambda *a, **k: None)


def test_backwards_year_range_is_refused():
    with pytest.raises(ValueError, match="before"):
        county_point_daily(41.5, -99.5, 2020, 2010, client=lambda *a, **k: None)


# ----------------------------------------------------------- spatial mean

class FakeGrid:
    """Minimal stand-in for the xarray Dataset get_bygeom returns."""

    def __init__(self, frame: pd.DataFrame, dims=("day", "y", "x")):
        self._frame = frame
        self.dims = dims

    def mean(self, dim, skipna=True):
        assert set(dim) == {"y", "x"}, f"should average over space, got {dim}"
        return FakeGrid(self._frame, dims=("day",))

    def to_dataframe(self):
        return self._frame


def test_county_mean_averages_over_space_not_time():
    frame = fake_point_response(days=30)
    captured = {}

    def fake(geometry, dates, variables=None, crs=None):
        captured["geometry"] = geometry
        return FakeGrid(frame)

    out = county_mean_daily("a-polygon", 2000, 2000, client=fake)
    assert captured["geometry"] == "a-polygon"
    assert len(out) == 30, "one row per day survives; the spatial dims collapse"
    assert list(out.columns) == ["date", "tmax_c", "tmin_c", "precip_mm"]


def test_spatial_dimensions_are_discovered_not_hardcoded():
    """The service may return lat/lon instead of x/y; both must work."""
    frame = fake_point_response(days=10)

    class LatLonGrid(FakeGrid):
        def mean(self, dim, skipna=True):
            assert set(dim) == {"lat", "lon"}
            return FakeGrid(frame, dims=("time",))

    out = _spatial_mean(LatLonGrid(frame, dims=("time", "lat", "lon")))
    assert len(out) == 10


def test_a_grid_with_no_spatial_dimensions_is_an_error():
    frame = fake_point_response(days=5)
    with pytest.raises(ValueError, match="no spatial dimensions"):
        _spatial_mean(FakeGrid(frame, dims=("day",)))


def test_a_grid_with_no_time_dimension_is_an_error():
    frame = fake_point_response(days=5)
    with pytest.raises(KeyError, match="time dimension"):
        _spatial_mean(FakeGrid(frame, dims=("y", "x")))


# ------------------------------------------------- the load-bearing property

def test_both_sources_feed_the_same_feature_code():
    """The comparison is only meaningful if nothing but the weather differs.

    Both products are summarised by `weather.growing_season_features`. This test
    runs the gridMET-shaped frame through it and asserts the output carries the
    same feature columns POWER's does - so a future refactor that gives gridMET
    its own summariser fails here rather than silently making every published
    comparison unattributable.
    """
    days = 400
    dates = pd.date_range("2000-01-01", periods=days, freq="D")
    rng = np.random.default_rng(0)
    power_like = pd.DataFrame({
        "date": dates,
        "tmax_c": rng.normal(25, 6, days),
        "tmin_c": rng.normal(12, 5, days),
        "precip_mm": rng.gamma(1.0, 2.0, days),
    })
    gridmet_like = _tidy(pd.DataFrame(
        {"tmmx": rng.normal(298, 6, days),
         "tmmn": rng.normal(285, 5, days),
         "pr": rng.gamma(1.0, 2.0, days)},
        index=pd.Index(dates, name="date")))

    from_power = weather.growing_season_features(power_like)
    from_gridmet = weather.growing_season_features(gridmet_like)
    assert list(from_power.columns) == list(from_gridmet.columns)
    assert len(from_power) == len(from_gridmet)


# ---------------------------------------------------------------- comparison

def test_compare_sources_reports_the_gap_per_feature():
    keys = pd.DataFrame({"fips": ["31001"] * 5, "year": range(2000, 2005)})
    power = keys.assign(gdd=[3000.0, 3100, 3200, 3300, 3400], precip_mm=400.0)
    grid = keys.assign(gdd=[3050.0, 3150, 3250, 3350, 3450], precip_mm=380.0)

    out = compare_sources(power, grid)
    assert out.loc["gdd", "mean_diff"] == pytest.approx(50.0)
    assert out.loc["precip_mm", "mean_diff"] == pytest.approx(-20.0)
    assert out.loc["gdd", "correlation"] == pytest.approx(1.0)
    assert out.loc["gdd", "mean_abs_diff"] == pytest.approx(50.0)


def test_compare_sources_ignores_columns_only_one_source_has():
    keys = pd.DataFrame({"fips": ["31001"] * 3, "year": range(2000, 2003)})
    power = keys.assign(gdd=1.0, county_name="Adams")
    grid = keys.assign(gdd=2.0)
    assert list(compare_sources(power, grid).index) == ["gdd"]
