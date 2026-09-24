"""Offline tests for the weather feature calculations."""

import pandas as pd

from yieldpred.weather import c_to_f, growing_season_features, spring_features


def make_daily(year: int = 2020, tmax_c: float = 30.0, tmin_c: float = 15.0,
               precip_mm: float = 2.0) -> pd.DataFrame:
    dates = pd.date_range(f"{year}-01-01", f"{year}-12-31", freq="D")
    return pd.DataFrame({"date": dates, "tmax_c": tmax_c, "tmin_c": tmin_c,
                         "precip_mm": precip_mm})


def test_c_to_f():
    assert c_to_f(pd.Series([0, 100])).tolist() == [32.0, 212.0]


def test_gdd_uses_base_50_and_cap_86():
    # tmax 40 C = 104 F -> capped at 86; tmin 5 C = 41 F -> floored at 50
    daily = make_daily(tmax_c=40.0, tmin_c=5.0)
    out = growing_season_features(daily)
    days = 30 + 31 + 30 + 31 + 31 + 30  # April-September
    expected = days * ((86 + 50) / 2 - 50)
    assert out["gdd"].item() == round(expected, 1)


def test_season_and_month_windows():
    daily = make_daily(precip_mm=2.0)
    out = growing_season_features(daily).iloc[0]
    assert out["precip_mm"] == 183 * 2          # 183 days, April-September
    assert out["precip_jul_mm"] == 31 * 2
    assert out["precip_aug_mm"] == 31 * 2
    assert out["heat_days_32"] == 0             # 30 C is not above 32
    assert out["dry_spell_max"] == 0            # 2 mm/day everywhere


def test_heat_days_and_dry_spell():
    daily = make_daily(tmax_c=36.0, precip_mm=0.0)
    daily.loc[daily["date"].dt.month == 7, "precip_mm"] = 5.0  # rain only in July
    out = growing_season_features(daily).iloc[0]
    assert out["heat_days_32"] == 62            # all of July and August
    assert out["heat_days_35"] == 62
    assert out["dry_spell_max"] == 91           # April through June, dry


def test_multiple_years():
    daily = pd.concat([make_daily(2019), make_daily(2020)], ignore_index=True)
    out = growing_season_features(daily)
    assert out["year"].tolist() == [2019, 2020]


# ------------------------------------------------- spring / planting features

def prepared(daily: pd.DataFrame) -> pd.DataFrame:
    """The frame shape `spring_features` expects, without re-deriving GDD by hand."""
    df = daily.copy()
    df["date"] = pd.to_datetime(df["date"])
    df["year"] = df["date"].dt.year
    df["month"] = df["date"].dt.month
    df["gdd"] = 0.0
    return df


def test_spring_windows_are_disjoint():
    out = spring_features(prepared(make_daily(precip_mm=2.0))).iloc[0]
    assert out["precip_mar_mm"] == 31 * 2                 # March only
    assert out["precip_apr_may_mm"] == (30 + 31) * 2      # April and May only


def test_spring_features_appear_in_the_season_summary():
    out = growing_season_features(make_daily()).iloc[0]
    assert {"precip_mar_mm", "workable_days", "last_frost_doy", "gdd_may"} <= set(out.index)


def test_spring_features_are_skipped_without_spring_data():
    """An April-September download still works - it just gets the summer features."""
    daily = make_daily()
    summer_only = daily[daily["date"].dt.month.between(4, 9)]
    assert spring_features(prepared(summer_only)) is None
    assert "precip_mar_mm" not in growing_season_features(summer_only).columns


def test_every_dry_day_in_the_window_is_workable():
    out = spring_features(prepared(make_daily(precip_mm=0.0))).iloc[0]
    assert out["workable_days"] == 21 + 31                # April 10-30, plus May


def test_daily_rain_alone_makes_a_day_unworkable():
    daily = make_daily(precip_mm=0.0)
    daily.loc[daily["date"] == "2020-05-01", "precip_mm"] = 3.0   # above 2.5 mm
    out = spring_features(prepared(daily)).iloc[0]
    assert out["workable_days"] == 21 + 31 - 1


def test_a_soaking_keeps_the_field_out_for_the_days_after():
    """The point of the feature: 20 mm on one day costs more than that one day."""
    daily = make_daily(precip_mm=0.0)
    daily.loc[daily["date"] == "2020-05-01", "precip_mm"] = 20.0
    out = spring_features(prepared(daily)).iloc[0]
    # The wet day itself, plus May 2nd and 3rd while the antecedent total stays high.
    assert out["workable_days"] == 21 + 31 - 3


def test_the_same_rain_spread_out_costs_far_less():
    """Same total, different timing - the comparison total rainfall cannot make."""
    spread = make_daily(precip_mm=0.0)
    for day in range(1, 11):
        spread.loc[spread["date"] == f"2020-05-{day:02d}", "precip_mm"] = 2.0

    soaked = make_daily(precip_mm=0.0)
    soaked.loc[soaked["date"] == "2020-05-01", "precip_mm"] = 20.0

    assert (spread["precip_mm"].sum() == soaked["precip_mm"].sum())
    assert (spring_features(prepared(spread))["workable_days"].item()
            > spring_features(prepared(soaked))["workable_days"].item())


def test_last_frost_is_the_latest_freeze_before_july():
    daily = make_daily(tmin_c=15.0)
    for date in ("2020-03-05", "2020-04-20", "2020-12-15"):
        daily.loc[daily["date"] == date, "tmin_c"] = -2.0
    out = spring_features(prepared(daily)).iloc[0]
    assert out["last_frost_doy"] == pd.Timestamp("2020-04-20").dayofyear


def test_no_frost_recorded_gives_the_sentinel():
    out = spring_features(prepared(make_daily(tmin_c=15.0))).iloc[0]
    assert out["last_frost_doy"] == 0
