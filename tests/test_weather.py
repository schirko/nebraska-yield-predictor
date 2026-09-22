"""Offline tests for the weather feature calculations."""

import pandas as pd

from yieldpred.weather import c_to_f, growing_season_features


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
