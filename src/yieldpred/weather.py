"""County growing-season weather features from NASA POWER.

NASA POWER (https://power.larc.nasa.gov) serves free daily gridded weather
(~0.5 degree) with no API key. We query one point per county - its internal
point from the Census Gazetteer file - and summarize each year's growing
season into the handful of variables that drive corn yields.

Limitation to document later: a single point per county ignores where the
corn actually is. A cropland-weighted average (CDL mask + zonal statistics)
is the planned improvement.
"""

from __future__ import annotations

import pandas as pd
import requests

POWER_URL = "https://power.larc.nasa.gov/api/temporal/daily/point"
GAZETTEER_URL = (
    "https://www2.census.gov/geo/docs/maps-data/data/gazetteer/"
    "2023_Gazetteer/2023_Gaz_counties_national.zip"
)

# T2M_MAX/MIN: daily max/min air temperature at 2 m (C)
# PRECTOTCORR:  bias-corrected total precipitation (mm/day)
POWER_PARAMS = ["T2M_MAX", "T2M_MIN", "PRECTOTCORR"]
POWER_FILL = -999.0  # POWER's missing-value flag

# Corn growing degree days, US convention (degrees F, base 50, cap 86)
GDD_BASE_F = 50.0
GDD_CAP_F = 86.0

GROWING_SEASON = (4, 9)  # April through September


def c_to_f(celsius: pd.Series) -> pd.Series:
    return celsius * 9.0 / 5.0 + 32.0


def county_points(state_fips: str = "31") -> pd.DataFrame:
    """Internal point (lat/lon) for each county in a state, from the Census Gazetteer."""
    gaz = pd.read_csv(GAZETTEER_URL, sep="\t", dtype={"GEOID": str})
    gaz.columns = [c.strip() for c in gaz.columns]
    gaz = gaz[gaz["GEOID"].str.startswith(state_fips)]
    return (gaz[["GEOID", "NAME", "INTPTLAT", "INTPTLONG"]]
            .rename(columns={"GEOID": "fips", "NAME": "county_name",
                             "INTPTLAT": "lat", "INTPTLONG": "lon"})
            .reset_index(drop=True))


def fetch_power_daily(lat: float, lon: float, start_year: int, end_year: int,
                      timeout: int = 120) -> pd.DataFrame:
    """Daily weather for one point. Returns columns: date, tmax_c, tmin_c, precip_mm."""
    params = {
        "parameters": ",".join(POWER_PARAMS),
        "community": "AG",
        "latitude": lat,
        "longitude": lon,
        "start": f"{start_year}0101",
        "end": f"{end_year}1231",
        "format": "JSON",
    }
    resp = requests.get(POWER_URL, params=params, timeout=timeout)
    resp.raise_for_status()
    block = resp.json()["properties"]["parameter"]

    df = pd.DataFrame({
        "tmax_c": pd.Series(block["T2M_MAX"]),
        "tmin_c": pd.Series(block["T2M_MIN"]),
        "precip_mm": pd.Series(block["PRECTOTCORR"]),
    })
    df = df.replace(POWER_FILL, pd.NA).astype(float)
    df.index = pd.to_datetime(df.index, format="%Y%m%d")
    return df.rename_axis("date").reset_index()


def growing_season_features(daily: pd.DataFrame) -> pd.DataFrame:
    """Summarize daily weather into one row per year.

    Features (April-September unless noted):
      gdd            accumulated corn growing degree days (F, base 50, cap 86)
      precip_mm      total precipitation
      precip_jul_mm  July precipitation (pollination - the critical month)
      precip_aug_mm  August precipitation (grain fill)
      tmax_jul_c     mean daily high in July
      heat_days_32   days with a high above 32 C (~90 F), July-August
      heat_days_35   days with a high above 35 C (~95 F), July-August
      dry_spell_max  longest run of days under 1 mm of rain
    """
    df = daily.copy()
    df["date"] = pd.to_datetime(df["date"])
    df["year"] = df["date"].dt.year
    df["month"] = df["date"].dt.month

    tmax_f = c_to_f(df["tmax_c"]).clip(upper=GDD_CAP_F)
    tmin_f = c_to_f(df["tmin_c"]).clip(lower=GDD_BASE_F)
    df["gdd"] = ((tmax_f + tmin_f) / 2 - GDD_BASE_F).clip(lower=0)

    start, end = GROWING_SEASON
    season = df[df["month"].between(start, end)]
    summer = df[df["month"].between(7, 8)]
    july = df[df["month"] == 7]
    august = df[df["month"] == 8]

    out = pd.DataFrame({
        "gdd": season.groupby("year")["gdd"].sum().round(1),
        "precip_mm": season.groupby("year")["precip_mm"].sum().round(1),
        "precip_jul_mm": july.groupby("year")["precip_mm"].sum().round(1),
        "precip_aug_mm": august.groupby("year")["precip_mm"].sum().round(1),
        "tmax_jul_c": july.groupby("year")["tmax_c"].mean().round(2),
        "heat_days_32": summer.assign(hot=summer["tmax_c"] > 32).groupby("year")["hot"].sum(),
        "heat_days_35": summer.assign(hot=summer["tmax_c"] > 35).groupby("year")["hot"].sum(),
        "dry_spell_max": pd.Series({year: _longest_dry_spell(group)
                                    for year, group in season.groupby("year")}, dtype=int),
    })
    out.index.name = "year"
    return out.sort_index().reset_index()


def _longest_dry_spell(group: pd.DataFrame, threshold_mm: float = 1.0) -> int:
    dry = (group["precip_mm"] < threshold_mm).astype(int)
    longest = run = 0
    for value in dry:
        run = run + 1 if value else 0
        longest = max(longest, run)
    return longest
