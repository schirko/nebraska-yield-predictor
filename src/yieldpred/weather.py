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

from pathlib import Path

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

# The spring window, which the growing-season features deliberately exclude.
# Corn is planted in late April and early May; what happens before that decides
# whether it gets planted on time, or at all. Nebraska's March 2019 flood and
# Iowa's record-wet 2013 spring are both invisible to an April-September window.
PLANTING_WINDOW = ("04-10", "05-31")

# A day is "workable" if it didn't rain much AND the soil has had a chance to dry.
# Thresholds follow the spirit of NASS's weekly "days suitable for fieldwork":
# machinery stays out of a wet field for a day or two after a soaking.
WORKABLE_RAIN_MM = 2.5          # rain on the day itself
WORKABLE_ANTECEDENT_MM = 12.5   # rain over the two preceding days
WORKABLE_LOOKBACK_DAYS = 2

FROST_C = 0.0
FROST_SEARCH_END_MONTH = 6  # a "spring" frost is one before July


def c_to_f(celsius: pd.Series) -> pd.Series:
    return celsius * 9.0 / 5.0 + 32.0


RAW = Path(__file__).resolve().parents[2] / "data" / "raw"


def county_points(state_fips: str = "31", use_cache: bool = True) -> pd.DataFrame:
    """Internal point (lat/lon) for each county in a state, from the Census Gazetteer.

    Cached to data/raw/ after the first successful download. County internal
    points do not change between Gazetteer editions in any way that matters here,
    and caching means a script that only needs gridMET does not also need Census
    to be reachable. Before the cache existed, a Census DNS failure could stop a
    gridMET fetch before it made a single gridMET request.

    The *internal point* is not the centroid: it is a point guaranteed to fall
    inside the polygon, which for a crescent-shaped county is somewhere the
    centroid is not. This matters because these are the exact coordinates sent to
    NASA POWER, so the gridMET point comparison has to use them too or it stops
    being a controlled comparison.
    """
    cache = RAW / f"county_points_{state_fips}.csv"
    if use_cache and cache.exists():
        return pd.read_csv(cache, dtype={"fips": str})

    try:
        gaz = pd.read_csv(GAZETTEER_URL, sep="\t", dtype={"GEOID": str})
    except Exception as err:                        # noqa: BLE001 - re-raised with context
        raise ConnectionError(
            f"Could not download the Census Gazetteer ({err}). This is the county "
            f"lat/lon lookup, not the weather service - check the connection and "
            f"retry. Once it succeeds it is cached to {cache} and never fetched "
            f"again.") from err

    gaz.columns = [c.strip() for c in gaz.columns]
    gaz = gaz[gaz["GEOID"].str.startswith(state_fips)]
    points = (gaz[["GEOID", "NAME", "INTPTLAT", "INTPTLONG"]]
              .rename(columns={"GEOID": "fips", "NAME": "county_name",
                               "INTPTLAT": "lat", "INTPTLONG": "lon"})
              .reset_index(drop=True))
    cache.parent.mkdir(parents=True, exist_ok=True)
    points.to_csv(cache, index=False)
    return points


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

    Spring features (see `spring_features`) are appended when the daily data
    covers them, which it does whenever the raw download was a full calendar year.
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
    out = out.sort_index()

    spring = spring_features(df)
    if spring is not None:
        out = out.join(spring, how="left")

    return out.reset_index()


def spring_features(daily: pd.DataFrame) -> pd.DataFrame | None:
    """Pre-season and planting-window features, one row per year.

    Expects the frame `growing_season_features` prepares: daily rows with `year`,
    `month` and a daily `gdd` column already computed.

    Returns None if the daily data doesn't reach back into spring, so a caller
    working from an April-September download still gets the summer features.

      precip_mar_mm      March precipitation - snowmelt and saturation before
                         anything is planted. Nebraska's March 2019 flood lives here.
      precip_apr_may_mm  April-May precipitation - rain during planting itself.
      workable_days      Days in the planting window a field could be worked:
                         little rain that day and little in the two days before.
      last_frost_doy     Day of year of the last spring freeze (0 if none recorded).
      gdd_may            May growing degree days - heat for emergence.

    The windows are deliberately disjoint. Overlapping windows produce features
    that are correlated by construction, which makes a feature study unreadable:
    you can no longer tell whether the second column added information or merely
    repeated the first.
    """
    months = set(daily["month"].unique())
    if not {3, 4, 5} <= months:
        return None

    march = daily[daily["month"] == 3]
    april_may = daily[daily["month"].isin((4, 5))]
    may = daily[daily["month"] == 5]

    out = pd.DataFrame({
        "precip_mar_mm": march.groupby("year")["precip_mm"].sum().round(1),
        "precip_apr_may_mm": april_may.groupby("year")["precip_mm"].sum().round(1),
        "gdd_may": may.groupby("year")["gdd"].sum().round(1),
        "workable_days": _workable_days(daily),
        "last_frost_doy": _last_spring_frost(daily),
    })
    out.index.name = "year"
    return out.sort_index()


def _workable_days(daily: pd.DataFrame) -> pd.Series:
    """Count days in the planting window when a field could actually be worked.

    Total rainfall can't tell 50 mm in one storm from 50 mm spread over three
    weeks. A planter can: it's the second that keeps you out of the field. The
    antecedent-rain term is what encodes that, and it's the whole reason this
    feature can say something `precip_apr_may_mm` cannot.

    Rolling sums are taken over the full daily series before the window is
    applied, so April 10th correctly sees the rain that fell on April 8th and 9th.
    """
    df = daily.sort_values("date")
    antecedent = (df["precip_mm"].shift(1)
                  .rolling(WORKABLE_LOOKBACK_DAYS, min_periods=1).sum())
    workable = ((df["precip_mm"] < WORKABLE_RAIN_MM)
                & (antecedent < WORKABLE_ANTECEDENT_MM))

    start, end = PLANTING_WINDOW
    day = df["date"].dt.strftime("%m-%d")
    in_window = day.between(start, end)

    counts = workable[in_window].groupby(df.loc[in_window, "year"]).sum()
    return counts.astype(int)


def _last_spring_frost(daily: pd.DataFrame) -> pd.Series:
    """Day of year of the last freeze before July; 0 when no freeze was recorded.

    0 is a sentinel meaning "earlier than any frost we saw", which is the right
    direction for a tree model: a lower value always means a longer season. It is
    also rare enough in Nebraska and Iowa to be worth knowing about if it appears.
    """
    spring = daily[daily["month"] <= FROST_SEARCH_END_MONTH]
    frosts = spring[spring["tmin_c"] <= FROST_C]
    last = frosts.groupby("year")["date"].max().dt.dayofyear

    all_years = pd.Index(sorted(daily["year"].unique()), name="year")
    return last.reindex(all_years).fillna(0).astype(int)


def _longest_dry_spell(group: pd.DataFrame, threshold_mm: float = 1.0) -> int:
    dry = (group["precip_mm"] < threshold_mm).astype(int)
    longest = run = 0
    for value in dry:
        run = run + 1 if value else 0
        longest = max(longest, run)
    return longest
