"""Alfalfa: county hay yields, the weather that drives them, and checks on the data.

Alfalfa is not corn. Corn is planted, grows once and is harvested once. An
alfalfa stand lives four to six years and is cut three to five times a season in
Nebraska, each cutting regrowing from the crown. So the weather that matters is
spread differently:

- The whole season counts, not one critical month: every cutting needs heat and water.
- Early summer (May-June) feeds the first two cuttings, which carry the most tonnage.
- Late summer (July-August) heat and drought slow regrowth for the later cuttings.
- Winter matters too: a hard, open freeze can kill crowns ("winterkill"), and the
  stand that survives is what gets cut the next year.

This module turns the daily NASA POWER files the corn model already downloaded
into those features, and gives the checks that belong before any modelling:
how many counties report each year, and whether the counties add up to the state.

NASS stopped publishing county hay estimates after the 2018 crop year, so the
county table ends there. The statewide series continues and is the check on it.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

LAST_COUNTY_YEAR = 2018          # NASS county hay estimates end with the 2018 crop

# Alfalfa growing degree days: base 5 C (41 F), the usual base for alfalfa and
# cool-season forage; capped at 30 C (86 F) like corn, above which growth slows.
GDD_BASE_C = 5.0
GDD_CAP_C = 30.0
HARD_FREEZE_C = -18.0            # about 0 F: the air temperature where winterkill risk climbs
HEAT_C = 32.0                    # about 90 F: summer slump for regrowth


def _gdd(tmax: pd.Series, tmin: pd.Series) -> pd.Series:
    hi = tmax.clip(upper=GDD_CAP_C)
    lo = tmin.clip(lower=GDD_BASE_C)
    return ((hi + lo) / 2 - GDD_BASE_C).clip(lower=0)


def season_features(daily: pd.DataFrame) -> pd.DataFrame:
    """One row of alfalfa weather per year from daily weather (date, tmax_c, tmin_c, precip_mm).

    Features:
      gdd_season       growing degree days (base 5 C), April-September
      precip_season_mm rain, April-September
      precip_early_mm  rain, May-June (first and second cuttings)
      precip_late_mm   rain, July-August (later cuttings)
      heat_days        days above 32 C, July-August
      winter_min_c     coldest night of the winter before (December of the year before
                       through February)
      freeze_days      nights below -18 C that winter

    Winter features need the December before, so the first year in the file has
    none (NaN) rather than a winter with a month missing.
    """
    df = daily.copy()
    df["date"] = pd.to_datetime(df["date"])
    df["year"] = df["date"].dt.year
    df["month"] = df["date"].dt.month
    df["gdd"] = _gdd(df["tmax_c"], df["tmin_c"])

    season = df[df["month"].between(4, 9)].groupby("year")
    early = df[df["month"].between(5, 6)].groupby("year")
    late = df[df["month"].between(7, 8)]

    out = pd.DataFrame({
        "gdd_season": season["gdd"].sum().round(1),
        "precip_season_mm": season["precip_mm"].sum().round(1),
        "precip_early_mm": early["precip_mm"].sum().round(1),
        "precip_late_mm": late.groupby("year")["precip_mm"].sum().round(1),
        "heat_days": late.assign(hot=late["tmax_c"] > HEAT_C).groupby("year")["hot"].sum().astype(int),
    })

    # The winter before crop year Y runs from December of Y-1 through February of Y.
    winter = df[df["month"].isin([12, 1, 2])].copy()
    winter["crop_year"] = np.where(winter["month"] == 12, winter["year"] + 1, winter["year"])
    first_year = df["year"].min()
    has_december = set(df.loc[df["month"] == 12, "year"] + 1)
    winter = winter[winter["crop_year"].isin(has_december) & (winter["crop_year"] > first_year)]
    w = winter.groupby("crop_year")
    out["winter_min_c"] = w["tmin_c"].min().round(1)
    out["freeze_days"] = w["tmin_c"].apply(lambda t: int((t < HARD_FREEZE_C).sum()))
    out.index.name = "year"
    return out.reset_index()


def weather_table(power_dir: Path, fips: list[str]) -> pd.DataFrame:
    """Alfalfa weather for every county that has a NASA POWER file in power_dir."""
    frames = []
    for code in sorted(fips):
        files = sorted(Path(power_dir).glob(f"{code}_*.csv"))
        if not files:
            continue
        feats = season_features(pd.read_csv(files[-1]))
        feats.insert(0, "fips", code)
        frames.append(feats)
    return pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()


def coverage(county: pd.DataFrame, state: pd.DataFrame | None = None) -> pd.DataFrame:
    """Per year: how many counties report a yield, and what share of the state's acres they hold.

    A county with its yield withheld still usually has its acres, so acres come
    from every row and yields only from rows that have one.
    """
    rows = county[county["practice"] == "all"]
    out = rows.groupby("year").agg(
        counties_with_yield=("yield_tons_acre", lambda s: int(s.notna().sum())),
        counties_listed=("fips", "nunique"),
        county_acres=("acres_harvested", "sum"),
    )
    if state is not None and not state.empty:
        st = state[state["practice"] == "all"].set_index("year")
        out["state_acres"] = st["acres_harvested"]
        out["share_of_state_acres"] = (out["county_acres"] / out["state_acres"]).round(3)
    return out.reset_index()


def compare_with_state(county: pd.DataFrame, state: pd.DataFrame) -> pd.DataFrame:
    """Acre-weighted average of the county yields beside NASS's own state yield.

    If the counties are a fair picture of the state, the two should be close.
    A county table that is mostly the big irrigated counties would run high.
    """
    rows = county[(county["practice"] == "all")].dropna(subset=["yield_tons_acre", "acres_harvested"])
    weighted = (rows.assign(w=rows["yield_tons_acre"] * rows["acres_harvested"])
                    .groupby("year")[["w", "acres_harvested"]].sum())
    out = pd.DataFrame({"county_weighted_yield": (weighted["w"] / weighted["acres_harvested"]).round(3)})
    st = state[state["practice"] == "all"].set_index("year")["yield_tons_acre"]
    out["state_yield"] = st
    out["difference"] = (out["county_weighted_yield"] - out["state_yield"]).round(3)
    return out.reset_index()


def model_table(county: pd.DataFrame, weather: pd.DataFrame) -> pd.DataFrame:
    """One row per county and year with a published all-practice yield, plus its weather."""
    rows = county[(county["practice"] == "all") & county["yield_tons_acre"].notna()]
    rows = rows[["fips", "county_name", "asd_desc", "year", "yield_tons_acre", "acres_harvested"]]
    return rows.merge(weather, on=["fips", "year"], how="inner").sort_values(["fips", "year"]).reset_index(drop=True)
