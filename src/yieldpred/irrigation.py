"""County irrigation share for corn: what fraction of harvested acres is irrigated.

The model needs to know which counties are buffered against drought. NASS reports
harvested acres by production practice from two programs:

* **SURVEY** - annual, but county-level practice detail stops after 2018 (the same
  cutoff that ends the irrigated/non-irrigated yield series).
* **CENSUS** - the Census of Agriculture, every five years (2002, 2007, 2012, 2017,
  2022), and still published.

Irrigation infrastructure changes slowly - a centre pivot is a multi-decade
investment - so treating the share as a smooth county characteristic and
interpolating between observations is reasonable. Every value is flagged as
observed or interpolated so the assumption stays visible.
"""

from __future__ import annotations

import pandas as pd

from yieldpred.nass import parse_value, query

PRACTICE_MAP = {"ALL PRODUCTION PRACTICES": "all", "IRRIGATED": "irrigated"}


def fetch_harvested_acres(source_desc: str = "SURVEY", state_alpha: str = "NE",
                          start_year: int = 2000, api_key: str | None = None) -> pd.DataFrame:
    """Corn-for-grain harvested acres by county and production practice."""
    params = {
        "source_desc": source_desc,
        "sector_desc": "CROPS",
        "commodity_desc": "CORN",
        "util_practice_desc": "GRAIN",
        "statisticcat_desc": "AREA HARVESTED",
        "unit_desc": "ACRES",
        "domain_desc": "TOTAL",
        "agg_level_desc": "COUNTY",
        "reference_period_desc": "YEAR",
        "state_alpha": state_alpha,
        "year__GE": start_year,
    }
    return tidy_acres(query(params, api_key), source_desc)


def tidy_acres(raw: pd.DataFrame, source_desc: str = "SURVEY") -> pd.DataFrame:
    """Clean raw acreage rows to: fips, year, practice, acres, source."""
    cols = ["fips", "year", "practice", "acres", "source"]
    if raw.empty:
        return pd.DataFrame(columns=cols)

    df = raw.copy()
    df = df[df["county_code"].astype(str) != "998"]
    df = df[df["prodn_practice_desc"].isin(PRACTICE_MAP)]

    df["fips"] = (df["state_fips_code"].astype(str).str.zfill(2)
                  + df["county_code"].astype(str).str.zfill(3))
    df["practice"] = df["prodn_practice_desc"].map(PRACTICE_MAP)
    df["year"] = df["year"].astype(int)
    df["acres"] = parse_value(df["Value"])
    df["source"] = source_desc

    return (df[cols].dropna(subset=["acres"])
            .drop_duplicates(subset=["fips", "year", "practice"])
            .reset_index(drop=True))


def observed_share(acres: pd.DataFrame) -> pd.DataFrame:
    """Irrigated acres / all acres, for county-years where both are reported."""
    wide = acres.pivot_table(index=["fips", "year"], columns="practice", values="acres")
    if "irrigated" not in wide or "all" not in wide:
        return pd.DataFrame(columns=["fips", "year", "irrigation_share"])

    wide = wide.dropna(subset=["all", "irrigated"])
    wide = wide[wide["all"] > 0]
    wide["irrigation_share"] = (wide["irrigated"] / wide["all"]).clip(0, 1)
    return wide.reset_index()[["fips", "year", "irrigation_share"]]


def build_share_series(observed: pd.DataFrame, years: range) -> pd.DataFrame:
    """Fill each county's share across all years by interpolating between observations.

    Interpolates linearly between known values and carries the first and last
    observations outward. `share_observed` marks which rows are real measurements.
    """
    if observed.empty:
        return pd.DataFrame(columns=["fips", "year", "irrigation_share", "share_observed"])

    frames = []
    for fips, group in observed.groupby("fips"):
        series = (group.set_index("year")["irrigation_share"]
                  .reindex(sorted(set(years) | set(group["year"]))))
        filled = series.interpolate(method="index", limit_direction="both")
        frames.append(pd.DataFrame({
            "fips": fips,
            "year": filled.index,
            "irrigation_share": filled.values.round(4),
            "share_observed": series.notna().values,
        }))

    out = pd.concat(frames, ignore_index=True)
    return out[out["year"].isin(years)].reset_index(drop=True)


def combine_sources(survey: pd.DataFrame, census: pd.DataFrame) -> pd.DataFrame:
    """Prefer annual SURVEY observations; fall back to CENSUS years."""
    combined = pd.concat([survey, census], ignore_index=True)
    return (combined.drop_duplicates(subset=["fips", "year"], keep="first")
            .sort_values(["fips", "year"]).reset_index(drop=True))
