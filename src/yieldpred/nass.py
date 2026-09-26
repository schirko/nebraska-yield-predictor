"""Client for the USDA NASS Quick Stats API.

Docs: https://quickstats.nass.usda.gov/api
The API key is read from the NASS_API_KEY environment variable (loaded from
a local .env file, which is git-ignored so the key never reaches GitHub).
"""

from __future__ import annotations

import os

import pandas as pd
import requests
from dotenv import load_dotenv

BASE_URL = "https://quickstats.nass.usda.gov/api"
MAX_RECORDS = 50_000  # Quick Stats refuses requests that would return more rows

# Codes NASS uses instead of a number, e.g. "(D)" = withheld to avoid
# disclosing individual operations. These become NaN.
SUPPRESSED_CODES = {"(D)", "(Z)", "(NA)", "(X)", "(S)", "(L)", "(H)"}

PRACTICE_MAP = {
    "ALL PRODUCTION PRACTICES": "all",
    "IRRIGATED": "irrigated",
    "NON-IRRIGATED": "non_irrigated",
}


class NASSError(RuntimeError):
    """Raised when the Quick Stats API returns an error."""


def get_api_key() -> str:
    load_dotenv()
    key = os.getenv("NASS_API_KEY", "").strip()
    if not key or key == "your-key-here":
        raise NASSError(
            "NASS_API_KEY is not set. Put your key in the .env file at the "
            "project root, e.g.  NASS_API_KEY=abc123..."
        )
    return key


def _get(endpoint: str, params: dict, api_key: str, timeout: int = 120) -> dict:
    try:
        resp = requests.get(
            f"{BASE_URL}/{endpoint}/",
            params={"key": api_key, **params},
            timeout=timeout,
        )
    except requests.RequestException as exc:
        # Re-raise without the request URL, which would contain the API key.
        raise NASSError(f"Could not reach NASS Quick Stats: {type(exc).__name__}") from None

    if resp.status_code != 200:
        try:
            detail = resp.json().get("error", resp.text)
        except ValueError:
            detail = resp.text
        raise NASSError(f"NASS API returned HTTP {resp.status_code}: {detail}")
    return resp.json()


def get_count(params: dict, api_key: str | None = None) -> int:
    """Number of rows a query would return (use before downloading)."""
    api_key = api_key or get_api_key()
    return int(_get("get_counts", params, api_key)["count"])


def query(params: dict, api_key: str | None = None) -> pd.DataFrame:
    """Run a Quick Stats query and return the raw rows as a DataFrame."""
    api_key = api_key or get_api_key()
    n = get_count(params, api_key)
    if n == 0:
        return pd.DataFrame()
    if n > MAX_RECORDS:
        raise NASSError(
            f"Query would return {n:,} rows (limit {MAX_RECORDS:,}). "
            "Narrow it, e.g. by year range."
        )
    data = _get("api_GET", {**params, "format": "JSON"}, api_key)["data"]
    return pd.DataFrame(data)


def parse_value(series: pd.Series) -> pd.Series:
    """Convert NASS 'Value' strings like '1,234.5' or '(D)' to floats/NaN."""
    s = series.astype(str).str.strip()
    s = s.where(~s.isin(SUPPRESSED_CODES))
    return pd.to_numeric(s.str.replace(",", "", regex=False), errors="coerce")


def tidy_county_yields(raw: pd.DataFrame) -> pd.DataFrame:
    """Clean raw county yield rows into a tidy long table.

    Columns: fips, state_alpha, county_name, asd_code, asd_desc, year,
    practice ('all' / 'irrigated' / 'non_irrigated'), yield_bu_acre
    """
    cols = ["fips", "state_alpha", "county_name", "asd_code", "asd_desc",
            "year", "practice", "yield_bu_acre"]
    if raw.empty:
        return pd.DataFrame(columns=cols)

    df = raw.copy()
    # "OTHER (COMBINED) COUNTIES" rows (county code 998) aren't real counties.
    df = df[df["county_code"].astype(str) != "998"]
    df = df[df["prodn_practice_desc"].isin(PRACTICE_MAP)]

    df["fips"] = (df["state_fips_code"].astype(str).str.zfill(2)
                  + df["county_code"].astype(str).str.zfill(3))
    df["practice"] = df["prodn_practice_desc"].map(PRACTICE_MAP)
    df["year"] = df["year"].astype(int)
    df["yield_bu_acre"] = parse_value(df["Value"])
    df["county_name"] = df["county_name"].str.title()

    return (df[cols]
            .dropna(subset=["yield_bu_acre"])
            .drop_duplicates(subset=["fips", "year", "practice"])
            .sort_values(["fips", "year", "practice"])
            .reset_index(drop=True))


def fetch_county_corn_yields(
    state_alpha: str = "NE",
    start_year: int = 2000,
    end_year: int | None = None,
    api_key: str | None = None,
) -> pd.DataFrame:
    """County corn-for-grain yields (bu/acre), all / irrigated / non-irrigated."""
    params = {
        "source_desc": "SURVEY",
        "sector_desc": "CROPS",
        "commodity_desc": "CORN",
        "util_practice_desc": "GRAIN",
        "statisticcat_desc": "YIELD",
        "unit_desc": "BU / ACRE",
        "agg_level_desc": "COUNTY",
        "reference_period_desc": "YEAR",
        "state_alpha": state_alpha,
        "year__GE": start_year,
    }
    if end_year is not None:
        params["year__LE"] = end_year
    return tidy_county_yields(query(params, api_key))


# ------------------------------------------------------------------ alfalfa hay
#
# NASS publishes alfalfa under commodity HAY, class ALFALFA, as three statistics.
# County estimates for hay stopped after the 2018 crop year (NASS notice of
# 2020-02-13: the Risk Management Agency ended the funding), so a county query
# returns 2018 as its last year. State totals continue.

ALFALFA_STATS = {
    ("YIELD", "TONS / ACRE"): "yield_tons_acre",
    ("AREA HARVESTED", "ACRES"): "acres_harvested",
    ("PRODUCTION", "TONS"): "production_tons",
}
ALFALFA_COLUMNS = list(ALFALFA_STATS.values())


def _alfalfa_params(state_alpha: str, level: str, start_year: int, end_year: int | None) -> dict:
    params = {
        "source_desc": "SURVEY",
        "sector_desc": "CROPS",
        "commodity_desc": "HAY",
        "class_desc": "ALFALFA",
        "agg_level_desc": level,
        "reference_period_desc": "YEAR",
        "state_alpha": state_alpha,
        "year__GE": start_year,
    }
    if end_year is not None:
        params["year__LE"] = end_year
    return params


def _alfalfa_wide(df: pd.DataFrame, keys: list[str]) -> pd.DataFrame:
    """Keep the three alfalfa statistics and put each in its own column."""
    df = df.copy()
    df["stat"] = [ALFALFA_STATS.get((s, u)) for s, u in zip(df["statisticcat_desc"], df["unit_desc"])]
    df = df[df["stat"].notna() & df["prodn_practice_desc"].isin(PRACTICE_MAP)]
    df["practice"] = df["prodn_practice_desc"].map(PRACTICE_MAP)
    df["year"] = df["year"].astype(int)
    df["value"] = parse_value(df["Value"])
    wide = (df.dropna(subset=["value"])
              .pivot_table(index=keys + ["year", "practice"], columns="stat", values="value", aggfunc="first")
              .reset_index())
    wide.columns.name = None
    for col in ALFALFA_COLUMNS:
        if col not in wide:
            wide[col] = float("nan")
    return wide.sort_values(keys + ["year", "practice"]).reset_index(drop=True)


def tidy_county_alfalfa(raw: pd.DataFrame) -> pd.DataFrame:
    """Clean raw county alfalfa rows into one row per county, year and practice.

    Columns: fips, state_alpha, county_name, asd_code, asd_desc, year, practice,
    yield_tons_acre, acres_harvested, production_tons (NaN where NASS withheld it).
    """
    keys = ["fips", "state_alpha", "county_name", "asd_code", "asd_desc"]
    if raw.empty:
        return pd.DataFrame(columns=keys + ["year", "practice"] + ALFALFA_COLUMNS)
    df = raw[raw["county_code"].astype(str) != "998"].copy()   # "other (combined) counties"
    df["fips"] = df["state_fips_code"].astype(str).str.zfill(2) + df["county_code"].astype(str).str.zfill(3)
    df["county_name"] = df["county_name"].str.title()
    return _alfalfa_wide(df, keys)


def tidy_state_alfalfa(raw: pd.DataFrame) -> pd.DataFrame:
    """Clean raw state alfalfa rows: one row per year and practice."""
    if raw.empty:
        return pd.DataFrame(columns=["state_alpha", "year", "practice"] + ALFALFA_COLUMNS)
    return _alfalfa_wide(raw, ["state_alpha"])


def fetch_county_alfalfa(state_alpha: str = "NE", start_year: int = 2000, end_year: int | None = None,
                         api_key: str | None = None) -> pd.DataFrame:
    """County alfalfa hay: yield (tons/acre), acres harvested and production."""
    return tidy_county_alfalfa(query(_alfalfa_params(state_alpha, "COUNTY", start_year, end_year), api_key))


def fetch_state_alfalfa(state_alpha: str = "NE", start_year: int = 2000, end_year: int | None = None,
                        api_key: str | None = None) -> pd.DataFrame:
    """Statewide alfalfa hay, which NASS still publishes every year."""
    return tidy_state_alfalfa(query(_alfalfa_params(state_alpha, "STATE", start_year, end_year), api_key))
