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
