"""Build the county-year modeling table by joining yields and weather."""

from __future__ import annotations

from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
PROCESSED = ROOT / "data" / "processed"

WEATHER_FEATURES = ["gdd", "precip_mm", "precip_jul_mm", "precip_aug_mm",
                    "tmax_jul_c", "heat_days_32", "heat_days_35", "dry_spell_max"]


def load_yields(state: str = "ne", practice: str = "all") -> pd.DataFrame:
    df = pd.read_parquet(PROCESSED / f"{state}_corn_yield_county.parquet")
    return df[df["practice"] == practice] if practice else df


def load_weather(state_fips: str = "31") -> pd.DataFrame:
    return pd.read_parquet(PROCESSED / f"weather_county_{state_fips}.parquet")


def load_irrigation(state: str = "ne") -> pd.DataFrame | None:
    """County irrigation share, if scripts/fetch_irrigation.py has been run."""
    path = PROCESSED / f"irrigation_share_{state}.parquet"
    return pd.read_parquet(path) if path.exists() else None


def load_irrigation_observed(state: str = "ne") -> pd.DataFrame | None:
    """The raw (non-interpolated) irrigation observations, if available.

    Kept separately so an experiment can rebuild the feature using only the
    years it is allowed to see, avoiding look-ahead in the temporal test.
    """
    path = PROCESSED / f"irrigation_observed_{state}.parquet"
    return pd.read_parquet(path) if path.exists() else None


def build_modeling_table(state: str = "ne", state_fips: str = "31",
                         practice: str = "all") -> pd.DataFrame:
    """One row per county-year: the yield to predict plus its weather.

    `year` is kept as a column so models can use the long-run yield trend
    (~2 bu/acre per year), either as a feature or via DetrendedRegressor.
    Irrigation share is joined when available.
    """
    yields = load_yields(state, practice)
    weather = load_weather(state_fips).drop(columns=["county_name"])

    df = yields.merge(weather, on=["fips", "year"], how="inner", validate="one_to_one")

    irrigation = load_irrigation(state)
    if irrigation is not None:
        df = df.merge(irrigation, on=["fips", "year"], how="left", validate="one_to_one")

    return df.sort_values(["fips", "year"]).reset_index(drop=True)


def feature_matrix(df: pd.DataFrame, extra: list[str] | None = None):
    """Split the table into X, y and the district codes used for spatial CV."""
    features = ["year"] + WEATHER_FEATURES + (extra or [])
    usable = df.dropna(subset=features + ["yield_bu_acre"])
    return usable[features], usable["yield_bu_acre"], usable["asd_code"], usable
