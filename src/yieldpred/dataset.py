"""Build the county-year modeling table by joining yields and weather."""

from __future__ import annotations

from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
PROCESSED = ROOT / "data" / "processed"

WEATHER_FEATURES = ["gdd", "precip_mm", "precip_jul_mm", "precip_aug_mm",
                    "tmax_jul_c", "heat_days_32", "heat_days_35", "dry_spell_max"]

# Spring and planting-window features, deliberately kept OUT of WEATHER_FEATURES.
# Two reasons: the feature study can then add them as one group and measure what
# they are worth, and every number already published for the weather-only model
# stays comparable. They exist only in weather files rebuilt after
# `weather.spring_features` was added.
SPRING_FEATURES = ["precip_mar_mm", "precip_apr_may_mm", "workable_days",
                   "last_frost_doy", "gdd_may"]


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


def output_path(name: str, state: str = "ne") -> Path:
    """Where a derived file lives for a given state.

    Nebraska keeps the original unsuffixed names so existing files and the app
    keep working; other states get a suffix. Slightly ugly, deliberately chosen
    over renaming files the app already reads.
    """
    stem = name if state == "ne" else f"{name}_{state}"
    return PROCESSED / f"{stem}.parquet"


def load_static(state: str = "ne") -> pd.DataFrame | None:
    """County characteristics that don't vary by year: soil rating, elevation."""
    path = PROCESSED / f"county_static_{state}.parquet"
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

    static = load_static(state)
    if static is not None:
        df = df.merge(static, on="fips", how="left", validate="many_to_one")

    return df.sort_values(["fips", "year"]).reset_index(drop=True)


def available_extras(df: pd.DataFrame) -> list[str]:
    """Optional feature columns present in this table, in the order they were added."""
    return [c for c in ("irrigation_share", "nccpi_corn", "soil_water_cm", "elevation_m")
            if c in df.columns]


def available_spring(df: pd.DataFrame) -> list[str]:
    """Spring/planting features present in this table, if the weather was rebuilt."""
    return [c for c in SPRING_FEATURES if c in df.columns]


def feature_groups(df: pd.DataFrame) -> list[tuple[str, list[str]]]:
    """The optional features, as named groups, in the order the study adds them.

    A group can be one column or several. Spring weather is five columns that
    belong to a single idea - what happened before planting - so adding them one
    at a time would make the ladder five rows longer without answering five
    separate questions.
    """
    groups = [(name, [name]) for name in available_extras(df)]
    spring = available_spring(df)
    if spring:
        groups.append((f"spring weather ({len(spring)})", spring))
    return groups


def feature_matrix(df: pd.DataFrame, extra: list[str] | None = None):
    """Split the table into X, y and the district codes used for spatial CV.

    Rows missing any chosen feature are dropped, so the row set depends on which
    features are asked for. That is why the feature study builds one matrix with
    everything present and then drops columns from it, rather than calling this
    once per rung: otherwise each rung would be scored on a different set of rows.
    """
    features = ["year"] + WEATHER_FEATURES + (extra or [])
    usable = df.dropna(subset=features + ["yield_bu_acre"])
    return usable[features], usable["yield_bu_acre"], usable["asd_code"], usable
