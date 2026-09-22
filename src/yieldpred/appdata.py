"""Data access and summaries for the Streamlit app.

Deliberately free of any Streamlit import: these are plain functions that read the
pre-computed files in data/processed/ and shape them for display. The app wraps them
in caching; the tests call them directly.

The app never downloads, trains or recomputes anything - it reads what the scripts
produced. That keeps it fast, keeps it working without an API key, and keeps it
deployable on Streamlit Community Cloud's small memory allowance.
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
PROCESSED = ROOT / "data" / "processed"
FIGURES = ROOT / "figures"

FEATURE_LABELS = {
    "year": "Year (trend)",
    "gdd": "Growing degree days",
    "precip_mm": "Season rainfall",
    "precip_jul_mm": "July rainfall",
    "precip_aug_mm": "August rainfall",
    "tmax_jul_c": "Mean July high",
    "heat_days_32": "Days above 32°C",
    "heat_days_35": "Days above 35°C",
    "dry_spell_max": "Longest dry spell",
    "irrigation_share": "Irrigation share",
}


def _read(name: str) -> pd.DataFrame | None:
    path = PROCESSED / name
    return pd.read_parquet(path) if path.exists() else None


def load_model_table() -> pd.DataFrame | None:
    return _read("model_table.parquet")


def load_errors() -> pd.DataFrame | None:
    return _read("model_errors.parquet")


def load_scores() -> pd.DataFrame | None:
    return _read("model_scores.parquet")


def load_irrigation_comparison() -> pd.DataFrame | None:
    return _read("irrigation_comparison.parquet")


def load_morans_by_year() -> pd.DataFrame | None:
    return _read("morans_by_year.parquet")


def load_morans_pooled() -> pd.DataFrame | None:
    return _read("morans_pooled.parquet")


def load_yields(state: str = "ne") -> pd.DataFrame | None:
    return _read(f"{state}_corn_yield_county.parquet")


def load_counties(state_fips: str = "31"):
    path = PROCESSED / f"counties_{state_fips}.parquet"
    if not path.exists():
        return None
    import geopandas as gpd
    return gpd.read_parquet(path)


def missing_data_message(available: dict[str, bool]) -> str | None:
    """A single instruction naming the script that produces whatever is missing."""
    scripts = {
        "yields": "python scripts/fetch_nass_yields.py",
        "weather": "python scripts/fetch_weather.py",
        "irrigation": "python scripts/fetch_irrigation.py",
        "model": "python scripts/train_baseline.py",
        "spatial": "python scripts/analyze_errors.py",
    }
    missing = [scripts[key] for key, ok in available.items() if not ok and key in scripts]
    if not missing:
        return None
    return "This page needs data that hasn't been generated yet. Run:\n\n```\n" + \
           "\n".join(missing) + "\n```"


# ----------------------------------------------------------------- summaries

def state_yield_history(yields: pd.DataFrame) -> pd.DataFrame:
    """Statewide average yield per year, by production practice."""
    return (yields.pivot_table(index="year", columns="practice",
                               values="yield_bu_acre", aggfunc="mean")
            .round(1))


def irrigation_gap(yields: pd.DataFrame) -> pd.DataFrame:
    """Per-year irrigated minus non-irrigated yield, for counties reporting both."""
    paired = (yields[yields["practice"] != "all"]
              .pivot_table(index=["fips", "year"], columns="practice",
                           values="yield_bu_acre")
              .dropna())
    if paired.empty:
        return pd.DataFrame(columns=["year", "gap", "irrigated", "non_irrigated", "counties"])
    paired["gap"] = paired["irrigated"] - paired["non_irrigated"]
    return (paired.groupby("year")
            .agg(gap=("gap", "mean"), irrigated=("irrigated", "mean"),
                 non_irrigated=("non_irrigated", "mean"), counties=("gap", "size"))
            .round(1).reset_index())


def county_history(model_table: pd.DataFrame, fips: str) -> pd.DataFrame:
    """One county's full record, most recent first."""
    cols = ["year", "yield_bu_acre", "gdd", "precip_mm", "precip_jul_mm",
            "heat_days_32", "dry_spell_max", "irrigation_share"]
    present = [c for c in cols if c in model_table.columns]
    return (model_table[model_table["fips"] == fips][present]
            .sort_values("year", ascending=False).reset_index(drop=True))


def county_choices(model_table: pd.DataFrame) -> pd.DataFrame:
    """fips + county_name, sorted by name, for a select box."""
    return (model_table[["fips", "county_name"]].drop_duplicates()
            .sort_values("county_name").reset_index(drop=True))


def feature_correlations(model_table: pd.DataFrame) -> pd.DataFrame:
    """Correlation of each feature with yield, strongest first."""
    features = [c for c in FEATURE_LABELS if c in model_table.columns]
    corr = (model_table[features]
            .apply(lambda col: col.corr(model_table["yield_bu_acre"]))
            .rename("correlation").reset_index()
            .rename(columns={"index": "feature"}))
    corr["label"] = corr["feature"].map(FEATURE_LABELS)
    return corr.reindex(corr["correlation"].abs().sort_values(ascending=False).index)


def scores_by_scheme(scores: pd.DataFrame, scheme: str) -> pd.DataFrame:
    """One validation scheme's columns, renamed for display."""
    cols = [c for c in scores.columns if c.startswith(f"{scheme}_")]
    out = scores[["model"] + cols].copy()
    return out.rename(columns={c: c.split("_", 1)[1] for c in cols}).set_index("model").round(2)


def worst_years(errors: pd.DataFrame, n: int = 8) -> pd.DataFrame:
    """Years with the largest average error, worst first."""
    by_year = (errors.groupby("year")
               .agg(mean_error=("error", "mean"), counties=("fips", "nunique"))
               .round(1).reset_index())
    return by_year.reindex(by_year["mean_error"].abs().sort_values(ascending=False).index).head(n)
