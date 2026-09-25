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
    "precip_mar_mm": "March rainfall",
    "precip_apr_may_mm": "April–May rainfall",
    "workable_days": "Workable planting days",
    "last_frost_doy": "Last spring frost",
    "gdd_may": "May growing degree days",
    "irrigation_share": "Irrigation share",
    "nccpi_corn": "Soil rating (NCCPI)",
    "elevation_m": "Elevation",
}


def _read(name: str) -> pd.DataFrame | None:
    path = PROCESSED / name
    return pd.read_parquet(path) if path.exists() else None


def _stem(name: str, state: str) -> str:
    """Nebraska keeps the unsuffixed filenames; other states get a suffix.

    Mirrors dataset.output_path, duplicated here so the app layer doesn't import
    the modelling layer just to build a filename.
    """
    return f"{name}.parquet" if state == "ne" else f"{name}_{state}.parquet"


def load_model_table(state: str = "ne") -> pd.DataFrame | None:
    return _read(_stem("model_table", state))


def load_errors(state: str = "ne") -> pd.DataFrame | None:
    return _read(_stem("model_errors", state))


def load_scores(state: str = "ne") -> pd.DataFrame | None:
    return _read(_stem("model_scores", state))


def load_irrigation_comparison(state: str = "ne") -> pd.DataFrame | None:
    return _read(_stem("irrigation_comparison", state))


def load_ablation(state: str = "ne") -> pd.DataFrame | None:
    return _read(_stem("feature_ablation", state))


def load_leak_control(state: str = "ne") -> pd.DataFrame | None:
    """The three-row leak comparison written by train_baseline.

    Absent when a state has no cell spanning a district boundary, which is a
    real answer rather than missing data - the page says so instead of warning.
    """
    return _read(_stem("leak_control", state))


def load_morans_by_year(state: str = "ne") -> pd.DataFrame | None:
    return _read(_stem("morans_by_year", state))


def load_morans_pooled(state: str = "ne") -> pd.DataFrame | None:
    return _read(_stem("morans_pooled", state))


def load_cross_state(train: str = "ne", test: str = "ia") -> pd.DataFrame | None:
    """County-year predictions from the train-on-one-state, test-on-another run."""
    return _read(f"cross_state_{train}_to_{test}.parquet")


def load_cross_state_scores(train: str = "ne", test: str = "ia") -> pd.DataFrame | None:
    return _read(f"cross_state_scores_{train}_to_{test}.parquet")


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
            "heat_days_32", "dry_spell_max", "precip_mar_mm", "precip_apr_may_mm",
            "workable_days", "last_frost_doy", "gdd_may", "irrigation_share"]
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


def map_frame(errors: pd.DataFrame, model_table: pd.DataFrame | None) -> pd.DataFrame:
    """Everything mappable, in one county-year table.

    Predictions and errors come from the model run; irrigation share, soil water
    capacity and elevation come from the modelling table. Joined here rather than in
    the page so the page stays layout-only and this can be tested.
    """
    out = errors.copy()
    if model_table is not None:
        extras = [c for c in ("irrigation_share", "soil_water_cm", "nccpi_corn",
                              "elevation_m") if c in model_table.columns]
        if extras:
            out = out.merge(model_table[["fips", "year"] + extras],
                            on=["fips", "year"], how="left")
    return out


def county_values(frame: pd.DataFrame, column: str, year=None) -> pd.DataFrame:
    """Values for one map: a single year, or the average across all years.

    Tooltip columns travel with the value, so hovering shows a county's actual and
    predicted yield alongside whatever layer is on screen.
    """
    carry = [c for c in ("yield_bu_acre", "predicted", "error", "irrigation_share")
             if c in frame.columns]
    columns = list(dict.fromkeys([column] + carry))

    if year is None:
        values = (frame.groupby(["fips", "county_name"], as_index=False)[columns]
                  .mean(numeric_only=True))
    else:
        values = (frame[frame["year"] == year][["fips", "county_name"] + columns]
                  .copy())
    return values.round(3)


def variance_decomposition(model_table: pd.DataFrame) -> dict[str, float]:
    """Split county-year yield variance into year, county and residual shares.

    A one-table description of what kind of problem you have. If most of the
    variation is between years, features describing *counties* can only ever
    address the rest, however clever they are.
    """
    y = model_table["yield_bu_acre"]
    total = y.var()
    if not total:
        return {"year": 0.0, "county": 0.0, "residual": 0.0}
    grand = y.mean()
    year_mean = model_table.groupby("year")["yield_bu_acre"].transform("mean")
    county_mean = model_table.groupby("fips")["yield_bu_acre"].transform("mean")
    year_share = (year_mean - grand).var() / total
    county_share = (county_mean - grand).var() / total
    return {"year": float(year_share), "county": float(county_share),
            "residual": float(max(0.0, 1.0 - year_share - county_share))}


def transfer_by_year(predictions: pd.DataFrame) -> pd.DataFrame:
    """Per-year summary of a cross-state transfer.

    `county_skill` is the correlation between predicted and actual across counties
    *within* that year - the question a decision tool actually faces, and one that
    pooled R² hides behind the much larger year-to-year swings.
    """
    rows = []
    for year, group in predictions.groupby("year"):
        # A correlation needs at least two counties and some variation in both
        # columns; a year with one reporting county simply has no ranking to score.
        scorable = (len(group) > 1 and group["predicted"].nunique() > 1
                    and group["yield_bu_acre"].nunique() > 1)
        skill = group["predicted"].corr(group["yield_bu_acre"]) if scorable else None
        rows.append({"year": int(year),
                     "mean_error": round(group["error"].mean(), 1),
                     "mean_abs_error": round(group["error"].abs().mean(), 1),
                     "county_skill": round(float(skill), 3) if skill is not None
                     and pd.notna(skill) else None,
                     "actual": round(group["yield_bu_acre"].mean(), 1),
                     "counties": int(group["fips"].nunique())})
    return pd.DataFrame(rows)


def county_skill(predictions: pd.DataFrame) -> float:
    """Median within-year county-ranking skill, a single summary of the above."""
    by_year = transfer_by_year(predictions)["county_skill"].dropna()
    return float(by_year.median()) if len(by_year) else float("nan")


def worst_years(errors: pd.DataFrame, n: int = 8) -> pd.DataFrame:
    """Years with the largest average error, worst first."""
    by_year = (errors.groupby("year")
               .agg(mean_error=("error", "mean"), counties=("fips", "nunique"))
               .round(1).reset_index())
    return by_year.reindex(by_year["mean_error"].abs().sort_values(ascending=False).index).head(n)
