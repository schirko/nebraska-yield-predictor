"""Build the modeling table and compare models under different validation schemes.

    python scripts/train_baseline.py

Three models, three ways of validating them. The point of the exercise is that
random cross-validation flatters a spatial model: neighbouring counties in the
same year are so similar that a random split leaks information. Holding out
whole agricultural districts, or whole years, is the honest test.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.dummy import DummyRegressor
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import GroupKFold, KFold, cross_val_predict
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from yieldpred.dataset import (PROCESSED, build_modeling_table, feature_groups,
                               feature_matrix, load_irrigation_observed, output_path)
from yieldpred.irrigation import build_share_series
from yieldpred.trend import DetrendedRegressor

ROOT = Path(__file__).resolve().parents[1]


def scores(y_true, y_pred) -> dict:
    return {
        "RMSE": float(np.sqrt(mean_squared_error(y_true, y_pred))),
        "MAE": float(mean_absolute_error(y_true, y_pred)),
        "R2": float(r2_score(y_true, y_pred)),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--state", default="NE", help="Two-letter state code")
    parser.add_argument("--state-fips", default="31")
    parser.add_argument("--split-spring", action="store_true",
                        help="Score spring rain and spring calendar as separate "
                             "groups, to test whether the calendar features are "
                             "acting as county fingerprints rather than weather")
    args = parser.parse_args()
    state = args.state.lower()

    df = build_modeling_table(state=state, state_fips=args.state_fips)
    out = output_path("model_table", state)
    df.to_parquet(out, index=False)
    print(f"Modeling table: {len(df):,} county-years, "
          f"{df['fips'].nunique()} counties, {df['year'].min()}-{df['year'].max()}")
    print(f"Saved to {out.relative_to(ROOT)}\n")

    X, y, groups, used = feature_matrix(df)

    print("=" * 70)
    print("CORRELATION OF EACH FEATURE WITH YIELD")
    print("=" * 70)
    corr = X.apply(lambda col: col.corr(y)).sort_values(key=abs, ascending=False)
    print(corr.round(3).to_string())

    def boosting():
        return HistGradientBoostingRegressor(
            max_iter=400, learning_rate=0.06, max_depth=6, random_state=0)

    models = {
        "Mean yield (baseline)": DummyRegressor(strategy="mean"),
        "Ridge (trend + weather)": make_pipeline(StandardScaler(), Ridge(alpha=1.0)),
        "Gradient boosting": boosting(),
        "Detrended boosting": DetrendedRegressor(boosting()),
    }

    trend = DetrendedRegressor(boosting()).fit(X, y)
    print(f"\nFitted yield trend: {trend.trend_slope_:.2f} bu/acre per year")

    print("\n" + "=" * 70)
    print("VALIDATION (lower RMSE is better; RMSE is in bu/acre)")
    print("=" * 70)

    rows = []
    for name, model in models.items():
        random_pred = cross_val_predict(model, X, y, cv=KFold(5, shuffle=True, random_state=0))
        spatial_pred = cross_val_predict(model, X, y, cv=GroupKFold(5), groups=groups)

        # Temporal holdout: train on 2000-2018, predict 2019 onward.
        train, test = used["year"] <= 2018, used["year"] >= 2019
        model.fit(X[train], y[train])
        future_pred = model.predict(X[test])

        rows.append({
            "model": name,
            **{f"random_{k}": v for k, v in scores(y, random_pred).items()},
            **{f"spatial_{k}": v for k, v in scores(y, spatial_pred).items()},
            **{f"future_{k}": v for k, v in scores(y[test], future_pred).items()},
        })

    results = pd.DataFrame(rows).set_index("model")
    results.reset_index().to_parquet(output_path("model_scores", state), index=False)
    for scheme, label in [("random", "Random 5-fold (optimistic)"),
                          ("spatial", "Leave-district-out (spatial)"),
                          ("future", "Train <=2018, test >=2019 (temporal)")]:
        cols = [c for c in results.columns if c.startswith(scheme)]
        print(f"\n{label}")
        print(results[cols].rename(columns=lambda c: c.split("_", 1)[1]).round(2).to_string())

    # What does each added feature buy? Build them up one group at a time, same
    # model, same folds, same rows - so every difference is attributable to one
    # thing. A "group" is usually one column; spring weather is five columns that
    # answer a single question, so they go in and come out together.
    study = feature_groups(df, split_spring=args.split_spring)
    extras = [column for _, columns in study for column in columns]
    if study:
        print("\n" + "=" * 78)
        print("WHAT EACH FEATURE ADDS (detrended boosting, cumulative)")
        print("=" * 78)

        X_all, y_all, groups_all, used_all = feature_matrix(df, extra=extras)
        print(f"Rows with every feature present: {len(X_all):,} of {len(df):,}")
        for name in extras:
            print(f"  correlation of {name} with yield: "
                  f"{X_all[name].corr(y_all):+.3f}")

        def evaluate(Xc, yc, gc, uc, label):
            model = DetrendedRegressor(boosting())
            spatial = cross_val_predict(model, Xc, yc, cv=GroupKFold(5), groups=gc)
            train, test = uc["year"] <= 2018, uc["year"] >= 2019
            model.fit(Xc[train], yc[train])
            future = model.predict(Xc[test])
            return {"features": label,
                    **{f"spatial_{k}": v for k, v in scores(yc, spatial).items()},
                    **{f"future_{k}": v for k, v in scores(yc[test], future).items()}}

        def without(*columns):
            """The full matrix minus these columns - never rebuilt, so rows match."""
            drop = [c for c in columns if c in X_all.columns]
            return X_all.drop(columns=drop)

        ladder = [evaluate(without(*extras), y_all, groups_all, used_all,
                           "weather only")]
        kept: list[str] = []
        for label, columns in study:
            kept += columns
            ladder.append(evaluate(without(*[c for c in extras if c not in kept]),
                                   y_all, groups_all, used_all, f"+ {label}"))

        full_row = ladder[-1]

        # Control: rebuild irrigation share from pre-2019 observations only, so the
        # temporal test cannot borrow information from the 2022 Census.
        observed = load_irrigation_observed(state)
        if observed is not None and "irrigation_share" in extras:
            past_only = build_share_series(
                observed[observed["year"] <= 2018],
                range(int(df["year"].min()), int(df["year"].max()) + 1))
            df_past = (df.drop(columns=["irrigation_share", "share_observed"],
                               errors="ignore")
                       .merge(past_only[["fips", "year", "irrigation_share"]],
                              on=["fips", "year"], how="left"))
            ladder.append(evaluate(*feature_matrix(df_past, extra=extras),
                                   "all features (no look-ahead)"))

        # Ablation: drop one group from the full set. The ladder says what a feature
        # ADDS to what came before; ablation says what is LOST when nothing else can
        # substitute for it. Two correlated features can each look big in the ladder
        # (when first) and small in ablation (because the other covers for it).
        ablation = [{"removed": "nothing (all features)",
                     **{k: v for k, v in full_row.items() if k != "features"}}]
        for label, columns in study:
            ablation.append({
                "removed": label,
                **{k: v for k, v in
                   evaluate(without(*columns), y_all, groups_all, used_all,
                            label).items() if k != "features"}})

        # The split run is an experiment, not the reported result, so it writes to
        # its own files. Otherwise it would quietly replace the tables the app and
        # the documentation read, and the two would disagree without anyone noticing.
        suffix = "_split" if args.split_spring else ""

        comparison_df = pd.DataFrame(ladder)
        comparison_df.to_parquet(output_path(f"irrigation_comparison{suffix}", state),
                                 index=False)
        print("\n" + comparison_df.set_index("features").round(3).to_string())
        print("\nEach row adds one feature (or one group) to the row above. Same "
              "model, same folds, same rows.")

        ablation_df = pd.DataFrame(ablation)
        ablation_df.to_parquet(output_path(f"feature_ablation{suffix}", state),
                               index=False)
        print("\n" + "=" * 78)
        print("WHAT EACH FEATURE IS WORTH ON ITS OWN (remove one, keep the rest)")
        print("=" * 78)
        print(ablation_df.set_index("removed").round(3).to_string())
        print("\nA feature that barely changes the score when removed is REDUNDANT - "
              "\nanother feature covers for it - not necessarily unimportant.")

    # Where does the best model miss? Save errors for mapping later.
    # Use every feature available, which is the model the app and the maps report on.
    X, y, groups, used = feature_matrix(df, extra=extras or None)
    best = DetrendedRegressor(boosting())
    pred = cross_val_predict(best, X, y, cv=GroupKFold(5), groups=groups)
    errors = used[["fips", "county_name", "year", "asd_desc", "yield_bu_acre"]].copy()
    errors["predicted"] = pred.round(1)
    errors["error"] = (errors["predicted"] - errors["yield_bu_acre"]).round(1)
    errors.to_parquet(output_path("model_errors", state), index=False)

    print("\n" + "=" * 70)
    print("WORST YEARS FOR THE MODEL (mean error, bu/acre)")
    print("=" * 70)
    by_year = errors.groupby("year")["error"].mean().round(1)
    print(by_year.reindex(by_year.abs().sort_values(ascending=False).index).head(6).to_string())
    print(f"\nSaved county-level errors to "
          f"{output_path('model_errors', state).relative_to(ROOT)}")


if __name__ == "__main__":
    main()
