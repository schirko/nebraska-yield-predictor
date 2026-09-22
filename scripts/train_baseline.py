"""Build the modeling table and compare models under different validation schemes.

    python scripts/train_baseline.py

Three models, three ways of validating them. The point of the exercise is that
random cross-validation flatters a spatial model: neighbouring counties in the
same year are so similar that a random split leaks information. Holding out
whole agricultural districts, or whole years, is the honest test.
"""

from __future__ import annotations

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

from yieldpred.dataset import PROCESSED, build_modeling_table, feature_matrix

ROOT = Path(__file__).resolve().parents[1]


def scores(y_true, y_pred) -> dict:
    return {
        "RMSE": float(np.sqrt(mean_squared_error(y_true, y_pred))),
        "MAE": float(mean_absolute_error(y_true, y_pred)),
        "R2": float(r2_score(y_true, y_pred)),
    }


def main() -> None:
    df = build_modeling_table()
    out = PROCESSED / "model_table.parquet"
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

    models = {
        "Mean yield (baseline)": DummyRegressor(strategy="mean"),
        "Ridge (trend + weather)": make_pipeline(StandardScaler(), Ridge(alpha=1.0)),
        "Gradient boosting": HistGradientBoostingRegressor(
            max_iter=400, learning_rate=0.06, max_depth=6, random_state=0),
    }

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
    for scheme, label in [("random", "Random 5-fold (optimistic)"),
                          ("spatial", "Leave-district-out (spatial)"),
                          ("future", "Train <=2018, test >=2019 (temporal)")]:
        cols = [c for c in results.columns if c.startswith(scheme)]
        print(f"\n{label}")
        print(results[cols].rename(columns=lambda c: c.split("_", 1)[1]).round(2).to_string())

    # Where does the best model miss? Save errors for mapping later.
    best = models["Gradient boosting"]
    pred = cross_val_predict(best, X, y, cv=GroupKFold(5), groups=groups)
    errors = used[["fips", "county_name", "year", "asd_desc", "yield_bu_acre"]].copy()
    errors["predicted"] = pred.round(1)
    errors["error"] = (errors["predicted"] - errors["yield_bu_acre"]).round(1)
    errors.to_parquet(PROCESSED / "model_errors.parquet", index=False)

    print("\n" + "=" * 70)
    print("WORST YEARS FOR THE MODEL (mean error, bu/acre)")
    print("=" * 70)
    by_year = errors.groupby("year")["error"].mean().round(1)
    print(by_year.reindex(by_year.abs().sort_values(ascending=False).index).head(6).to_string())
    print("\nSaved county-level errors to data/processed/model_errors.parquet")


if __name__ == "__main__":
    main()
