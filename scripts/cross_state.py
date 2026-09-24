"""Train on one state, predict another - the hardest generalization test here.

    python scripts/cross_state.py --train NE --test IA

Leave-district-out cross-validation asks whether the model can predict an unseen
county *within* the state it learned from. This asks something stricter: can it
predict a state it has never seen at all, with different soils, different rainfall
and (in Iowa's case) almost no irrigation?

Three numbers are reported, and the gaps between them are the point:

1. **Home performance** - the test state's own model, scored leave-district-out.
   The ceiling: what's achievable when the model has seen that state.
2. **Transferred** - trained entirely on the training state, applied cold.
3. **Transferred, level-corrected** - the same predictions after subtracting their
   mean error. This separates two different failures: getting the *level* wrong
   (a constant offset, easily fixed with one year of local data) from getting the
   *pattern* wrong (which county does better than which, and in what year - the
   part that actually matters).
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.dummy import DummyRegressor
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import GroupKFold, cross_val_predict

from yieldpred.dataset import (available_extras, available_spring,
                               build_modeling_table, feature_matrix, output_path)
from yieldpred.trend import DetrendedRegressor

ROOT = Path(__file__).resolve().parents[1]
STATE_FIPS = {"ne": "31", "ia": "19", "ks": "20", "mo": "29", "sd": "46", "il": "17"}


def boosting():
    return HistGradientBoostingRegressor(max_iter=400, learning_rate=0.06,
                                         max_depth=6, random_state=0)


def scores(y_true, y_pred) -> dict:
    return {"RMSE": float(np.sqrt(mean_squared_error(y_true, y_pred))),
            "MAE": float(mean_absolute_error(y_true, y_pred)),
            "R2": float(r2_score(y_true, y_pred)),
            "bias": float(np.mean(np.asarray(y_pred) - np.asarray(y_true)))}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--train", default="NE")
    parser.add_argument("--test", default="IA")
    args = parser.parse_args()
    train_state, test_state = args.train.lower(), args.test.lower()

    tables = {}
    for state in (train_state, test_state):
        fips = STATE_FIPS.get(state)
        if fips is None:
            raise SystemExit(f"Unknown state {state!r}; add it to STATE_FIPS.")
        try:
            tables[state] = build_modeling_table(state=state, state_fips=fips)
        except FileNotFoundError as exc:
            raise SystemExit(f"Missing data for {state.upper()}: {exc}\n"
                             f"Run the fetch scripts with --state {state.upper()} first.")

    # Only features both states actually have. Irrigation share drops out here
    # whenever the test state doesn't irrigate, which is the point: the transfer
    # has to manage without the training state's most valuable column.
    def optional(state: str) -> list[str]:
        return available_extras(tables[state]) + available_spring(tables[state])

    shared = [f for f in optional(train_state) if f in optional(test_state)]
    print(f"Training on {train_state.upper()} ({len(tables[train_state]):,} county-years), "
          f"testing on {test_state.upper()} ({len(tables[test_state]):,})")
    print(f"Shared features beyond weather: {', '.join(shared) or 'none'}\n")

    X_tr, y_tr, g_tr, _ = feature_matrix(tables[train_state], extra=shared)
    X_te, y_te, g_te, used_te = feature_matrix(tables[test_state], extra=shared)

    rows = []

    # 1. Ceiling: the test state's own model, honestly validated.
    home = cross_val_predict(DetrendedRegressor(boosting()), X_te, y_te,
                             cv=GroupKFold(5), groups=g_te)
    rows.append({"model": f"{test_state.upper()} own model (leave-district-out)",
                 **scores(y_te, home)})

    # 2. Floor: predict the training state's average everywhere.
    dummy = DummyRegressor(strategy="mean").fit(X_tr, y_tr)
    rows.append({"model": f"{train_state.upper()} average, applied to {test_state.upper()}",
                 **scores(y_te, dummy.predict(X_te))})

    # 3. The transfer itself.
    model = DetrendedRegressor(boosting()).fit(X_tr, y_tr)
    transferred = model.predict(X_te)
    rows.append({"model": f"Trained on {train_state.upper()}, applied cold",
                 **scores(y_te, transferred)})

    # 4. The same predictions with their average offset removed.
    corrected = transferred - np.mean(transferred - np.asarray(y_te))
    rows.append({"model": "...after removing the average offset",
                 **scores(y_te, corrected)})

    results = pd.DataFrame(rows).set_index("model")
    print("=" * 88)
    print(f"CAN A {train_state.upper()} MODEL PREDICT {test_state.upper()}?")
    print("=" * 88)
    print(results.round(2).to_string())
    print("\nbias = mean(predicted - actual): positive means the model runs high.")

    # Save predictions for mapping and for the app.
    out = used_te[["fips", "county_name", "year", "yield_bu_acre"]].copy()
    out["predicted"] = np.round(transferred, 1)
    out["error"] = (out["predicted"] - out["yield_bu_acre"]).round(1)
    out["predicted_corrected"] = np.round(corrected, 1)
    path = output_path(f"cross_state_{train_state}_to_{test_state}", "ne")
    out.to_parquet(path, index=False)
    results.reset_index().to_parquet(
        output_path(f"cross_state_scores_{train_state}_to_{test_state}", "ne"), index=False)

    print(f"\nSaved {len(out):,} predictions to {path.relative_to(ROOT)}")

    print("\n" + "=" * 88)
    print("WHERE THE TRANSFER STRUGGLES")
    print("=" * 88)

    # Two different failures, reported separately. `mean_error` is the level: how far
    # off the whole state was that year. `county_skill` is the pattern: within that one
    # year, does the model know which counties do better? Pooled R2 blends them, and a
    # decision tool cares mostly about the second.
    def within_year_skill(frame: pd.DataFrame) -> pd.Series:
        """Correlation of predicted with actual ACROSS counties, one value per year."""
        return pd.Series({year: group["predicted"].corr(group["yield_bu_acre"])
                          for year, group in frame.groupby("year")}, name="county_skill")

    by_year = out.groupby("year").agg(mean_error=("error", "mean"),
                                      counties=("fips", "nunique"))
    by_year["mean_abs_error"] = out.assign(a=out["error"].abs()).groupby("year")["a"].mean()
    by_year["county_skill"] = within_year_skill(out)
    print(by_year.round({"mean_error": 1, "mean_abs_error": 1,
                         "county_skill": 3}).to_string())

    home_frame = pd.DataFrame({"year": used_te["year"].to_numpy(),
                               "predicted": home,
                               "yield_bu_acre": np.asarray(y_te)})
    print(f"\nMedian within-year county skill: transferred "
          f"{by_year['county_skill'].median():.3f}, "
          f"{test_state.upper()}'s own model {within_year_skill(home_frame).median():.3f}")
    print("A transfer that keeps most of the home model's within-year skill has learned\n"
          "relationships. Years where it goes negative are worth naming individually.")


if __name__ == "__main__":
    main()
