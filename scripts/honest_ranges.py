"""Honest yield ranges: an 80% range for every prediction, and how often it really held.

    python scripts/honest_ranges.py                       # Nebraska
    python scripts/honest_ranges.py --state IA --state-fips 19
    python scripts/honest_ranges.py --crop soybeans

Uses the same model and the same leave-district-out folds as train_baseline.py (detrended boosting,
every feature), so the predictions match the ones the app already shows. Writes:

- ranges(_ia).parquet: one row per county-year: yield, prediction, 80% range, whether it held
- ranges_coverage(_ia).parquet: how often the ranges held, overall and by irrigation group, for a
  single width and (where irrigation varies) for widths scaled to irrigation; the one whose
  worst group lands closest to 80% is the one saved and shown (`shown` column)
- ranges_forward(_ia).parquet: the next-season test, season by season

Method and reasons: src/yieldpred/ranges.py.
"""

from __future__ import annotations

import argparse

import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.model_selection import GroupKFold

from yieldpred import ranges
from yieldpred import twocrop
from yieldpred.crops import get_crop
from yieldpred.dataset import feature_groups, feature_matrix, output_path
from yieldpred.trend import DetrendedRegressor

FIRST_FORWARD_YEAR = 2006   # the next-season test starts once six seasons are known


def boosting():
    """The model train_baseline.py reports on (same settings)."""
    return HistGradientBoostingRegressor(max_iter=400, learning_rate=0.06, max_depth=6, random_state=0)


def spatial_predictions(X, y, groups) -> tuple[np.ndarray, np.ndarray]:
    """Leave-district-out predictions and the fold each row was predicted in."""
    pred, fold_of = np.empty(len(y)), np.empty(len(y), dtype=int)
    for k, (train, test) in enumerate(GroupKFold(5).split(X, y, groups)):
        model = DetrendedRegressor(boosting()).fit(X.iloc[train], y.iloc[train])
        pred[test], fold_of[test] = model.predict(X.iloc[test]), k
    return pred, fold_of


def forward_predictions(X, y, used) -> pd.DataFrame:
    """Each season from FIRST_FORWARD_YEAR predicted by a model trained only on earlier seasons."""
    frames = []
    for year in sorted(used["year"].unique()):
        if year < FIRST_FORWARD_YEAR:
            continue
        train, test = (used["year"] < year).to_numpy(), (used["year"] == year).to_numpy()
        model = DetrendedRegressor(boosting()).fit(X[train], y[train])
        frame = used.loc[test, ["fips", "year"]].copy()
        frame["error"] = y[test].to_numpy() - model.predict(X[test])
        if "irrigation_share" in used:
            frame["irrigation_share"] = used.loc[test, "irrigation_share"].to_numpy()
        frames.append(frame)
    return pd.concat(frames, ignore_index=True)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--state", default="NE")
    parser.add_argument("--state-fips", default="31")
    parser.add_argument("--crop", default="corn", help="corn (default) or soybeans")
    args = parser.parse_args()
    state = args.state.lower()
    crop = get_crop(args.crop).key

    df = pd.read_parquet(output_path("model_table", state, crop))
    extras = [column for _, columns in feature_groups(df) for column in columns]
    X, y, groups, used = feature_matrix(df, extra=extras or None)
    used = used.reset_index(drop=True)
    X, y = X.reset_index(drop=True), y.reset_index(drop=True)
    scaled = "irrigation_share" in used and used["irrigation_share"].std() > 0.05

    pred, folds = spatial_predictions(X, y, groups)
    both = twocrop.uses_both_crops(state, crop)
    if both:   # Nebraska soybeans: the average with the two-crop model (see yieldpred/twocrop.py)
        corn = pd.read_parquet(output_path("model_table", state, "corn"))
        pred = twocrop.spatial_blend(X, y, groups, corn)
        print("Predictions: average of the soybean model and the two-crop model")
    errors = y.to_numpy() - pred
    table = used[["fips", "county_name", "year", "asd_desc", "yield_bu_acre"]].copy()
    table["predicted"] = pred.round(1)
    table["error"] = errors
    if "irrigation_share" in used:
        table["irrigation_share"] = used["irrigation_share"]
        table["irrigation_group"] = ranges.irrigation_group(used["irrigation_share"]).to_numpy()

    print(f"{state.upper()} {crop}: {len(table):,} county-years, RMSE {np.sqrt(np.mean(errors ** 2)):.1f} bu/acre "
          f"(leave-district-out)")
    coverage_rows, frames = [], {}
    variants = [("one width", None)] + ([("width scaled to irrigation", used["irrigation_share"])] if scaled else [])
    for label, irrigation in variants:
        low, high = ranges.cross_conformal(errors, folds, irrigation)
        frame = table.assign(low=low, high=high)
        parts = [ranges.coverage(frame).reset_index(drop=True).assign(group="all")]
        if "irrigation_group" in frame:
            parts.append(ranges.coverage(frame, "irrigation_group").rename_axis("group").reset_index())
        result = pd.concat(parts, ignore_index=True)[["group", "rows", "held_pct", "width_bu"]]
        result.insert(0, "ranges", label)
        coverage_rows.append(result)
        by_year = frame.assign(inside=(frame.error >= frame.low) & (frame.error <= frame.high)).groupby("year")["inside"].mean()
        print(f"\n{label}: held {100 * ((frame.error >= frame.low) & (frame.error <= frame.high)).mean():.1f}% "
              f"(aim 80), average width {np.mean(high - low):.1f} bu/acre; "
              f"worst season {100 * by_year.min():.0f}% ({by_year.idxmin()})")
        print(result.drop(columns="ranges").to_string(index=False))
        frames[label] = frame

    coverage_table = pd.concat(coverage_rows, ignore_index=True)
    # Show the variant whose worst group is closest to the aim. Chosen by the data, not by habit:
    # scaling to irrigation fixed corn's dryland counties but made soybeans' worse.
    best, gap = ranges.pick_variant(coverage_table)
    coverage_table["shown"] = coverage_table["ranges"] == best
    chosen, scaled = frames[best], best != "one width"
    print(f"\nShown in the app: {best} (worst group {gap:.1f} points from {100 * ranges.LEVEL:.0f}%)")
    coverage_table.to_parquet(output_path("ranges_coverage", state, crop), index=False)

    out = chosen[["fips", "county_name", "year", "asd_desc", "yield_bu_acre", "predicted"]].copy()
    out["low"] = (chosen["predicted"] + chosen["low"]).round(1)
    out["high"] = (chosen["predicted"] + chosen["high"]).round(1)
    out["inside"] = (out["yield_bu_acre"] >= out["low"]) & (out["yield_bu_acre"] <= out["high"])
    out.to_parquet(output_path("ranges", state, crop), index=False)

    print("\nNEXT-SEASON TEST (train on earlier seasons only, predict the next)")
    forward = (twocrop.next_season_blend(X, y, used, corn, FIRST_FORWARD_YEAR) if both
               else forward_predictions(X, y, used))
    print(f"RMSE {np.sqrt(np.mean(forward.error ** 2)):.1f} bu/acre over {forward.year.nunique()} seasons")
    seasons = ranges.forward_years(forward, scaled=scaled)
    seasons["season_share"] = round(ranges.season_share(forward), 3)   # same for every row
    seasons.to_parquet(output_path("ranges_forward", state, crop), index=False)
    print(f"held {seasons.held_pct.mean():.1f}% on average (aim 80); "
          f"{(seasons.held_pct < 60).sum()} of {len(seasons)} seasons under 60%; "
          f"{100 * ranges.season_share(forward):.0f}% of the miss was shared by the whole state")
    print(seasons.to_string(index=False))
    print(f"\nSaved {output_path('ranges', state, crop).name}, {output_path('ranges_coverage', state, crop).name}, "
          f"{output_path('ranges_forward', state, crop).name}")


if __name__ == "__main__":
    main()
