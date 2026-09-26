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

from yieldpred.dataset import (PROCESSED, available_rotation, build_modeling_table,
                               feature_groups, feature_matrix,
                               load_irrigation_observed, output_path)
from yieldpred.irrigation import build_share_series
from yieldpred.leak import LeakFreeGroupKFold, leak_report, power_cells
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
    parser.add_argument("--compare-weather", action="store_true",
                        help="Score POWER against the gridMET variants on one "
                             "row set, to measure what the 55 km grid costs")
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
            return {"features": label, "rows": len(Xc),
                    **{f"spatial_{k}": v for k, v in scores(yc, spatial).items()},
                    **{f"future_{k}": v for k, v in scores(yc[test], future).items()}}

        def without(*columns):
            """The full matrix minus these columns - never rebuilt, so rows match."""
            drop = [c for c in columns if c in X_all.columns]
            return X_all.drop(columns=drop)

        # CONTROL ROW, AND IT BELONGS FIRST.
        #
        # Optional features are not available for every county-year, so the study
        # runs on the subset where all of them exist. That subset is not a random
        # sample - NASS suppresses small, marginal counties, which are the hardest
        # ones to predict - so simply having fewer rows RAISES the score. On
        # Nebraska the weather-only spatial R2 moved 0.316 -> 0.451 on nothing but
        # the row change, which is larger than most features in this project.
        #
        # Reporting weather-only twice, on both row sets, makes that visible. Any
        # gain in the rows below has to be read against this line, not against the
        # published all-rows number.
        X_full, y_full, g_full, u_full = feature_matrix(df)
        ladder = [evaluate(X_full, y_full, g_full, u_full,
                           "weather only, ALL rows (control)")]
        ladder.append(evaluate(without(*extras), y_all, groups_all, used_all,
                               "weather only, study rows"))
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
        print("\nEach row adds one feature (or one group) to the row above.")
        sample_effect = (comparison_df.iloc[1]["spatial_R2"]
                         - comparison_df.iloc[0]["spatial_R2"])
        lost = comparison_df.iloc[0]["rows"] - comparison_df.iloc[1]["rows"]
        print(f"The first two rows are the SAME features on different rows: "
              f"{lost:,.0f} county-years drop out")
        print(f"for want of an optional feature, and that alone moves spatial R2 "
              f"by {sample_effect:+.3f}.")
        if abs(sample_effect) > 0.02:
            print("Read every gain below against the second row, never the first.")

        # ---------------------------------------------- is a carried feature real?
        #
        # Suppressed rotation years are carried forward from the county's last
        # measured value, which keeps the rows in the model but makes part of the
        # feature stale. In Nebraska only ~72% of rotation values are measured
        # (Iowa ~94%), so the question is whether the feature is doing work or
        # whether the carried values are diluting it into nothing.
        #
        # Comparing "with rotation" to "without rotation" ON THE SAME MEASURED
        # ROWS is the test. Comparing the measured subset to the full set would
        # confound the feature with the sample, which is the trap the control row
        # at the top of the ladder exists to expose.
        rotation_cols = available_rotation(df)
        if rotation_cols and "rotation_observed" in used_all.columns:
            measured = used_all["rotation_observed"].fillna(False).to_numpy(dtype=bool)
            share = measured.mean()
            print("\n" + "=" * 78)
            print("IS THE CARRIED ROTATION FEATURE DOING ANYTHING? (measured rows only)")
            print("=" * 78)
            print(f"{measured.sum():,} of {len(measured):,} rows ({share:.0%}) have a "
                  f"measured rotation; the rest were carried forward.")

            # `dropped` rather than `without`: assigning to the name would shadow
            # the without() helper and blow up on the call two lines above.
            def delta(rows, label):
                kept = evaluate(X_all[rows], y_all[rows], groups_all[rows],
                                used_all[rows], label)
                dropped = evaluate(without(*rotation_cols)[rows], y_all[rows],
                                   groups_all[rows], used_all[rows], label)
                return kept, dropped

            rows_all = slice(None)
            full_with, full_without = delta(rows_all, "all rows")
            obs_with, obs_without = delta(measured, "measured rows")

            comparison = pd.DataFrame([
                {"rows scored": f"all ({len(measured):,})",
                 "with rotation": full_with["spatial_R2"],
                 "without": full_without["spatial_R2"],
                 "rotation is worth": full_with["spatial_R2"] - full_without["spatial_R2"]},
                {"rows scored": f"measured only ({measured.sum():,})",
                 "with rotation": obs_with["spatial_R2"],
                 "without": obs_without["spatial_R2"],
                 "rotation is worth": obs_with["spatial_R2"] - obs_without["spatial_R2"]},
            ]).set_index("rows scored")
            print("\n" + comparison.round(3).to_string())
            comparison.reset_index().to_parquet(
                output_path(f"rotation_carried_control{suffix}", state), index=False)

            gain_all = full_with["spatial_R2"] - full_without["spatial_R2"]
            gain_obs = obs_with["spatial_R2"] - obs_without["spatial_R2"]
            print("\nIf the second number is much larger, the carried values are")
            print("diluting a real feature. If they are close, carrying is harmless.")
            if gain_obs > gain_all + 0.02:
                print(f"-> DILUTED: rotation is worth {gain_obs:+.3f} where it is "
                      f"measured, {gain_all:+.3f} once carried rows are mixed in.")
            elif abs(gain_obs - gain_all) <= 0.02:
                print(f"-> HARMLESS: {gain_obs:+.3f} measured vs {gain_all:+.3f} "
                      f"overall - carrying costs nothing.")
            else:
                print(f"-> ODD: rotation looks BETTER with carried rows "
                      f"({gain_all:+.3f}) than without ({gain_obs:+.3f}); the "
                      f"measured subset is probably just harder.")

        ablation_df = pd.DataFrame(ablation)
        ablation_df.to_parquet(output_path(f"feature_ablation{suffix}", state),
                               index=False)
        print("\n" + "=" * 78)
        print("WHAT EACH FEATURE IS WORTH ON ITS OWN (remove one, keep the rest)")
        print("=" * 78)
        print(ablation_df.set_index("removed").round(3).to_string())
        print("\nA feature that barely changes the score when removed is REDUNDANT - "
              "\nanother feature covers for it - not necessarily unimportant.")

    # ---- What is the coarse weather grid costing? ----------------------------
    # POWER serves ~55 km cells; gridMET serves ~4 km. Swapping one for the other
    # and reporting the new number would tell us nothing about which choice earned
    # the change. So score every available source on THE SAME county-years with
    # THE SAME folds and the same model - only the weather columns differ.
    if args.compare_weather:
        print("\n" + "=" * 78)
        print("WHAT IS THE 55 km WEATHER GRID COSTING? (same rows, same folds)")
        print("=" * 78)

        sources, tables = {}, {}
        for source in ("power", "gridmet_point", "gridmet_mean"):
            try:
                tables[source] = build_modeling_table(
                    state=state, state_fips=args.state_fips, weather_source=source)
            except FileNotFoundError as err:
                print(f"  {source}: not available ({err})")

        if len(tables) < 2:
            print("\n  Need at least two sources to compare. "
                  "Run scripts/fetch_gridmet.py first.")
        else:
            # The row sets can differ if one product covers a county-year the other
            # doesn't. Intersect them, or this repeats the row-set mistake the
            # feature ladder already has a control for.
            common = None
            for frame in tables.values():
                keys = set(map(tuple, frame[["fips", "year"]].to_numpy()))
                common = keys if common is None else (common & keys)
            print(f"\nScoring {len(common):,} county-years present in all "
                  f"{len(tables)} sources.")
            for source, frame in tables.items():
                dropped = len(frame) - len(common)
                if dropped:
                    print(f"  ({source} had {dropped:,} rows the others lacked)")

            rows = []
            for source, frame in tables.items():
                mask = [tuple(k) in common for k in frame[["fips", "year"]].to_numpy()]
                shared = frame[mask].sort_values(["fips", "year"]).reset_index(drop=True)
                Xc, yc, gc, uc = feature_matrix(shared, extra=extras or None)
                model = DetrendedRegressor(boosting())
                spatial = cross_val_predict(model, Xc, yc, cv=GroupKFold(5), groups=gc)
                train, test = uc["year"] <= 2018, uc["year"] >= 2019
                model.fit(Xc[train], yc[train])
                future = model.predict(Xc[test])
                rows.append({
                    "weather source": source,
                    "rows": len(Xc),
                    **{f"spatial_{k}": v for k, v in scores(yc, spatial).items()},
                    **{f"future_{k}": v for k, v in scores(yc[test], future).items()},
                })

            table = pd.DataFrame(rows).set_index("weather source")
            print("\n" + table.round(3).to_string())
            table.reset_index().to_parquet(
                output_path("weather_source_comparison", state), index=False)

            if "power" in table.index:
                base = table.loc["power", "spatial_R2"]
                print("\nAgainst POWER (55 km):")
                for source in table.index:
                    if source == "power":
                        continue
                    delta = table.loc[source, "spatial_R2"] - base
                    print(f"  {source:14s} {delta:+.3f} spatial R2")
                print("\nRegistered beforehand: the POINT variant was predicted NOT to")
                print("beat POWER (a 55 km cell is already a spatial average, which is")
                print("what a county-average target wants), and the MEAN variant was")
                print("predicted to beat both. If the point variant wins, that")
                print("prediction was wrong and resolution matters more than averaging.")

    # ---- Is leave-district-out leaking weather? ------------------------------
    # Counties smaller than a NASA POWER grid cell can be served by the same cell,
    # in which case their weather rows are identical. If two such counties sit in
    # different districts, holding one out leaves its exact weather vector in the
    # training data. Rerun the spatial CV with those training rows removed - test
    # folds untouched, so the two scores are measured on identical rows.
    X, y, groups, used = feature_matrix(df, extra=extras or None)
    weather = pd.read_parquet(PROCESSED / f"weather_county_{args.state_fips}.parquet")
    cells_by_county = power_cells(weather)
    report = leak_report(cells_by_county.reindex(used["fips"].unique()),
                         used.drop_duplicates("fips").set_index("fips")["asd_desc"])

    print("\n" + "=" * 78)
    print("DOES THE SPATIAL HOLDOUT LEAK WEATHER?")
    print("=" * 78)
    print(f"{report['counties']} modelled counties served by {report['cells']} "
          f"distinct POWER cells")
    print(f"cells spanning more than one district: {report['cells_crossing_districts']}")
    print(f"counties whose exact weather sits in another district: "
          f"{report['counties_leaking']} ({report['share_leaking']:.0%})")

    leak_rows = None
    if report["counties_leaking"]:
        cells_by_row = used["fips"].map(cells_by_county).to_numpy()
        control = DetrendedRegressor(boosting())
        standard = cross_val_predict(control, X, y, cv=GroupKFold(5), groups=groups)
        splitter = LeakFreeGroupKFold(5, cells_by_row)
        leakfree = cross_val_predict(DetrendedRegressor(boosting()), X, y,
                                     cv=splitter, groups=groups)
        cost = splitter.training_cost()

        # The control that makes the comparison mean something. Removing rows
        # lowers a score on its own, so drop the SAME number at random and see
        # what that costs. Several seeds, because one draw is not a control.
        thinned = []
        for seed in range(5):
            pred = cross_val_predict(DetrendedRegressor(boosting()), X, y,
                                     cv=splitter.matched_control(seed=seed),
                                     groups=groups)
            thinned.append(scores(y, pred))
        thin_s = {k: float(np.mean([t[k] for t in thinned])) for k in thinned[0]}

        std_s, lf_s = scores(y, standard), scores(y, leakfree)
        leak_rows = pd.DataFrame([
            {"spatial CV": "standard (leaky)", **std_s},
            {"spatial CV": "same rows dropped, chosen at random (control)", **thin_s},
            {"spatial CV": "leaking counties dropped from training", **lf_s},
        ]).set_index("spatial CV")
        print("\n" + leak_rows.round(3).to_string())
        print(f"\nThe filter cost {cost['share_of_training_lost']:.0%} of the training "
              f"rows ({cost['rows_dropped_total']:,} of "
              f"{cost['rows_dropped_total'] + sum(splitter.trained_):,}), "
              f"per fold {cost['rows_dropped_per_fold']}. The control row above "
              f"loses the same count at random, averaged over 5 seeds.")

        raw = std_s["R2"] - lf_s["R2"]
        from_size = std_s["R2"] - thin_s["R2"]
        attributable = raw - from_size
        print(f"\nTotal drop when the leak is removed:      {raw:+.3f} R2")
        print(f"Drop explained by the smaller training set: {from_size:+.3f} R2")
        print(f"-> attributable to the leak:               {attributable:+.3f} R2")
        if attributable <= 0.005:
            print("\nThe leaking rows were worth no more than any other rows. The")
            print("spatial score is not meaningfully inflated by shared weather.")
        else:
            print(f"\nThe spatial score is inflated by roughly {attributable:.3f} R2.")
            print("Report the corrected figure, or report both.")
        leak_rows.reset_index().to_parquet(
            output_path("leak_control", state), index=False)
    else:
        print("\nNothing to correct: no cell spans a district boundary here.")

    # Where does the best model miss? Save errors for mapping later.
    # Use every feature available, which is the model the app and the maps report on.
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
