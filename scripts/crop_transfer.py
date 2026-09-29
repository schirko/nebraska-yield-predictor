"""Does what the corn model learned help predict soybeans? A transfer-learning test.

    python scripts/crop_transfer.py                       # Nebraska
    python scripts/crop_transfer.py --state IA --state-fips 19

Needs both crops' modeling tables (train_baseline.py, then train_baseline.py --crop soybeans).
Everything is scored the honest way: leave-district-out on soybean county-years, and any corn
model used for a held-out district was trained without that district.

Four ways to predict soybeans:

1. **Soybean model** - the usual detrended boosting, trained on soybeans. The benchmark.
2. **Corn model, applied cold** - no soybean weather training at all. The corn model's
   anomaly (how far above or below trend, as a share of trend) is applied to the soybean trend.
   If corn and soybeans answered weather the same way, this would do as well as (1).
3. **Soybean model + corn signal** - (1) plus one more input: the corn model's predicted
   anomaly for that county-year (stacking).
4. **One model for both crops** - trained on both crops' anomalies as shares of trend, with a
   crop flag, so each crop's data helps the other (multi-task learning).

Also prints which inputs matter most for each crop (permutation importance on held-out
districts), and how closely the two crops' real anomalies move together.
"""

from __future__ import annotations

import argparse

import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.inspection import permutation_importance
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.model_selection import GroupKFold

from yieldpred.twocrop import pooled_fit
from yieldpred.dataset import WEATHER_FEATURES, feature_groups, output_path
from yieldpred.trend import DetrendedRegressor


def boosting():
    return HistGradientBoostingRegressor(max_iter=400, learning_rate=0.06, max_depth=6, random_state=0)


def load(state: str) -> tuple[pd.DataFrame, pd.DataFrame, list[str]]:
    corn = pd.read_parquet(output_path("model_table", state, "corn"))
    soy = pd.read_parquet(output_path("model_table", state, "soybeans"))
    extras = [c for _, cols in feature_groups(soy) for c in cols if c in corn]
    features = ["year"] + WEATHER_FEATURES + extras
    corn = corn.dropna(subset=features + ["yield_bu_acre"]).reset_index(drop=True)
    soy = soy.dropna(subset=features + ["yield_bu_acre"]).reset_index(drop=True)
    return corn, soy, features


def relative_anomaly(model: DetrendedRegressor, X: pd.DataFrame) -> np.ndarray:
    """The model's weather effect as a share of its trend (0.05 = 5% above trend)."""
    trend = model.trend_.predict(np.asarray(X["year"], dtype=float).reshape(-1, 1))
    return model.estimator_.predict(X.drop(columns=["year"])) / trend


def score(y, pred) -> dict:
    return {"RMSE": round(float(np.sqrt(mean_squared_error(y, pred))), 2), "R2": round(float(r2_score(y, pred)), 3)}


FIRST_FORWARD_YEAR = 2006   # as in honest_ranges.py: six seasons known before the first forecast


def season_bootstrap(frame: pd.DataFrame, a: str, b: str, draws: int = 2000) -> tuple[float, float, float, float]:
    """RMSE of column `b` minus RMSE of column `a`, resampling whole seasons. Returns the mean
    difference, a 90% interval and the share of draws where `b` was better (lower)."""
    rng = np.random.default_rng(0)
    by_year = {y: g for y, g in frame.groupby("year")}
    years = list(by_year)
    out = []
    for _ in range(draws):
        g = pd.concat([by_year[y] for y in rng.choice(years, size=len(years))])
        out.append(np.sqrt(np.mean((g["actual"] - g[b]) ** 2)) - np.sqrt(np.mean((g["actual"] - g[a]) ** 2)))
    out = np.asarray(out)
    return float(out.mean()), float(np.quantile(out, .05)), float(np.quantile(out, .95)), float((out < 0).mean())


def next_season(corn: pd.DataFrame, soy: pd.DataFrame, features: list[str]) -> pd.DataFrame:
    """The harder test: every season from FIRST_FORWARD_YEAR, predicted by models trained only on
    earlier seasons (both crops' earlier seasons for the two-crop model). One row per soybean
    county-year with the actual yield and each approach's prediction."""
    frames = []
    for year in sorted(soy["year"].unique()):
        if year < FIRST_FORWARD_YEAR:
            continue
        s_tr, s_te = soy[soy["year"] < year], soy[soy["year"] == year]
        c_tr = corn[corn["year"] < year]
        soy_model = DetrendedRegressor(boosting()).fit(s_tr[features], s_tr["yield_bu_acre"])
        pooled, trends = pooled_fit([(0, c_tr[features], c_tr["yield_bu_acre"]),
                                     (1, s_tr[features], s_tr["yield_bu_acre"])])
        trend_te = trends[1].predict(s_te[["year"]].to_numpy(float))
        rel = pooled.predict(s_te[features].drop(columns=["year"]).assign(is_soy=1))
        frames.append(pd.DataFrame({"fips": s_te["fips"].to_numpy(), "year": year,
                                    "actual": s_te["yield_bu_acre"].to_numpy(),
                                    "soy": soy_model.predict(s_te[features]),
                                    "pooled": trend_te * (1 + rel)}))
    return pd.concat(frames, ignore_index=True)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--state", default="NE")
    parser.add_argument("--state-fips", default="31")
    args = parser.parse_args()
    state = args.state.lower()

    corn, soy, features = load(state)
    Xs, ys, gs = soy[features], soy["yield_bu_acre"], soy["asd_code"]
    print(f"{state.upper()}: {len(soy):,} soybean and {len(corn):,} corn county-years, "
          f"{len(features) - 1} inputs\n")

    preds = {k: np.empty(len(soy)) for k in ("soy", "cold", "stacked", "pooled")}
    importance = {"corn": [], "soybeans": []}
    for train, test in GroupKFold(5).split(Xs, ys, gs):
        held_out = set(gs.iloc[test])
        corn_train = corn[~corn["asd_code"].isin(held_out)]
        corn_model = DetrendedRegressor(boosting()).fit(corn_train[features], corn_train["yield_bu_acre"])
        soy_model = DetrendedRegressor(boosting()).fit(Xs.iloc[train], ys.iloc[train])
        preds["soy"][test] = soy_model.predict(Xs.iloc[test])

        # 2. corn's anomaly on the soybean trend (the soybean trend uses soybean yields and years only)
        soy_trend = LinearRegression().fit(Xs.iloc[train][["year"]].to_numpy(float), ys.iloc[train])
        trend_test = soy_trend.predict(Xs.iloc[test][["year"]].to_numpy(float))
        preds["cold"][test] = trend_test * (1 + relative_anomaly(corn_model, Xs.iloc[test]))

        # 3. stacking: the corn signal as one more input (for training rows too, from a corn model
        #    that never saw the held-out districts, so nothing leaks)
        signal_tr = relative_anomaly(corn_model, Xs.iloc[train])
        signal_te = relative_anomaly(corn_model, Xs.iloc[test])
        stacked = DetrendedRegressor(boosting()).fit(Xs.iloc[train].assign(corn_signal=signal_tr), ys.iloc[train])
        preds["stacked"][test] = stacked.predict(Xs.iloc[test].assign(corn_signal=signal_te))

        # 4. one model, both crops, anomalies as shares of each crop's own trend
        frames = []
        for flag, frame, y in ((0, corn_train[features], corn_train["yield_bu_acre"]),
                               (1, Xs.iloc[train], ys.iloc[train])):
            t = LinearRegression().fit(frame[["year"]].to_numpy(float), y)
            rel = y.to_numpy() / t.predict(frame[["year"]].to_numpy(float)) - 1
            frames.append(frame.drop(columns=["year"]).assign(is_soy=flag, target=rel))
        both = pd.concat(frames, ignore_index=True)
        pooled = boosting().fit(both.drop(columns=["target"]), both["target"])
        rel_te = pooled.predict(Xs.iloc[test].drop(columns=["year"]).assign(is_soy=1))
        preds["pooled"][test] = trend_test * (1 + rel_te)

        # which inputs matter, measured on held-out districts only
        for crop, model, X, y in (("soybeans", soy_model, Xs.iloc[test], ys.iloc[test]),):
            r = permutation_importance(model, X, y, n_repeats=5, random_state=0, scoring="neg_root_mean_squared_error")
            importance[crop].append(pd.Series(r.importances_mean, index=features))
        corn_test = corn[corn["asd_code"].isin(held_out)]
        r = permutation_importance(corn_model, corn_test[features], corn_test["yield_bu_acre"], n_repeats=5,
                                   random_state=0, scoring="neg_root_mean_squared_error")
        importance["corn"].append(pd.Series(r.importances_mean, index=features))

    preds["blend"] = (preds["soy"] + preds["pooled"]) / 2
    rows = [("Soybean model (benchmark)", "soy"), ("Corn model applied cold", "cold"),
            ("Soybean model + corn signal", "stacked"), ("One model for both crops", "pooled"),
            ("Average of soybean and both-crops models", "blend")]
    table = pd.DataFrame([{"approach": label, **score(ys, preds[key])} for label, key in rows])
    # Is a difference bigger than luck? Resample whole seasons (errors within a season move
    # together, so resampling single rows would overstate certainty) and see how often each
    # approach beats the soybean model.
    rng = np.random.default_rng(0)
    years = soy["year"].to_numpy()
    by_year = {y: np.flatnonzero(years == y) for y in np.unique(years)}
    draws = []
    for _ in range(2000):
        idx = np.concatenate([by_year[y] for y in rng.choice(list(by_year), size=len(by_year))])
        base = np.sqrt(np.mean((ys.to_numpy()[idx] - preds["soy"][idx]) ** 2))
        draws.append({key: np.sqrt(np.mean((ys.to_numpy()[idx] - preds[key][idx]) ** 2)) - base
                      for _, key in rows})
    draws = pd.DataFrame(draws)
    table["vs soybean model"] = [round(float(draws[key].mean()), 2) for _, key in rows]
    table["90% interval"] = [f"{draws[key].quantile(.05):+.2f} to {draws[key].quantile(.95):+.2f}" for _, key in rows]
    table["share of draws better"] = [f"{(draws[key] < 0).mean():.0%}" for _, key in rows]
    print("PREDICTING SOYBEANS IN DISTRICTS NEVER SEEN (RMSE in bu/acre; negative = better)")
    print(table.to_string(index=False))

    # how much of each crop's error comes from each input: bu/acre of RMSE lost when it is shuffled,
    # and as a share of the crop's mean yield so the two crops compare
    imp = pd.DataFrame({crop: pd.concat(v, axis=1).mean(axis=1) for crop, v in importance.items()})
    means = {"corn": corn["yield_bu_acre"].mean(), "soybeans": soy["yield_bu_acre"].mean()}
    rel = pd.DataFrame({f"{c} %": (100 * imp[c] / means[c]).round(2) for c in imp})
    rel["corn rank"] = rel["corn %"].rank(ascending=False).astype(int)
    rel["soy rank"] = rel["soybeans %"].rank(ascending=False).astype(int)
    print("\nWHAT EACH INPUT IS WORTH (RMSE lost when shuffled, % of mean yield)")
    print(rel.sort_values("soybeans %", ascending=False).to_string())

    paired = corn.merge(soy, on=["fips", "year"], suffixes=("_corn", "_soy"))
    anomalies = {}
    for crop in ("corn", "soy"):
        t = LinearRegression().fit(paired[["year"]].to_numpy(float), paired[f"yield_bu_acre_{crop}"])
        anomalies[crop] = paired[f"yield_bu_acre_{crop}"] / t.predict(paired[["year"]].to_numpy(float)) - 1
    together = np.corrcoef(anomalies["corn"], anomalies["soy"])[0, 1]
    print(f"\nReal anomalies of the two crops, same county and year ({len(paired):,} pairs): "
          f"correlation {together:.2f}")

    print("\nNEXT SEASON: trained only on earlier seasons, predicting the next (soybeans, bu/acre)")
    forward = next_season(corn, soy, features)
    forward["blend"] = (forward["soy"] + forward["pooled"]) / 2
    records = [{"approach": "Soybean model (benchmark)", "key": "soy"},
               {"approach": "One model for both crops", "key": "pooled"},
               {"approach": "Average of soybean and both-crops models", "key": "blend"}]
    for r in records:
        k = r.pop("key")
        r["RMSE"] = round(float(np.sqrt(np.mean((forward["actual"] - forward[k]) ** 2))), 2)
        if k == "soy":
            r.update({"vs soybean model": 0.0, "90% interval": "-", "share of draws better": "-", "seasons won": "-"})
            continue
        mean, lo, hi, share = season_bootstrap(forward, "soy", k)
        won = forward.groupby("year").apply(
            lambda g, k=k: np.mean((g["actual"] - g[k]) ** 2) < np.mean((g["actual"] - g["soy"]) ** 2),
            include_groups=False)
        r.update({"vs soybean model": round(mean, 2), "90% interval": f"{lo:+.2f} to {hi:+.2f}",
                  "share of draws better": f"{share:.0%}", "seasons won": f"{int(won.sum())} of {len(won)}"})
    forward_table = pd.DataFrame(records)
    print(forward_table.to_string(index=False))
    forward_table.to_parquet(output_path("crop_transfer_forward", state), index=False)

    table.to_parquet(output_path("crop_transfer", state), index=False)
    rel.reset_index(names="input").to_parquet(output_path("crop_importance", state), index=False)
    print(f"\nSaved {output_path('crop_transfer', state).name}, {output_path('crop_importance', state).name}, "
          f"{output_path('crop_transfer_forward', state).name}")


if __name__ == "__main__":
    main()
