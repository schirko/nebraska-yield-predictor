"""Soybean predictions that borrow from corn, where the test says it helps.

`scripts/crop_transfer.py` tried four ways of using corn to predict soybeans. One held up on both
honest tests in Nebraska and nowhere else: the **average of the soybean model and one model trained
on both crops** (each crop's yield as a share of its own trend, plus a crop flag).

    Nebraska soybeans, RMSE bu/acre   unseen districts          next season
    soybean model                     6.18                      6.38
    average with the two-crop model   5.94 (better, 100%)       6.08 (better, 97%)

(% = share of 2,000 season resamples in which it was better.) The two-crop model alone gained as
much on average but was erratic from season to season: a big win in the unusual seasons (the 2012
and 2022 droughts, 2009) and small losses in ordinary ones. Averaging keeps most of the win and
gives back little of the loss. In Iowa the average's gain was 0.04 to 0.06 bu/acre, too small to be
worth the complexity, so Iowa soybeans use the soybean model alone.

Honesty note: the average was chosen after seeing the next-season result, so that result alone
could flatter it. The unseen-district test is the independent check, and it agreed.

`BOTH_CROPS` says where the app uses the average. train_baseline.py and honest_ranges.py call
`spatial_blend` / `next_season_blend` there, so the predictions, the error maps and the honest
ranges all come from the same model.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import GroupKFold

from yieldpred.trend import DetrendedRegressor

BOTH_CROPS = {("ne", "soybeans")}     # (state, crop) where the average won both tests


def uses_both_crops(state: str, crop: str) -> bool:
    return (state.lower(), crop) in BOTH_CROPS


def boosting():
    """The project's boosting settings (as train_baseline.py)."""
    return HistGradientBoostingRegressor(max_iter=400, learning_rate=0.06, max_depth=6, random_state=0)


def pooled_fit(frames):
    """One model for both crops. `frames`: [(is_soy, X with a year column, y)]. Each crop's yield
    becomes a share of its own linear trend, a crop flag is added, and one booster learns both.
    Returns the booster and each crop's trend (keyed by the flag)."""
    rows, trends = [], {}
    for flag, X, y in frames:
        years = X[["year"]].to_numpy(float)
        trends[flag] = LinearRegression().fit(years, y)
        rel = np.asarray(y, dtype=float) / trends[flag].predict(years) - 1
        rows.append(X.drop(columns=["year"]).assign(is_soy=flag, target=rel))
    both = pd.concat(rows, ignore_index=True)
    return boosting().fit(both.drop(columns=["target"]), both["target"]), trends


def pooled_predict(model, trends, X) -> np.ndarray:
    """Soybean yields (bu/acre) from the two-crop model: the soybean trend times (1 + anomaly)."""
    rel = model.predict(X.drop(columns=["year"]).assign(is_soy=1))
    return trends[1].predict(X[["year"]].to_numpy(float)) * (1 + rel)


def _blend(soy_X, soy_y, corn_X, corn_y, test_X) -> np.ndarray:
    soy_model = DetrendedRegressor(boosting()).fit(soy_X, soy_y)
    pooled, trends = pooled_fit([(0, corn_X, corn_y), (1, soy_X, soy_y)])
    return (soy_model.predict(test_X) + pooled_predict(pooled, trends, test_X)) / 2


def spatial_blend(X: pd.DataFrame, y: pd.Series, groups: pd.Series, corn: pd.DataFrame,
                  folds: int = 5) -> np.ndarray:
    """Leave-district-out soybean predictions from the average. Same GroupKFold folds as the
    soybean model; the corn rows used for a fold exclude that fold's districts, so nothing about a
    held-out district reaches the prediction through corn either. `corn` has the X columns,
    yield_bu_acre and asd_code."""
    features = list(X.columns)
    corn = corn.dropna(subset=features + ["yield_bu_acre"])
    pred = np.empty(len(y))
    for train, test in GroupKFold(folds).split(X, y, groups):
        c = corn[~corn["asd_code"].isin(set(groups.iloc[test]))]
        pred[test] = _blend(X.iloc[train], y.iloc[train], c[features], c["yield_bu_acre"], X.iloc[test])
    return pred


def next_season_blend(X: pd.DataFrame, y: pd.Series, used: pd.DataFrame, corn: pd.DataFrame,
                      first_year: int) -> pd.DataFrame:
    """Each season from `first_year`, predicted by the average of models trained only on earlier
    seasons of both crops. Returns fips, year, error (actual - predicted), plus irrigation_share
    when present, like honest_ranges.forward_predictions."""
    features = list(X.columns)
    corn = corn.dropna(subset=features + ["yield_bu_acre"])
    frames = []
    for year in sorted(used["year"].unique()):
        if year < first_year:
            continue
        train, test = (used["year"] < year).to_numpy(), (used["year"] == year).to_numpy()
        c = corn[corn["year"] < year]
        pred = _blend(X[train], y[train], c[features], c["yield_bu_acre"], X[test])
        frame = used.loc[test, ["fips", "year"]].copy()
        frame["error"] = y[test].to_numpy() - pred
        if "irrigation_share" in used:
            frame["irrigation_share"] = used.loc[test, "irrigation_share"].to_numpy()
        frames.append(frame)
    return pd.concat(frames, ignore_index=True)
