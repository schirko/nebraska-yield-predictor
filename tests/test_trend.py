"""Tests for DetrendedRegressor, focused on the extrapolation problem it solves."""

import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingRegressor

from yieldpred.trend import DetrendedRegressor

SLOPE = 2.0  # bu/acre per year


def make_data(years=range(2000, 2019), n_per_year=40, seed=0) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    rows = []
    for year in years:
        heat = rng.normal(28, 8, n_per_year)
        yield_ = 100 + SLOPE * (year - 2000) - 1.5 * heat + rng.normal(0, 5, n_per_year)
        rows.append(pd.DataFrame({"year": year, "heat_days_32": heat, "y": yield_}))
    return pd.concat(rows, ignore_index=True)


def test_recovers_the_trend_slope():
    df = make_data()
    model = DetrendedRegressor(HistGradientBoostingRegressor(max_iter=50)).fit(
        df[["year", "heat_days_32"]], df["y"])
    assert abs(model.trend_slope_ - SLOPE) < 0.3


def test_extrapolates_to_unseen_years_where_plain_boosting_cannot():
    train = make_data()
    future = make_data(years=range(2019, 2026), seed=1)
    X_train, y_train = train[["year", "heat_days_32"]], train["y"]
    X_future, y_future = future[["year", "heat_days_32"]], future["y"]

    plain = HistGradientBoostingRegressor(max_iter=200, random_state=0).fit(X_train, y_train)
    detrended = DetrendedRegressor(
        HistGradientBoostingRegressor(max_iter=200, random_state=0)).fit(X_train, y_train)

    plain_error = np.abs(plain.predict(X_future) - y_future).mean()
    detrended_error = np.abs(detrended.predict(X_future) - y_future).mean()

    # Plain boosting flat-lines at the last training year and drifts further off
    # every year; detrending should cut the error by more than half.
    assert detrended_error < plain_error / 2


def test_base_estimator_does_not_see_the_year_column():
    df = make_data()
    model = DetrendedRegressor(HistGradientBoostingRegressor(max_iter=20)).fit(
        df[["year", "heat_days_32"]], df["y"])
    assert model.estimator_.n_features_in_ == 1
