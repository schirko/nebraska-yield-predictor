"""Separate the long-run yield trend from weather-driven variation.

Corn yields rise roughly 2 bu/acre per year from better genetics and management.
Tree-based models cannot extrapolate: they predict by averaging training rows, so
a model trained through 2018 treats 2024 as if it were 2018 and misses six years
of gains. Linear models extrapolate fine but can't capture interactions between
weather variables.

DetrendedRegressor gets both. A linear trend in `year` is fitted first, the base
model learns only the residual - how much better or worse than trend a county-year
turned out, given its weather - and predictions add the trend back.

This mirrors standard practice in the crop-yield forecasting literature, where
models are usually fitted to yield anomalies rather than raw yields.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, RegressorMixin, clone
from sklearn.linear_model import LinearRegression


class DetrendedRegressor(BaseEstimator, RegressorMixin):
    """Fit `estimator` to yield anomalies around a linear time trend.

    Parameters
    ----------
    estimator : any scikit-learn regressor, fitted to the residuals.
    year_col : name of the column holding the year.
    drop_year : if True (default) the year column is hidden from the base
        estimator, so it models weather effects only.
    """

    def __init__(self, estimator, year_col: str = "year", drop_year: bool = True):
        self.estimator = estimator
        self.year_col = year_col
        self.drop_year = drop_year

    def _years(self, X: pd.DataFrame) -> np.ndarray:
        return np.asarray(X[self.year_col], dtype=float).reshape(-1, 1)

    def _features(self, X: pd.DataFrame) -> pd.DataFrame:
        return X.drop(columns=[self.year_col]) if self.drop_year else X

    def fit(self, X: pd.DataFrame, y):
        y = np.asarray(y, dtype=float)
        self.trend_ = LinearRegression().fit(self._years(X), y)
        residuals = y - self.trend_.predict(self._years(X))
        self.estimator_ = clone(self.estimator).fit(self._features(X), residuals)
        return self

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        return self.trend_.predict(self._years(X)) + self.estimator_.predict(self._features(X))

    @property
    def trend_slope_(self) -> float:
        """Fitted trend in bushels per acre per year."""
        return float(self.trend_.coef_[0])
