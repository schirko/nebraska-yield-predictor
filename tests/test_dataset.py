"""Offline tests for the modeling-table helpers."""

import pandas as pd

from yieldpred.dataset import WEATHER_FEATURES, feature_matrix


def make_table(n: int = 4) -> pd.DataFrame:
    base = {"fips": ["31109"] * n, "county_name": ["Lancaster"] * n,
            "asd_code": ["80"] * n, "asd_desc": ["Southeast"] * n,
            "year": list(range(2000, 2000 + n)), "yield_bu_acre": [150.0] * n}
    base.update({feat: [1.0] * n for feat in WEATHER_FEATURES})
    return pd.DataFrame(base)


def test_feature_matrix_columns_and_groups():
    X, y, groups, used = feature_matrix(make_table())
    assert list(X.columns) == ["year"] + WEATHER_FEATURES
    assert len(X) == len(y) == len(groups) == len(used)
    assert groups.unique().tolist() == ["80"]


def test_rows_with_missing_values_are_dropped():
    df = make_table()
    df.loc[0, "gdd"] = None
    df.loc[1, "yield_bu_acre"] = None
    X, y, _, _ = feature_matrix(df)
    assert len(X) == 2


def test_extra_features_are_included():
    df = make_table().assign(irrigation_share=0.5)
    X, _, _, _ = feature_matrix(df, extra=["irrigation_share"])
    assert "irrigation_share" in X.columns
