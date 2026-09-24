"""Offline tests for the modeling-table helpers."""

import pandas as pd

from yieldpred.dataset import (SPRING_FEATURES, WEATHER_FEATURES, available_extras,
                               available_spring, feature_groups, feature_matrix)


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


def test_spring_features_are_not_part_of_the_weather_baseline():
    """The weather-only rung must keep meaning what it meant before spring existed."""
    assert not set(SPRING_FEATURES) & set(WEATHER_FEATURES)
    df = make_table().assign(**{feat: 1.0 for feat in SPRING_FEATURES})
    X, _, _, _ = feature_matrix(df)
    assert not set(SPRING_FEATURES) & set(X.columns)


def test_available_spring_finds_only_what_is_there():
    assert available_spring(make_table()) == []
    df = make_table().assign(workable_days=40, gdd_may=300.0)
    assert available_spring(df) == ["workable_days", "gdd_may"]


def test_feature_groups_bundles_spring_and_keeps_the_rest_singular():
    df = make_table().assign(irrigation_share=0.5, elevation_m=400.0,
                             **{feat: 1.0 for feat in SPRING_FEATURES})
    groups = feature_groups(df)

    labels = [label for label, _ in groups]
    assert labels[:2] == ["irrigation_share", "elevation_m"]
    assert labels[-1].startswith("spring weather")

    columns = [column for _, cols in groups for column in cols]
    assert columns == available_extras(df) + SPRING_FEATURES
    assert groups[-1][1] == SPRING_FEATURES


def test_feature_groups_is_empty_without_optional_columns():
    assert feature_groups(make_table()) == []
