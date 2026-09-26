"""Offline tests for the modeling-table helpers."""

import pandas as pd

from yieldpred.dataset import (SPRING_CALENDAR, SPRING_FEATURES, SPRING_RAIN,
                               WEATHER_FEATURES, available_extras, available_spring,
                               feature_groups, feature_matrix)


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


def spring_table() -> pd.DataFrame:
    return make_table().assign(irrigation_share=0.5, elevation_m=400.0,
                               **{feat: 1.0 for feat in SPRING_FEATURES})


def test_feature_groups_bundles_spring_and_keeps_the_rest_singular():
    groups = feature_groups(spring_table())

    labels = [label for label, _ in groups]
    assert labels[:2] == ["irrigation_share", "elevation_m"]
    assert labels[-1].startswith("spring weather")

    columns = [column for _, cols in groups for column in cols]
    assert columns == available_extras(spring_table()) + SPRING_FEATURES
    assert groups[-1][1] == SPRING_FEATURES


def test_feature_groups_is_empty_without_optional_columns():
    assert feature_groups(make_table()) == []


def test_split_spring_separates_rain_from_calendar():
    groups = feature_groups(spring_table(), split_spring=True)

    labels = [label for label, _ in groups]
    assert labels[-2].startswith("spring rain")
    assert labels[-1].startswith("spring calendar")
    assert groups[-2][1] == SPRING_RAIN
    assert groups[-1][1] == SPRING_CALENDAR


def test_splitting_does_not_change_which_columns_are_used():
    """The experiment must differ only in grouping, or its comparison means nothing."""
    df = spring_table()
    combined = [c for _, cols in feature_groups(df) for c in cols]
    split = [c for _, cols in feature_groups(df, split_spring=True) for c in cols]
    assert combined == split


def test_split_spring_skips_halves_that_are_absent():
    df = make_table().assign(**{feat: 1.0 for feat in SPRING_RAIN})
    labels = [label for label, _ in feature_groups(df, split_spring=True)]
    assert labels == ["spring rain (3)"]


def test_gridmet_filenames_follow_the_project_convention(tmp_path, monkeypatch):
    """Regression: the loader and the fetch script must spell one filename alike.

    `fetch_gridmet.py` writes via `output_path`, which gives Nebraska the
    unsuffixed name and every other state a suffix. An earlier `WEATHER_SOURCES`
    hardcoded "gridmet_point_{state}" and so looked for `gridmet_point_ne.parquet`
    while the file on disk was `gridmet_point.parquet`. Nothing raised - the fetch
    reported success, and the comparison reported "not available". A silent
    disagreement between two spellings of one name is worse than a crash.
    """
    from yieldpred import dataset

    for state, expected in [("ne", "gridmet_point.parquet"),
                            ("ia", "gridmet_point_ia.parquet"),
                            ("ks", "gridmet_point_ks.parquet")]:
        assert dataset.output_path("gridmet_point", state).name == expected

    # And the loader must ask for exactly that path, not build its own.
    monkeypatch.setattr(dataset, "PROCESSED", tmp_path)
    for state, stem in [("ne", "gridmet_mean"), ("ia", "gridmet_mean_ia")]:
        try:
            dataset.load_weather("31", "gridmet_mean", state)
        except FileNotFoundError as err:
            assert f"{stem}.parquet" in str(err), \
                f"loader looked for the wrong file for {state}: {err}"
        else:
            raise AssertionError("expected FileNotFoundError from an empty directory")


def test_power_still_keyed_by_fips_not_state():
    """POWER predates --state and is named by FIPS; don't 'tidy' it into the
    state convention or every existing weather file stops being found."""
    from yieldpred import dataset
    assert dataset.WEATHER_SOURCES["power"].format(fips="31") == "weather_county_31"
