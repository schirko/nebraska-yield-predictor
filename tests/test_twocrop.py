"""The two-crop average: honest folds, and only where it earned its place."""

import numpy as np
import pandas as pd

from yieldpred import twocrop


def fake(n_districts=10, years=range(2000, 2012), seed=0, level=50.0):
    rng = np.random.default_rng(seed)
    rows = []
    for d in range(n_districts):
        for year in years:
            rain = rng.normal(400, 80)
            rows.append({"fips": f"{d:05d}", "asd_code": str(d), "year": year, "rain": rain,
                         "yield_bu_acre": level * (1 + 0.01 * (year - 2000)) * (1 + (rain - 400) / 2000)
                         + rng.normal(0, 1)})
    return pd.DataFrame(rows)


def test_only_nebraska_soybeans_use_both_crops():
    assert twocrop.uses_both_crops("NE", "soybeans")
    assert not twocrop.uses_both_crops("ia", "soybeans")
    assert not twocrop.uses_both_crops("ne", "corn")


def test_spatial_blend_is_honest_and_sensible():
    soy, corn = fake(), fake(seed=1, level=170.0)
    X, y, g = soy[["year", "rain"]], soy["yield_bu_acre"], soy["asd_code"]
    pred = twocrop.spatial_blend(X, y, g, corn)
    assert pred.shape == (len(soy),) and np.isfinite(pred).all()
    assert np.sqrt(np.mean((y - pred) ** 2)) < y.std()        # learned something

    # a held-out district's corn must not reach its prediction: poison it and nothing changes much
    poisoned = corn.copy()
    poisoned.loc[poisoned["asd_code"] == "0", "yield_bu_acre"] *= 5
    again = twocrop.spatial_blend(X, y, g, poisoned)
    district0 = (g == "0").to_numpy()
    assert np.allclose(pred[district0], again[district0])


def test_next_season_blend_uses_only_earlier_seasons():
    soy, corn = fake(), fake(seed=1, level=170.0)
    X, y = soy[["year", "rain"]], soy["yield_bu_acre"]
    future = corn.copy()
    future.loc[future["year"] >= 2011, "yield_bu_acre"] = 1e6   # corn from the season itself
    a = twocrop.next_season_blend(X, y, soy, corn, 2008)
    b = twocrop.next_season_blend(X, y, soy, future, 2008)
    assert set(a["year"]) == {2008, 2009, 2010, 2011}
    assert np.allclose(a["error"], b["error"])                  # later corn never used
