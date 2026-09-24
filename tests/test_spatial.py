"""Offline tests for the spatial statistics, using a synthetic grid of squares."""

import geopandas as gpd
import numpy as np
import pandas as pd
import pytest
from shapely.geometry import box

from yieldpred.spatial import align_to_geometry, morans_i, queen_weights

SIDE = 6  # 6x6 grid = 36 "counties"


def grid() -> gpd.GeoDataFrame:
    cells = [{"fips": f"{row}{col}", "row": row, "col": col,
              "geometry": box(col, row, col + 1, row + 1)}
             for row in range(SIDE) for col in range(SIDE)]
    return gpd.GeoDataFrame(cells, crs="EPSG:4326")


def test_queen_weights_neighbour_counts():
    w = queen_weights(grid())
    counts = sorted({len(v) for v in w.neighbors.values()})
    # Corners touch 3 cells, edges 5, interior 8 (corner contact counts for queen)
    assert counts == [3, 5, 8]


def test_clustered_values_give_positive_morans_i():
    gdf = grid()
    values = gdf["row"].to_numpy(dtype=float)  # smooth north-south gradient
    result = morans_i(values, queen_weights(gdf))
    assert result.i > 0.5
    assert result.p_value < 0.05
    assert result.verdict == "clustered"


def test_random_values_give_no_pattern():
    gdf = grid()
    rng = np.random.default_rng(42)
    result = morans_i(rng.normal(size=len(gdf)), queen_weights(gdf))
    assert abs(result.i) < 0.3
    assert result.verdict == "no significant spatial pattern"


def test_alternating_stripes_are_dispersed():
    # Alternating columns: every side neighbour differs, so I goes strongly negative.
    # (A true checkerboard is NOT dispersed under queen weights - its diagonal
    # neighbours match, which cancels most of the effect. A nice reminder that the
    # neighbour definition is part of the hypothesis.)
    gdf = grid()
    values = (gdf["col"] % 2).to_numpy(dtype=float)
    result = morans_i(values, queen_weights(gdf))
    assert result.i < -0.3
    assert result.verdict == "dispersed"


def test_align_to_geometry_keeps_order_and_intersection():
    gdf = grid()
    values = gdf[["fips"]].head(10).assign(error=range(10))
    merged, array = align_to_geometry(gdf, values)
    assert len(merged) == 10
    assert array.tolist() == list(range(10))
    assert merged["fips"].tolist() == values["fips"].tolist()


def test_moran_result_string_is_readable():
    gdf = grid()
    result = morans_i(gdf["row"].to_numpy(dtype=float), queen_weights(gdf))
    text = str(result)
    assert "Moran's I" in text and "p =" in text


@pytest.mark.parametrize("permutations", [99, 499])
def test_permutations_do_not_change_the_statistic(permutations):
    gdf = grid()
    values = gdf["col"].to_numpy(dtype=float)
    result = morans_i(values, queen_weights(gdf), permutations=permutations)
    assert result.i == pytest.approx(morans_i(values, queen_weights(gdf)).i)


def test_coverage_simplification_preserves_neighbours():
    """Simplifying a tiling must not pull the tiles apart.

    Shared boundaries are simplified once and applied to both sides, so the
    neighbour structure survives and counties don't become islands.
    """
    import geopandas as gpd
    from yieldpred.geo import neighbor_report, simplify_coverage

    wobbly = gpd.GeoDataFrame(
        [{"fips": f"{r}{c}", "geometry": box(c, r, c + 1, r + 1).segmentize(0.25)}
         for r in range(5) for c in range(5)], crs="EPSG:5070")

    before = neighbor_report(wobbly)
    simplified = wobbly.copy()
    simplified["geometry"] = simplify_coverage(wobbly.geometry, 0.3)
    after = neighbor_report(simplified)

    assert after["islands"] == 0
    assert after["mean_neighbors"] == before["mean_neighbors"]
    assert (simplified.geometry.count_coordinates().sum()
            <= wobbly.geometry.count_coordinates().sum())


# ----------------------------------------------- joins: statistics vs maps

def three_counties():
    gdf = gpd.GeoDataFrame({"fips": ["31001", "31003", "31075"]},
                           geometry=[box(i, 0, i + 1, 1) for i in range(3)],
                           crs="EPSG:4326")
    values = pd.DataFrame({"fips": ["31001", "31003"], "error": [1.0, -1.0]})
    return gdf, values


def test_inner_join_drops_counties_without_values():
    """Moran's I needs a value for every unit in the weights graph."""
    merged, array = align_to_geometry(*three_counties(), value_col="error")
    assert len(merged) == 2 and len(array) == 2
    assert "31075" not in set(merged["fips"])


def test_left_join_keeps_every_polygon_for_maps():
    """A county dropped from a map leaves nothing, not a readable gap.

    Grant and Hooker really did vanish from every figure in this project until
    a rendered map was looked at closely.
    """
    merged, array = align_to_geometry(*three_counties(), value_col="error",
                                      how="left")
    assert len(merged) == 3
    assert merged.loc[merged["fips"] == "31075", "error"].isna().all()
    assert np.isnan(array).sum() == 1
