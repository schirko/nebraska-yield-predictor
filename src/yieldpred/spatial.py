"""Spatial statistics: is a map clustered, or is it noise?

Moran's I is the standard test. It compares each place's value to the average of
its neighbours:

* **I near +1** - neighbours resemble each other (clustering)
* **I near 0**  - no spatial pattern; the map could have been shuffled
* **I negative** - neighbours differ systematically (a checkerboard)

Applied to model *errors*, it is a diagnostic rather than a description. Errors
that cluster mean some spatially-varying driver is missing from the features.
Errors that look random mean the spatial signal has been used up - what's left is
noise, or something that doesn't vary smoothly across space.

"Neighbours" needs defining. Queen contiguity counts two counties as neighbours if
their boundaries touch at all, even at a corner - named after the chess piece,
like rook contiguity (shared edges only).
"""

from __future__ import annotations

from dataclasses import dataclass

import geopandas as gpd
import numpy as np


@dataclass
class MoranResult:
    """Moran's I with its significance test."""
    i: float          # the statistic
    expected: float   # what I would be under no spatial pattern (approx -1/(n-1))
    p_value: float    # from a permutation test
    z_score: float
    n: int

    @property
    def verdict(self) -> str:
        if self.p_value > 0.05:
            return "no significant spatial pattern"
        return "clustered" if self.i > self.expected else "dispersed"

    def __str__(self) -> str:
        return (f"Moran's I = {self.i:+.3f} (expected {self.expected:+.3f}), "
                f"p = {self.p_value:.4f} -> {self.verdict}")


def queen_weights(gdf: gpd.GeoDataFrame, silence_warnings: bool = True):
    """Row-standardized queen-contiguity weights for a set of polygons.

    Row standardization means each county's neighbour weights sum to 1, so a county
    with eight neighbours doesn't count for more than one with three.

    Islands (counties with no neighbours) are legitimate when analysing a single
    year, because counties that didn't report are absent and can leave a survivor
    surrounded by gaps. They are NOT legitimate for the full county set - there, an
    island means the geometry is broken. libpysal's per-call warnings are silenced
    here; use `geo.neighbor_report` to check the full layer deliberately instead.
    """
    from libpysal.weights import Queen

    w = Queen.from_dataframe(gdf, use_index=False, silence_warnings=silence_warnings)
    w.transform = "r"
    return w


def island_count(w) -> int:
    """How many units have no neighbours under these weights."""
    return sum(1 for neighbors in w.neighbors.values() if not neighbors)


def morans_i(values, w, permutations: int = 999, seed: int = 0) -> MoranResult:
    """Moran's I with a permutation test for significance.

    The permutation test shuffles the values across counties many times and asks how
    often a shuffled map is as clustered as the real one. That avoids assuming any
    particular distribution - useful, since model errors are rarely well behaved.
    """
    from esda.moran import Moran

    np.random.seed(seed)
    moran = Moran(np.asarray(values, dtype=float), w, permutations=permutations)
    return MoranResult(i=float(moran.I), expected=float(moran.EI),
                       p_value=float(moran.p_sim), z_score=float(moran.z_sim),
                       n=int(len(values)))


def align_to_geometry(gdf: gpd.GeoDataFrame, values: "object",
                      key: str = "fips", value_col: str = "error",
                      how: str = "inner"):
    """Join a values table to geometry, keeping the row order of the two in step.

    Spatial weights and value vectors must be in the same order; building them from
    a single joined frame is the safe way to guarantee that.

    `how` exists because statistics and maps want different things, and the
    difference is easy to get wrong:

    * **"inner" (default) for statistics.** Moran's I needs a value for every unit
      in the weights matrix - a contiguity-weighted average of its neighbours
      cannot be computed where a neighbour is missing. Counties without data must
      be dropped from the graph, not carried as NaN.
    * **"left" for maps.** Dropping a county from a map doesn't leave a gap the
      viewer can interpret, it leaves *nothing* - the polygon simply isn't drawn
      and the page shows through. Nebraska's Grant and Hooker counties are in the
      Sandhills, grow almost no corn, and are never reported by NASS; with an
      inner join they silently vanished from every figure in this project until
      somebody looked closely at a rendered map. A left join keeps them and lets
      the renderer mark them as "no data", which is a statement rather than a hole.
    """
    merged = gdf.merge(values, on=key, how=how)
    return merged, merged[value_col].to_numpy()
