"""Is a single weather point per county good enough? And what does sharing one cost?

Run before building cropland-weighted weather, to find out whether the thing it
would fix is actually broken:

    python scripts/probe_weather_grid.py

`weather.py` queries NASA POWER once per county, at the county's internal point
from the Census Gazetteer. POWER serves the MERRA-2 grid - 0.5 degrees latitude
by 0.625 degrees longitude, roughly 55 km - so a query returns the value of the
cell containing that point, however much of the county lies outside it. A typical
Nebraska county is 39 km across. Two consequences follow, and this script measures
both from data already on disk. No downloads, no API keys.

**1. Counties can share a cell.** If two counties' internal points land in the same
cell, `weather.py` fetched the same numbers twice. Their weather rows are then
byte-identical, which is how this script recovers the true grid membership without
knowing POWER's exact grid alignment: hash each county's whole weather record and
group by the hash. That is more reliable than reconstructing the grid arithmetic,
because the Gazetteer point is not the county centroid and near a cell boundary the
difference decides which cell you get.

Sharing matters because of *how the model is validated*. Leave-district-out holds
out whole NASS agricultural districts so the model must predict counties it has
never seen. But if a held-out county shares a POWER cell with a training county in
a different district, its exact weather vector is in the training data. The county
is unseen; its weather is not.

**2. A single point may describe the county badly.** Three ways of asking whether
that hurts, in ascending strength:

* `sampled coverage` - what share of the county lies inside the one cell queried.
* `cells spanned`    - how many cells cover at least 1% of the county.
* `local gradient`   - mean |county - neighbour| weather difference, in SD units,
                       over queen-contiguity neighbours. This is the one that
                       matters: coverage only bites where neighbouring cells
                       genuinely differ, and this measures that directly.

Each is correlated against the county's mean absolute prediction error. If point
sampling is driving the clustered residuals, badly-sampled counties should be
predicted worse.

WHAT THIS FOUND (2026-09, both states)

Sharing is real and asymmetric. Nebraska: 88 modelled counties served by 66 distinct
cells, but only 2% share a cell across a district boundary. Iowa: 98 counties, 61
cells, and **44% share their exact weather vector with a county in another district**.
Iowa's districts are small and its cells straddle them constantly. Worse, the shared
cells chain all nine Iowa districts into a single connected component, so there is no
way to group counties that both respects districts and never splits a cell. Iowa's
0.781 - the best spatial score in the project - is measured across that leak.

The sampling quality tests are null. Coverage vs MAE: r = -0.02 (NE), +0.05 (IA).
Local gradient vs MAE: r = -0.04 (NE), +0.04 (IA). Counties with under half their
area in the sampled cell are predicted no worse than counties wholly inside one, and
in both states the sharpest-gradient quartile is predicted slightly *better*. Three
tests, two states, nothing.

The honest reading of that null is not "sampling doesn't matter" but "at 55 km there
is nothing to weight". Cropland weighting moves a county's query point between cells
that are mostly the same cell. The limitation is the resolution of the weather
product, not the choice of point within it - which is an argument for a finer product
(gridMET at 4 km, Daymet at 1 km) rather than for better weighting of this one.

Two caveats on the null, both worth stating:

* The errors come from a model that already carries per-county static features
  (soil, elevation, irrigation share). A county whose weather series is consistently
  wrong can have that offset partly absorbed by those columns, in exactly the
  county-fingerprint way Iowa's NCCPI demonstrated. The model may be hiding the
  problem rather than not having one - and it would resurface on a genuinely new
  county.
* `sampled coverage` and `cells spanned` use a reconstructed grid and an approximate
  internal point, so a handful of counties near cell boundaries are misassigned. The
  `local gradient` test uses neither and agrees with them.
"""

from __future__ import annotations

import sys
from collections import defaultdict
from pathlib import Path

import geopandas as gpd
import numpy as np
import pandas as pd
from scipy import stats

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from yieldpred.spatial import queen_weights  # noqa: E402

PROCESSED = Path(__file__).resolve().parents[1] / "data" / "processed"

# POWER's MERRA-2 grid. Used only for the coverage measures, which are approximate
# by construction; the leak analysis recovers membership from the data instead.
DLAT, DLON = 0.5, 0.625

# Variables whose joint series identifies a cell. Any subset works - identical is
# identical - but three independent ones make an accidental hash collision absurd.
FINGERPRINT = ["gdd", "precip_mm", "tmax_jul_c"]

# Variables the gradient test runs on, chosen to cover both moisture and heat.
GRADIENT_VARS = ["precip_mm", "heat_days_32", "tmax_jul_c", "gdd"]

STATES = [
    ("NE", "counties_31", "weather_county_31", "model_errors"),
    ("IA", "counties_19", "weather_county_19", "model_errors_ia"),
]


def load(name: str):
    path = PROCESSED / f"{name}.parquet"
    if not path.exists():
        return None
    return gpd.read_parquet(path) if name.startswith("counties") else pd.read_parquet(path)


def cell_membership(weather: pd.DataFrame) -> pd.DataFrame:
    """Recover which counties POWER served from the same grid cell.

    Identical weather across 26 years is not a coincidence - it means one cell was
    queried twice. Hashing the full record is exact and needs no grid arithmetic.
    """
    keys = (weather.sort_values("year")
                   .groupby("fips")
                   .apply(lambda d: hash(tuple(np.round(d[FINGERPRINT].to_numpy().ravel(), 6))),
                          include_groups=False)
                   .rename("cell").reset_index())
    return keys


def connected_groups(*groupings: dict) -> dict:
    """Union-find: merge counties linked by ANY of the groupings given."""
    parent: dict = {}

    def find(x):
        parent.setdefault(x, x)
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    for grouping in groupings:
        for members in grouping.values():
            for m in members[1:]:
                a, b = find(members[0]), find(m)
                if a != b:
                    parent[a] = b
    return {k: find(k) for k in parent}


def coverage(counties: gpd.GeoDataFrame) -> pd.DataFrame:
    """For each county: how many POWER cells it spans, and how much of it is sampled."""
    from shapely.geometry import box

    g = counties.to_crs(4326)
    pts = g.geometry.representative_point()
    ea = g.to_crs(5070)                      # Conus Albers: equal area, so ratios are honest
    rows = []
    for (_, c), (_, e), pt in zip(g.iterrows(), ea.iterrows(), pts):
        minx, miny, maxx, maxy = c.geometry.bounds
        i0, j0 = int((miny + 90) / DLAT), int((minx + 180) / DLON)
        i1, j1 = int((maxy + 90) / DLAT), int((maxx + 180) / DLON)
        total, parts = e.geometry.area, {}
        for i in range(i0 - 1, i1 + 2):
            for j in range(j0 - 1, j1 + 2):
                lat, lon = -90 + i * DLAT, -180 + j * DLON
                cb = gpd.GeoSeries(
                    [box(lon - DLON / 2, lat - DLAT / 2, lon + DLON / 2, lat + DLAT / 2)],
                    crs=4326).to_crs(5070).iloc[0]
                inter = e.geometry.intersection(cb)
                if not inter.is_empty and inter.area / total > 1e-6:
                    parts[(i, j)] = inter.area / total
        si = int(round((pt.y + 90) / DLAT))
        sj = int(round((pt.x + 180) / DLON))
        rows.append(dict(fips=c.fips, area_km2=c.area_km2,
                         n_cells=sum(1 for f in parts.values() if f >= 0.01),
                         sampled_frac=parts.get((si, sj), 0.0)))
    return pd.DataFrame(rows)


def local_gradient(counties: gpd.GeoDataFrame, weather: pd.DataFrame) -> pd.DataFrame:
    """Mean |county - neighbour| weather difference, in SD units.

    A county in a place where weather changes sharply over short distances is one
    that a single point describes badly. This needs no grid reconstruction at all.
    """
    w = queen_weights(counties)
    fips = counties.fips.tolist()
    out = {}
    for var in GRADIENT_VARS:
        piv = weather.pivot_table(index="year", columns="fips", values=var).reindex(columns=fips)
        sd = np.nanstd(piv.to_numpy())
        col = {}
        for i, f in enumerate(fips):
            nb = [fips[k] for k in w.neighbors.get(i, [])]
            if nb:
                col[f] = np.nanmean(np.abs(piv[nb].sub(piv[f], axis=0).to_numpy())) / sd
        out[var] = pd.Series(col)
    df = pd.DataFrame(out)
    df["gradient"] = df.mean(axis=1)
    df.index.name = "fips"
    return df.reset_index()


def report(state, counties_name, weather_name, errors_name) -> None:
    counties, weather, errors = (load(counties_name), load(weather_name), load(errors_name))
    if counties is None or weather is None:
        print(f"\n{state}: missing data - run the fetch scripts first.")
        return

    print(f"\n{'=' * 66}\n{state}\n{'=' * 66}")

    cells = cell_membership(weather)
    print(f"{len(cells)} counties served by {cells.cell.nunique()} distinct POWER cells")
    shared = cells.groupby("cell").filter(lambda d: len(d) > 1)
    print(f"{len(shared)} counties share a cell with at least one other county")

    if errors is None:
        print("no model errors on disk - skipping the leak and sampling tests")
        return

    districts = errors[["fips", "asd_desc"]].drop_duplicates()
    m = cells.merge(districts, on="fips")     # modelled counties only

    # ---- leak: does a held-out county's weather appear in another district? -----
    spread = m.groupby("cell")["asd_desc"].agg(["nunique", "size"])
    crossing = spread[(spread["size"] > 1) & (spread["nunique"] > 1)]
    leaky = m[m.cell.isin(crossing.index)]
    print(f"\nLEAVE-DISTRICT-OUT LEAK")
    print(f"  cells spanning more than one district: {len(crossing)} of {m.cell.nunique()}")
    print(f"  counties whose exact weather vector sits in another district: "
          f"{len(leaky)} of {len(m)} ({100 * len(leaky) / len(m):.0f}%)")

    by_district, by_cell = defaultdict(list), defaultdict(list)
    for _, r in m.iterrows():
        by_district[r.asd_desc].append(r.fips)
        by_cell[r.cell].append(r.fips)
    comp = connected_groups(by_district, by_cell)
    n_groups = len(set(comp.values()))
    print(f"  grouping that respects districts AND never splits a cell: "
          f"{n_groups} group(s)")
    if n_groups < 2:
        print("  -> impossible: the shared cells chain every district together.")
        print("     The fix is to drop leaking counties from TRAINING per fold,")
        print("     holding the test set fixed, so the two scores stay comparable.")

    # ---- is a single point good enough? ----------------------------------------
    mae = (errors.assign(abs_err=errors.error.abs())
                 .groupby("fips").agg(mae=("abs_err", "mean")).reset_index())
    d = (mae.merge(coverage(counties), on="fips")
            .merge(local_gradient(counties, weather), on="fips"))

    print(f"\nIS ONE POINT PER COUNTY GOOD ENOUGH?  (n={len(d)})")
    for col, label in [("sampled_frac", "sampled coverage"),
                       ("n_cells", "cells spanned   "),
                       ("gradient", "local gradient  ")]:
        r, p = stats.pearsonr(d[col], d.mae)
        flag = "" if p < 0.05 else "   (not significant)"
        print(f"  MAE vs {label}  r = {r:+.3f}  p = {p:.3f}{flag}")

    lo, hi = d[d.sampled_frac < 0.5], d[d.sampled_frac >= 0.5]
    if len(lo) and len(hi):
        _, p = stats.ttest_ind(lo.mae, hi.mae, equal_var=False)
        print(f"  MAE under 50% coverage (n={len(lo):2d}): {lo.mae.mean():5.2f} bu/acre")
        print(f"  MAE over  50% coverage (n={len(hi):2d}): {hi.mae.mean():5.2f} bu/acre")
        print(f"  difference {lo.mae.mean() - hi.mae.mean():+.2f}  (Welch p = {p:.3f})")


def main() -> None:
    for args in STATES:
        report(*args)
    print("\nSee this file's docstring for what these numbers meant in September 2026.")


if __name__ == "__main__":
    main()
