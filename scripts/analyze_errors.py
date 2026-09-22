"""Map the model's errors and test whether they cluster geographically.

    python scripts/analyze_errors.py

Downloads county boundaries on first run (cached afterwards), then:
  1. Moran's I on the average error per county - do the misses cluster?
  2. Moran's I year by year - is clustering a permanent flaw or a bad-year effect?
  3. Choropleth maps written to figures/
"""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

from yieldpred.geo import (county_boundaries, load_boundaries, neighbor_report,
                           save_boundaries)
from yieldpred.spatial import (align_to_geometry, island_count, morans_i,
                               queen_weights)
from yieldpred.viz import choropleth

ROOT = Path(__file__).resolve().parents[1]
PROCESSED = ROOT / "data" / "processed"
FIGURES = ROOT / "figures"

SOURCE = "Data: USDA NASS Quick Stats, NASA POWER, US Census Bureau"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--rebuild-boundaries", action="store_true",
                        help="Re-download and rebuild the county geometry")
    args = parser.parse_args()

    errors = pd.read_parquet(PROCESSED / "model_errors.parquet")

    counties = None if args.rebuild_boundaries else load_boundaries()
    if counties is None:
        print("Building county boundaries...")
        counties = county_boundaries()
        print(f"Saved to {save_boundaries(counties).relative_to(ROOT)}")
    print(f"{len(counties)} county polygons, {len(errors):,} county-year predictions")

    # Geometry check: Nebraska counties tile the state, so every county must have
    # neighbours. Islands mean the polygons have been damaged (see geo.simplify_coverage).
    report = neighbor_report(counties)
    print(f"Geometry check: {report['mean_neighbors']} neighbours on average, "
          f"minimum {report['min_neighbors']}, {report['islands']} islands")
    if report["islands"]:
        print("  WARNING: islands in a county tiling mean broken geometry - "
              "re-run with --rebuild-boundaries")
    print()

    # ------------------------------------------------------------------ pooled
    county_mean = (errors.groupby(["fips", "county_name"], as_index=False)
                   .agg(error=("error", "mean"),
                        actual=("yield_bu_acre", "mean"),
                        n_years=("year", "nunique")))

    merged, values = align_to_geometry(counties, county_mean, value_col="error")
    weights = queen_weights(merged)
    result = morans_i(values, weights)

    print("=" * 70)
    print("DO THE ERRORS CLUSTER? (average error per county)")
    print("=" * 70)
    print(result)
    print(f"Counties: {result.n}   Average neighbours: "
          f"{sum(len(v) for v in weights.neighbors.values()) / result.n:.1f}"
          f"   Islands: {island_count(weights)}")

    # -------------------------------------------------------------- by year
    print("\n" + "=" * 70)
    print("BY YEAR (clustered years point at a missing driver that year)")
    print("=" * 70)
    rows = []
    for year, group in errors.groupby("year"):
        year_merged, year_values = align_to_geometry(
            counties, group[["fips", "error"]], value_col="error")
        if len(year_merged) < 30:
            continue
        year_weights = queen_weights(year_merged)
        year_result = morans_i(year_values, year_weights, permutations=499)
        rows.append({"year": year, "counties": year_result.n,
                     "islands": island_count(year_weights),
                     "morans_i": round(year_result.i, 3),
                     "p_value": round(year_result.p_value, 4),
                     "mean_error": round(group["error"].mean(), 1),
                     "verdict": year_result.verdict})
    by_year = pd.DataFrame(rows).set_index("year")
    print(by_year.to_string())

    clustered = by_year[by_year["p_value"] < 0.05]
    print(f"\nSignificantly clustered in {len(clustered)} of {len(by_year)} years")

    # ------------------------------------------------------------------ maps
    print("\n" + "=" * 70)
    print("MAPS")
    print("=" * 70)

    choropleth(merged, "error",
               "Where the model misses",
               "Average prediction error by county, 2000-2025. "
               "Blue = model predicts too low, orange = too high.",
               SOURCE, diverging=True, legend_label="bu/acre (predicted - actual)",
               out_path=FIGURES / "error_map.png")

    choropleth(merged, "actual",
               "Average corn yield by county",
               "Nebraska, 2000-2025", SOURCE,
               legend_label="bu/acre", out_path=FIGURES / "yield_map.png")

    for year in (2012, 2019):
        year_rows = errors[errors["year"] == year]
        if year_rows.empty:
            continue
        year_merged, _ = align_to_geometry(counties, year_rows[["fips", "error"]],
                                           value_col="error")
        choropleth(year_merged, "error",
                   f"Prediction error in {year}",
                   {2012: "The drought year", 2019: "The flood year"}.get(year, ""),
                   SOURCE, diverging=True, legend_label="bu/acre (predicted - actual)",
                   out_path=FIGURES / f"error_map_{year}.png")

    share_path = PROCESSED / "irrigation_share_ne.parquet"
    if share_path.exists():
        share = pd.read_parquet(share_path)
        latest = share[share["year"] == share["year"].max()][["fips", "irrigation_share"]]
        share_merged, _ = align_to_geometry(counties, latest, value_col="irrigation_share")
        choropleth(share_merged, "irrigation_share",
                   "Share of corn acres irrigated",
                   f"{share['year'].max()}, interpolated from NASS survey and census data",
                   SOURCE, legend_label="share of harvested acres",
                   out_path=FIGURES / "irrigation_map.png")

    for path in sorted(FIGURES.glob("*.png")):
        print(f"  {path.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
