"""Download county soil productivity (NCCPI) and elevation.

    python scripts/fetch_soil_terrain.py
    python scripts/fetch_soil_terrain.py --skip-elevation

Both are static county characteristics - they don't change year to year - so they are
saved once and joined to every year of the modeling table.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

from yieldpred.soils import SoilDataError, fetch_soil_rating, probe, to_counties
from yieldpred.terrain import county_elevations
from yieldpred.weather import county_points

ROOT = Path(__file__).resolve().parents[1]
PROCESSED = ROOT / "data" / "processed"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--state", default="NE")
    parser.add_argument("--state-fips", default="31")
    parser.add_argument("--skip-soils", action="store_true")
    parser.add_argument("--skip-elevation", action="store_true")
    parser.add_argument("--probe-soils", action="store_true",
                        help="Diagnose which part of the soil query SDA rejects, then exit")
    args = parser.parse_args()

    if args.probe_soils:
        print("Probing Soil Data Access, simplest query first:")
        probe()
        return

    PROCESSED.mkdir(parents=True, exist_ok=True)
    yields_path = PROCESSED / f"{args.state.lower()}_corn_yield_county.parquet"
    if not yields_path.exists():
        raise SystemExit("Run scripts/fetch_nass_yields.py first — county names come "
                         "from the yield data.")
    lookup = (pd.read_parquet(yields_path)[["fips", "county_name"]]
              .drop_duplicates("fips"))

    frames = []

    # ------------------------------------------------------------------ soils
    if not args.skip_soils:
        print("Querying USDA Soil Data Access for soil productivity...")
        try:
            areas, value_col = fetch_soil_rating(args.state)
            print(f"  {len(areas)} soil survey areas, measure: {value_col}")
            soils = to_counties(areas, lookup, value_col)
            print(f"  mapped to {len(soils)} counties "
                  f"({value_col} {soils[value_col].min():.2f}-{soils[value_col].max():.2f}, "
                  f"mean {soils[value_col].mean():.2f})")
            frames.append(soils[["fips", value_col]])

            print("\n  Best soils:")
            print(soils.nlargest(5, value_col)[["county_name", value_col]]
                  .to_string(index=False))
            print("\n  Poorest soils:")
            print(soils.nsmallest(5, value_col)[["county_name", value_col]]
                  .to_string(index=False))

            missing = set(lookup["fips"]) - set(soils["fips"])
            if missing:
                print(f"\n  No soil rating for {len(missing)} counties — they will fall "
                      "out of any model that uses this feature.")
        except SoilDataError as exc:
            print(f"  FAILED: {exc}")
            print("  Run with --probe-soils to see which part of the query is rejected.")
            print("  Continuing without soils; re-run later to add them.")

    # -------------------------------------------------------------- elevation
    if not args.skip_elevation:
        print("\nLooking up county elevations (USGS)...")
        points = county_points(args.state_fips)
        elevations = county_elevations(points)
        if not elevations.empty:
            print(f"  {len(elevations)} counties, "
                  f"{elevations['elevation_m'].min():.0f}–"
                  f"{elevations['elevation_m'].max():.0f} m")
            print("\n  Highest:")
            print(elevations.nlargest(5, "elevation_m")[["county_name", "elevation_m"]]
                  .to_string(index=False))
            print("\n  Lowest:")
            print(elevations.nsmallest(5, "elevation_m")[["county_name", "elevation_m"]]
                  .to_string(index=False))
            frames.append(elevations[["fips", "elevation_m"]])

    if not frames:
        raise SystemExit("\nNothing downloaded.")

    # ----------------------------------------------------------------- merge
    out = frames[0]
    for frame in frames[1:]:
        out = out.merge(frame, on="fips", how="outer")

    path = PROCESSED / f"county_static_{args.state.lower()}.parquet"
    if path.exists():
        # Keep any columns a previous run produced (e.g. soils when this run skipped them)
        previous = pd.read_parquet(path)
        keep = [c for c in previous.columns if c == "fips" or c not in out.columns]
        out = previous[keep].merge(out, on="fips", how="outer")

    out.to_parquet(path, index=False)
    print(f"\nSaved {len(out)} counties to {path.relative_to(ROOT)}")
    print(f"Columns: {', '.join(c for c in out.columns if c != 'fips')}")


if __name__ == "__main__":
    main()
