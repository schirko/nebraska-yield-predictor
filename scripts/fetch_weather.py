"""Download daily weather for every Nebraska county and build growing-season features.

One NASA POWER request per county (about 90 of them), cached in data/raw/power/
so re-runs are fast and an interrupted run can be resumed.

    python scripts/fetch_weather.py
    python scripts/fetch_weather.py --state-fips 19 --start 2000   # Iowa
"""

from __future__ import annotations

import argparse
import time
from pathlib import Path

import pandas as pd
import requests

from yieldpred.weather import county_points, fetch_power_daily, growing_season_features

ROOT = Path(__file__).resolve().parents[1]
RAW = ROOT / "data" / "raw" / "power"
PROCESSED = ROOT / "data" / "processed"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--state-fips", default="31", help="State FIPS (31 = Nebraska)")
    parser.add_argument("--start", type=int, default=2000)
    parser.add_argument("--end", type=int, default=2025)
    parser.add_argument("--pause", type=float, default=1.0,
                        help="Seconds between requests (be polite to NASA)")
    args = parser.parse_args()

    RAW.mkdir(parents=True, exist_ok=True)
    PROCESSED.mkdir(parents=True, exist_ok=True)

    print("Looking up county centre points...")
    points = county_points(args.state_fips)
    print(f"{len(points)} counties\n")

    frames, failures = [], []
    for i, row in points.iterrows():
        cache = RAW / f"{row.fips}_{args.start}_{args.end}.csv"

        if cache.exists():
            daily = pd.read_csv(cache, parse_dates=["date"])
            status = "cached"
        else:
            try:
                daily = fetch_power_daily(row.lat, row.lon, args.start, args.end)
            except (requests.RequestException, KeyError, ValueError) as exc:
                print(f"[{i + 1:>2}/{len(points)}] {row.county_name:<18} FAILED ({type(exc).__name__})")
                failures.append(row.fips)
                continue
            daily.to_csv(cache, index=False)
            time.sleep(args.pause)
            status = "downloaded"

        feats = growing_season_features(daily)
        feats.insert(0, "fips", row.fips)
        feats.insert(1, "county_name", row.county_name)
        frames.append(feats)
        print(f"[{i + 1:>2}/{len(points)}] {row.county_name:<18} {status}: "
              f"{feats['year'].min()}-{feats['year'].max()}")

    if not frames:
        raise SystemExit("\nNo weather downloaded - check your internet connection.")

    weather = pd.concat(frames, ignore_index=True)
    out = PROCESSED / f"weather_county_{args.state_fips}.parquet"
    weather.to_parquet(out, index=False)

    print(f"\nSaved {len(weather):,} county-years to {out.relative_to(ROOT)}")
    if failures:
        print(f"Failed for {len(failures)} counties: {failures}")
        print("Re-run the script to retry them (finished counties are cached).")

    print("\nStatewide averages by year (last 8):")
    summary = (weather.groupby("year")[["gdd", "precip_mm", "precip_jul_mm",
                                        "heat_days_32", "dry_spell_max"]]
               .mean().round(1).tail(8))
    print(summary.to_string())


if __name__ == "__main__":
    main()
