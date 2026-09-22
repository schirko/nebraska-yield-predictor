"""Build the county irrigation-share feature from NASS harvested-acre data.

    python scripts/fetch_irrigation.py

Pulls annual SURVEY acreage and five-yearly CENSUS OF AGRICULTURE acreage, computes
irrigated / total harvested acres per county-year, and interpolates across the gaps.
"""

from __future__ import annotations

import argparse
from pathlib import Path

from yieldpred.irrigation import (build_share_series, combine_sources,
                                  fetch_harvested_acres, observed_share)
from yieldpred.nass import NASSError

ROOT = Path(__file__).resolve().parents[1]
PROCESSED = ROOT / "data" / "processed"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--state", default="NE")
    parser.add_argument("--start", type=int, default=2000)
    parser.add_argument("--end", type=int, default=2025)
    args = parser.parse_args()

    frames = {}
    for source in ("SURVEY", "CENSUS"):
        try:
            acres = fetch_harvested_acres(source, args.state, args.start)
        except NASSError as exc:
            print(f"{source}: {exc}")
            continue
        share = observed_share(acres)
        frames[source] = share
        if share.empty:
            print(f"{source}: no usable rows")
        else:
            print(f"{source}: {len(share):,} county-years with both totals, "
                  f"{share['year'].min()}-{share['year'].max()}, "
                  f"{share['fips'].nunique()} counties")

    if not frames:
        raise SystemExit("No acreage data returned.")

    observed = combine_sources(frames.get("SURVEY", frames["CENSUS"]),
                               frames.get("CENSUS", frames["SURVEY"]))
    years = range(args.start, args.end + 1)
    series = build_share_series(observed, years)

    PROCESSED.mkdir(parents=True, exist_ok=True)
    out = PROCESSED / f"irrigation_share_{args.state.lower()}.parquet"
    series.to_parquet(out, index=False)
    # Keep the raw observations too, so the feature can be rebuilt using only
    # the years a given experiment is allowed to see (see train_baseline.py).
    observed.to_parquet(PROCESSED / f"irrigation_observed_{args.state.lower()}.parquet",
                        index=False)

    print(f"\nSaved {len(series):,} county-years to {out.relative_to(ROOT)}")
    print(f"Observed values: {int(series['share_observed'].sum()):,} "
          f"({series['share_observed'].mean():.0%}); the rest interpolated")

    latest = series[series["year"] == series["year"].max()].copy()
    # Attach county names if the yield data has been downloaded, so the sanity
    # check below is readable rather than a list of FIPS codes.
    yields_path = PROCESSED / f"{args.state.lower()}_corn_yield_county.parquet"
    if yields_path.exists():
        import pandas as pd
        names = (pd.read_parquet(yields_path)[["fips", "county_name"]]
                 .drop_duplicates("fips"))
        latest = latest.merge(names, on="fips", how="left")
    cols = [c for c in ["fips", "county_name", "irrigation_share"] if c in latest.columns]
    print(f"\nIrrigation share in {series['year'].max()}: "
          f"mean {latest['irrigation_share'].mean():.2f}, "
          f"median {latest['irrigation_share'].median():.2f}")
    print("\nMost irrigated counties (FIPS, share):")
    print(latest.nlargest(8, "irrigation_share")[cols].to_string(index=False))
    print("\nLeast irrigated counties (FIPS, share):")
    print(latest.nsmallest(8, "irrigation_share")[cols].to_string(index=False))


if __name__ == "__main__":
    main()
