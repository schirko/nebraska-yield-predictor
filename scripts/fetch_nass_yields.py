"""Download county corn yields from NASS Quick Stats and save them locally.

Run from the project root (with the virtual environment active):

    python scripts/fetch_nass_yields.py
    python scripts/fetch_nass_yields.py --state IA --start 2005
"""

from __future__ import annotations

import argparse
from pathlib import Path

from yieldpred.nass import NASSError, fetch_county_corn_yields

ROOT = Path(__file__).resolve().parents[1]
PROCESSED = ROOT / "data" / "processed"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--state", default="NE", help="Two-letter state code (default NE)")
    parser.add_argument("--start", type=int, default=2000, help="First year (default 2000)")
    parser.add_argument("--end", type=int, default=None, help="Last year (default: latest)")
    args = parser.parse_args()

    print(f"Requesting {args.state} county corn yields from {args.start}...")
    try:
        df = fetch_county_corn_yields(args.state, args.start, args.end)
    except NASSError as exc:
        raise SystemExit(f"Error: {exc}")

    if df.empty:
        raise SystemExit("No rows returned - check the state code and years.")

    PROCESSED.mkdir(parents=True, exist_ok=True)
    stem = f"{args.state.lower()}_corn_yield_county"
    df.to_parquet(PROCESSED / f"{stem}.parquet", index=False)
    df.to_csv(PROCESSED / f"{stem}.csv", index=False)

    # Quick summary so you can sanity-check the download.
    print(f"\nSaved {len(df):,} rows to data/processed/{stem}.parquet (+ .csv)")
    print(f"Counties: {df['fips'].nunique()}   Years: {df['year'].min()}-{df['year'].max()}")
    print("\nRows per practice:")
    print(df["practice"].value_counts().to_string())
    print("\nState-wide mean yield (bu/acre), last 5 years:")
    recent = df[df["year"] >= df["year"].max() - 4]
    print(recent.pivot_table(index="year", columns="practice",
                             values="yield_bu_acre", aggfunc="mean").round(1).to_string())


if __name__ == "__main__":
    main()
