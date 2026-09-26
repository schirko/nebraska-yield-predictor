"""Download Nebraska alfalfa hay from NASS and build the alfalfa tables.

Run from the project root (with the virtual environment active):

    python scripts/fetch_alfalfa.py
    python scripts/fetch_alfalfa.py --no-download     # rebuild the tables from files already saved

Step 1 (needs NASS_API_KEY in .env, like fetch_nass_yields.py):
    data/processed/ne_alfalfa_county.parquet (+ .csv)   county yield, acres, production
    data/processed/ne_alfalfa_state.parquet  (+ .csv)   the statewide series

Step 2 (no internet; reads the NASA POWER files the corn model already downloaded):
    data/processed/alfalfa_weather_ne.parquet           alfalfa weather per county and year
    data/processed/alfalfa_table_ne.parquet             yields joined to weather, ready to model
"""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

from yieldpred.alfalfa import LAST_COUNTY_YEAR, compare_with_state, coverage, model_table, weather_table
from yieldpred.nass import NASSError, fetch_county_alfalfa, fetch_state_alfalfa

ROOT = Path(__file__).resolve().parents[1]
PROCESSED = ROOT / "data" / "processed"
POWER = ROOT / "data" / "raw" / "power"


def save(df: pd.DataFrame, stem: str) -> None:
    df.to_parquet(PROCESSED / f"{stem}.parquet", index=False)
    df.to_csv(PROCESSED / f"{stem}.csv", index=False)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--state", default="NE")
    parser.add_argument("--start", type=int, default=2000)
    parser.add_argument("--no-download", action="store_true", help="skip NASS; use the saved files")
    args = parser.parse_args()
    st = args.state.lower()
    PROCESSED.mkdir(parents=True, exist_ok=True)

    if not args.no_download:
        print(f"Requesting {args.state} county alfalfa from {args.start}...")
        try:
            county = fetch_county_alfalfa(args.state, args.start)
            state = fetch_state_alfalfa(args.state, args.start)
        except NASSError as exc:
            raise SystemExit(f"Error: {exc}")
        if county.empty:
            raise SystemExit("No county rows returned - check the state code and years.")
        save(county, f"{st}_alfalfa_county")
        save(state, f"{st}_alfalfa_state")
        last = int(county["year"].max())
        print(f"Saved {len(county):,} county rows, {county['fips'].nunique()} counties, "
              f"{county['year'].min()}-{last}; {len(state)} state rows, {state['year'].min()}-{state['year'].max()}")
        if last > LAST_COUNTY_YEAR:
            print(f"Note: county data runs to {last}, past the {LAST_COUNTY_YEAR} cutoff NASS announced. "
                  "Worth a look: the cutoff may have been lifted for hay.")
    else:
        county = pd.read_parquet(PROCESSED / f"{st}_alfalfa_county.parquet")
        state = pd.read_parquet(PROCESSED / f"{st}_alfalfa_state.parquet")

    print("\nCounties reporting a yield, by year:")
    print(coverage(county, state).to_string(index=False))
    print("\nCounty average (acre-weighted) against NASS's state yield, tons/acre:")
    print(compare_with_state(county, state).to_string(index=False))

    weather = weather_table(POWER, sorted(county["fips"].unique()))
    if weather.empty:
        raise SystemExit(f"No NASA POWER files found in {POWER}. Run scripts/fetch_weather.py first.")
    save(weather, f"alfalfa_weather_{st}")
    table = model_table(county, weather)
    save(table, f"alfalfa_table_{st}")
    print(f"\nModel table: {len(table):,} county-years, {table['fips'].nunique()} counties "
          f"-> data/processed/alfalfa_table_{st}.parquet")


if __name__ == "__main__":
    main()
