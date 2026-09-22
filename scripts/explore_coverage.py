"""Check how complete the NASS yield data is, by year and practice.

Run from the project root:  python scripts/explore_coverage.py
"""

from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data" / "processed" / "ne_corn_yield_county.parquet"

pd.set_option("display.width", 120)

df = pd.read_parquet(DATA)

print("=" * 64)
print("COUNTIES REPORTING, BY YEAR AND PRACTICE")
print("=" * 64)
counts = df.pivot_table(index="year", columns="practice", values="fips",
                        aggfunc="nunique").fillna(0).astype(int)
print(counts.to_string())

print("\n" + "=" * 64)
print("YEAR RANGE PER PRACTICE")
print("=" * 64)
span = df.groupby("practice")["year"].agg(["min", "max", "nunique"])
print(span.to_string())

print("\n" + "=" * 64)
print("MEAN YIELD BY YEAR AND PRACTICE (bu/acre)")
print("=" * 64)
means = df.pivot_table(index="year", columns="practice",
                       values="yield_bu_acre", aggfunc="mean").round(1)
print(means.to_string())

print("\n" + "=" * 64)
print("COUNTIES WITH BOTH IRRIGATED AND NON-IRRIGATED IN THE SAME YEAR")
print("=" * 64)
both = (df[df["practice"] != "all"]
        .pivot_table(index=["fips", "year"], columns="practice",
                     values="yield_bu_acre")
        .dropna())
print(f"{len(both):,} county-year pairs across {both.index.get_level_values('fips').nunique()} counties")
if not both.empty:
    gap = (both["irrigated"] - both["non_irrigated"])
    print(f"Irrigation advantage: mean {gap.mean():.1f} bu/acre, "
          f"median {gap.median():.1f}, min {gap.min():.1f}, max {gap.max():.1f}")
    print("\nMean irrigation advantage by year:")
    print(gap.groupby("year").mean().round(1).to_string())

print("\n" + "=" * 64)
print("MOST AND LEAST COMPLETE COUNTIES (all-practice series)")
print("=" * 64)
per_county = (df[df["practice"] == "all"]
              .groupby(["fips", "county_name"])["year"].nunique()
              .sort_values())
print(f"Years per county: min {per_county.min()}, median {int(per_county.median())}, "
      f"max {per_county.max()}")
print("\nFewest years:")
print(per_county.head(5).to_string())
