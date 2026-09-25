"""Build the corn-soybean rotation features from NASS acreage.

    python scripts/fetch_rotation.py
    python scripts/fetch_rotation.py --state IA
    python scripts/fetch_rotation.py --probe        # what does NASS actually have?

Corn following soybeans out-yields corn following corn. The model can't see field
histories, but a county's acreage split says how much of its corn is rotated and
how much is continuous - and that is reconstructable from data NASS publishes.

Also writes the planted-vs-harvested ratio, which is NOT a model feature: it is
the measurement that tests whether Iowa's wet springs remove the worst ground from
the harvested-acre denominator. See src/yieldpred/rotation.py.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

from yieldpred.nass import NASSError
from yieldpred.rotation import (CORN_ANY, CORN_GRAIN, CORN_SILAGE, SOYBEANS,
                                fetch_acres, fill_gaps, harvested_ratio,
                                rotation_features)

ROOT = Path(__file__).resolve().parents[1]
PROCESSED = ROOT / "data" / "processed"

# Probe from simple to complex - the technique that found the SSURGO problem in ten
# minutes instead of an afternoon. Each row is one question, and the first failure
# names the cause instead of leaving a silent empty frame.
PROBES = [
    ("corn planted, SURVEY", CORN_ANY, "AREA PLANTED", "SURVEY"),
    ("soybeans planted, SURVEY", SOYBEANS, "AREA PLANTED", "SURVEY"),
    ("corn grain harvested, SURVEY", CORN_GRAIN, "AREA HARVESTED", "SURVEY"),
    ("corn silage harvested, SURVEY", CORN_SILAGE, "AREA HARVESTED", "SURVEY"),
    ("corn planted, CENSUS", CORN_ANY, "AREA PLANTED", "CENSUS"),
    ("soybeans planted, CENSUS", SOYBEANS, "AREA PLANTED", "CENSUS"),
]


def probe(state: str, start: int) -> None:
    print("=" * 74)
    print(f"WHAT NASS ACTUALLY HAS FOR {state.upper()}")
    print("=" * 74)
    for label, commodity, statistic, source in PROBES:
        try:
            df = fetch_acres(commodity, statistic, source, state, start)
        except NASSError as exc:
            print(f"  {label:32s} FAILED  {exc}")
            continue
        if df.empty:
            print(f"  {label:32s} empty")
        else:
            print(f"  {label:32s} {len(df):>5,} county-years, "
                  f"{df['year'].min()}-{df['year'].max()}, "
                  f"{df['fips'].nunique()} counties")


def load(label: str, commodity: dict, statistic: str, state: str,
         start: int) -> pd.DataFrame:
    """Try SURVEY first, fall back to CENSUS, and say which one answered."""
    for source in ("SURVEY", "CENSUS"):
        try:
            df = fetch_acres(commodity, statistic, source, state, start)
        except NASSError as exc:
            print(f"  {label} ({source}): {exc}")
            continue
        if not df.empty:
            print(f"  {label}: {len(df):,} county-years from {source}, "
                  f"{df['year'].min()}-{df['year'].max()}, "
                  f"{df['fips'].nunique()} counties")
            return df
    print(f"  {label}: nothing returned")
    return pd.DataFrame(columns=["fips", "year", "acres"])


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--state", default="NE")
    parser.add_argument("--end", type=int, default=2025)
    parser.add_argument("--start", type=int, default=1999,
                        help="One year before the modelling window, so the "
                             "lagged feature has a value in the first year")
    parser.add_argument("--probe", action="store_true",
                        help="Report what NASS has and stop")
    args = parser.parse_args()
    state = args.state.lower()

    if args.probe:
        probe(args.state, args.start)
        return

    print(f"Fetching acreage for {args.state.upper()}...")
    corn = load("corn planted", CORN_ANY, "AREA PLANTED", args.state, args.start)
    soy = load("soybeans planted", SOYBEANS, "AREA PLANTED", args.state, args.start)
    corn_harvested = load("corn grain harvested", CORN_GRAIN, "AREA HARVESTED",
                          args.state, args.start)
    corn_silage = load("corn silage harvested", CORN_SILAGE, "AREA HARVESTED",
                       args.state, args.start)

    if corn.empty or soy.empty:
        raise SystemExit(
            "\nNo usable acreage. Run with --probe to see what NASS has for this "
            "state; a state that grows no soybeans has no rotation to measure.")

    PROCESSED.mkdir(parents=True, exist_ok=True)

    features = rotation_features(corn, soy)

    # Fill suppressed years rather than letting them drop rows from the model.
    # Dropping is not neutral: it removes the hardest counties and flatters every
    # score. See rotation.fill_gaps.
    before = len(features)
    model_years = range(args.start + 1, args.end + 1)
    features = fill_gaps(features, model_years)
    carried = int((~features["rotation_observed"]).sum())
    print(f"\nGap filling: {before:,} observed county-years -> {len(features):,} "
          f"after carrying forward ({carried:,} carried, "
          f"{1 - carried / max(len(features), 1):.0%} observed)")

    out = PROCESSED / f"rotation_{state}.parquet"
    features.to_parquet(out, index=False)
    print(f"\nSaved {len(features):,} county-years to {out.relative_to(ROOT)}")

    # Make the one judgement call in this feature visible every time it runs.
    corn_only = sorted(set(corn["fips"]) - set(soy["fips"]))
    if corn_only:
        print(f"\n{len(corn_only)} counties report corn and never soybeans; their "
              f"corn_share is set to 1.0 (continuous corn), not dropped.")
        print(f"  {', '.join(corn_only)}")
    unobserved = int((~features["rotation_observed"]).sum())
    print(f"Rotation observed for {len(features) - unobserved:,} county-years "
          f"({1 - unobserved / max(len(features), 1):.0%}); the rest are the "
          f"corn-only counties above.")

    print("\n" + "=" * 74)
    print("ROTATION INTENSITY (corn share of corn + soybean planted acres)")
    print("=" * 74)
    print("0.50 = clean two-year rotation.  Above 0.60 = more continuous corn.\n")
    by_year = features.groupby("year")["corn_share"].agg(["mean", "median", "count"])
    print(by_year.round(3).tail(10).to_string())

    latest = features[features["year"] == features["year"].max()]
    print(f"\nMost continuous corn in {features['year'].max()}:")
    print(latest.nlargest(6, "corn_share")[["fips", "corn_share"]].to_string(index=False))
    print("Most rotated:")
    print(latest.nsmallest(6, "corn_share")[["fips", "corn_share"]].to_string(index=False))

    # ------------------------------------------------ the abandonment diagnostic
    if not corn_harvested.empty:
        ratio = harvested_ratio(corn, corn_harvested, corn_silage)
        ratio.to_parquet(PROCESSED / f"harvested_ratio_{state}.parquet", index=False)

        print("\n" + "=" * 74)
        print("HARVESTED / PLANTED  (not a feature - the prevented-planting test)")
        print("=" * 74)
        print("A wet spring that stops planting should show up as a LOW ratio.")
        if ratio["silage_corrected"].any():
            print("Silage acres are removed from the denominator, so this measures")
            print("acres MEANT for grain that failed - not corn chopped for feed.")
            print(f"Mean silage share: {ratio['silage_share'].mean():.1%} of planted "
                  f"acres.\n")
        else:
            print("WARNING: no silage data - the ratio is understated by the")
            print("silage share, and falls in drought years for reasons that")
            print("are not crop failure.\n")
        by_year = ratio.groupby("year")["harvested_ratio"].mean().round(3)
        print(by_year.to_string())
        print("\nLowest years:")
        print(by_year.nsmallest(5).to_string())

        weather = PROCESSED / f"weather_county_{'31' if state == 'ne' else '19'}.parquet"
        if weather.exists():
            w = pd.read_parquet(weather)
            if "workable_days" in w.columns:
                joined = (ratio.groupby("year")["harvested_ratio"].mean()
                          .to_frame()
                          .join(w.groupby("year")["workable_days"].mean()))
                r = joined["harvested_ratio"].corr(joined["workable_days"])
                print(f"\ncorr(harvested ratio, workable planting days) = {r:+.3f}")
                print("Positive means fewer workable days went with more abandonment,")
                print("which is the prevented-planting story. Near zero means it didn't.")


if __name__ == "__main__":
    main()
