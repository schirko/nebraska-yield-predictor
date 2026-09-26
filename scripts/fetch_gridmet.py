"""gridMET weather at 4 km, fetched two ways, to measure what 55 km was costing.

    python scripts/fetch_gridmet.py --probe          # ONE county, then stop
    python scripts/fetch_gridmet.py                  # Nebraska, both variants
    python scripts/fetch_gridmet.py --state IA --state-fips 19

Needs `pip install pygridmet`. Raw daily data is cached per county under
data/raw/gridmet/, the same way `fetch_weather.py` caches POWER, so re-running is
cheap and an interrupted run resumes.

WHY THIS EXISTS

`weather.py` gets weather from NASA POWER, whose grid cells are ~55 km across -
wider than most counties here. `scripts/probe_weather_grid.py` established that
this is the ceiling on the whole pipeline, and that cropland-weighting a 55 km
product is precision at the wrong stage. gridMET is ~4 km.

The obvious move is to swap one for the other. The move this project makes instead
is to fetch gridMET *alongside* POWER and score both on identical rows, because
"the finer dataset is better" is an assumption until it is measured, and the
measurement is more interesting than the upgrade. Nobody publishes how much a
coarse weather grid costs a county-level yield model. After this runs, we will
know for two states.

TWO VARIANTS, AND WHY BOTH

  --variant point   one 4 km pixel, at the same internal point POWER is asked for
  --variant mean    every 4 km pixel inside the county polygon, averaged
  --variant both    (default) fetch both

PREDICTIONS, REGISTERED BEFORE THE FIRST FETCH

1. **The point variant will NOT beat POWER, and may lose to it.** The target is a
   county-*average* yield. A 55 km cell is already a spatial average - crude, and
   centred in the wrong place, but an average. A single 4 km pixel is a sharper
   measurement of a smaller place, which is not the same thing as a better estimate
   of a county. If the gain were about resolution alone, this variant would show it.
2. **The county-mean variant will beat both**, and by more on spatial validation
   (unseen counties) than on temporal. Averaging ~100 pixels should mostly remove
   the sampling noise that a single point carries, and sampling noise is exactly
   what makes one county look different from its neighbour for no real reason.
3. **The gain will be larger in Nebraska than Iowa.** Nebraska has a real
   east-west moisture gradient, so where you sample inside a county matters more;
   Iowa is comparatively uniform.
4. **The features will differ most in precipitation, least in temperature.**
   Rain is spatially patchy at scales far below 55 km - a thunderstorm is a few km
   across - while temperature varies smoothly. `compare_sources` reports this
   before any model is fitted, so prediction 4 is settled cheaply and separately.

If prediction 1 holds and 2 fails, the honest conclusion is that this pipeline is
not sampling-limited at all, and the clustered errors are something else entirely.
That would be a more valuable result than a small win.

A NOTE ON WHAT IS COMPARABLE

Both products are summarised by the *same* functions in `weather.py`. Nothing here
computes a feature of its own. That is deliberate: if gridMET had its own
summariser, a score difference could be the product or could be the code, and the
whole exercise would prove nothing.
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from yieldpred import gridmet, weather  # noqa: E402
from yieldpred.dataset import PROCESSED, output_path  # noqa: E402

RAW = ROOT / "data" / "raw" / "gridmet"


def counties_with_geometry(state_fips: str):
    """County polygons, for the county-mean variant. Falls back to points only."""
    path = PROCESSED / f"counties_{state_fips}.parquet"
    if not path.exists():
        return None
    import geopandas as gpd
    return gpd.read_parquet(path).to_crs(4326)


def summarize(daily: pd.DataFrame) -> pd.DataFrame:
    """One row per year, using weather.py's functions and nothing else.

    `growing_season_features` already calls `spring_features` itself, on a frame it
    has prepared with `year`, `month` and a daily `gdd` column. An earlier version
    here called `spring_features` a second time on the raw daily frame and died with
    `KeyError: 'month'` - the probe caught it on the first county. One call is both
    correct and the whole point: POWER and gridMET must go through identical code.
    """
    return weather.growing_season_features(daily)


def fetch_one(row, geometry, variant: str, start: int, end: int,
              pause: float) -> pd.DataFrame | None:
    cache = RAW / f"{row.fips}_{variant}_{start}_{end}.csv"
    if cache.exists():
        return pd.read_csv(cache, parse_dates=["date"])

    if variant == "point":
        daily = gridmet.county_point_daily(row.lat, row.lon, start, end)
    else:
        if geometry is None:
            return None
        daily = gridmet.county_mean_daily(geometry, start, end)

    cache.parent.mkdir(parents=True, exist_ok=True)
    daily.to_csv(cache, index=False)
    time.sleep(pause)
    return daily


def run_variant(points, geoms, variant: str, args) -> pd.DataFrame | None:
    print(f"\n{'=' * 70}\n{variant.upper()} variant\n{'=' * 70}")
    frames, failures = [], []
    for i, row in enumerate(points.itertuples(), start=1):
        geometry = None
        if geoms is not None:
            match = geoms[geoms.fips == row.fips]
            geometry = match.geometry.iloc[0] if len(match) else None
        try:
            daily = fetch_one(row, geometry, variant, args.start, args.end, args.pause)
            if daily is None:
                failures.append((row.fips, "no geometry"))
                continue
            summary = summarize(daily)
            summary.insert(0, "county_name", row.county_name)
            summary.insert(0, "fips", row.fips)
            frames.append(summary)
            print(f"  [{i:3d}/{len(points)}] {row.county_name:22s} "
                  f"{len(summary)} years")
        except Exception as err:                       # noqa: BLE001 - reported, not swallowed
            failures.append((row.fips, str(err)[:90]))
            print(f"  [{i:3d}/{len(points)}] {row.county_name:22s} FAILED: "
                  f"{str(err)[:60]}")

    if failures:
        print(f"\n{len(failures)} counties failed:")
        for fips, why in failures[:10]:
            print(f"    {fips}: {why}")
        print("  Finished counties are cached, so re-running only retries these.")
    return pd.concat(frames, ignore_index=True) if frames else None


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--state", default="NE")
    parser.add_argument("--state-fips", default="31")
    parser.add_argument("--start", type=int, default=2000)
    parser.add_argument("--end", type=int, default=2025)
    parser.add_argument("--variant", choices=["point", "mean", "both"], default="both")
    parser.add_argument("--pause", type=float, default=0.5,
                        help="Seconds between requests, to be a good citizen")
    parser.add_argument("--probe", action="store_true",
                        help="Fetch ONE county both ways, print what came back, and "
                             "stop. Run this first: it costs one request and tells "
                             "you whether the units, dates and shapes are what this "
                             "code expects before you spend an hour on 93 counties.")
    args = parser.parse_args()
    state = args.state.lower()

    points = weather.county_points(args.state_fips)
    geoms = counties_with_geometry(args.state_fips)
    if geoms is None and args.variant in ("mean", "both"):
        print(f"No counties_{args.state_fips}.parquet — run fetch_nass_yields.py "
              f"first, or use --variant point.")
        if args.variant == "mean":
            return

    # ------------------------------------------------------------------ probe
    if args.probe:
        row = next(points.itertuples())
        print(f"Probing {row.county_name} ({row.fips}) at {row.lat:.3f}, {row.lon:.3f}\n")
        for variant in ("point", "mean"):
            geometry = None
            if geoms is not None:
                match = geoms[geoms.fips == row.fips]
                geometry = match.geometry.iloc[0] if len(match) else None
            if variant == "mean" and geometry is None:
                print("  mean: no geometry available, skipped")
                continue
            try:
                daily = fetch_one(row, geometry, variant, args.start, args.start, 0)
                span = (daily["date"].max() - daily["date"].min()).days + 1
                gap = span - len(daily)
                note = "" if gap == 0 else f"  ** {gap} day(s) missing inside the range **"
                print(f"  {variant}: {len(daily)} daily rows, "
                      f"{daily['date'].min().date()} to {daily['date'].max().date()}"
                      f"{note}")
                print(f"    tmax_c {daily['tmax_c'].min():6.1f} to "
                      f"{daily['tmax_c'].max():6.1f}   "
                      f"precip_mm total {daily['precip_mm'].sum():7.1f}")
                # A sanity range: if the Kelvin conversion were missed these would
                # read about 273 too high, and a whole run would be wasted.
                if not (-50 < daily["tmax_c"].min() < 60):
                    print("    ^^ that temperature range looks wrong — check units")
                print(f"    features: {summarize(daily).iloc[0].to_dict()}")
            except Exception as err:                   # noqa: BLE001
                print(f"  {variant}: FAILED — {err}")
        print("\nIf both look sane, drop --probe and run for real.")
        return

    # ------------------------------------------------------------------- run
    variants = ["point", "mean"] if args.variant == "both" else [args.variant]
    written = {}
    for variant in variants:
        if variant == "mean" and geoms is None:
            continue
        frame = run_variant(points, geoms, variant, args)
        if frame is None:
            continue
        out = output_path(f"gridmet_{variant}", state)
        frame.to_parquet(out, index=False)
        written[variant] = frame
        print(f"\nSaved {len(frame):,} county-years to {out.relative_to(ROOT)}")

    # --------------------------------------------- how different are they?
    power_path = PROCESSED / f"weather_county_{args.state_fips}.parquet"
    if power_path.exists() and written:
        power = pd.read_parquet(power_path)
        for variant, frame in written.items():
            print(f"\n{'=' * 70}")
            print(f"POWER (55 km) vs gridMET {variant} (4 km) — features only, no model")
            print("=" * 70)
            print(gridmet.compare_sources(power, frame).round(3).to_string())
        print("\nPrediction 4 said precipitation would diverge most and temperature "
              "least.\nThat is settled by the table above, before any model is fitted.")
        print("\nNext: python scripts/train_baseline.py --compare-weather")


if __name__ == "__main__":
    main()
