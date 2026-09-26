"""The Yield Predictor's card on the farm-account home page: each county's trend corn yield.

One number per county, with an honest range:

1. Fit a straight line through the county's NASS corn yields (all practices) since 2000. Yields rise
   over time with better genetics and practices, so the line says what a "normal" year looks like now.
   Its value for next season is the TREND YIELD.
2. Express every past year as a ratio to its trend value (0.92 = 8% below trend). The 10th percentile
   of those ratios, times next season's trend yield, is what a bad year (1 in 10) looks like.

This is not a forecast of this season (the model has never been validated in-season; see the README).
It is the county's recent normal and its downside, which is exactly what a farmer plans around.

    python scripts/export_suite_card.py      # writes data/processed/suite_card.json
"""

from __future__ import annotations

import csv
import json
import statistics
from datetime import date
from pathlib import Path

FORMAT = 1
MIN_YEARS = 10       # fewer and the line is mostly noise
RECENT_YEARS = 5     # the county must have a report in the last 5 seasons (NASS has dropped many)
APP_URL = "https://nebraska-yield-predictor-izzr8x4rxcppkcfjmqmiej.streamlit.app/"
SOURCE = "USDA NASS county corn yields (all practices), 2000 on"


def fit_line(xs: list[float], ys: list[float]) -> tuple[float, float]:
    """Ordinary least squares: returns (slope, intercept)."""
    mx, my = statistics.fmean(xs), statistics.fmean(ys)
    sxx = sum((x - mx) ** 2 for x in xs)
    slope = sum((x - mx) * (y - my) for x, y in zip(xs, ys)) / sxx
    return slope, my - slope * mx


def quantile(values: list[float], q: float) -> float:
    """Linear-interpolation quantile (numpy's default), without numpy."""
    v = sorted(values)
    pos = q * (len(v) - 1)
    lo = int(pos)
    hi = min(lo + 1, len(v) - 1)
    return v[lo] + (v[hi] - v[lo]) * (pos - lo)


def county_card(history: dict[int, float], name: str, state: str, season: int) -> dict:
    """The card for one county. `history` = {year: bu/acre}."""
    years = sorted(history)
    slope, intercept = fit_line(years, [history[y] for y in years])
    trend = intercept + slope * season
    ratios = [history[y] / (intercept + slope * y) for y in years]
    low = trend * quantile(ratios, 0.10)
    direction = "rise" if slope >= 0 else "fall"
    return {
        "label": f"Trend corn yield, {season}",
        "headline": f"{trend:.0f} bu/acre",
        "detail": (f"{name}, {state} yields since {years[0]} {direction} about {abs(slope):.1f} bu/acre a year. "
                   f"In 1 year in 10 the county falls below {low:.0f}."),
        "value": round(trend, 1), "low": round(low, 1), "unit": "bu/acre",
        "years": len(years), "last_year": years[-1],
    }


def read_yields(paths: list[Path]) -> dict[str, dict]:
    """fips -> {name, state, history{year: yield}} from the processed NASS CSVs (practice == all)."""
    out: dict[str, dict] = {}
    for path in paths:
        with open(path, newline="", encoding="utf-8") as f:
            for r in csv.DictReader(f):
                if r["practice"] != "all" or not r["yield_bu_acre"]:
                    continue
                c = out.setdefault(r["fips"], {"name": f"{r['county_name'].title()} County",
                                               "state": r["state_alpha"], "history": {}})
                c["history"][int(r["year"])] = float(r["yield_bu_acre"])
    return out


def build(paths: list[Path], made: date | None = None) -> dict:
    counties = read_yields(paths)
    latest = max(y for c in counties.values() for y in c["history"])
    season = latest + 1
    cards, skipped = {}, {}
    for fips, c in sorted(counties.items()):
        h = c["history"]
        if len(h) < MIN_YEARS:
            skipped[fips] = f"only {len(h)} years of yields"
        elif max(h) < latest - RECENT_YEARS + 1:
            skipped[fips] = f"no NASS yield since {max(h)}"
        else:
            cards[fips] = county_card(h, c["name"], c["state"], season)
    return {
        "format": FORMAT, "app": "corn-yield-predictor", "made": (made or date.today()).isoformat(),
        "season": season, "source": SOURCE, "link": APP_URL,
        "method": "Straight-line trend through the county's yields; the low end is the 10th percentile of each "
                  "year's yield relative to its trend. County averages, not a forecast of this season.",
        "not_covered": "Nebraska and Iowa counties with recent USDA yield reports today.",
        "cards": cards, "skipped": skipped,
    }


def write(data: dict, path: Path) -> None:
    path.write_text(json.dumps(data, indent=1, sort_keys=True) + "\n", encoding="utf-8")
