"""Honest yield ranges: an 80% range around every prediction, checked against what really happened.

Every prediction the app shows was made by a model that never saw that county's district (spatial
cross-validation), so each one has a real, honest miss. A range is built from those misses and then
tested the only way that counts: how often did the real yield land inside?

Two ideas carry the module:

- **Cross-conformal ranges.** For the counties in one fold, the range comes from the misses in the
  OTHER four folds (split-conformal prediction, with the finite-sample rank: with n misses use the
  ceil((n+1) x 0.9)-th smallest for the top, not the plain 90th percentile). The range never uses
  the county's own miss, so checking it is fair.
- **Ranges that fit the county.** One width for everyone is right on average and wrong almost
  everywhere: in Nebraska a single 80% range held 68% of the time for dryland counties and 86% for
  irrigated ones, because rain decides the dryland crop. So misses are first divided by an expected
  miss size (a straight line in irrigation share, fitted on the other folds), the range is found on
  that scale, then multiplied back (normalized conformal prediction). Dryland ranges get wider,
  irrigated ones narrower, and each group holds close to 80%. Letting every feature set the width
  was tried and did worse (70% overall): with 2,000 rows it overfits the misses.

`forward_years` is the harder test: train on every year before, predict the next season, build the
range from earlier seasons' misses, and adapt the level after each season (adaptive conformal
inference). There a whole season can surprise every county at once, which is why those ranges hold
less often; the app says so. `season_share` measures it: the part of the next-season miss that
every county in the state shared (about a quarter in Nebraska, half in Iowa). A shared miss can't be
averaged away across counties, so in a shock season most ranges miss together, and in a calm season
almost all hold. Shorter look-back windows and faster adaptation were tried; neither fixes that,
because the shock isn't knowable before the season.
"""

from __future__ import annotations

import math

import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression

LEVEL = 0.80
MIN_SCALE = 2.0          # bu/acre: the smallest expected miss a county can be given
MIN_ERRORS = 20          # a range needs at least this many past misses
ADAPT_STEP = 0.02        # forward years: how far one season's hit rate moves the next level
                         # (0.05 chased Iowa's shock seasons and left later ranges near 100%)
LEVEL_LIMITS = (0.5, 0.98)
IRRIGATION_GROUPS = ([-0.01, 0.2, 0.6, 1.01], ["dryland", "mixed", "irrigated"])


def conformal_bounds(errors, level: float = LEVEL) -> tuple[float, float] | None:
    """Lower and upper error bounds that hold `level` of future misses (split-conformal, each tail
    (1 - level) / 2, with the finite-sample rank). None with fewer than MIN_ERRORS misses."""
    errors = np.sort(np.asarray(errors, dtype=float))
    n = len(errors)
    if n < MIN_ERRORS:
        return None
    k = min(n, math.ceil((n + 1) * (1 - (1 - level) / 2)))
    return float(errors[n - k]), float(errors[k - 1])


def expected_miss(train_irrigation, train_errors, target_irrigation) -> np.ndarray:
    """Expected size of a miss (bu/acre) from irrigation share: a straight line fitted to the absolute
    misses of the training rows, floored at MIN_SCALE."""
    model = LinearRegression().fit(np.asarray(train_irrigation, dtype=float).reshape(-1, 1),
                                   np.abs(np.asarray(train_errors, dtype=float)))
    return np.clip(model.predict(np.asarray(target_irrigation, dtype=float).reshape(-1, 1)), MIN_SCALE, None)


def cross_conformal(errors, folds, irrigation=None, level: float = LEVEL) -> tuple[np.ndarray, np.ndarray]:
    """Per-row error bounds (low, high) where each fold's bounds come only from the other folds' misses.
    With `irrigation`, the bounds are scaled to each county's expected miss (normalized conformal).
    errors = actual - predicted."""
    errors, folds = np.asarray(errors, dtype=float), np.asarray(folds)
    low, high = np.full(len(errors), np.nan), np.full(len(errors), np.nan)
    for fold in np.unique(folds):
        train, test = folds != fold, folds == fold
        if irrigation is None:
            scale_train, scale_test = np.ones(train.sum()), np.ones(test.sum())
        else:
            irrigation = np.asarray(irrigation, dtype=float)
            scale_train = expected_miss(irrigation[train], errors[train], irrigation[train])
            scale_test = expected_miss(irrigation[train], errors[train], irrigation[test])
        bounds = conformal_bounds(errors[train] / scale_train, level)
        if bounds is None:
            continue
        low[test], high[test] = bounds[0] * scale_test, bounds[1] * scale_test
    return low, high


def irrigation_group(share) -> pd.Series:
    edges, labels = IRRIGATION_GROUPS
    return pd.cut(pd.Series(share, dtype=float), edges, labels=labels)


def coverage(frame: pd.DataFrame, by: str | None = None) -> pd.DataFrame:
    """How often the range held, and how wide it was. `frame` needs error, low, high (+ `by`)."""
    inside = (frame["error"] >= frame["low"]) & (frame["error"] <= frame["high"])
    data = frame.assign(inside=inside, width=frame["high"] - frame["low"])
    groups = data.groupby(by, observed=True) if by else data.groupby(lambda _: "all")
    return pd.DataFrame({
        "rows": groups.size(),
        "held_pct": (100 * groups["inside"].mean()).round(1),
        "width_bu": groups["width"].mean().round(1),
    })


def pick_variant(coverage_table: pd.DataFrame, level: float = LEVEL) -> tuple[str, float]:
    """Which kind of range to show: the one whose worst group lands closest to the aim.

    `coverage_table` has one row per (ranges, group) with held_pct. Returns the variant's name and
    how many points its worst group is from the aim. Ties go to the first variant listed (the
    simpler one). Chosen by the data: irrigation-scaled widths fixed Nebraska corn's dryland
    counties (68% -> 81%) but made Nebraska soybeans' worse (72% -> 68%)."""
    gaps = (coverage_table["held_pct"] - 100 * level).abs().groupby(coverage_table["ranges"], sort=False).max()
    best = gaps.idxmin()
    return best, float(gaps[best])


def forward_years(frame: pd.DataFrame, level: float = LEVEL, step: float = ADAPT_STEP,
                  scaled: bool = False, min_years: int = 4) -> pd.DataFrame:
    """The next-season test. `frame`: one row per county-year with year, error (actual - predicted by a
    model trained only on earlier years) and, if `scaled`, irrigation_share. For each season, build the
    range from all earlier seasons' misses, count how many counties it held, then move the level by
    `step` x (share missed - share allowed) before the next season (adaptive conformal inference).
    Returns one row per season: year, counties, held_pct, width_bu, bias_bu (mean miss), level_used."""
    rows, target = [], level
    for year in sorted(frame["year"].unique()):
        past, now = frame[frame["year"] < year], frame[frame["year"] == year]
        if past["year"].nunique() < min_years:
            continue
        if scaled:
            s_past = expected_miss(past["irrigation_share"], past["error"], past["irrigation_share"])
            s_now = expected_miss(past["irrigation_share"], past["error"], now["irrigation_share"])
        else:
            s_past, s_now = np.ones(len(past)), np.ones(len(now))
        bounds = conformal_bounds(past["error"].to_numpy() / s_past, target)
        if bounds is None:
            continue
        low, high = bounds[0] * s_now, bounds[1] * s_now
        held = ((now["error"].to_numpy() >= low) & (now["error"].to_numpy() <= high)).mean()
        rows.append({"year": int(year), "counties": len(now), "held_pct": round(100 * float(held), 1),
                     "width_bu": round(float(np.mean(high - low)), 1),
                     "bias_bu": round(float(now["error"].mean()), 1), "level_used": round(target, 3)})
        target = float(np.clip(target + step * ((1 - held) - (1 - level)), *LEVEL_LIMITS))
    return pd.DataFrame(rows, columns=["year", "counties", "held_pct", "width_bu", "bias_bu", "level_used"])


def season_share(frame: pd.DataFrame) -> float:
    """Share of the squared miss that is the season's statewide average miss (0 to 1). `frame`: year,
    error. Near 0: counties miss independently. Near 1: the whole state misses together."""
    total = float((frame["error"] ** 2).mean())
    if not total:
        return 0.0
    shared = frame.groupby("year")["error"].transform("mean")
    return float((shared ** 2).mean() / total)
