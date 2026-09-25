"""Leave-district-out has a leak, and this module measures it.

The spatial validation scheme holds out whole NASS agricultural districts so the
model has to predict counties it has never seen. That is the right idea, and it
has a hole in it that only shows up once you know how the weather was fetched.

`weather.py` queries NASA POWER once per county. POWER serves the MERRA-2 grid -
0.5 degrees latitude by 0.625 of longitude, about 55 km - and a county is usually
smaller than that, so **two counties can be served by the same cell**. When they
are, their weather rows are not merely similar, they are identical: the same
numbers were fetched twice.

Now put those two counties in different districts. Hold one district out, and the
held-out county's exact weather vector is sitting in the training data attached to
a real yield. The county is unseen. Its weather is not.

In Iowa this affects 43 of 98 modelled counties (44%); in Nebraska, 2 of 88.
Iowa's districts are small and regular, so cells straddle their boundaries
constantly. See `scripts/probe_weather_grid.py` for how that was measured.

**Why not just regroup?** Because you cannot. Requiring that no cell ever be split
across folds chains all nine Iowa districts into one connected component - there is
no grouping of Iowa counties that both respects districts and keeps cells whole. So
the leak cannot be designed away; it has to be measured.

**How this measures it.** `LeakFreeGroupKFold` wraps `GroupKFold` and changes one
thing: after the fold is chosen, any training county sharing a POWER cell with a
test county is dropped **from training only**. The test set is untouched, so both
variants predict exactly the same rows and the two scores are directly comparable.
The difference between them is what the leak was worth.

That is the same discipline as the carried-value control in the rotation work:
change one thing, on one row set, and report the gap. It is not free - the leak-free
run trains on fewer rows, so part of any drop is simply less training data rather
than the leak. `training_cost()` reports how many rows each fold lost so that part
can be weighed honestly rather than assumed away.
"""

from __future__ import annotations

import hashlib

import numpy as np
import pandas as pd
from sklearn.model_selection import GroupKFold

# Three variables is plenty to identify a cell - identical is identical - but using
# several independent ones makes an accidental collision absurd rather than merely
# unlikely.
FINGERPRINT = ("gdd", "precip_mm", "tmax_jul_c")

# POWER reports to a fixed number of decimals; rounding before hashing keeps the
# identity robust to float round-trips through parquet.
ROUND = 6


def power_cells(weather: pd.DataFrame,
                columns: tuple[str, ...] = FINGERPRINT) -> pd.Series:
    """Map each county to an id for the POWER grid cell it was served from.

    Recovered from the data rather than from the grid geometry, deliberately.
    Reconstructing the grid needs POWER's exact alignment *and* the Census
    Gazetteer internal point that `weather.py` queried - and near a cell boundary
    the difference between that point and the centroid decides which cell you get.
    Identical weather across every year in the record is a fact about what was
    fetched, and needs neither.

    Returns a Series indexed by fips whose values are opaque cell ids; only
    equality between them is meaningful.
    """
    missing = [c for c in columns if c not in weather.columns]
    if missing:
        raise KeyError(f"weather table is missing {missing}")

    ordered = weather.sort_values(["fips", "year"])
    out = {}
    for fips, block in ordered.groupby("fips", sort=False):
        arr = np.round(block[list(columns)].to_numpy(dtype="float64"), ROUND)
        out[fips] = hashlib.sha1(arr.tobytes()).hexdigest()[:12]
    return pd.Series(out, name="power_cell").rename_axis("fips")


def leak_report(cells: pd.Series, districts: pd.Series) -> dict:
    """How much of the spatial holdout is compromised, before fixing anything.

    `cells` and `districts` are both indexed by fips and cover the modelled
    counties only - the leak is a property of the rows the model actually scores,
    not of every county in the state.
    """
    frame = pd.DataFrame({"cell": cells, "district": districts}).dropna()
    per_cell = frame.groupby("cell")["district"].agg(["nunique", "size"])
    crossing = per_cell[(per_cell["size"] > 1) & (per_cell["nunique"] > 1)]
    leaking = frame[frame.cell.isin(crossing.index)]
    return {
        "counties": len(frame),
        "cells": frame.cell.nunique(),
        "cells_crossing_districts": len(crossing),
        "counties_leaking": len(leaking),
        "share_leaking": len(leaking) / len(frame) if len(frame) else 0.0,
    }


class LeakFreeGroupKFold:
    """GroupKFold, minus any training row whose weather is in the test set.

    Deliberately not a subclass. scikit-learn's splitters promise a partition of
    the data, and this breaks that promise on purpose: the dropped training rows
    belong to neither side. Inheriting would invite a caller to assume the parent's
    guarantees still hold.

    Parameters
    ----------
    n_splits : int
        Passed through to the underlying GroupKFold.
    cells : array-like
        One cell id per ROW, aligned with X - so the same shape as `groups`, not
        one entry per county. Build it with
        ``used["fips"].map(power_cells(weather)).to_numpy()``.
    """

    def __init__(self, n_splits: int, cells) -> None:
        self.n_splits = int(n_splits)
        self.cells = np.asarray(cells)
        self.dropped_: list[int] = []      # training rows removed, per fold
        self.trained_: list[int] = []      # training rows kept, per fold

    def get_n_splits(self, X=None, y=None, groups=None) -> int:
        return self.n_splits

    def split(self, X, y=None, groups=None):
        if len(self.cells) != len(X):
            raise ValueError(
                f"cells has {len(self.cells)} entries but X has {len(X)} rows; "
                "cells must be per-row, not per-county")
        self.dropped_, self.trained_ = [], []
        for train, test in GroupKFold(self.n_splits).split(X, y, groups):
            contaminated = np.isin(self.cells[train], np.unique(self.cells[test]))
            kept = train[~contaminated]
            if kept.size == 0:
                raise ValueError(
                    "a fold lost its entire training set to the leak filter; "
                    "the districts and the weather grid are too entangled to "
                    "validate this way")
            self.dropped_.append(int(contaminated.sum()))
            self.trained_.append(int(kept.size))
            yield kept, test

    def matched_control(self, seed: int = 0) -> "ThinnedGroupKFold":
        """The control this comparison needs: same folds, same losses, random rows.

        Call after a full split pass, once `dropped_` is populated.
        """
        if not self.dropped_:
            raise RuntimeError("run split() before asking for the matched control")
        return ThinnedGroupKFold(self.n_splits, list(self.dropped_), seed=seed)

    def training_cost(self) -> dict:
        """What the filter cost in training rows - call after a full split pass.

        Part of any score drop is less training data rather than the leak itself.
        Reporting this is what keeps the comparison honest.
        """
        if not self.dropped_:
            return {"folds": 0}
        total_dropped = sum(self.dropped_)
        total_kept = sum(self.trained_)
        return {
            "folds": len(self.dropped_),
            "rows_dropped_total": total_dropped,
            "rows_dropped_per_fold": list(self.dropped_),
            "share_of_training_lost": total_dropped / (total_dropped + total_kept),
        }


class ThinnedGroupKFold:
    """GroupKFold with a fixed number of training rows removed at random per fold.

    This is the control that makes the leak measurement mean anything.

    `LeakFreeGroupKFold` removes the leaking training rows, and the score falls.
    But it removed rows, and removing rows lowers scores by itself - so a raw
    before-and-after cannot distinguish "the leak was doing work" from "the model
    had less to learn from". Running the same folds again, dropping the same number
    of training rows *chosen at random*, separates the two:

    * random thinning costs about as much as the leak filter -> the leak was worth
      little, and the drop is mostly the missing data.
    * random thinning costs much less -> the dropped rows were special, which is
      what a leak looks like.

    Use several seeds and average; one draw is a coin flip, not a control.
    """

    def __init__(self, n_splits: int, drop_counts: list[int], seed: int = 0) -> None:
        self.n_splits = int(n_splits)
        self.drop_counts = list(drop_counts)
        self.seed = seed
        if len(self.drop_counts) != self.n_splits:
            raise ValueError(
                f"need one drop count per fold: got {len(self.drop_counts)} "
                f"for {self.n_splits} splits")

    def get_n_splits(self, X=None, y=None, groups=None) -> int:
        return self.n_splits

    def split(self, X, y=None, groups=None):
        rng = np.random.default_rng(self.seed)
        for (train, test), n_drop in zip(
                GroupKFold(self.n_splits).split(X, y, groups), self.drop_counts):
            n_drop = min(n_drop, max(train.size - 1, 0))
            if n_drop:
                drop = rng.choice(train.size, size=n_drop, replace=False)
                keep = np.delete(train, drop)
            else:
                keep = train
            yield keep, test
