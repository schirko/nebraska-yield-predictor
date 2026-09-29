# Honest Ranges, in Plain Words

Every yield the Yield Predictor shows comes with an **80% range**. This page explains what that
means, how the range is built, how it's checked, and where it falls short. The code is
`src/yieldpred/ranges.py` (run by `scripts/honest_ranges.py`), and the app shows the results under
Model & Validation → Honest Ranges. README section 8 has the summary table.

## One row of the ranges file

`data/processed/ranges.parquet` has one row per county and year (2,002 rows for Nebraska corn):
the prediction, a low and a high end, the real yield, and whether it landed inside. Hall County,
Nebraska, corn:

| Year | Actual | Predicted | 80% range | Inside? |
|---|---|---|---|---|
| 2018 | 217.5 | 209.0 | 191.3 to 226.6 | Yes |
| 2019 | 180.4 | 199.7 | 181.7 to 217.6 | **No** |
| 2020 | 201.8 | 202.3 | 184.0 to 220.5 | Yes |

Columns: `fips`, `county_name`, `year`, `asd_desc` (the USDA district), `yield_bu_acre` (actual),
`predicted`, `low`, `high`, `inside`. The other files follow the same pattern: `ranges_ia` for
Iowa, `ranges_soy` and `ranges_ia_soy` for soybeans.

**"80%" is a promise about the long run:** over many county-years, the real yield should land
inside about 8 times in 10. Missing 2019 isn't a failure; about 1 year in 5 *should* miss. A range
that never missed would just be wider than it needs to be.

## How a range is built (conformal prediction)

1. **Collect honest misses.** Every prediction comes from a model that never saw that county's
   district (the "unseen districts" test, spatial cross-validation). So each county-year has a real
   miss: actual minus predicted.
2. **Sort the misses and take the edges.** For 80%, keep the middle 80% of past misses: about the
   10th percentile for the low end and the 90th for the high end. If past misses ran from −18 to
   +18 bu/acre, a prediction of 199.7 gets a range of about 181.7 to 217.6. With `n` misses the
   code uses the `ceil((n + 1) × 0.9)`-th smallest rather than the plain percentile (the
   "finite-sample rank"), which keeps the promise honest when there are few misses. A range needs
   at least 20 past misses (`MIN_ERRORS`).
3. **Never use the county's own miss.** The data is split into five groups ("folds"). A county's
   range is built only from misses in the *other* four, so checking whether its own yield landed
   inside is a fair test. That's the "cross" in **cross-conformal**.

Why this method: it doesn't assume the misses follow a bell curve or any other shape. It only
uses how big the misses really were.

## The twist: one width didn't fit everyone

The first version gave every county the same width. `ranges_coverage.parquet` shows it failing
by group:

| Nebraska corn | Held (goal 80%) | Average width |
|---|---|---|
| One width, dryland counties | 68.4% | 45.9 bu/acre |
| One width, irrigated counties | 85.9% | 46.5 bu/acre |
| **Scaled to irrigation, dryland** | **81.2%** | 60.1 bu/acre |
| **Scaled to irrigation, mixed** | **75.9%** | 50.2 bu/acre |
| **Scaled to irrigation, irrigated** | **78.5%** | 39.2 bu/acre |

Dryland corn depends on rain, so its misses are bigger; irrigated corn is steadier. With one width,
dryland ranges were too narrow and irrigated ones too wide.

The fix is **normalized conformal prediction**: divide each miss by that county's expected miss
size (a straight line in irrigation share, fitted on the other folds; never below 2 bu/acre), find
the range on that scale, then multiply back. Dryland ranges widen, irrigated ones tighten, and
each group holds close to 80%. The overall rate barely moved (78.3% to 78.1%).

**The lesson: a range can be right on average and wrong for everyone in particular.** Checking by
group is how you catch it. Letting every input set the width was also tried and did worse (70%
overall): with about 2,000 rows it fits the noise in the misses.

`honest_ranges.py` builds both kinds and keeps the one whose worst group is closest to 80%
(`ranges.pick_variant`). Corn in Nebraska uses the irrigation-scaled ranges; soybeans don't,
because scaling made soybeans' dryland coverage worse.

## The honest limit: next season

`ranges_forward*.parquet` is the harder, real-life test: train only on earlier seasons, predict
the next one, and build its range from earlier seasons' misses. After each season the level is
nudged up or down by how that season went (**adaptive conformal inference**, step 0.02).

| Next-season test | Nebraska | Iowa |
|---|---|---|
| Held, on average | 74.4% | 75.6% |
| Seasons under 60% | 1 of 16 | 5 of 16 |
| Share of the miss every county shared | 25% | **52%** |

An unusual season surprises every county at once: in Nebraska's worst next-season test (2021) only 46% of ranges held, and Iowa ran under 55% four seasons in a row (2013 to 2016). That shared part of
the miss doesn't average out across counties, so in a shock year most ranges miss together, and in
a calm year nearly all hold. Shorter look-back windows and faster adaptation were tried; neither
fixes it, because the shock isn't knowable before the season. That's why the app says **"8 in 10
over many seasons, not 8 in 10 counties this season."**

## Words worth knowing

- **Coverage:** how often the real value landed inside the range. The number to check, not assume.
- **Conformal prediction:** building ranges from a model's real past misses, with no assumption
  about their shape.
- **Split / cross-conformal:** the misses used to set a range come from data the prediction didn't
  use; "cross" rotates that through every fold so every row gets a fair range.
- **Normalized conformal:** scale misses by an expected size first, so ranges can be wider where
  predictions are shakier.
- **Adaptive conformal:** adjust the level season by season when the future doesn't behave like
  the past.
