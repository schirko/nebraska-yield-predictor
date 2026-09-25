"""Corn-soybean rotation: what a county's acreage split says about its yield.

Corn following soybeans out-yields corn following corn, by something like 10-15%
in the Corn Belt. Three mechanisms stack: a nitrogen credit from the legume, a
disease and pest break (corn rootworm above all), and better soil structure.

The model cannot see any individual field's history, and doesn't need to. At
county resolution the question collapses to how the acres split:

    corn_share = corn planted acres / (corn + soybean planted acres)

Near 0.5 is a county in a clean two-year rotation. Drifting up past 0.6 means more
continuous corn - which in Nebraska tends to mean irrigated ground, and ground near
ethanol plants and feedlots.

This is the same shape of feature as irrigation share: not published as a feature,
reconstructed from acreage ratios, and motivated by knowing how the farms actually
operate rather than by anything visible in the yield series.

PLANTED, NOT HARVESTED - AND THE DISTINCTION IS THE POINT
---------------------------------------------------------
Planted acres are known by late June, months before harvest, so a feature built
from them is legitimate for a yield model.

Harvested acres are not. The gap between planted and harvested is *abandonment* -
acres that went in and never came out - which is an outcome of the season, not an
input to it. Using it to predict yield would be predicting the harvest from the
harvest.

So `harvested_ratio` is computed here and deliberately kept OUT of the feature set.
It exists for one diagnostic: testing whether Iowa's wet springs show the
prevented-planting selection effect that may explain the drift documented in
METHODS.md. A number can be evidence without being a feature.
"""

from __future__ import annotations

import pandas as pd

from yieldpred.nass import parse_value, query

# NASS's `commodity_desc: "CORN"` is FIELD corn. Sweet corn and popcorn are
# separate commodities, so they are never in scope here.
#
# Utilization is where the care is needed. NASS reports PLANTED acres for corn as
# a whole and HARVESTED acres split by utilization, so "corn planted for grain"
# does not exist as a series. Every ratio below has to account for that.
CORN_GRAIN = {"commodity_desc": "CORN", "util_practice_desc": "GRAIN"}
CORN_SILAGE = {"commodity_desc": "CORN", "util_practice_desc": "SILAGE"}
CORN_ANY = {"commodity_desc": "CORN"}          # grain + silage
SOYBEANS = {"commodity_desc": "SOYBEANS"}


def fetch_acres(commodity: dict, statistic: str = "AREA PLANTED",
                source_desc: str = "SURVEY", state_alpha: str = "NE",
                start_year: int = 2000, api_key: str | None = None) -> pd.DataFrame:
    """County acreage for one commodity and one statistic.

    `statistic` is "AREA PLANTED" or "AREA HARVESTED". Note that NASS reports
    corn planted acres for corn as a whole (grain plus silage) but harvested acres
    separately for grain - so the planted series should not carry
    `util_practice_desc`, or it comes back empty.
    """
    params = {
        "source_desc": source_desc,
        "sector_desc": "CROPS",
        "statisticcat_desc": statistic,
        "unit_desc": "ACRES",
        "domain_desc": "TOTAL",
        "agg_level_desc": "COUNTY",
        "reference_period_desc": "YEAR",
        "state_alpha": state_alpha,
        "year__GE": start_year,
        **commodity,
    }
    return tidy_acres(query(params, api_key))


def tidy_acres(raw: pd.DataFrame) -> pd.DataFrame:
    """Clean raw acreage rows to: fips, year, acres.

    Only all-practice rows are kept. A county that reports irrigated and
    non-irrigated separately would otherwise be counted two or three times, and
    the resulting share would be silently wrong rather than obviously wrong.
    """
    cols = ["fips", "year", "acres"]
    if raw.empty:
        return pd.DataFrame(columns=cols)

    df = raw.copy()
    df = df[df["county_code"].astype(str) != "998"]
    if "prodn_practice_desc" in df.columns:
        df = df[df["prodn_practice_desc"] == "ALL PRODUCTION PRACTICES"]

    df["fips"] = (df["state_fips_code"].astype(str).str.zfill(2)
                  + df["county_code"].astype(str).str.zfill(3))
    df["year"] = df["year"].astype(int)
    df["acres"] = parse_value(df["Value"])

    return (df.dropna(subset=["acres"])[cols]
            .groupby(["fips", "year"], as_index=False)["acres"].sum()
            .sort_values(["fips", "year"])
            .reset_index(drop=True))


def rotation_features(corn_planted: pd.DataFrame,
                      soy_planted: pd.DataFrame) -> pd.DataFrame:
    """County-year rotation intensity, plus last year's soybean share.

    Returns fips, year, corn_share, soy_share_prev, rotation_acres.

    `soy_share_prev` is the mechanism stated directly: this year's corn partly
    lives on last year's soybeans. Shifting within each county - never across the
    whole frame - matters, because a plain shift would hand one county's value to
    the next county in the table.
    """
    cols = ["fips", "year", "corn_share", "soy_share_prev", "rotation_acres",
            "rotation_observed"]
    if corn_planted.empty or soy_planted.empty:
        return pd.DataFrame(columns=cols)

    merged = corn_planted.merge(soy_planted, on=["fips", "year"], how="left",
                                suffixes=("_corn", "_soy"))

    # An absent soybean row means one of two different things, and treating them
    # alike would be wrong in opposite directions.
    #
    # NASS reports soybeans in 82 Nebraska counties and corn in 91. The nine
    # without soybeans are Sandhills ranching counties where soybeans are not
    # grown at all - so "no soybean acres" is not missing data, it is the answer:
    # that county's corn is 100% continuous, which is precisely the condition this
    # feature exists to detect. Dropping those counties would delete the clearest
    # examples of the thing being measured.
    #
    # A county that usually reports soybeans and is missing a single year is a
    # different case - that is likely a disclosure suppression, genuinely unknown,
    # and it stays missing.
    grows_soybeans = set(soy_planted["fips"])
    never_soybeans = ~merged["fips"].isin(grows_soybeans)
    merged.loc[never_soybeans, "acres_soy"] = 0.0
    merged["rotation_observed"] = ~never_soybeans

    merged = merged.dropna(subset=["acres_soy"])
    merged["rotation_acres"] = merged["acres_corn"] + merged["acres_soy"]
    merged = merged[merged["rotation_acres"] > 0]
    merged["corn_share"] = merged["acres_corn"] / merged["rotation_acres"]

    merged = merged.sort_values(["fips", "year"])
    previous = merged.groupby("fips")["corn_share"].shift(1)
    merged["soy_share_prev"] = 1.0 - previous

    # A gap in the record makes "last year" wrong rather than missing, so only
    # a genuinely consecutive year counts.
    consecutive = merged.groupby("fips")["year"].diff() == 1
    merged.loc[~consecutive, "soy_share_prev"] = pd.NA
    merged["soy_share_prev"] = merged["soy_share_prev"].astype("Float64").astype(float)

    return merged[cols].reset_index(drop=True)


def fill_gaps(features: pd.DataFrame, years: "object" = None) -> pd.DataFrame:
    """Carry a county's last observed rotation forward across suppressed years.

    WHY THIS EXISTS, WHICH IS NOT "TO MAKE THE NUMBERS NICER"
    ---------------------------------------------------------
    NASS suppresses a county-year when too few operations report it, and those are
    small, marginal counties. Leaving the gaps as missing drops those rows from the
    model - and dropping them is not neutral. Measured directly on Nebraska, the
    weather-only spatial R2 went from 0.316 on all 2,069 rows to 0.451 on the 1,694
    rows that survive, purely because the hard counties left. That +0.135 is larger
    than almost every feature in this project.

    So the choice is between a feature with gaps that silently changes which
    counties the model is scored on, and a feature that is complete but partly
    carried forward. The second is honest if - and only if - the carrying is
    flagged, which is what `rotation_observed` is for, and what lets a control run
    re-fit on observed rows only and check that nothing hinges on it.

    Carrying forward (rather than interpolating between) is deliberate: it uses no
    information from after the year being filled, so a temporal holdout cannot
    borrow from its own future.
    """
    if features.empty:
        return features

    out = features.sort_values(["fips", "year"]).copy()
    if years is not None:
        grid = pd.MultiIndex.from_product(
            [sorted(out["fips"].unique()), list(years)], names=["fips", "year"])
        out = (out.set_index(["fips", "year"]).reindex(grid).reset_index())

    out["rotation_observed"] = out["rotation_observed"].fillna(False).astype(bool)
    filled = out["corn_share"].isna()
    out["rotation_observed"] &= ~filled

    for column in ("corn_share", "soy_share_prev", "rotation_acres"):
        out[column] = out.groupby("fips")[column].ffill()

    # A county whose record never started has nothing to carry.
    return out.dropna(subset=["corn_share"]).reset_index(drop=True)


def harvested_ratio(planted: pd.DataFrame, harvested: pd.DataFrame,
                    silage: pd.DataFrame | None = None) -> pd.DataFrame:
    """Grain harvested over acres that were *meant* for grain: the abandonment signal.

    NOT a model feature - see the module docstring. This is the measurement that
    tests the prevented-planting hypothesis.

    SILAGE IS THE TRAP HERE, AND THE FIRST VERSION FELL IN IT
    ---------------------------------------------------------
    Planted acres cover grain and silage together; harvested acres are grain only.
    Dividing one by the other therefore subtracts the silage share from every
    county in every year - a county at 10% silage reads 0.90 with no abandonment
    at all - and Nebraska, with its feedlots and dairies, is a heavier silage state
    than Iowa.

    Worse, the silage share moves with the weather. Drought-stressed corn gets
    chopped for silage rather than left to fill: if the ears won't make grain you
    salvage the plant as feed. So an uncorrected ratio falls in dry years for a
    reason that is not crop failure, and reading it as abandonment turns a
    utilization decision into a disaster.

    Passing `silage` removes silage-harvested acres from the denominator, leaving
    the ratio measuring what it claims: of the acres intended for grain, how many
    made it. `silage_share` is returned alongside, because it is a decent read on a
    county's livestock intensity in its own right.
    """
    cols = ["fips", "year", "acres_planted", "acres_harvested", "acres_silage",
            "acres_for_grain", "harvested_ratio", "silage_share",
            "silage_corrected"]
    if planted.empty or harvested.empty:
        return pd.DataFrame(columns=cols)

    df = planted.merge(harvested, on=["fips", "year"], how="inner",
                       suffixes=("_planted", "_harvested"))
    df = df[df["acres_planted"] > 0].copy()

    if silage is not None and not silage.empty:
        df = df.merge(silage.rename(columns={"acres": "acres_silage"}),
                      on=["fips", "year"], how="left")
        df["acres_silage"] = df["acres_silage"].fillna(0.0)
        df["silage_corrected"] = True
    else:
        df["acres_silage"] = 0.0
        df["silage_corrected"] = False

    df["silage_share"] = df["acres_silage"] / df["acres_planted"]
    df["acres_for_grain"] = df["acres_planted"] - df["acres_silage"]

    # A county can report more silage than planted in a thin year - rounding, or
    # acres bought standing from a neighbour. Those rows cannot be interpreted.
    df = df[df["acres_for_grain"] > 0]
    df["harvested_ratio"] = df["acres_harvested"] / df["acres_for_grain"]

    return df[cols].reset_index(drop=True)
