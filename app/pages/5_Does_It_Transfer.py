"""The hardest tests in the app: train on Nebraska, predict Iowa (corn); train on corn, predict
soybeans (soybeans)."""

import sys
from pathlib import Path

# Make src/ importable even if the editable install (-e .) didn't take, which can
# happen on hosted runtimes. Harmless locally, where the install does work.
_SRC = Path(__file__).resolve().parents[2] / "src"
if str(_SRC) not in sys.path:
    sys.path.insert(0, str(_SRC))


import pandas as pd
import streamlit as st

from yieldpred.appdata import (load_crop_transfer, load_crop_transfer_forward, county_skill, load_counties, load_cross_state,
                               load_cross_state_scores, load_errors,
                               load_irrigation_comparison, load_model_table,
                               transfer_by_year, variance_decomposition)
from yieldpred.disclaimer import FOOTER
from yieldpred.brand import current_crop, page_footer, page_heading, page_setup
from yieldpred.geo import display_geometry
from yieldpred.spatial import align_to_geometry
from yieldpred.viz import interactive_choropleth
from yieldpred.yearcharts import year_chart

page_setup("Does It Transfer?")
crop = current_crop()

# ------------------------------------------------------ soybeans: corn -> soybeans
if crop == "soybeans":
    page_heading("Does It Transfer?")
    st.markdown("""
For soybeans the transfer question is between crops: **does what the corn model learned help
predict soybeans?** The same counties grow both, often on the same field in alternate years, so
if the two crops answered weather the same way, the corn model would do the soybean model's job.

Four ways to predict soybean yields in districts the model never saw:

1. **Soybean model:** trained on soybeans. The benchmark.
2. **Corn model, applied cold:** no soybean weather training at all; the corn model's \"how far
   above or below trend\" is applied to the soybean trend.
3. **Soybean model + corn signal:** the corn model's prediction as one more input.
4. **One model for both crops:** trained on both at once, so each crop's data helps the other.
""")
    with st.expander("In plain words: what is a \"model for both crops\"?", expanded=True):
        st.markdown("""
Nothing changes on the farm. It is only about which past records the computer learns from.

Think of a model as a student studying old report cards: each one is a county's yield for one year,
next to that year's weather.

- **The soybean model** studied only soybean report cards.
- **The model for both crops** studied the soybean *and* the corn report cards, each labeled with
  its crop. To make them comparable, each yield is turned into "how far above or below normal for
  that crop", since corn yields about three times the bushels.

Why add corn? Nebraska has had only a few really bad drought years, like 2012 and 2022, so a
soybean-only student has seen very few examples of how bad things can get. The corn records add
more drought examples, and in Nebraska a dry summer hurts both crops.

**The average** then asks both students and splits the difference: if one says 52 bu/acre and the
other 48, the app shows 50. One knows soybeans best; the other has seen more droughts. Together
they miss less than either alone.
""")
    for code, name in (("ne", "Nebraska"), ("ia", "Iowa")):
        table = load_crop_transfer(code)
        st.subheader(name)
        if table is None:
            st.info("Run `python scripts/crop_transfer.py" + ("" if code == "ne" else " --state IA --state-fips 19")
                    + "` to build this comparison.")
            continue
        show = table.rename(columns={"approach": "Approach", "RMSE": "Typical miss (RMSE, bu/acre)",
                                     "vs soybean model": "vs soybean model (bu/acre)",
                                     "90% interval": "90% interval (season resampling)",
                                     "share of draws better": "Better in"})
        st.dataframe(show, hide_index=True, width="stretch")
        forward = load_crop_transfer_forward(code)
        if forward is not None:
            st.markdown("**The harder test: next season** (trained only on earlier seasons)")
            st.dataframe(forward.rename(columns={
                "approach": "Approach", "RMSE": "Typical miss (RMSE, bu/acre)",
                "vs soybean model": "vs soybean model (bu/acre)", "90% interval": "90% interval (season resampling)",
                "share of draws better": "Better in", "seasons won": "Seasons won"}), hide_index=True, width="stretch")
    st.caption("Negative means better than the soybean model. The interval comes from resampling whole "
               "seasons 2,000 times, because misses within one season move together.")
    st.markdown("""
**What it means.** In **Nebraska**, corn's data teaches soybeans something: the two crops share the
same droughts, and even the corn model applied cold comes close. The two-crop model alone is
erratic from season to season, winning big in unusual seasons (the 2012 and 2022 droughts) and
losing a little in ordinary ones. **Averaging it with the soybean model** keeps most of the win: it
is better on both tests, and it is what this app shows for Nebraska soybeans. (It was picked after
seeing the next-season result; the unseen-district test is the independent check, and it agreed.) In **Iowa** nothing beats the soybean model,
and corn applied cold is about 30% worse: soybeans there depend on August rain, when they fill
their pods, and corn on July rain, when it pollinates. **Transfer is a hypothesis to test, not a
free lunch**, and the same test gives opposite answers one state apart.
""")
    st.divider()
    st.caption(FOOTER)
    page_footer()
    st.stop()


@st.cache_data
def data():
    predictions = load_cross_state("ne", "ia")
    scores = load_cross_state_scores("ne", "ia")
    counties = load_counties("19")
    return (predictions, scores,
            display_geometry(counties) if counties is not None else None,
            load_model_table("ne"), load_model_table("ia"),
            load_errors("ne"), load_errors("ia"),
            load_irrigation_comparison("ia"))


(predictions, scores, iowa_shapes, ne_table, ia_table,
 ne_errors, ia_errors, ia_ladder) = data()

page_heading("Does It Transfer?")
st.markdown("""
Every other page answers one question: *how well does this model do on Nebraska?* This page
answers a harder one: **did it learn agronomy, or did it learn Nebraska?**

The test is to train on Nebraska only and predict Iowa — a state the model has never seen,
with better and far more uniform soils, more reliable rain, and **essentially no irrigation**,
which means the transfer runs without the single most valuable feature in the Nebraska model.
""")

if predictions is None or scores is None:
    st.warning("This page needs the transfer run. From the project root:\n\n"
               "```\npython scripts/fetch_nass_yields.py  --state IA --state-fips 19\n"
               "python scripts/fetch_weather.py      --state IA --state-fips 19\n"
               "python scripts/fetch_soil_terrain.py --state IA --state-fips 19\n"
               "python scripts/train_baseline.py     --state IA --state-fips 19\n"
               "python scripts/cross_state.py --train NE --test IA\n```")
    st.stop()

# ------------------------------------------------------------ the four numbers
st.subheader("Four yardsticks, and the gaps between them")

table = scores.set_index("model")
st.dataframe(table.round(2), width="stretch")

by_key = {}
for name in table.index:
    key = ("ceiling" if "own model" in name else
           "floor" if "average" in name else
           "cold" if "cold" in name else "corrected")
    by_key[key] = table.loc[name]

if {"floor", "ceiling", "cold"} <= by_key.keys():
    floor, ceiling, cold = by_key["floor"]["R2"], by_key["ceiling"]["R2"], by_key["cold"]["R2"]
    columns = st.columns(4)
    columns[0].metric("Floor (Nebraska's average)", f"{floor:+.3f}",
                      help="R² from knowing nothing about Iowa except Nebraska's mean yield")
    columns[1].metric("Cold transfer", f"{cold:+.3f}",
                      help="Trained entirely on Nebraska, applied to Iowa with no adjustment")
    columns[2].metric("Ceiling (Iowa's own model)", f"{ceiling:+.3f}",
                      help="What a model that got to study Iowa achieves")
    columns[3].metric("Distance floor → ceiling covered",
                      f"{(cold - floor) / (ceiling - floor):.0%}")

st.markdown("""
**A transfer score alone means nothing.** An R² on another state is impossible to judge until
it's bracketed, which is why all four rows are reported.

- **The floor** is what you'd get by applying Nebraska's average yield to Iowa. Its bias of
  **−14.1 bu/acre** is a real measurement: that is how much Iowa out-yields Nebraska.
- **The ceiling** is what a model trained *on Iowa* and honestly validated achieves. It is not
  1.0, and treating 1.0 as the target makes every transfer look like a failure.
- **The cold transfer** lands most of the way from one to the other — on a state it has never
  seen, without irrigation share.
- **Bias fell from −14.1 to −4.1.** The model recovered about **70% of the level difference
  between two states from its inputs alone**: it read Iowa's higher soil ratings and milder,
  wetter summers and raised its expectations. It did not memorize "Nebraska ≈ 161 bushels."
""")

with st.expander("Why the fourth row — 'after removing the average offset' — exists"):
    st.markdown("""
Subtracting the mean error separates two failures that a single R² blends together:

- A **calibration** failure — every prediction shifted by roughly the same constant. Cheap to
  fix: one season of local data, or a single offset term.
- A **pattern** failure — the model ranks counties and years wrongly. Expensive: it means the
  relationships themselves don't hold in the new place.

Here the cold and corrected scores are **nearly identical**, so almost none of the remaining
error is a level problem. That's the less comfortable answer and the more useful one. Had it
been 0.10 cold and 0.60 corrected, the honest conclusion would have been "transfers beautifully
in shape, needs re-levelling" — a completely different engineering task.
""")

# --------------------------------------------------------- variance structure
if ne_table is not None and ia_table is not None:
    st.subheader("Two states, two different problems")

    shares = pd.DataFrame(
        {"Nebraska": variance_decomposition(ne_table),
         "Iowa": variance_decomposition(ia_table)}).T
    shares.columns = ["Between years", "Between counties", "Residual"]
    st.dataframe(shares.style.format("{:.1%}"), width="stretch")

    st.markdown("""
Total county-year yield variation, split into the part explained by **which year it was**, the
part explained by **which county it is**, and what's left.

In Nebraska, *where* you farm matters about as much as *when* — because half the state
irrigates and half doesn't, one fact that splits the counties into two populations. In Iowa the
counties are far more alike (county average yields span 54 bu/acre; Nebraska's span 112), and
the dominant question is what kind of season it was.

**Iowa is a weather problem. Nebraska is a weather-and-geography problem.** That single
difference explains the anomaly below — and it's worth computing for any dataset with an
entity and a time dimension, because it tells you where your model's opportunity actually is.
""")

# ------------------------------------------------------------ the anomaly
if ia_ladder is not None:
    st.subheader("The anomaly: adding features made Iowa's spatial score worse")
    st.dataframe(ia_ladder.set_index("features").round(3), width="stretch")
    st.markdown("""
Spatial R² goes **down** as features are added (0.694 → 0.654) while temporal R² goes **up**
(0.161 → 0.357). In Nebraska both went up. Both movements here are real, and they have
different causes.

**Why spatial got worse.** Leave-district-out asks the model to predict counties it has never
seen. A county-level feature earns its place there only if it separates counties in a way that
generalizes. Iowa's soil rating runs 0.58–0.89 — nearly constant, a quarter as variable as
Nebraska's. It can't tell unseen counties apart, but it *can* give the trees extra places to
split and fit noise.

**Why temporal got better.** The temporal test keeps every county in training and asks about
new years. There, a value that never changes within a county works as a **county fingerprint**:
having seen one county's 0.81 next to its yields for nineteen years, the model can carry that
county's level forward. That's legitimate — a soil rating is known long before harvest — but
it's *identity*, not agronomy, and it's why the temporal score more than doubles.

> **The rule worth keeping:** a feature's value isn't a property of the feature. It's a
> property of the feature **and what you hold out**. Report both, or you'll believe whichever
> number flatters it.
""")

# ------------------------------------------------------- year-by-year transfer
st.subheader("Where the transfer holds, and where it breaks")

yearly = transfer_by_year(predictions)

tab_skill, tab_bias = st.tabs(["County ranking skill", "Level error"])

with tab_skill:
    st.altair_chart(year_chart(yearly.set_index("year")[["county_skill"]]
                               .rename(columns={"county_skill": "within-year correlation"}),
                               y_label="within-year correlation", height=280), width="stretch")
    medians = {"Nebraska's model on Nebraska": county_skill(ne_errors) if ne_errors is not None else None,
               "Iowa's own model on Iowa": county_skill(ia_errors) if ia_errors is not None else None,
               "Nebraska's model on Iowa": county_skill(predictions)}
    st.dataframe(
        pd.DataFrame([{"Model": k, "Median within-year county correlation": round(v, 2)}
                      for k, v in medians.items() if v is not None]),
        hide_index=True, width="stretch")
    st.markdown("""
Pooled R² rewards a model for knowing that 2012 was a bad year — which the trend and the
weather largely give away. The narrower question for a decision tool is: **within one year,
does it know which counties do better?**

Two Iowa years fail outright, and neither is random:

- **2013** — a record-wet spring; planting ran weeks late across much of the state.
- **2020** — the August 10 derecho, a straight-line windstorm that flattened corn across
  central Iowa.

The weather window starts in April, so a wet planting season is barely represented, and there
is **no wind variable in the feature set at all**. Both are missing-feature failures, not
modelling failures — and that's the most useful triage question in applied modelling: *when
the model missed, was the information even available to it?* Nebraska's 2019 flood year gave
the same answer independently, which is much stronger evidence than finding the same gap twice
in one place.
""")

with tab_bias:
    st.altair_chart(year_chart(yearly.set_index("year")[["mean_error"]]
                               .rename(columns={"mean_error": "mean error (bu/acre)"}),
                               y_label="mean error (bu/acre)", kind="bar", height=280), width="stretch")
    st.caption("Positive means the Nebraska-trained model predicted Iowa too high. Mean "
               "transfer error correlates −0.36 with Iowa's actual yield that year: the "
               "model under-predicts Iowa's best years and over-predicts its worst — "
               "classic shrinkage toward the distribution it was trained on.")
    st.dataframe(yearly.rename(columns={
        "year": "Year", "mean_error": "Mean error", "mean_abs_error": "Mean abs error",
        "county_skill": "County skill", "actual": "Actual yield",
        "counties": "Counties"}), hide_index=True, width="stretch", height=300)

# ------------------------------------------------------------------- the map
if iowa_shapes is not None:
    st.subheader("Where in Iowa the Nebraska model misses")

    years = sorted(predictions["year"].unique())
    choice = st.select_slider("Year (or the average across all years)",
                              options=["All years"] + [int(y) for y in years],
                              value="All years")

    if choice == "All years":
        values = (predictions.groupby(["fips", "county_name"], as_index=False)
                  .agg(error=("error", "mean"), yield_bu_acre=("yield_bu_acre", "mean"),
                       predicted=("predicted", "mean")))
    else:
        values = predictions[predictions["year"] == choice][
            ["fips", "county_name", "error", "yield_bu_acre", "predicted"]].copy()

    merged, _ = align_to_geometry(iowa_shapes, values.round(1), value_col="error")
    st.altair_chart(
        interactive_choropleth(
            merged, "error", "bu/acre (predicted − actual)",
            tooltips=[("county_name", "N", "County"), ("error", "Q", "Error"),
                      ("yield_bu_acre", "Q", "Actual"), ("predicted", "Q", "Predicted")],
            diverging=True))
    st.caption("Blue = the Nebraska model predicted too low, orange = too high. Hover a "
               "county for its numbers.")

st.divider()
st.markdown("""
#### Why this page exists

Most portfolio models stop at a random train/test split. The ladder of generalization runs:

1. **Random 5-fold** — predict rows nearly identical to ones you trained on (leaks).
2. **Leave-district-out** — predict counties you've never seen, same state, same years.
3. **Temporal holdout** — predict years that hadn't happened yet.
4. **Cross-state transfer** — predict a different place entirely.

Each rung removes a different kind of memorization. A model that survives the fourth has
learned relationships rather than a lookup table — and the honest way to report it is with the
floor and the ceiling beside it.
""")

# ------------------------------------------------------------------ disclaimer
st.divider()
st.caption(FOOTER)
page_footer()
