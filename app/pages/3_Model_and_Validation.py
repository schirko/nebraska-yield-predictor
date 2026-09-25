"""The results, and what they mean - including why the three numbers differ."""

import sys
from pathlib import Path

# Make src/ importable even if the editable install (-e .) didn't take, which can
# happen on hosted runtimes. Harmless locally, where the install does work.
_SRC = Path(__file__).resolve().parents[2] / "src"
if str(_SRC) not in sys.path:
    sys.path.insert(0, str(_SRC))


import streamlit as st

from yieldpred.appdata import (feature_correlations, load_ablation,
                               load_irrigation_comparison, load_model_table,
                               load_morans_by_year, load_morans_pooled,
                               load_errors, load_scores, missing_data_message,
                               scores_by_scheme, worst_years)

from yieldpred.disclaimer import FOOTER
st.set_page_config(page_title="Model & Validation", page_icon="📊", layout="wide")


@st.cache_data
def data():
    return (load_scores(), load_model_table(), load_errors(),
            load_irrigation_comparison(), load_ablation(),
            load_morans_by_year(), load_morans_pooled())


scores, model_table, errors, comparison, ablation, morans_year, morans_pooled = data()

st.title("Model & Validation")

message = missing_data_message({"model": scores is not None})
if message:
    st.warning(message)
    st.stop()

# --------------------------------------------------------------- the results
st.subheader("The same model, scored three ways")
st.markdown("""
A model's accuracy depends on what you ask it to do. These three columns use identical
models and identical data — only the choice of what to hide during training changes.
""")

scheme = st.radio(
    "Validation scheme",
    [("random", "Random 5-fold"), ("spatial", "Leave-district-out"),
     ("future", "Train ≤2018, test ≥2019")],
    format_func=lambda pair: pair[1], horizontal=True)

EXPLANATIONS = {
    "random": ("**Optimistic, and misleading here.** A random split puts some counties' "
               "neighbours from the same year in the training data. Those rows are near "
               "duplicates, so the model has effectively seen the answer."),
    "spatial": ("**The fair spatial test.** Whole agricultural districts are held out, so "
                "the model must predict counties it has never encountered, using weather "
                "and irrigation rather than familiarity."),
    "future": ("**The forecasting test.** Train on 2000–2018, predict 2019 onward. This is "
               "what a real tool faces, and it exposes models that cannot extrapolate."),
}
st.info(EXPLANATIONS[scheme[0]])
st.dataframe(scores_by_scheme(scores, scheme[0]), width="stretch")

st.caption("RMSE and MAE are in bu/acre — lower is better. R² is the share of variation "
           "explained: 1.0 is perfect, 0 means no better than always guessing the average, "
           "and negative means worse than that guess.")

with st.expander("Is this any good? Four yardsticks"):
    st.markdown("""
"Is R² 0.64 good?" has no answer in the abstract. It has four in context.

**1. Against the baseline.** A dummy that always predicts the average scores −0.06 here, and
weather alone got 0.31. The score is only meaningful as a distance from the floor.

**2. In the units of the problem.** Spatial RMSE of about 19 bu/acre against yields averaging
161 means typical misses near **12%** — for a county the model has never seen, knowing only
its weather, irrigation share, soil rating and elevation.

**3. Under which test.** These come from holding out whole agricultural districts, and from
predicting years after the training cutoff. Published crop-yield studies often report similar
or higher numbers using random splits — which aren't comparable, since this project's own
random-split score is higher too. Comparing an honest number to an optimistic one is a
category error.

**4. How the errors behave.** They're mixed in sign rather than all overpredictions, so there
is little systematic bias left. And every gain came with a mechanism — irrigation buffers
drought, soil holds water, elevation shortens the season — which is more trustworthy than a
gain from tuning.

#### Where "good" stops

- **Not good enough to act on commercially.** A 20 bu/acre miss at roughly $4.50/bushel is
  about $90 an acre. No grain merchandiser or crop insurer would trade on it.
- **The errors still cluster geographically**, so something spatial is still missing.
- **It does do worse outside Nebraska.** Applied cold to Iowa it scores 0.50 against an
  Iowa-native 0.65 — see the *Does It Transfer?* page. Good evidence it learned agronomy, and
  equally good evidence that a one-state model is not a Corn Belt model.

The defensible claim is *good for a project built from public data, honestly validated, with
its limits stated* — not *good enough to deploy*.
""")

with st.expander("Why does the same model score so differently?"):
    st.markdown("""
**Gradient boosting scores 0.57, 0.40 and −0.42** across the three schemes.

The drop from 0.57 to 0.40 is **spatial autocorrelation**: nearby counties resemble each
other, so a random split leaks. The honest number for "predict an unseen county" is 0.40.

The collapse to −0.42 is different — it's **extrapolation**. A tree predicts by averaging
training rows, so it can never output a value outside the range it saw. Trained through
2018, it treats 2024 as if it were 2018 and misses six years of yield trend. Ridge
regression fits a slope and keeps extending it, which is why the simpler model wins there.

**Detrended boosting** fixes this by fitting a straight-line trend first and letting the
trees model only the deviation from it — the linear part extrapolates, the trees handle
interactions. That moved the future score from −0.42 to −0.05, and to **+0.35** once
irrigation share was added.
""")

# --------------------------------------------------------- irrigation effect
if comparison is not None:
    st.subheader("What each feature was worth")
    st.dataframe(comparison.set_index("features").round(3), width="stretch")
    st.markdown("""
Each row adds one feature to the row above — same model, same folds, same rows, so every
difference is attributable to the one column that changed.

**Irrigation share nearly doubled the spatial score, from 0.31 to 0.58.** That is a bigger
gain than any amount of model tuning produced, and it came from knowing something about
Nebraska rather than about machine learning.

**The soil rating (NCCPI) is second, and its effect is lopsided:** about +0.02 spatially but
**+0.18** on future years. A county's soil rating never changes, so once the model has seen a
county in training it can use that value to carry the county's level forward — useful, and
legitimate, but closer to identity than to agronomy.

**Elevation adds nothing measurable** once the others are present. That's evidence it's
*redundant* here, not that it's meaningless.

The last row rebuilds irrigation share using only observations from 2018 and earlier, testing
whether interpolating across the gap let future information leak backwards. It didn't — the
score held up, and if anything improved, because a share that stops updating is steadier and
centre-pivot acreage barely moves year to year.
""")

if ablation is not None:
    with st.expander("A ladder isn't enough: what each feature is worth on its own"):
        st.dataframe(ablation.set_index("removed").round(3), width="stretch")
        st.markdown("""
The table above builds features **up** one at a time; this one takes each feature **out** of
the full set and leaves the rest.

The two answer different questions, and with correlated features they disagree. A feature
added first can look enormous in the ladder and tiny in the ablation — because by then
another feature has learned to stand in for it. The ladder measures what a feature *adds to
what came before*; ablation measures what is *lost when nothing else can substitute*.

A feature that barely changes the score when removed is **redundant**, which is not the same
as unimportant. Reporting only one of these tables is how features get over- or under-sold.
""")

# ------------------------------------------------------------------ features
if model_table is not None:
    st.subheader("Which inputs carry signal")
    corr = feature_correlations(model_table)
    st.bar_chart(corr.set_index("label")["correlation"], height=320,
                 x_label="correlation with yield")
    st.markdown("""
**The instructive pair:** days above 32°C correlate at about **−0.50**, while accumulated
growing degree days manage only **−0.11**. Both measure heat. Corn tolerates warm weather
but loses pollination during heat spikes, so *extreme* heat matters and *average* heat
mostly doesn't. How you summarize a variable often matters more than which variable you
picked.

Correlation only measures straight-line relationships with one variable at a time, so it
understates irrigation share, whose importance comes from its **interaction** with drought.
""")

# ------------------------------------------------------------------- spatial
if morans_pooled is not None and morans_year is not None:
    st.subheader("Do the errors cluster?")
    pooled = morans_pooled.iloc[0]
    cols = st.columns(3)
    cols[0].metric("Moran's I", f"{pooled['morans_i']:+.3f}",
                   help="Near +1 means neighbouring counties have similar errors; "
                        "0 means no spatial pattern")
    cols[1].metric("Expected if random", f"{pooled['expected']:+.3f}")
    cols[2].metric("Years clustered",
                   f"{int((morans_year['p_value'] < 0.05).sum())} of {len(morans_year)}")

    st.line_chart(morans_year.set_index("year")[["morans_i"]]
                  .rename(columns={"morans_i": "Moran's I"}), height=260)
    st.markdown("""
Clustered errors are a **diagnostic, not a verdict**. They mean some spatially-varying
driver is missing from the features — if the model had captured everything geographic, the
leftovers would look like random static.

The east–west gradient in the error map is what prompted adding soil productivity and
elevation. Both helped, and the errors still cluster, so something spatial remains: the
leading suspects now are where the corn actually grows within each county (cropland-weighted
weather) and the spring planting window.
""")

# ------------------------------------------------------------- worst years
if errors is not None:
    st.subheader("Where the model misses worst")
    st.dataframe(worst_years(errors).rename(columns={
        "year": "Year", "mean_error": "Mean error (bu/acre)", "counties": "Counties"}),
        hide_index=True, width="stretch")
    st.caption("Positive means the model predicted too high. 2019 was Nebraska's March "
               "flood year — planting was delayed or prevented, which no April–September "
               "weather feature can see.")

# ------------------------------------------------------------------ disclaimer
st.divider()
st.caption(FOOTER)
