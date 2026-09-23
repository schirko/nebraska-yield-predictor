"""The results, and what they mean - including why the three numbers differ."""

import sys
from pathlib import Path

# Make src/ importable even if the editable install (-e .) didn't take, which can
# happen on hosted runtimes. Harmless locally, where the install does work.
_SRC = Path(__file__).resolve().parents[2] / "src"
if str(_SRC) not in sys.path:
    sys.path.insert(0, str(_SRC))


import streamlit as st

from yieldpred.appdata import (feature_correlations, load_irrigation_comparison,
                               load_model_table, load_morans_by_year, load_morans_pooled,
                               load_errors, load_scores, missing_data_message,
                               scores_by_scheme, worst_years)

st.set_page_config(page_title="Model & validation", page_icon="📊", layout="wide")


@st.cache_data
def data():
    return (load_scores(), load_model_table(), load_errors(),
            load_irrigation_comparison(), load_morans_by_year(), load_morans_pooled())


scores, model_table, errors, comparison, morans_year, morans_pooled = data()

st.title("Model & validation")

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
    st.subheader("What irrigation share was worth")
    st.dataframe(comparison.set_index("features").round(2), width="stretch")
    st.markdown("""
Same model, same folds, one feature added. The spatial score nearly doubled, from
**0.31 to 0.58** — a bigger gain than any amount of model tuning produced, and it came
from knowing something about Nebraska rather than about machine learning.

The third row rebuilds the feature using only observations from 2018 and earlier, testing
whether interpolating across the gap let future information leak backwards. It didn't:
the score held up.
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
leftovers would look like random static. Soil productivity and elevation are the leading
candidates, and adding them is the next step.
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
