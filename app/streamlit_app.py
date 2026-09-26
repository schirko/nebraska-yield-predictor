"""Nebraska Corn Yield Predictor - home page.

Run from the project root:  streamlit run app/streamlit_app.py

The app reads pre-computed files from data/processed/. It never calls an API or
trains a model, so it starts instantly and needs no API key.
"""

import sys
from pathlib import Path

# Make src/ importable even if the editable install (-e .) didn't take, which can
# happen on hosted runtimes. Harmless locally, where the install does work.
_SRC = Path(__file__).resolve().parents[1] / "src"
if str(_SRC) not in sys.path:
    sys.path.insert(0, str(_SRC))


import streamlit as st

from yieldpred.appdata import (irrigation_gap, load_morans_pooled, load_scores,
                               load_yields, missing_data_message, state_yield_history)

from yieldpred.disclaimer import FOOTER
from yieldpred.brand import PAGE_ICON, show_logo
st.set_page_config(page_title="Nebraska Corn Yield Predictor", page_icon=PAGE_ICON,
                   layout="wide")
show_logo()


@st.cache_data
def data():
    return load_yields(), load_scores(), load_morans_pooled()


yields, scores, morans = data()

st.title("Nebraska Corn Yield Predictor")
st.caption("Predicting county corn yields from weather, irrigation and soils — "
           "and being honest about how well that works")

message = missing_data_message({"yields": yields is not None, "model": scores is not None})
if message:
    st.warning(message)
    st.stop()

# ------------------------------------------------------------------ headline
left, mid, right = st.columns(3)
gap = irrigation_gap(yields)
if not gap.empty:
    left.metric("Irrigation advantage", f"{gap['gap'].mean():.0f} bu/acre",
                help="Average yield difference between irrigated and non-irrigated corn "
                     "in counties reporting both, 2000–2018")
    worst = gap.loc[gap["gap"].idxmax()]
    mid.metric(f"In {int(worst['year'])} (drought)", f"{worst['gap']:.0f} bu/acre",
               delta=f"{worst['gap'] - gap['gap'].mean():+.0f} vs average",
               help="Irrigation matters most in the years it is needed most")
if morans is not None:
    right.metric("Errors cluster (Moran's I)", f"{morans['morans_i'].iloc[0]:+.2f}",
                 help="Model errors are geographically clustered, which means a "
                      "spatially-varying driver is still missing")

st.divider()

# ------------------------------------------------------------------ the story
story, sidebar = st.columns([3, 2])

with story:
    st.subheader("What this project does")
    st.markdown("""
Nebraska is a natural experiment. It is a top corn state, it is heavily irrigated,
and rainfall drops sharply from east to west — so the same weather produces very
different outcomes depending on whether a field has water.

This project predicts **county corn yields** from growing-season weather, irrigation
share and the long-run yield trend, then asks how much of that prediction can be
trusted. Three questions get separate answers:

1. Can it fill in gaps for counties it has seen? *(random cross-validation)*
2. Can it predict a county it has **never** seen? *(hold out whole districts)*
3. Can it predict a year that hasn't happened? *(train through 2018, test after)*
4. Can it predict **a different state entirely?** *(train on Nebraska, test on Iowa)*

Those answers differ a lot, and the gap between them is the most useful thing here.
""")

    st.subheader("What the data says")
    st.markdown("""
- **Irrigation is worth about 85 bu/acre on average** — and roughly 140 in the 2012
  drought. Irrigation's value rises exactly when weather turns bad.
- **Extreme heat matters more than total heat.** Days above 32°C correlate with yield
  at −0.50; accumulated growing degree days only −0.11.
- **The model's errors form a map.** They cluster in all 26 years, overpredicting in
  the sandy, high-elevation west and underpredicting in the loess-soil northeast —
  which is what pointed at soil quality and elevation as the features to add next.
- **It transfers to Iowa.** Trained on Nebraska and applied to Iowa counties it had
  never seen — without irrigation, which Iowa doesn't do — it scored R² **0.50**
  against a floor of −0.26 and an Iowa-native ceiling of 0.65, and closed 70% of the
  14 bu/acre gap between the two states from its inputs alone.
""")

with sidebar:
    st.subheader("Statewide yields")
    trend = state_yield_history(yields)
    st.line_chart(trend, y_label="bu/acre", height=260)
    st.caption("Irrigated and non-irrigated series end in 2018, when USDA stopped "
               "publishing county estimates by practice.")

    st.subheader("Explore")
    st.page_link("pages/1_Maps.py", label="Maps", icon="🗺️")
    st.page_link("pages/2_County_Explorer.py", label="County Explorer", icon="🔎")
    st.page_link("pages/3_Model_and_Validation.py", label="Model & Validation", icon="📊")
    st.page_link("pages/4_How_It_Works.py", label="How It Works", icon="🛠️")
    st.page_link("pages/5_Does_It_Transfer.py", label="Does It Transfer?", icon="🔁")

st.divider()
st.caption("Data: USDA NASS Quick Stats · NASA POWER · US Census Bureau. "
           "Built with Python, scikit-learn, GeoPandas and Streamlit.")

# ------------------------------------------------------------------ disclaimer
st.divider()
st.caption(FOOTER)
