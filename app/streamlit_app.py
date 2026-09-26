"""Yield Predictor - home page.

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
                               load_yields, missing_data_message,
                               state_yield_history)

from yieldpred.disclaimer import FOOTER
from yieldpred.brand import page_footer, page_heading, page_setup, start_here, state_name
state = page_setup("Home")


@st.cache_data
def data(state: str):
    """Keyed on the state: without that argument the cache would hand Iowa
    Nebraska's tables, which is the quietest possible way to be wrong."""
    return (load_yields(state), load_scores(state), load_morans_pooled(state))


yields, scores, morans = data(state)

page_heading("Home", show_title=False)
start_here()

message = missing_data_message({"yields": yields is not None, "model": scores is not None})
if message:
    st.warning(message)
    st.stop()

# ------------------------------------------------------------------ headline
# The tiles are built from whatever the state actually has, rather than being a
# fixed set of three. Iowa has no irrigated/non-irrigated split (it barely
# irrigates) and no saved Moran's I, so a hardcoded row left an empty strip
# under the Iowa heading - a blank band is worse than a different tile.
tiles = []
gap = irrigation_gap(yields)
if not gap.empty:
    worst = gap.loc[gap["gap"].idxmax()]
    tiles.append(("Irrigation advantage", f"{gap['gap'].mean():.0f} bu/acre", None,
                  "Average yield difference between irrigated and non-irrigated corn "
                  "in counties reporting both, 2000–2018"))
    tiles.append((f"In {int(worst['year'])} (drought)", f"{worst['gap']:.0f} bu/acre",
                  f"{worst['gap'] - gap['gap'].mean():+.0f} vs average",
                  "Irrigation matters most in the years it is needed most"))

if morans is not None:
    tiles.append(("Errors cluster (Moran's I)", f"{morans['morans_i'].iloc[0]:+.2f}", None,
                  "Model errors are geographically clustered, which means a "
                  "spatially-varying driver is still missing"))

# Generic tiles, used to fill the row for a state that has no irrigation split.
# Deliberately figures that are read straight off the yield table rather than
# model scores: the scores table reports the *model comparison* and the feature
# ladder reports different numbers, and a headline tile that quietly disagreed
# with the Model & Validation page would be worse than no tile.
if yields is not None and not yields.empty:
    tiles.append(("Counties", f"{yields['fips'].nunique():,}", None,
                  "Counties with a reported corn yield in this state"))
    tiles.append(("Years covered", f"{int(yields['year'].min())}–{int(yields['year'].max())}",
                  None, "Span of USDA NASS county yield estimates used here"))
    tiles.append(("Average yield", f"{yields['yield_bu_acre'].mean():.0f} bu/acre", None,
                  "Mean across every county and year in the table, all practices"))

for column, (label, value, delta, note) in zip(st.columns(min(len(tiles), 3)), tiles[:3]):
    column.metric(label, value, delta=delta, help=note)

st.divider()

# ------------------------------------------------------------------ the story
story, aside = st.columns([3, 2])   # "aside" since the app no longer has a sidebar

with story:
    # Two states, two stories. The Nebraska text was being shown under an "Iowa"
    # heading the moment the state switch existed, which made half of it false -
    # Iowa barely irrigates and has no east-west moisture gradient to speak of.
    st.subheader("What This Project Does")
    if state == "ne":
        st.markdown("""
Nebraska is a natural experiment. It is a top corn state, it is heavily irrigated,
and rainfall drops sharply from east to west — so the same weather produces very
different outcomes depending on whether a field has water.
""")
    else:
        st.markdown("""
Iowa is the control. It is the biggest corn state, it is almost entirely rainfed,
and its soils are some of the most uniformly productive in the world — so the
things that dominate in Nebraska barely move here, and what's left has to explain
the yield on its own.
""")

    st.markdown("""
This project predicts **county corn yields** from growing-season weather, soils and
the long-run yield trend, then asks how much of that prediction can be trusted.
Four questions get separate answers:

1. Can it fill in gaps for counties it has seen? *(random cross-validation)*
2. Can it predict a county it has **never** seen? *(hold out whole districts)*
3. Can it predict a year that hasn't happened? *(train through 2018, test after)*
4. Can it predict **a different state entirely?** *(train on Nebraska, test on Iowa)*

Those answers differ a lot, and the gap between them is the most useful thing here.
""")

    st.subheader("What The Data Says")
    if state == "ne":
        st.markdown("""
- **Irrigation is worth about 85 bu/acre on average** — and roughly 140 in the 2012
  drought. Irrigation's value rises exactly when weather turns bad.
- **Extreme heat matters more than total heat.** Days above 32°C correlate with yield
  at −0.50; accumulated growing degree days only −0.11.
- **Soil quality makes the model worse here.** Removing the NCCPI soil rating *improves*
  both scores, in four independent runs. Nebraska's soil signal is already carried by
  irrigation, which it correlates with at −0.81.
- **The model's errors form a map.** They cluster in all 26 years, overpredicting in
  the sandy, high-elevation west and underpredicting in the loess-soil northeast.
""")
    else:
        st.markdown("""
- **Crop rotation is the best feature after weather** — worth **+0.115** unseen-county
  R², against +0.006 for the same column in Nebraska. Where irrigation isn't masking
  it, what you planted last year matters.
- **Spring weather is a trade, not a win.** Including it scores better on unseen
  counties (0.781 against 0.769) and distinctly worse on unseen *years* (0.342 against
  0.480). Which model is right depends on what you want it for.
- **That 0.781 is optimistic by about 0.023.** Iowa counties are smaller than a NASA
  POWER weather cell, so 44% of them share byte-identical weather with a county in
  another district — the "unseen" county wasn't quite unseen. Corrected: about 0.758.
- **Something changed after 2018.** Every spring-weather relationship weakened, vanished
  or flipped sign, and none of the explanations tried so far survive. Nebraska's
  equivalents are stable and point the opposite way.
""")

with aside:
    st.subheader("Statewide yields")
    trend = state_yield_history(yields)
    st.line_chart(trend, y_label="bu/acre", height=260)
    st.caption("Irrigated and non-irrigated series end in 2018, when USDA stopped "
               "publishing county estimates by practice.")
    # The "Explore" list of page links used to sit here. It moved into the menu
    # across the top of every page (yieldpred.brand.nav_bar), so this column is
    # the chart it started as and nothing else. The data-sources line that used
    # to follow moved into the green footer band for the same reason: one copy,
    # on every page, instead of one copy on the home page.

# ------------------------------------------------------------------ disclaimer
st.divider()
st.caption(FOOTER)
page_footer()
