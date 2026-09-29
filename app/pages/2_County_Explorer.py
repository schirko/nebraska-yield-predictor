"""One county at a time: its yields, its weather, and how the model did."""

import sys
from pathlib import Path

# Make src/ importable even if the editable install (-e .) didn't take, which can
# happen on hosted runtimes. Harmless locally, where the install does work.
_SRC = Path(__file__).resolve().parents[2] / "src"
if str(_SRC) not in sys.path:
    sys.path.insert(0, str(_SRC))


import altair as alt
import pandas as pd
import streamlit as st

from yieldpred.appdata import (county_choices, county_history, load_errors,
                               load_model_table, load_ranges, load_yields,
                               missing_data_message, range_record)

from yieldpred.disclaimer import FOOTER
from yieldpred.brand import current_crop, crop_word, page_footer, page_heading, page_setup
from yieldpred.yearcharts import year_chart
state = page_setup("County Explorer")
crop = current_crop()


@st.cache_data
def data(state: str, crop: str):
    return load_model_table(state, crop), load_errors(state, crop), load_yields(state, crop), load_ranges(state, crop)


model_table, errors, yields, ranges = data(state, crop)

page_heading("County Explorer")

message = missing_data_message({"model": model_table is not None,
                                "yields": yields is not None})
if message:
    st.warning(message)
    st.stop()

choices = county_choices(model_table)
name = st.selectbox("County", choices["county_name"])
fips = choices.loc[choices["county_name"] == name, "fips"].iloc[0]

history = county_history(model_table, fips)
county_errors = errors[errors["fips"] == fips] if errors is not None else pd.DataFrame()

# ------------------------------------------------------------------- summary
cols = st.columns(4)
cols[0].metric("Years of data", len(history))
cols[1].metric("Average yield", f"{history['yield_bu_acre'].mean():.0f} bu/acre")
if "irrigation_share" in history:
    cols[2].metric("Irrigated share", f"{history['irrigation_share'].iloc[0]:.0%}",
                   help=f"Share of harvested {crop_word(crop)} acres under irrigation, most recent year")
if not county_errors.empty:
    cols[3].metric("Model's typical miss", f"{county_errors['error'].abs().mean():.1f} bu/acre",
                   help="Average absolute error, predicted by a model that never saw "
                        "this county's district in training")

# ----------------------------------------------------------------- yield plot
st.subheader("Yield history")
county_ranges = ranges[ranges["fips"] == fips] if ranges is not None else pd.DataFrame()
if not county_ranges.empty:
    record = range_record(ranges, fips)
    base = alt.Chart(county_ranges).encode(x=alt.X("year:O", title=None))
    band = base.mark_area(opacity=0.22, color="#e08a2e").encode(
        y=alt.Y("low:Q", title="bu/acre", scale=alt.Scale(zero=False)), y2="high:Q",
        tooltip=[alt.Tooltip("year:O", title="Year"),
                 alt.Tooltip("low:Q", title="Range low"), alt.Tooltip("high:Q", title="Range high")])
    predicted = base.mark_line(color="#e08a2e", strokeDash=[5, 3]).encode(y="predicted:Q")
    actual = base.mark_line(color="#2f5d3a").encode(y="yield_bu_acre:Q")
    dots = base.mark_circle(size=55).encode(
        y="yield_bu_acre:Q",
        color=alt.Color("inside:N", title="Inside range",
                        scale=alt.Scale(domain=[True, False], range=["#2f5d3a", "#c0392b"]),
                        legend=alt.Legend(orient="bottom", labelExpr="datum.value ? 'Yes' : 'No'")),
        tooltip=[alt.Tooltip("year:O", title="Year"),
                 alt.Tooltip("yield_bu_acre:Q", title="Actual"),
                 alt.Tooltip("predicted:Q", title="Predicted"),
                 alt.Tooltip("low:Q", title="Range low"), alt.Tooltip("high:Q", title="Range high")])
    st.altair_chart((band + predicted + actual + dots).properties(height=320), width="stretch")
    st.markdown(f"**Range held {record['held']} of {record['years']} years.** "
                "The shaded band is the 80% range around each prediction (dashed line); "
                "the solid line is what the county really harvested. A red dot is a year it "
                "landed outside.")
    st.caption("An honest 80% range should miss about 1 year in 5. Missing none is not "
               "better — it means the band is wider than it needs to be. See Model & "
               "Validation → Honest Ranges for how the ranges are built and checked.")
elif not county_errors.empty:
    chart = (county_errors.set_index("year")[["yield_bu_acre", "predicted"]]
             .rename(columns={"yield_bu_acre": "Actual", "predicted": "Predicted"}))
    st.altair_chart(year_chart(chart, y_label="bu/acre", height=300), width="stretch")
    st.caption("Where the lines diverge, the model was surprised — usually a year when "
               "something outside the weather window mattered.")
else:
    st.altair_chart(year_chart(history.set_index("year")[["yield_bu_acre"]].rename(columns={"yield_bu_acre": "Yield"}),
                               y_label="bu/acre"), width="stretch")

# ------------------------------------------------------- irrigated vs dryland
practice = yields[yields["fips"] == fips]
paired = (practice[practice["practice"] != "all"]
          .pivot_table(index="year", columns="practice", values="yield_bu_acre"))
if not paired.dropna().empty:
    st.subheader("Irrigated vs. non-irrigated")
    st.altair_chart(year_chart(paired.rename(columns={"irrigated": "Irrigated",
                                                      "non_irrigated": "Non-irrigated"}),
                               y_label="bu/acre", height=280), width="stretch")
    gap = (paired["irrigated"] - paired["non_irrigated"]).dropna()
    if not gap.empty:
        st.caption(f"Average advantage in this county: **{gap.mean():.0f} bu/acre**. "
                   f"Largest: **{gap.max():.0f}** in {int(gap.idxmax())}. "
                   "USDA stopped publishing this breakdown after 2018.")

# ------------------------------------------------------------------- weather
st.subheader("Growing-season weather")
weather_cols = [c for c in ["precip_mm", "precip_jul_mm", "heat_days_32", "dry_spell_max",
                            "precip_mar_mm", "precip_apr_may_mm", "workable_days",
                            "last_frost_doy", "gdd_may"]
                if c in history]
labels = {"precip_mm": "Season rainfall (mm)", "precip_jul_mm": "July rainfall (mm)",
          "heat_days_32": "Days above 32°C", "dry_spell_max": "Longest dry spell (days)",
          "precip_mar_mm": "March rainfall (mm)",
          "precip_apr_may_mm": "April–May rainfall (mm)",
          "workable_days": "Workable planting days",
          "last_frost_doy": "Last spring frost (day of year)",
          "gdd_may": "May growing degree days"}
picked = st.selectbox("Variable", weather_cols, format_func=lambda c: labels[c])
st.altair_chart(year_chart(history.set_index("year")[[picked]].rename(columns=labels),
                           y_label=labels[picked], kind="bar", height=260), width="stretch")

with st.expander("Full record"):
    st.dataframe(history, hide_index=True, width="stretch")

# ------------------------------------------------------------------ disclaimer
st.divider()
st.caption(FOOTER)
page_footer()
