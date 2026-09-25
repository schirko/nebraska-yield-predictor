"""One county at a time: its yields, its weather, and how the model did."""

import sys
from pathlib import Path

# Make src/ importable even if the editable install (-e .) didn't take, which can
# happen on hosted runtimes. Harmless locally, where the install does work.
_SRC = Path(__file__).resolve().parents[2] / "src"
if str(_SRC) not in sys.path:
    sys.path.insert(0, str(_SRC))


import pandas as pd
import streamlit as st

from yieldpred.appdata import (county_choices, county_history, load_errors,
                               load_model_table, load_yields, missing_data_message)

st.set_page_config(page_title="County Explorer", page_icon="🔎", layout="wide")


@st.cache_data
def data():
    return load_model_table(), load_errors(), load_yields()


model_table, errors, yields = data()

st.title("County Explorer")

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
                   help="Share of harvested corn acres under irrigation, most recent year")
if not county_errors.empty:
    cols[3].metric("Model's typical miss", f"{county_errors['error'].abs().mean():.1f} bu/acre",
                   help="Average absolute error, predicted by a model that never saw "
                        "this county's district in training")

# ----------------------------------------------------------------- yield plot
st.subheader("Yield history")
if not county_errors.empty:
    chart = (county_errors.set_index("year")[["yield_bu_acre", "predicted"]]
             .rename(columns={"yield_bu_acre": "Actual", "predicted": "Predicted"}))
    st.line_chart(chart, y_label="bu/acre", height=300)
    st.caption("Where the lines diverge, the model was surprised — usually a year when "
               "something outside the weather window mattered.")
else:
    st.line_chart(history.set_index("year")[["yield_bu_acre"]], y_label="bu/acre")

# ------------------------------------------------------- irrigated vs dryland
practice = yields[yields["fips"] == fips]
paired = (practice[practice["practice"] != "all"]
          .pivot_table(index="year", columns="practice", values="yield_bu_acre"))
if not paired.dropna().empty:
    st.subheader("Irrigated vs. non-irrigated")
    st.line_chart(paired.rename(columns={"irrigated": "Irrigated",
                                         "non_irrigated": "Non-irrigated"}),
                  y_label="bu/acre", height=280)
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
st.bar_chart(history.set_index("year")[[picked]].rename(columns=labels),
             y_label=labels[picked], height=260)

with st.expander("Full record"):
    st.dataframe(history, hide_index=True, width="stretch")
