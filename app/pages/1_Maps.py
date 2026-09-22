"""Maps: yields, predictions and where the model misses."""

import pandas as pd
import streamlit as st

from yieldpred.appdata import load_counties, load_errors, missing_data_message
from yieldpred.viz import choropleth

st.set_page_config(page_title="Maps", page_icon="🗺️", layout="wide")


@st.cache_data
def data():
    return load_errors(), load_counties()


errors, counties = data()

st.title("Maps")

message = missing_data_message({"model": errors is not None,
                                "spatial": counties is not None})
if message:
    st.warning(message)
    st.stop()

LAYERS = {
    "Actual yield": ("yield_bu_acre", False, "bu/acre",
                     "What the county actually produced."),
    "Predicted yield": ("predicted", False, "bu/acre",
                        "What the model predicted, from a model that never saw this "
                        "county's district during training."),
    "Prediction error": ("error", True, "bu/acre (predicted − actual)",
                         "Orange means the model predicted too high, blue too low, "
                         "grey close to right."),
}

controls, _ = st.columns([2, 1])
with controls:
    choice = st.radio("Layer", list(LAYERS), horizontal=True)
    years = sorted(errors["year"].unique())
    year = st.select_slider("Year", options=["All years (average)"] + years,
                            value="All years (average)")

column, diverging, label, explanation = LAYERS[choice]

if year == "All years (average)":
    values = (errors.groupby(["fips", "county_name"], as_index=False)
              .agg(**{column: (column, "mean")}))
    subtitle = "Average, 2000–2025"
else:
    values = errors[errors["year"] == year][["fips", "county_name", column]]
    subtitle = str(year)

merged = counties.merge(values, on="fips", how="left")

fig = choropleth(merged, column, f"{choice} — {subtitle}", explanation,
                 "Data: USDA NASS Quick Stats, NASA POWER, US Census Bureau",
                 diverging=diverging, legend_label=label, figsize=(10, 5.6))
st.pyplot(fig, width="stretch")

with st.expander("How to read this map"):
    st.markdown("""
**Colour choices are deliberate.** Error is a *diverging* quantity — it has a
meaningful zero — so it gets two hues with neutral grey at zero, and the scale is
symmetric so grey lands exactly on nothing-wrong. Yields are a *magnitude*, so they
get a single hue from light to dark. Rainbow scales are avoided because people read
the bands as unordered categories.

**Counties in pale grey have no data** for the selected year. USDA publishes county
estimates only where enough farms report; coverage fell from 91 counties in 2000 to
46 in 2025.

**The error map is the interesting one.** If the model had learned everything spatial,
the errors would look like random static. They don't — they form an east–west gradient,
which says something real and geographic is still missing. Soil productivity and
elevation are the leading suspects.
""")

if year != "All years (average)":
    st.subheader(f"{year} in numbers")
    year_rows = errors[errors["year"] == year]
    cols = st.columns(4)
    cols[0].metric("Counties reporting", f"{year_rows['fips'].nunique()}")
    cols[1].metric("Average yield", f"{year_rows['yield_bu_acre'].mean():.0f} bu/acre")
    cols[2].metric("Average error", f"{year_rows['error'].mean():+.1f} bu/acre")
    cols[3].metric("Typical miss", f"{year_rows['error'].abs().mean():.1f} bu/acre")

    worst = year_rows.reindex(year_rows["error"].abs().sort_values(ascending=False).index)
    st.caption("Largest misses this year")
    st.dataframe(
        worst.head(8)[["county_name", "yield_bu_acre", "predicted", "error"]]
        .rename(columns={"county_name": "County", "yield_bu_acre": "Actual",
                         "predicted": "Predicted", "error": "Error"}),
        hide_index=True, width="stretch")
