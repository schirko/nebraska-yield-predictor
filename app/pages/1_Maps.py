"""Maps: yields, predictions, model errors, and the county characteristics behind them."""

import sys
from pathlib import Path

# Make src/ importable even if the editable install (-e .) didn't take, which can
# happen on hosted runtimes. Harmless locally, where the install does work.
_SRC = Path(__file__).resolve().parents[2] / "src"
if str(_SRC) not in sys.path:
    sys.path.insert(0, str(_SRC))


import streamlit as st

from yieldpred.appdata import (county_values, load_counties, load_errors,
                               load_model_table, map_frame, missing_data_message)
from yieldpred.disclaimer import FOOTER
from yieldpred.brand import page_footer, page_heading, page_setup
from yieldpred.geo import display_geometry
from yieldpred.viz import choropleth, interactive_choropleth

page_setup("Maps")


@st.cache_data
def data():
    errors = load_errors()
    counties = load_counties()
    frame = map_frame(errors, load_model_table()) if errors is not None else None
    # A lighter geometry for the interactive chart, which embeds its shapes in the page.
    display = display_geometry(counties) if counties is not None else None
    return frame, counties, display


frame, counties, display = data()

page_heading("Maps")

message = missing_data_message({"model": frame is not None,
                                "spatial": counties is not None})
if message:
    st.warning(message)
    st.stop()

# column, diverging, varies by year, legend label, explanation, tooltip format
LAYERS = {
    "Actual yield": ("yield_bu_acre", False, True, "bu/acre",
                     "What the county actually produced.", ".0f"),
    "Predicted yield": ("predicted", False, True, "bu/acre",
                        "What the model predicted — from a model that never saw this "
                        "county's district during training.", ".0f"),
    "Prediction error": ("error", True, True, "bu/acre (predicted − actual)",
                         "Orange = predicted too high, blue = too low, grey = close.",
                         "+.1f"),
    "Irrigation share": ("irrigation_share", False, True, "share of corn acres",
                         "Fraction of harvested corn acres under irrigation.", ".2f"),
    "Soil water capacity": ("soil_water_cm", False, False, "cm of water",
                            "Plant-available water the top 150 cm of soil can hold — "
                            "about 10 cm in Sandhills sand, 30+ in deep loess.", ".1f"),
    "Soil productivity (NCCPI)": ("nccpi_corn", False, False, "0–1 index",
                                  "USDA's rating of how productive the soil is for corn.",
                                  ".2f"),
    "Elevation": ("elevation_m", False, False, "metres",
                  "County centre elevation. Higher ground means cooler nights and a "
                  "shorter frost-free season.", ".0f"),
}
available = {name: spec for name, spec in LAYERS.items() if spec[0] in frame.columns}

# Side by side rather than stacked: the layer picker and the year slider are
# one decision, and stacking them pushed the map itself below the fold.
layer_col, year_col = st.columns([3, 2], gap="large")
with layer_col:
    choice = st.radio("Layer", list(available), horizontal=True)

column, diverging, by_year, label, explanation, fmt = available[choice]

if by_year:
    years = sorted(frame["year"].unique())
    with year_col:
        picked = st.select_slider("Year", options=["All years (average)"] + years,
                                  value="All years (average)")
    year = None if picked == "All years (average)" else picked
    subtitle = "Average, 2000–2025" if year is None else str(year)
else:
    year = None
    subtitle = "Fixed county characteristic — the same in every year"
    st.caption("This layer doesn't change by year, so the year slider is hidden.")

values = county_values(frame, column, year)
merged = display.merge(values, on="fips", how="left")
merged["County"] = merged["county_name_y"].fillna(merged["county_name_x"])

TOOLTIP_LABELS = {"yield_bu_acre": "Actual (bu/acre)", "predicted": "Predicted (bu/acre)",
                  "error": "Error (bu/acre)", "irrigation_share": "Irrigated share",
                  "soil_water_cm": "Soil water (cm)", "elevation_m": "Elevation (m)"}

tooltips = [("County", "N", "County"), (column, "Q", TOOLTIP_LABELS.get(column, choice))]
for extra in ("yield_bu_acre", "predicted", "error", "irrigation_share"):
    if extra != column and extra in merged.columns:
        tooltips.append((extra, "Q", TOOLTIP_LABELS[extra]))

st.subheader(f"{choice} — {subtitle}")
st.caption(explanation + "  Hover a county for its name and numbers.")

keep = ["County", "geometry"] + [t[0] for t in tooltips if t[0] != "County"]
chart = interactive_choropleth(merged[keep], column, legend_label=label,
                               tooltips=tooltips, diverging=diverging, value_format=fmt)
st.altair_chart(chart)

with st.expander("How to read this map"):
    st.markdown("""
**Colour choices are deliberate.** Error is a *diverging* quantity — it has a meaningful
zero — so it gets two hues with neutral grey at zero, and the scale is symmetric so grey
lands exactly on nothing-wrong. Everything else is a *magnitude*, so it gets a single hue
from light to dark. Rainbow scales are avoided because people read the bands as unordered
categories.

**Counties with no colour have no data** for the selected year. USDA publishes county
estimates only where enough farms report; coverage fell from 91 counties in 2000 to 46 in
2025.

**Try this comparison.** Look at *Prediction error*, then at *Soil water capacity* and
*Elevation*. The error map's east–west gradient lines up with both — which is exactly why
those two were added as features, and why the model improved when they were.
""")

if year is not None:
    st.subheader(f"{year} in numbers")
    rows = frame[frame["year"] == year]
    cols = st.columns(4)
    cols[0].metric("Counties reporting", f"{rows['fips'].nunique()}")
    cols[1].metric("Average yield", f"{rows['yield_bu_acre'].mean():.0f} bu/acre")
    cols[2].metric("Average error", f"{rows['error'].mean():+.1f} bu/acre")
    cols[3].metric("Typical miss", f"{rows['error'].abs().mean():.1f} bu/acre")

    worst = rows.reindex(rows["error"].abs().sort_values(ascending=False).index)
    st.caption("Largest misses this year")
    st.dataframe(
        worst.head(8)[["county_name", "yield_bu_acre", "predicted", "error"]]
        .rename(columns={"county_name": "County", "yield_bu_acre": "Actual",
                         "predicted": "Predicted", "error": "Error"}),
        hide_index=True, width="stretch")

with st.expander("Static version (for reports and the README)"):
    fig = choropleth(counties.merge(values, on="fips", how="left"), column,
                     f"{choice} — {subtitle}", explanation,
                     "Data: USDA NASS Quick Stats, NASA POWER, USDA Soil Data Access, "
                     "USGS, US Census Bureau",
                     diverging=diverging, legend_label=label, figsize=(10, 5.6))
    st.pyplot(fig, width="stretch")

# ------------------------------------------------------------------ disclaimer
st.divider()
st.caption(FOOTER)
page_footer()
