"""Streamlit front end (placeholder - maps and the model come later).

Run from the project root:  streamlit run app/streamlit_app.py
"""

from pathlib import Path

import pandas as pd
import streamlit as st

DATA = Path(__file__).resolve().parents[1] / "data" / "processed" / "ne_corn_yield_county.parquet"

st.set_page_config(page_title="Nebraska Corn Yield Predictor", page_icon="🌽")
st.title("Nebraska Corn Yield Predictor")
st.caption("County corn yields from USDA NASS - irrigated vs. non-irrigated")

if not DATA.exists():
    st.info("No data yet. Run `python scripts/fetch_nass_yields.py` first.")
    st.stop()


@st.cache_data
def load() -> pd.DataFrame:
    return pd.read_parquet(DATA)


df = load()

st.subheader("State-wide average yield")
trend = df.pivot_table(index="year", columns="practice", values="yield_bu_acre", aggfunc="mean")
st.line_chart(trend, y_label="bu/acre")

st.subheader("County detail")
county = st.selectbox("County", sorted(df["county_name"].unique()))
st.dataframe(
    df[df["county_name"] == county]
    .pivot_table(index="year", columns="practice", values="yield_bu_acre")
    .sort_index(ascending=False),
    width="stretch",
)
