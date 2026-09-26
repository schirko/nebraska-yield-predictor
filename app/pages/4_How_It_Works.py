"""The teaching page: architecture, data sources, and the vocabulary."""

import sys
from pathlib import Path

# Make src/ importable even if the editable install (-e .) didn't take, which can
# happen on hosted runtimes. Harmless locally, where the install does work.
_SRC = Path(__file__).resolve().parents[2] / "src"
if str(_SRC) not in sys.path:
    sys.path.insert(0, str(_SRC))


import streamlit as st

from yieldpred.disclaimer import DATA_SOURCES, FOOTER, LIMITATIONS
from yieldpred.brand import page_footer, page_setup

page_setup("How It Works")

st.title("How It Works")
st.caption("The architecture, the data, and the ideas — for anyone who wants to know "
           "what's under the hood")

tab_flow, tab_data, tab_model, tab_words = st.tabs(
    ["Data flow", "Data sources", "The model", "Vocabulary"])

with tab_flow:
    st.subheader("From API to app")
    st.code("""
USDA NASS Quick Stats  ─┐
NASA POWER             ─┤
USDA Soil Data Access  ─┼──► data/processed/*.parquet ──► model ──► this app
USGS elevation         ─┤
US Census (boundaries) ─┘

scripts/fetch_nass_yields.py   county corn yields, 2000-2025
scripts/fetch_weather.py       daily weather -> growing-season features
scripts/fetch_irrigation.py    irrigated share of corn acres
scripts/fetch_soil_terrain.py  soil water capacity and elevation
scripts/train_baseline.py      models, validation, per-county errors
scripts/analyze_errors.py      Moran's I and the maps
""", language="text")

    st.markdown("""
**Every stage writes a file.** That means any step can be re-run without repeating the
ones before it, downloads are cached, and this app starts instantly because it only
*reads* results — it never trains anything or calls an API.

**The code is layered deliberately:**

| Layer | Location | Job |
|---|---|---|
| Core logic | `src/yieldpred/` | Data access, features, models, maps. Knows nothing about any UI |
| Steps | `scripts/` | Thin wrappers: parse arguments, call the logic, print a summary |
| Interface | `app/` | This app. Presentation only |
| Tests | `tests/` | 45 tests that run offline in seconds, no API key needed |

Keeping the logic out of the UI is what makes it testable, reusable in a notebook, and
movable behind an API later. If the yield-cleaning code lived inside a Streamlit page, none
of that would be possible.
""")

with tab_data:
    st.subheader("Where the numbers come from")
    st.markdown("""
| Source | Provides | Notes |
|---|---|---|
| **USDA NASS Quick Stats** | County corn yields, harvested acres by practice | Free API key. County estimates thin out over time: 91 counties in 2000, 46 in 2025 |
| **NASA POWER** | Daily temperature and rainfall, ~0.5° grid | No key needed. One point per county |
| **USDA Census of Agriculture** | Irrigated acres every five years | Carries irrigation share past the 2018 survey cutoff |
| **USDA Soil Data Access** | Soil water capacity per county | A public SQL endpoint over the national soil database — no multi-gigabyte download |
| **USGS Elevation Point Query** | County elevation | Stands in for season length and cooler nights |
| **US Census Bureau** | County boundaries | Used for maps and for defining which counties neighbour which |

**Three data decisions worth knowing about:**

1. **Withheld values are missing, not zero.** USDA publishes `(D)` when reporting a number
   would expose an individual farm. Treating that as zero would invent crop failures.
2. **Irrigation share is interpolated between observations**, because centre pivots are
   multi-decade investments rather than annual decisions. Every value is flagged as measured
   or interpolated.
3. **Missing counties are not random.** The counties that stop reporting are Sandhills
   ranching counties with little corn, so results describe Nebraska's corn-growing counties
   rather than all of Nebraska.
""")

with tab_model:
    st.subheader("What the model actually does")
    st.markdown("""
For each county-year the model sees a dozen numbers — the year, eight weather summaries,
the irrigation share, and two county characteristics (soil water capacity and elevation) —
and predicts yield in bushels per acre.

**Two models working together.** Corn yields rise about **2.2 bu/acre per year** from better
genetics and management. A linear model captures that trend and can extend it into future
years. Gradient boosting captures the interactions — how heat, rain and irrigation combine —
but *cannot* produce a value outside the range it was trained on, so it can't extend a trend.

So the two are composed: fit the trend with a line, subtract it, let hundreds of small
decision trees model the leftover deviation, then add the trend back. Each does the part it
is good at.

**Following one prediction through:**

1. Assemble the row: 2023, 3,510 growing degree days, 417 mm of rain, 30 days above 32°C,
   78% irrigated, and the rest.
2. The trend line gives the year's baseline expectation.
3. The trees read the weather and irrigation and return a deviation — positive for a kind
   season, negative for a hostile one.
4. Add them. That's the prediction.
5. Actual minus predicted is the error, which gets mapped and tested for clustering.

**The weather features, and why each is there:**

| Feature | Why |
|---|---|
| Growing degree days | Accumulated heat drives crop development |
| Season / July / August rainfall | July is pollination, the most drought-sensitive stage |
| Mean July high | Heat stress during pollination |
| Days above 32°C and 35°C | Extreme heat damage, which averages hide |
| Longest dry spell | *When* the rain failed, not just how much fell |

**The spring features, and why they were added later:**

| Feature | Why |
|---|---|
| March rainfall | Snowmelt and saturation before anything is planted — Nebraska's 2019 flood |
| April–May rainfall | Rain during the planting window itself |
| Workable planting days | Days a field could be worked: little rain that day, little in the two days before |
| Last spring frost | Season length at the front end |
| May growing degree days | Heat for emergence and stand establishment |

The model's two biggest unexplained misses — Nebraska 2019 and Iowa 2013 — were both spring
problems, invisible to a weather window that starts in April. **Workable days** is the
interesting one: total rainfall can't tell 50 mm in one storm from 50 mm spread over three
weeks, but a planter can. Counting the days soil was dry enough to carry machinery turns a
rainfall column into a planting-delay column, and it picks Iowa 2013 as the worst planting
spring in the 26-year record without being told to.

**The county characteristics, and why each is there:**

| Feature | Why |
|---|---|
| Irrigation share | Irrigated fields shrug off the drought signal the model relies on |
| Soil water capacity | Sandy soil holds ~10 cm of water, deep loess 30+; it decides how long a crop lasts between rains |
| Elevation | A proxy for season length and cooler nights — the panhandle sits 1,200 m above the southeast corner |

**A known limitation:** weather is sampled at one point per county, which ignores where the
corn actually grows — a real issue in large western counties that mix cropland and
rangeland. Weighting by the USDA Cropland Data Layer is the planned improvement.
""")

with tab_words:
    st.subheader("Vocabulary")
    st.markdown("""
| Term | What it means here |
|---|---|
| **Feature** | An input column, such as days above 32°C |
| **Target** | The value being predicted — yield in bu/acre |
| **Fit / train** | Adjust a model until it predicts the examples well |
| **Hyperparameter** | A setting chosen before training, such as how many trees to build |
| **Residual** | Actual minus predicted — what's left over |
| **Cross-validation** | Repeatedly hold out part of the data, train on the rest, score the held-out part |
| **Leakage** | Information about the test data reaching the model during training — see Model & Validation for a real one found in this project |
| **R²** | Share of variation explained: 1 perfect, 0 no better than the average, negative worse |
| **RMSE** | Typical error size in bu/acre, with large misses penalized extra |
| **Spatial autocorrelation** | Nearby places resemble each other, so their data isn't independent |
| **Moran's I** | A number summarizing whether a map is clustered (+), random (0) or alternating (−) |
| **Queen contiguity** | Two counties are neighbours if their borders touch anywhere, even at a corner |
| **Zonal statistics** | Summarizing a grid of pixels within a boundary, like average rainfall over a county |
| **FIPS code** | The federal ID for a county — 31109 is Lancaster County, Nebraska |
| **CRS** | Coordinate reference system: how points on a round Earth become a flat map |
""")

# ------------------------------------------------------------------ disclaimer
st.divider()
st.subheader("Limitations & Disclaimer")
st.markdown(LIMITATIONS)
with st.expander("Data sources and endorsement"):
    st.markdown(DATA_SOURCES)
st.caption(FOOTER)
page_footer()
