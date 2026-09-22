# Nebraska Corn Yield Predictor

Predicting county-level corn yields across Nebraska from weather, soils and cropland data, with a focus on how **irrigated and non-irrigated** fields respond differently to growing-season conditions.

> 🚧 Work in progress. Built as a portfolio project during my M.S. in Data Science.

## Why Nebraska?

Nebraska is one of the top corn-producing states and among the most heavily irrigated states in the U.S. USDA NASS reports county yields separately for irrigated and non-irrigated corn, and rainfall drops sharply from east to west across the state. Together, those make it a natural experiment in how much weather matters when irrigation isn't there to buffer it.

## Approach (planned)

1. **Target:** county corn yields (bu/acre) from USDA NASS Quick Stats, 2000–present
2. **Features:** growing-season weather (PRISM / NASA POWER), soil productivity (gSSURGO), drought (US Drought Monitor), all summarized over cropland only using the USDA Cropland Data Layer
3. **Models:** baseline trend model → gradient boosting
4. **Spatial validation:** hold out whole regions (spatial cross-validation) instead of random rows
5. **Diagnostics:** maps of prediction error and Moran's I to check whether errors cluster geographically
6. **App:** interactive Streamlit map of predicted vs. actual yields

## Data sources

| Source | Used for |
|---|---|
| [USDA NASS Quick Stats](https://quickstats.nass.usda.gov/) | County yields (target) |
| [USDA Cropland Data Layer](https://www.nass.usda.gov/Research_and_Science/Cropland/SARS1a.php) | Cropland mask |
| gSSURGO | Soil properties |
| PRISM / NASA POWER | Weather |
| US Drought Monitor | Drought severity |
| Census TIGER | County boundaries |

## What the data shows so far

- **4,819 county-year yield records** across 91 counties, 2000–2025.
- **Irrigated vs. non-irrigated yields are reported only through 2018**, so the model
  targets all-practice yields and treats irrigation as a county characteristic.
- Over the 1,349 paired county-years, **irrigated corn out-yielded non-irrigated corn by
  84.5 bu/acre on average** — and the gap widens sharply in drought years (139.7 in 2012,
  119.7 in 2002, versus about 60 in wet years).

See [docs/METHODS.md](docs/METHODS.md) for data cleaning decisions, feature definitions,
and the validation strategy.

## Project structure

```
app/                  Streamlit app
src/yieldpred/        Core logic
  nass.py             USDA NASS Quick Stats client and cleaning
  weather.py          NASA POWER download and growing-season features
  dataset.py          Joins yields and weather into the modeling table
scripts/              Runnable steps (download, explore, train)
docs/METHODS.md       Data decisions, features, validation
notebooks/            Exploration
data/raw/             Raw downloads and API cache (not committed)
data/processed/       Small cleaned files used by the app
tests/                Unit tests (no network or API key needed)
```

## Getting started

```bash
python -m venv .venv          # Python 3.12
.venv\Scripts\activate        # Windows (macOS/Linux: source .venv/bin/activate)
pip install -r requirements.txt

copy .env.example .env        # then paste your NASS API key into .env
python scripts/fetch_nass_yields.py    # county yields
python scripts/explore_coverage.py     # what the data covers
python scripts/fetch_weather.py        # ~4 min, cached afterwards
python scripts/train_baseline.py       # models + validation
streamlit run app/streamlit_app.py
```

Run the tests with `pytest`.

## Roadmap

- [x] Project setup
- [x] Download NASS county yields
- [x] Data coverage analysis
- [x] Growing-season weather features
- [x] Baseline models with spatial and temporal validation
- [ ] Irrigation share as a feature
- [ ] Soil productivity (gSSURGO)
- [ ] Cropland masking and zonal statistics
- [ ] Moran's I on model errors
- [ ] Interactive map in Streamlit
- [ ] Deploy to Streamlit Community Cloud

## Author

Scott — add your LinkedIn / GitHub links here
