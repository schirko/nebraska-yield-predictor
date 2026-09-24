# Corn Yield Predictor — Nebraska, Tested On Iowa

County-level corn yield prediction from weather, soils and irrigation, built to answer a
question most yield models skip: **does the model generalize, or has it memorized one state?**

A Nebraska-trained model, applied cold to Iowa counties it had never seen, reached
**R² 0.503** against a floor of −0.26 and an Iowa-native ceiling of 0.654 — and closed 70% of
the 14 bu/acre level gap between the two states from its inputs alone.

Built as a portfolio project during my M.S. in Data Science. Python · scikit-learn ·
GeoPandas · PySAL · Altair · Streamlit.

## Why This Project

Nebraska is one of the top corn-producing states and among the most heavily irrigated in the
U.S. USDA NASS reports county yields separately for irrigated and non-irrigated corn, and
rainfall drops sharply from east to west across the state. That makes it a natural experiment
in how much weather matters when irrigation isn't there to buffer it.

Iowa is the control: it borders Nebraska, grows more corn than any other state, and differs
in the ways that matter — better and far more uniform soils, more reliable rainfall, and
essentially no irrigation.

## Headline Results

### 1. How You Validate Changes The Answer

Every model is scored three ways, because the three answers differ and only two of them are
honest. Random cross-validation leaks: neighbouring counties in the same year are near
duplicates. Holding out whole agricultural districts, or whole years, is the real test.

| Model | Random 5-fold | Leave-district-out | Train ≤2018 / test ≥2019 |
|---|---|---|---|
| Mean yield (dummy) | −0.00 | −0.06 | −0.60 |
| Ridge (trend + weather) | 0.405 | 0.249 | 0.201 |
| Gradient boosting | 0.565 | 0.395 | **−0.419** |
| Detrended boosting | 0.537 | 0.313 | −0.052 |

Random cross-validation inflates the spatial score by about 40%. And the model that wins
under both random and spatial splits is the one that fails worst on future years — because
**tree models cannot extrapolate**. Trained through 2018, gradient boosting scored worse than
guessing the average, since it can't produce a yield outside the range it has seen. Fitting a
linear trend first and modelling the residual (`DetrendedRegressor`) fixed it.

### 2. Domain Knowledge Beat Model Tuning

Each row adds one feature to the row above — same model, same folds, same rows:

| Feature set | Spatial RMSE | Spatial R² | Temporal R² |
|---|---|---|---|
| Weather only | 26.53 | 0.313 | −0.052 |
| + irrigation share | 20.69 | **0.582** | 0.317 |
| + NCCPI soil rating | 20.22 | 0.601 | **0.499** |
| + elevation | 20.23 | 0.601 | 0.500 |
| All features, no look-ahead control | 19.33 | **0.635** | **0.533** |

Irrigation share — reconstructed from NASS harvested-acre ratios, since it isn't published as
a feature — lifted the spatial score by **+0.27 R²**, more than any amount of hyperparameter
tuning. It came from knowing what Nebraska farming looks like, not from the model.

The last row is a leakage control: the irrigation series is rebuilt using only observations
from 2018 or earlier, so the temporal test can't borrow from the 2022 Census. The score
didn't fall, so the leak wasn't doing any work.

### 3. The Errors Are A Map, Not Noise

![Average prediction error by county](figures/error_map_ne.png)

Moran's I on the pooled county errors is **+0.598 (p = 0.001)** — strongly clustered, and
significantly clustered in all 26 years individually. Random error looks like static; this
looks like a map of something real: overprediction in the sandy, high-elevation west,
underprediction in the loess-soil northeast. That map is what motivated adding soil
productivity and elevation.

![Share of corn acres irrigated](figures/irrigation_map_ne.png)

The irrigation feature, recovered from acreage ratios alone, reproduces Nebraska's real
agricultural geography: heavy irrigation up the central Platte Valley and west, near zero in
the wetter southeast. A feature that reproduces a geography you already know is probably
measuring what you think it is.

### 4. Does It Transfer? Iowa Says Mostly Yes

`scripts/cross_state.py --train NE --test IA` trains on Nebraska only and predicts Iowa cold —
without irrigation share, because Iowa doesn't irrigate.

| | RMSE | MAE | R² | bias |
|---|---|---|---|---|
| Iowa's own model (leave-district-out) — *the ceiling* | 16.37 | 12.57 | **0.654** | −0.62 |
| Nebraska's average yield applied to Iowa — *the floor* | 31.22 | 25.62 | **−0.258** | −14.14 |
| Trained on Nebraska, applied cold | 19.61 | 15.49 | **0.503** | −4.11 |
| …same predictions, average offset removed | 19.18 | 14.61 | 0.525 | 0.00 |

- The transfer lands **77% of the way from floor to ceiling** on a state it has never seen.
- **Bias fell from −14.1 to −4.1**: the model recovered ~70% of the level difference between
  the states from its inputs, rather than memorizing Nebraska's average.
- **Cold and offset-corrected scores are nearly identical**, which says the residual error is
  *pattern* error, not a calibration offset — the harder kind, and the honest one to report.

Where it breaks is as informative as where it works. Scoring the transfer by how well it ranks
counties *within* each year, two years fail outright: **2013** (Iowa's record-wet, late-planted
spring) and **2020** (the August 10 derecho). The model has no planting-window features and no
wind variable, so both events are invisible to it. Missing features, not a broken model — and
the same diagnosis Nebraska's 2019 flood year produced, found independently in another state.

### 5. The Same Feature Can Mean Opposite Things

| Feature | Nebraska | Iowa |
|---|---|---|
| `precip_mm` | **+0.313** | **+0.036** |
| `dry_spell_max` | **−0.108** | **+0.195** |

In Nebraska a long dry spell is a drought. In Iowa, where rain is rarely the binding
constraint, it means sunshine, fewer leaf-disease days, and fields that aren't waterlogged.
Same column, same units, same code, opposite sign. A model fitted on pooled Corn Belt data
would average these into nothing.

## What I'd Fix Next

Honest limitations, in the order they'd matter:

1. **One weather point per county.** Large western counties mix cropland and rangeland; the
   fix is weighting weather by the USDA Cropland Data Layer.
2. **No spring.** The weather window starts in April, so a March flood (Nebraska 2019) or a
   record-wet planting season (Iowa 2013) is invisible.
3. **No wind.** The 2020 derecho is the clearest single failure in the project.
4. **Coverage is not missing at random.** NASS reported 91 Nebraska counties in 2000 and 46
   in 2025, and the counties that stop reporting are the ones that grow little corn.

## Data Sources

| Source | Used for |
|---|---|
| [USDA NASS Quick Stats](https://quickstats.nass.usda.gov/) | County yields, harvested acres by practice |
| [NASA POWER](https://power.larc.nasa.gov/) | Daily growing-season weather |
| [USDA Soil Data Access](https://sdmdataaccess.sc.egov.usda.gov/) | NCCPI soil productivity, available water capacity |
| [USGS EPQS](https://apps.nationalmap.gov/epqs/) | County elevation |
| [Census TIGER / Gazetteer](https://www.census.gov/geographies/mapping-files/time-series/geo/tiger-line-file.html) | County boundaries and centroids |
| USDA Cropland Data Layer | Planned: cropland-weighted weather |

## Project Structure

```
app/                  Streamlit app (multipage)
src/yieldpred/        Core logic
  nass.py             USDA NASS Quick Stats client and cleaning
  weather.py          NASA POWER download and growing-season features
  irrigation.py       Irrigated share of corn acres from NASS acreage
  soils.py            NCCPI and available water from USDA Soil Data Access
  terrain.py          County elevation from USGS
  dataset.py          Joins everything into the county-year modeling table
  trend.py            DetrendedRegressor - the trend/deviation split
  spatial.py          Contiguity weights and Moran's I
  geo.py              County boundaries, topology-safe simplification
  viz.py              Static and interactive choropleths
  appdata.py          Streamlit-free data access for the app
scripts/              Runnable steps, all --state aware
  fetch_*.py          Downloads
  train_baseline.py   Models, validation, feature ladder, ablation
  analyze_errors.py   Moran's I and error maps
  cross_state.py      Train on one state, predict another
notebooks/            Exploration
data/raw/             Raw downloads and API cache (not committed)
data/processed/       Small cleaned files used by the app
figures/              Generated maps
tests/                Unit tests (no network or API key needed)
```

## Getting Started

```bash
python -m venv .venv          # Python 3.12
.venv\Scripts\activate        # Windows (macOS/Linux: source .venv/bin/activate)
pip install -r requirements-dev.txt   # full pipeline (requirements.txt is app-only)

copy .env.example .env        # then paste your NASS API key into .env
python scripts/fetch_nass_yields.py    # county yields
python scripts/explore_coverage.py     # what the data covers
python scripts/fetch_weather.py        # ~4 min, cached afterwards
python scripts/fetch_irrigation.py     # irrigation share
python scripts/fetch_soil_terrain.py   # NCCPI + elevation
python scripts/train_baseline.py       # models, validation, feature study
python scripts/analyze_errors.py       # spatial diagnostics + maps
streamlit run app/streamlit_app.py
```

Any script runs for another state by passing `--state IA --state-fips 19`. The transfer test
is `python scripts/cross_state.py --train NE --test IA`.

Run the tests with `pytest` — they use synthetic fixtures, so no network or API key is needed.

## Roadmap

- [x] NASS county yields and coverage analysis
- [x] Growing-season weather features
- [x] Baseline models with spatial and temporal validation
- [x] Irrigation share as a feature, with a no-look-ahead control
- [x] Moran's I on model errors and error maps
- [x] NCCPI soil productivity and elevation
- [x] Interactive Streamlit app
- [x] Second state (Iowa) end to end
- [x] Cross-state transfer test
- [ ] Deploy to Streamlit Community Cloud
- [ ] Cropland-weighted weather (Cropland Data Layer)
- [ ] Spring / planting-window features
- [ ] Wind and storm damage

## Author

Scott — add your LinkedIn / GitHub links here
