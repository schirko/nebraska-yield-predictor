"""gridMET weather at 4 km, as a measured alternative to NASA POWER at 55 km.

`weather.py` gets daily weather from NASA POWER, which serves the MERRA-2 grid:
0.5 degrees latitude by 0.625 of longitude, roughly **55 km** on a side. A typical
county in Nebraska or Iowa is about 39 km across, so a county is *smaller than a
grid cell* - see `scripts/probe_weather_grid.py` for what that costs.

gridMET (University of Idaho, via the Climatology Lab) serves the same kind of
daily surface weather on a **4 km** grid, 1979 to yesterday, for the contiguous US.
That is roughly 14x finer on a side and about 190x finer by area: a county goes
from a fraction of one cell to roughly a hundred of them.

**This module does not replace `weather.py`. It sits beside it**, because "we
switched to a better dataset" is an assumption until someone measures it. Both
sources feed the *same* feature code (`weather.growing_season_features` and
`weather.spring_features`), so a comparison between them isolates the weather
product and nothing else. That is the whole point: if the summarisation differed
too, a score change would be unattributable.

To make that work, every fetch here returns the contract `fetch_power_daily`
returns - columns `date`, `tmax_c`, `tmin_c`, `precip_mm` - and nothing
downstream needs to know which source it came from.

TWO WAYS TO SAMPLE, AND WHY BOTH ARE HERE

* `county_point_daily` asks gridMET for **one pixel**, at the same internal point
  `weather.py` sends to POWER. Changing only the product, holding sampling fixed.
* `county_mean_daily` averages **every gridMET pixel inside the county polygon**.
  Changing the sampling too.

Keeping them apart matters because of a prediction registered before any of this
ran (see `scripts/fetch_gridmet.py`): a single 4 km pixel may well score *worse*
than a single 55 km one. The target is a county-average yield, and a 55 km cell is
already a kind of spatial average - a crude one, centred wrong, but an average. One
4 km pixel is a sharper measurement of a smaller place, which is not the same thing
as a better estimate of the county. If that is right, the gain lives in the
*averaging* rather than the resolution, and only the county-mean variant shows it.

Getting a null from the point variant alone and concluding "resolution doesn't
help" would be the wrong lesson from the right experiment.

UNITS

gridMET reports temperature in **Kelvin** and precipitation in mm. POWER reports
Celsius and mm. Everything here converts to Celsius on the way out, because the
feature code's thresholds (32 C, 35 C, 0 C frost) are in Celsius and a silent unit
mismatch would not crash - it would just quietly return zero heat days forever.
"""

from __future__ import annotations

import pandas as pd

# gridMET's names for what POWER calls T2M_MAX / T2M_MIN / PRECTOTCORR.
GRIDMET_VARS = ["tmmx", "tmmn", "pr"]

# gridMET temperatures are Kelvin.
KELVIN_OFFSET = 273.15

# The dataset's own limits, worth failing loudly against rather than getting
# empty frames back from a request for 1band978 or next week.
GRIDMET_FIRST_YEAR = 1979

# Data inside this window is published but still subject to revision, which is
# a caveat to carry into any result that leans on the most recent season.
PRELIMINARY_DAYS = 60


def _to_celsius(kelvin: pd.Series) -> pd.Series:
    return kelvin - KELVIN_OFFSET


def _find_columns(columns) -> dict[str, str]:
    """Map each wanted variable to the column that actually carries it.

    Found the hard way, on the first real request: `get_bycoords` labels its
    columns with units - `tmmx (K)`, `pr (mm)` - while `get_bygeom` returns the
    bare names. Matching on the exact string worked against the polygon path and
    failed against the point path, which is the kind of difference no amount of
    reading the docs would have surfaced.

    So the name is matched on its first token instead, which accepts both spellings
    and any future unit suffix. Anything genuinely absent still raises, because a
    silently missing variable would be far worse than a loud one.
    """
    lookup = {}
    for name in columns:
        head = str(name).split("(")[0].strip()
        if head in GRIDMET_VARS and head not in lookup:
            lookup[head] = name
    return lookup


def _tidy(frame: pd.DataFrame) -> pd.DataFrame:
    """Rename and convert gridMET's columns into the contract weather.py uses.

    The whole point of this module is that the returned frame is
    indistinguishable from `weather.fetch_power_daily`'s, so the identical
    feature code runs over both.
    """
    found = _find_columns(frame.columns)
    missing = [v for v in GRIDMET_VARS if v not in found]
    if missing:
        raise KeyError(
            f"gridMET response is missing {missing}; got {list(frame.columns)}. "
            "The service renames variables occasionally - check pygridmet's docs "
            "before assuming the request was wrong.")

    out = pd.DataFrame({
        "date": pd.to_datetime(frame.index if frame.index.name else frame["date"]),
        "tmax_c": _to_celsius(frame[found["tmmx"]].astype(float)).round(2),
        "tmin_c": _to_celsius(frame[found["tmmn"]].astype(float)).round(2),
        "precip_mm": frame[found["pr"]].astype(float).round(2),
    })
    return out.reset_index(drop=True).sort_values("date").reset_index(drop=True)


def _check_years(start_year: int, end_year: int) -> None:
    if start_year < GRIDMET_FIRST_YEAR:
        raise ValueError(
            f"gridMET starts in {GRIDMET_FIRST_YEAR}; asked for {start_year}")
    if end_year < start_year:
        raise ValueError(f"end_year {end_year} is before start_year {start_year}")


def county_point_daily(lat: float, lon: float, start_year: int, end_year: int,
                       *, client=None) -> pd.DataFrame:
    """One gridMET pixel, at the point `weather.py` sends to POWER.

    Same sampling strategy, different product - so a comparison against the POWER
    series isolates the resolution of the source and holds everything else fixed.

    `client` is an injection point for tests; production passes nothing and the
    real `pygridmet` is imported lazily, so importing this module costs nothing
    and the dependency stays optional.
    """
    _check_years(start_year, end_year)
    get_bycoords = client or _lazy("get_bycoords")
    raw = get_bycoords(
        (lon, lat),                                   # pygridmet takes (lon, lat)
        (f"{start_year}-01-01", f"{end_year}-12-31"),
        variables=GRIDMET_VARS,
        crs=4326,
    )
    return _tidy(pd.DataFrame(raw))


def county_mean_daily(geometry, start_year: int, end_year: int,
                      *, client=None) -> pd.DataFrame:
    """Every gridMET pixel inside the county, averaged to one daily series.

    This is the variant that can actually beat POWER, for a reason worth stating
    plainly: the target is a county-*average* yield, so the input wants to be a
    county-average too. A single pixel - however fine - measures one place.

    Averaging over the whole county is deliberately the *unweighted* version.
    Weighting by where the corn actually grows (a Cropland Data Layer mask) is the
    obvious next step, and it only becomes meaningful once the grid is fine enough
    to have something to weight, which at 55 km it was not.
    """
    _check_years(start_year, end_year)
    get_bygeom = client or _lazy("get_bygeom")
    grid = get_bygeom(
        geometry,
        (f"{start_year}-01-01", f"{end_year}-12-31"),
        variables=GRIDMET_VARS,
        crs=4326,
    )
    return _tidy(_spatial_mean(grid))


def _spatial_mean(grid) -> pd.DataFrame:
    """Collapse a gridded daily cube to one value per day, per variable.

    Averages over every dimension that isn't time. The spatial dimensions get
    named differently depending on the projection the service returns (`x`/`y`,
    `lat`/`lon`), so they are discovered rather than hard-coded - a hard-coded
    name would work until the service changed and then produce a confusing error
    far from its cause.
    """
    time_dim = next((d for d in grid.dims if str(d).lower() in ("time", "day", "date")), None)
    if time_dim is None:
        raise KeyError(f"no time dimension among {list(grid.dims)}")
    spatial = [d for d in grid.dims if d != time_dim]
    if not spatial:
        raise ValueError("the grid has no spatial dimensions to average over; "
                         "this looks like a point response, not a polygon one")

    averaged = grid.mean(dim=spatial, skipna=True)
    frame = averaged.to_dataframe()
    # to_dataframe can leave scalar coords as extra columns; keep the variables.
    keep = [c for c in frame.columns if c in GRIDMET_VARS]
    frame = frame[keep]
    frame.index = pd.to_datetime(frame.index)
    frame.index.name = "date"
    return frame


def _lazy(name: str):
    """Import pygridmet only when a fetch actually happens.

    Keeps `import yieldpred.gridmet` free for the app and the tests, which never
    call the service, and means a missing optional dependency surfaces as a clear
    message at the point of use rather than an ImportError at startup.
    """
    try:
        import pygridmet
    except ImportError as err:  # pragma: no cover - environment-dependent
        raise ImportError(
            "gridMET support needs pygridmet: pip install pygridmet") from err
    return getattr(pygridmet, name)


def compare_sources(power: pd.DataFrame, grid: pd.DataFrame,
                    on: tuple[str, ...] = ("fips", "year")) -> pd.DataFrame:
    """How far apart are the two products, feature by feature?

    Before asking which one predicts yield better, it is worth knowing whether
    they even disagree. If the features come out nearly identical, any score
    difference is noise and the whole exercise is answered cheaply; if they
    diverge sharply, the size of the divergence tells you where to look.

    Returns one row per shared feature column with the mean signed difference
    (is one product systematically warmer or wetter?), the mean absolute
    difference, and the correlation between them.
    """
    keys = list(on)
    shared = [c for c in power.columns
              if c in grid.columns and c not in keys
              and pd.api.types.is_numeric_dtype(power[c])]
    merged = power[keys + shared].merge(
        grid[keys + shared], on=keys, suffixes=("_power", "_gridmet"))

    rows = []
    for col in shared:
        a, b = merged[f"{col}_power"], merged[f"{col}_gridmet"]
        # A constant column has no variance to correlate, and asking anyway
        # gives NaN plus a divide-by-zero warning from inside pandas. Say
        # "undefined" deliberately instead of letting a warning say it.
        constant = a.nunique(dropna=True) < 2 or b.nunique(dropna=True) < 2
        rows.append({
            "feature": col,
            "power_mean": a.mean(),
            "gridmet_mean": b.mean(),
            "mean_diff": (b - a).mean(),
            "mean_abs_diff": (b - a).abs().mean(),
            "correlation": float("nan") if constant else a.corr(b),
        })
    return pd.DataFrame(rows).set_index("feature")
