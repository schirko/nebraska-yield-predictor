"""County elevation from the USGS Elevation Point Query Service.

Elevation is the other half of the east-west story the error map told. The Nebraska
panhandle sits above 1,200 m; the southeast corner is near 300 m. Higher ground means
cooler nights, a shorter frost-free window, and less time for corn to fill grain -
none of which the weather features capture, since growing degree days count heat
without knowing when the season ends.

Elevation is also a cheap proxy for several correlated things at once: continentality,
aridity, and the shift from row crops to rangeland. A feature that stands in for a
bundle of drivers is fine for prediction, but worth remembering when interpreting
importance - it doesn't isolate any single mechanism.
"""

from __future__ import annotations

import time

import pandas as pd
import requests

EPQS_URL = "https://epqs.nationalmap.gov/v1/json"


class ElevationError(RuntimeError):
    """Raised when the elevation service returns something unusable."""


def point_elevation(lat: float, lon: float, timeout: int = 60) -> float:
    """Elevation in metres at one point."""
    params = {"x": lon, "y": lat, "wkid": 4326, "units": "Meters", "includeDate": "false"}
    try:
        resp = requests.get(EPQS_URL, params=params, timeout=timeout)
    except requests.RequestException as exc:
        raise ElevationError(f"Could not reach the elevation service: "
                             f"{type(exc).__name__}") from None
    if resp.status_code != 200:
        raise ElevationError(f"Elevation service returned HTTP {resp.status_code}")

    try:
        return float(resp.json()["value"])
    except (ValueError, KeyError, TypeError):
        raise ElevationError(f"Unexpected response: {resp.text[:200]}") from None


def county_elevations(points: pd.DataFrame, pause: float = 0.3,
                      progress: bool = True) -> pd.DataFrame:
    """Elevation for each county centre point.

    `points` needs fips, county_name, lat, lon - the frame `weather.county_points`
    returns. Counties that fail are left out rather than failing the whole run, so a
    flaky service costs a few rows instead of the batch.
    """
    rows, failures = [], []
    for i, point in enumerate(points.itertuples(), start=1):
        try:
            metres = point_elevation(point.lat, point.lon)
        except ElevationError:
            failures.append(point.fips)
            continue
        rows.append({"fips": point.fips, "county_name": point.county_name,
                     "elevation_m": round(metres, 1)})
        if progress and i % 10 == 0:
            print(f"  {i}/{len(points)} counties")
        time.sleep(pause)

    if failures:
        print(f"  elevation lookup failed for {len(failures)} counties: {failures[:5]}"
              f"{'...' if len(failures) > 5 else ''}")
    return pd.DataFrame(rows)
