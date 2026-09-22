"""County soil productivity from USDA Soil Data Access (SDA).

The error map showed the model overpredicting in the sandy west and underpredicting
in the loess-soil northeast - the signature of a missing soil variable.

**NCCPI** (National Commodity Crop Productivity Index) is USDA's 0-1 rating of how
inherently productive a soil is for a given crop, built from soil properties and
climate. `nccpi3corn` is the corn-specific version. It is exactly the "how good is
the dirt here" number the model lacks.

Getting it without downloading gSSURGO (tens of GB) means querying **Soil Data
Access**, a public SQL endpoint over the national soil database. We ask for an
acre-weighted average NCCPI per soil survey area, then map survey areas to counties.

**The join caveat:** SSURGO is organized by survey area, not county. In Nebraska most
survey areas are a single county, but a few cover two or three. Those get the same
value assigned to each of their counties, which is an approximation - a cleaner
approach would intersect soil polygons with county boundaries, at much greater cost.
Flagged in the output so the limitation stays visible.
"""

from __future__ import annotations

import pandas as pd
import requests

SDA_URL = "https://SDMDataAccess.sc.egov.usda.gov/Tabular/post.rest"

# Acre-weighted mean of the corn NCCPI for every map unit in each survey area.
# muacres is the map unit's acreage, so this weights productive-but-tiny units
# appropriately rather than treating every polygon as equal.
NCCPI_QUERY = """
SELECT l.areasymbol, l.areaname,
       SUM(v.nccpi3corn * mu.muacres) / NULLIF(SUM(CASE WHEN v.nccpi3corn IS NULL
                                                        THEN 0 ELSE mu.muacres END), 0)
           AS nccpi_corn,
       SUM(CASE WHEN v.nccpi3corn IS NULL THEN 0 ELSE mu.muacres END) AS rated_acres,
       SUM(mu.muacres) AS total_acres
FROM legend l
JOIN mapunit mu ON mu.lkey = l.lkey
LEFT JOIN valu1 v ON v.mukey = mu.mukey
WHERE l.areasymbol LIKE '{state}%'
GROUP BY l.areasymbol, l.areaname
"""


class SoilDataError(RuntimeError):
    """Raised when Soil Data Access returns something unusable."""


def query_sda(sql: str, timeout: int = 300) -> pd.DataFrame:
    """Run a SQL query against Soil Data Access and return a DataFrame."""
    payload = {"SERVICE": "query", "FORMAT": "JSON+COLUMNNAME", "QUERY": sql}
    try:
        resp = requests.post(SDA_URL, json=payload, timeout=timeout)
    except requests.RequestException as exc:
        raise SoilDataError(f"Could not reach Soil Data Access: {type(exc).__name__}") from None

    if resp.status_code != 200:
        # SDA reports SQL errors as an OGC ServiceException; the useful part is the
        # text inside the tag, and it can appear well into the document.
        detail = resp.text
        if "<ServiceException>" in detail:
            detail = detail.split("<ServiceException>", 1)[1].split("</ServiceException>")[0]
        raise SoilDataError(f"Soil Data Access returned HTTP {resp.status_code}: "
                            f"{detail.strip()[:600]}")
    try:
        table = resp.json()["Table"]
    except (ValueError, KeyError):
        raise SoilDataError(f"Unexpected response: {resp.text[:300]}") from None

    if not table:
        raise SoilDataError("Query returned no rows.")
    # First row is the column names when FORMAT is JSON+COLUMNNAME.
    return pd.DataFrame(table[1:], columns=table[0])


# Fallback when the NCCPI table is unavailable: available water storage in the top
# 150 cm, from the map unit aggregated attribute table. It measures how much water
# the soil can hold for the crop - the property that matters most for dryland corn,
# and the main reason sandy western soils yield less than eastern loess.
AWS_QUERY = """
SELECT l.areasymbol, l.areaname,
       SUM(m.aws0150wta * mu.muacres) / NULLIF(SUM(CASE WHEN m.aws0150wta IS NULL
                                                        THEN 0 ELSE mu.muacres END), 0)
           AS soil_water_mm,
       SUM(CASE WHEN m.aws0150wta IS NULL THEN 0 ELSE mu.muacres END) AS rated_acres,
       SUM(mu.muacres) AS total_acres
FROM legend l
JOIN mapunit mu ON mu.lkey = l.lkey
LEFT JOIN muaggatt m ON m.mukey = mu.mukey
WHERE l.areasymbol LIKE '{state}%'
GROUP BY l.areasymbol, l.areaname
"""

PROBES = [
    ("legend table", "SELECT TOP 1 areasymbol, areaname FROM legend"),
    ("state filter", "SELECT TOP 1 areasymbol FROM legend WHERE areasymbol LIKE 'NE%'"),
    ("mapunit join", "SELECT TOP 1 mu.mukey, mu.muacres FROM mapunit mu "
                     "JOIN legend l ON mu.lkey = l.lkey WHERE l.areasymbol LIKE 'NE%'"),
    ("valu1 (NCCPI)", "SELECT TOP 1 mukey, nccpi3corn FROM valu1"),
    ("muaggatt (water)", "SELECT TOP 1 mukey, aws0150wta FROM muaggatt"),
]


def probe() -> None:
    """Run progressively more demanding queries to find which part SDA rejects.

    When a composite query fails with a vague message, the fastest diagnosis is to
    ask the simplest possible question and add one thing at a time.
    """
    for label, sql in PROBES:
        try:
            df = query_sda(sql, timeout=120)
            print(f"  OK    {label}: {df.iloc[0].to_dict()}")
        except SoilDataError as exc:
            print(f"  FAIL  {label}: {exc}")


def fetch_soil_rating(state_alpha: str = "NE") -> tuple[pd.DataFrame, str]:
    """Soil productivity per survey area, preferring NCCPI and falling back to water storage.

    Returns the frame and the name of the value column, so callers know which measure
    they got.
    """
    try:
        df = query_sda(NCCPI_QUERY.format(state=state_alpha))
        value_col = "nccpi_corn"
    except SoilDataError as exc:
        print(f"  NCCPI query failed ({exc})")
        print("  Falling back to available water storage (muaggatt.aws0150wta)...")
        df = query_sda(AWS_QUERY.format(state=state_alpha))
        value_col = "soil_water_mm"

    for col in (value_col, "rated_acres", "total_acres"):
        df[col] = pd.to_numeric(df[col], errors="coerce")
    return df.dropna(subset=[value_col]), value_col


def fetch_nccpi(state_alpha: str = "NE") -> pd.DataFrame:
    """Acre-weighted corn NCCPI per soil survey area (no fallback)."""
    df = query_sda(NCCPI_QUERY.format(state=state_alpha))
    for col in ("nccpi_corn", "rated_acres", "total_acres"):
        df[col] = pd.to_numeric(df[col], errors="coerce")
    return df.dropna(subset=["nccpi_corn"])


def counties_in_area(areaname: str) -> list[str]:
    """Extract county names from a survey area name.

    'Adams County, Nebraska'                  -> ['Adams']
    'Boyd and Keya Paha Counties, Nebraska'   -> ['Boyd', 'Keya Paha']
    'Cherry County, Nebraska, Northern Part'  -> ['Cherry']
    """
    name = areaname.split(",")[0].strip()
    name = name.replace(" Counties", "").replace(" County", "")
    parts = [part.strip() for chunk in name.split(",") for part in chunk.split(" and ")]
    return [part for part in parts if part]


def to_counties(areas: pd.DataFrame, county_lookup: pd.DataFrame,
                value_col: str = "nccpi_corn") -> pd.DataFrame:
    """Map survey-area ratings onto counties.

    `county_lookup` needs columns fips and county_name. A survey area covering several
    counties gives each of them its value; a county covered by several survey areas
    gets the acre-weighted average of them.
    """
    rows = []
    for area in areas.itertuples():
        for county in counties_in_area(area.areaname):
            rows.append({"county_name": county,
                         value_col: getattr(area, value_col),
                         "rated_acres": area.rated_acres,
                         "areasymbol": area.areasymbol})
    if not rows:
        return pd.DataFrame(columns=["fips", "county_name", value_col, "survey_areas"])

    expanded = pd.DataFrame(rows)
    lookup = county_lookup.copy()
    lookup["county_name"] = lookup["county_name"].str.title()
    expanded["county_name"] = expanded["county_name"].str.title()

    merged = expanded.merge(lookup, on="county_name", how="inner")
    merged["weighted"] = merged[value_col] * merged["rated_acres"]

    grouped = (merged.groupby(["fips", "county_name"], as_index=False)
               .agg(weighted=("weighted", "sum"), acres=("rated_acres", "sum"),
                    unweighted=(value_col, "mean"),
                    survey_areas=("areasymbol", "nunique")))

    # Acre-weighted where acreage is known, plain mean as a fallback.
    grouped[value_col] = (grouped["weighted"] / grouped["acres"]).where(
        grouped["acres"] > 0, grouped["unweighted"]).round(4)

    return grouped[["fips", "county_name", value_col, "survey_areas"]]
