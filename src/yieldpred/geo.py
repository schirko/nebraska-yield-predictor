"""County boundaries for mapping and for building spatial weights.

Source: the Census cartographic boundary file (a generalized version of TIGER,
small enough for a web app). Downloaded once into data/raw/ and cached.

CRS notes - the thing that trips people up in GIS work:
* **EPSG:4326** (lat/lon degrees) is what web maps expect.
* **EPSG:5070** (Albers Equal Area, conterminous US) is what you must use for any
  measurement - area, distance, sensible simplification - because degrees are not
  a constant distance apart.

So: measure in 5070, display in 4326.
"""

from __future__ import annotations

from pathlib import Path

import geopandas as gpd
import requests

CB_URL = "https://www2.census.gov/geo/tiger/GENZ2023/shp/cb_2023_us_county_500k.zip"
ROOT = Path(__file__).resolve().parents[2]
RAW = ROOT / "data" / "raw"
PROCESSED = ROOT / "data" / "processed"

EQUAL_AREA = "EPSG:5070"
WEB = "EPSG:4326"


def download_county_shapefile(cache_dir: Path = RAW) -> Path:
    cache_dir.mkdir(parents=True, exist_ok=True)
    path = cache_dir / "cb_2023_us_county_500k.zip"
    if not path.exists():
        resp = requests.get(CB_URL, timeout=300)
        resp.raise_for_status()
        path.write_bytes(resp.content)
    return path


def county_boundaries(state_fips: str = "31", simplify_m: float = 0.0,
                      cache_dir: Path = RAW) -> gpd.GeoDataFrame:
    """County polygons for one state: fips, county_name, area_km2, geometry (WGS84).

    `simplify_m` defaults to 0 - no simplification - because ordinary simplification
    is topology-destroying. See `simplify_coverage` below; only use a non-zero value
    for display copies, never for the geometry that spatial weights are built from.
    """
    gdf = gpd.read_file(download_county_shapefile(cache_dir))
    gdf = gdf[gdf["STATEFP"] == state_fips].copy()

    projected = gdf.to_crs(EQUAL_AREA)
    projected["area_km2"] = (projected.area / 1e6).round(1)
    if simplify_m:
        projected["geometry"] = simplify_coverage(projected.geometry, simplify_m)

    out = projected.to_crs(WEB)
    return (out.rename(columns={"GEOID": "fips", "NAME": "county_name"})
            [["fips", "county_name", "area_km2", "geometry"]]
            .sort_values("fips")
            .reset_index(drop=True))


def simplify_coverage(geometry: gpd.GeoSeries, tolerance_m: float) -> gpd.GeoSeries:
    """Simplify adjacent polygons *without* pulling them apart.

    Plain `.simplify()` treats each polygon on its own, so a boundary shared by two
    counties is simplified twice - slightly differently each time. The polygons stop
    touching, gaps and slivers appear, and any contiguity-based analysis silently
    breaks: neighbours go missing and counties turn into "islands".

    Coverage simplification (shapely >= 2.1) simplifies each shared edge once and
    applies the same result to both sides, so the tiling stays a tiling.
    """
    from shapely import coverage_simplify

    simplified = coverage_simplify(geometry.to_numpy(), tolerance=tolerance_m)
    return gpd.GeoSeries(simplified, index=geometry.index, crs=geometry.crs)


def display_geometry(gdf: gpd.GeoDataFrame, tolerance_m: float = 500.0) -> gpd.GeoDataFrame:
    """A lighter copy of the boundaries for interactive maps.

    An interactive chart embeds its geometry in the page, so full-resolution
    outlines make every render slower. 500 m of coverage simplification is
    invisible at state scale and shrinks the payload substantially - and being
    coverage simplification, it doesn't tear the counties apart (see
    `simplify_coverage`). Analysis still uses the unsimplified geometry.
    """
    out = gdf.to_crs(EQUAL_AREA)
    out["geometry"] = simplify_coverage(out.geometry, tolerance_m)
    return out.to_crs(WEB)


def neighbor_report(gdf: gpd.GeoDataFrame) -> dict:
    """Sanity-check a polygon layer before using it for contiguity analysis.

    Nebraska counties are a tiling of the state, so every county must have at least
    two neighbours. Islands mean the geometry is broken, not that the map is unusual.
    """
    from libpysal.weights import Queen

    w = Queen.from_dataframe(gdf, use_index=False)
    counts = [len(v) for v in w.neighbors.values()]
    return {"counties": len(gdf),
            "mean_neighbors": round(sum(counts) / len(counts), 2),
            "min_neighbors": min(counts),
            "islands": sum(1 for c in counts if c == 0)}


def save_boundaries(gdf: gpd.GeoDataFrame, state_fips: str = "31") -> Path:
    """Save as GeoParquet - geometry plus CRS in one compact file."""
    PROCESSED.mkdir(parents=True, exist_ok=True)
    path = PROCESSED / f"counties_{state_fips}.parquet"
    gdf.to_parquet(path, index=False)
    return path


def load_boundaries(state_fips: str = "31") -> gpd.GeoDataFrame | None:
    path = PROCESSED / f"counties_{state_fips}.parquet"
    return gpd.read_parquet(path) if path.exists() else None
