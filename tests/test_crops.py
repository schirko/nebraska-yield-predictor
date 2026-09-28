"""--crop: corn keeps every file name it always had; soybeans get their own files and NASS query."""

import pandas as pd
import pytest

from yieldpred import dataset
from yieldpred.crops import CROPS, crop_stem, get_crop, yields_file
from yieldpred.dataset import build_modeling_table, output_path
from yieldpred.nass import yield_params


def test_corn_keeps_every_existing_file_name():
    """The app and the published results read these names; the crop option must not move them."""
    assert output_path("model_table", "ne").name == "model_table.parquet"
    assert output_path("model_table", "ne", "corn").name == "model_table.parquet"
    assert output_path("model_errors", "ia", "corn").name == "model_errors_ia.parquet"
    assert yields_file("NE") == "ne_corn_yield_county.parquet"
    assert crop_stem("irrigation_share_ne") == "irrigation_share_ne"


def test_soybeans_get_their_own_files_after_the_state():
    assert output_path("model_table", "ne", "soybeans").name == "model_table_soy.parquet"
    assert output_path("model_table", "ia", "soybeans").name == "model_table_ia_soy.parquet"
    assert yields_file("IA", "soybeans") == "ia_soybeans_yield_county.parquet"
    assert crop_stem("irrigation_share_ne", "soybeans") == "irrigation_share_ne_soy"


def test_crop_names_are_forgiving_and_unknown_crops_are_refused():
    assert get_crop("Soy").key == get_crop("soybean").key == "soybeans"
    assert get_crop(CROPS["corn"]) is CROPS["corn"]
    with pytest.raises(ValueError, match="wheat"):
        get_crop("wheat")


def test_every_crop_has_a_distinct_suffix():
    suffixes = [c.suffix for c in CROPS.values()]
    assert len(set(suffixes)) == len(suffixes)
    assert CROPS["corn"].suffix == ""


def test_nass_query_picks_the_crop():
    corn, soy = yield_params("corn", "NE", 2000), yield_params("soybeans", "IA", 2005, 2024)
    assert (corn["commodity_desc"], corn["util_practice_desc"]) == ("CORN", "GRAIN")
    assert soy["commodity_desc"] == "SOYBEANS"
    assert "util_practice_desc" not in soy          # soybeans have no grain/silage split
    assert soy["state_alpha"] == "IA" and soy["year__LE"] == 2024
    for p in (corn, soy):
        assert (p["statisticcat_desc"], p["unit_desc"], p["agg_level_desc"]) == ("YIELD", "BU / ACRE", "COUNTY")


def _yields(fips, value):
    return pd.DataFrame({"fips": [fips], "state_alpha": ["NE"], "county_name": ["Adams"], "asd_code": ["50"],
                         "asd_desc": ["South Central"], "year": [2020], "practice": ["all"],
                         "yield_bu_acre": [value]})


def test_modeling_table_reads_the_crops_own_yields_and_irrigation(tmp_path, monkeypatch):
    monkeypatch.setattr(dataset, "PROCESSED", tmp_path)
    _yields("31001", 190.0).to_parquet(tmp_path / "ne_corn_yield_county.parquet")
    _yields("31001", 62.0).to_parquet(tmp_path / "ne_soybeans_yield_county.parquet")
    pd.DataFrame({"fips": ["31001"], "year": [2020], "gdd": [1500.0]}).to_parquet(tmp_path / "weather_county_31.parquet")
    pd.DataFrame({"fips": ["31001"], "year": [2020], "irrigation_share": [0.7]}).to_parquet(
        tmp_path / "irrigation_share_ne.parquet")
    pd.DataFrame({"fips": ["31001"], "year": [2020], "irrigation_share": [0.4]}).to_parquet(
        tmp_path / "irrigation_share_ne_soy.parquet")

    corn = build_modeling_table("ne", "31")
    soy = build_modeling_table("ne", "31", crop="soybeans")
    assert (corn["yield_bu_acre"].item(), corn["irrigation_share"].item()) == (190.0, 0.7)
    assert (soy["yield_bu_acre"].item(), soy["irrigation_share"].item()) == (62.0, 0.4)
    assert soy["gdd"].item() == corn["gdd"].item()   # same county, same summer
