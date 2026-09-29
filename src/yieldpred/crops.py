"""The crops the pipeline can model, and the one place their names and NASS codes live.

Every script takes `--crop` (default `corn`). Corn keeps every file name the project has always
used, so nothing already published moves; another crop adds its `suffix` to each derived file:

    corn       model_table.parquet        model_table_ia.parquet       ne_corn_yield_county.parquet
    soybeans   model_table_soy.parquet    model_table_ia_soy.parquet   ne_soybeans_yield_county.parquet

Deliberately plain data, no imports from the rest of the package, so the app layer can use it
without pulling in the modelling code.

What changes between crops and what doesn't:

- **Changes:** the NASS query (commodity, and for corn the "grain" utilization, since corn is also
  cut for silage), the acreage behind the irrigation share, the file names, and the words the app
  shows.
- **Doesn't:** the weather (a county's summer is the same summer for both crops; the features
  already include August rain, which matters most for soybeans), soils, terrain, the model and
  the validation. Whether the corn feature set is right for soybeans is a question for the data,
  not an assumption: scripts/train_baseline.py --crop soybeans runs the same feature study.
"""

from __future__ import annotations

from dataclasses import dataclass, field


@dataclass(frozen=True)
class Crop:
    key: str                    # what --crop takes, and the word in the yield file name
    name: str                   # for people: "Corn", "Soybeans"
    suffix: str                 # added to derived file names; "" keeps the original names
    nass: dict = field(default_factory=dict)   # Quick Stats filters that pick this crop
    unit: str = "bu/acre"
    word: str = ""              # in a sentence, before a noun: "corn yields", "soybean yields"


CROPS = {
    "corn": Crop("corn", "Corn", "", {"commodity_desc": "CORN", "util_practice_desc": "GRAIN"}, word="corn"),
    "soybeans": Crop("soybeans", "Soybeans", "soy", {"commodity_desc": "SOYBEANS"}, word="soybean"),
}
DEFAULT_CROP = "corn"


def get_crop(key: str | Crop = DEFAULT_CROP) -> Crop:
    """The Crop for a --crop value (case-insensitive; "soy" and "soybean" also work)."""
    if isinstance(key, Crop):
        return key
    k = str(key).strip().lower()
    k = {"soy": "soybeans", "soybean": "soybeans"}.get(k, k)
    if k not in CROPS:
        raise ValueError(f"unknown crop {key!r}; choose from {', '.join(CROPS)}")
    return CROPS[k]


def crop_stem(name: str, crop: str | Crop = DEFAULT_CROP) -> str:
    """`name` with the crop's suffix: unchanged for corn, `name_soy` for soybeans."""
    suffix = get_crop(crop).suffix
    return f"{name}_{suffix}" if suffix else name


def yields_file(state: str, crop: str | Crop = DEFAULT_CROP) -> str:
    """The county yield download: ne_corn_yield_county.parquet, ne_soybeans_yield_county.parquet."""
    return f"{state.lower()}_{get_crop(crop).key}_yield_county.parquet"
