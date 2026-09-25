"""Sky & Soil - the one place colour is defined.

Every chart, map and page in this project reads its colours from here, and the
same file is meant to be copied unchanged into the other apps in the suite so
they look like siblings rather than cousins.

Why a module rather than a few hex codes sprinkled around: colour is a system,
not a decoration. When the diverging scale on the error map and the diverging
scale on an Altair chart come from different constants, they drift apart, and
the viewer ends up learning two visual languages for one idea.

THE PALETTE IS VALIDATED, NOT CHOSEN BY EYE. The six categorical slots below
were run through a colour-vision-deficiency checker (protan/deutan simulation,
OKLab distance) against the actual chart surfaces in both light and dark mode.
Every adjacent pair clears the separation target, every slot sits inside the
lightness band and above the chroma floor, and every slot clears 3:1 contrast
against its surface. The first three additionally clear the harder all-pairs
gate, which is what scatter plots and choropleths need. See docs/DESIGN-SYSTEM.md
for the numbers and how to re-run the check.

Past six series, fold the rest into "Other" or use small multiples. A seventh
generated hue is how palettes quietly become unreadable.
"""

from __future__ import annotations

from matplotlib.colors import LinearSegmentedColormap

# --------------------------------------------------------------- brand anchors

PLAINS_BLUE = "#1A5B96"   # primary - sky over the Platte
TURNED_EARTH = "#A8622D"  # secondary - the other pole of the diverging scale
RIPE_WHEAT = "#A8861E"    # accent, used sparingly

# ------------------------------------------------------------------- surfaces

LIGHT = {
    "surface": "#FBFAF8",      # chart surface - warm paper, not clinical white
    "plane": "#F4F2EE",        # page behind the chart
    "ink": "#1C2024",          # primary text
    "ink_secondary": "#4A5159",
    "muted": "#7C848C",        # axis labels, captions
    "grid": "#E4E0D9",         # hairline gridlines
    "axis": "#C8C3B9",
    "boundary": "#FFFFFF",     # county borders on a filled map
    "no_data": "#D5CFC5",   # must NOT resemble the diverging midpoint
}

DARK = {
    "surface": "#15181B",
    "plane": "#0E1013",
    "ink": "#F5F3EF",
    "ink_secondary": "#C4C8CC",
    "muted": "#8A9298",
    "grid": "#272B2F",
    "axis": "#3A3F44",
    "boundary": "#15181B",
    "no_data": "#2E3237",   # must NOT resemble the diverging midpoint
}

# ------------------------------------------------- categorical (fixed order)
#
# Fixed order is the colour-blind-safety mechanism, not a style choice: the
# sequence was validated pair by pair. Assign slot 1 to the first series, slot 2
# to the second, and never cycle or reorder by rank - a filter that drops a
# series must not repaint the survivors.

CATEGORICAL = [PLAINS_BLUE, TURNED_EARTH, "#1B8F7A", RIPE_WHEAT, "#D46A8B", "#5A4BA8"]
CATEGORICAL_DARK = ["#4A90D9", "#C5722F", "#1F9E7E", "#B78F1D", "#D26C8F", "#8B7CDC"]

MAX_SERIES = 6  # past this, fold into "Other" or facet

# --------------------------------------------------------------- sequential
#
# One hue, light to dark. Magnitude has an order, and a single hue is the only
# encoding a viewer reads as ordered without being told. Rainbow scales get read
# as unordered categories, which is why they are never used here.

SEQUENTIAL_STEPS = ["#D6E4F2", "#AFC9E4", "#85A9D2", "#5A88BF", "#3670A8",
                    PLAINS_BLUE, "#0F4273"]
SEQUENTIAL_STEPS_DARK = ["#12304D", "#1A4670", "#215E95", "#3178B8", "#4A90D9",
                         "#79AFE4", "#A9CCEF"]

# --------------------------------------------------------------- diverging
#
# Two hues and a NEUTRAL midpoint, always on a symmetric domain so the neutral
# lands exactly on zero. An asymmetric diverging scale silently misreports which
# places are above and below.

DIVERGING_STEPS = [PLAINS_BLUE, "#6D9BC4", "#EDEAE4", "#D2A176", TURNED_EARTH]
DIVERGING_STEPS_DARK = ["#4A90D9", "#2E6A9E", "#33383D", "#9A6236", "#C5722F"]

# ------------------------------------------------------------------- status
#
# Reserved. A status colour never doubles as "series 7", and never carries
# meaning on its own - it ships with an icon or a word beside it.

STATUS = {"good": "#1F7A33", "warning": "#D39E00",
          "serious": "#C25A22", "critical": "#B3332F"}


def colormap(kind: str = "sequential", dark: bool = False) -> LinearSegmentedColormap:
    """A matplotlib colormap from the same steps the web charts use.

    Keeping matplotlib and Altair on one set of hex values is the whole point of
    this module: the PNG in the README and the interactive map in the app should
    not be two different visual languages.
    """
    steps = {
        ("sequential", False): SEQUENTIAL_STEPS,
        ("sequential", True): SEQUENTIAL_STEPS_DARK,
        ("diverging", False): DIVERGING_STEPS,
        ("diverging", True): DIVERGING_STEPS_DARK,
    }[(kind, dark)]
    return LinearSegmentedColormap.from_list(f"skysoil_{kind}", steps)


def altair_range(kind: str = "sequential", dark: bool = False) -> list[str]:
    """The same steps as a plain list, for `alt.Scale(range=...)`."""
    return {
        ("sequential", False): SEQUENTIAL_STEPS,
        ("sequential", True): SEQUENTIAL_STEPS_DARK,
        ("diverging", False): DIVERGING_STEPS,
        ("diverging", True): DIVERGING_STEPS_DARK,
        ("categorical", False): CATEGORICAL,
        ("categorical", True): CATEGORICAL_DARK,
    }[(kind, dark)]


def series_colors(n: int, dark: bool = False) -> list[str]:
    """Colours for `n` series, in fixed slot order.

    Raises past the series cap rather than inventing a hue, because a generated
    seventh colour is indistinguishable from one of the first six for a
    meaningful share of viewers.
    """
    palette = CATEGORICAL_DARK if dark else CATEGORICAL
    if n > len(palette):
        raise ValueError(
            f"{n} series exceeds the validated palette ({len(palette)} slots). "
            "Fold the tail into 'Other', or use small multiples.")
    return palette[:n]
