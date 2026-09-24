"""Map rendering shared by the analysis scripts and the Streamlit app.

Design rules followed here (they are conventions, not preferences):

* **Diverging data gets two hues and a neutral midpoint.** Model error is diverging
  - it has a meaningful zero - so over- and under-prediction get opposite hues and
  zero gets neutral grey. The blue/orange pair stays distinguishable under the
  common forms of colour blindness, unlike red/green.
* **Diverging scales are symmetric.** Limits are +/- the largest absolute value, so
  the neutral colour lands exactly on zero. An asymmetric diverging scale silently
  lies about which places are above and below.
* **Magnitude data gets one hue, light to dark.** Never a rainbow: viewers read
  rainbow bands as categories and can't order them.
* **The data is the loudest thing on the page.** Boundaries are thin and grey, the
  frame is gone, the title states the finding rather than the variable name.
"""

from __future__ import annotations

from pathlib import Path

import matplotlib
matplotlib.use("Agg")  # render to file; no desktop window needed
import matplotlib.pyplot as plt

from yieldpred.theme import LIGHT, altair_range, colormap

# Colour comes from yieldpred.theme so the PNGs in the README and the interactive
# charts in the app speak one visual language. That module explains why the
# palette is validated rather than chosen.
DIVERGING = colormap("diverging")
SEQUENTIAL = colormap("sequential")

INK = LIGHT["ink"]
MUTED = LIGHT["muted"]
BOUNDARY = LIGHT["boundary"]
NO_DATA = LIGHT["no_data"]
SURFACE = LIGHT["surface"]


def _style(ax, title: str, subtitle: str | None, source: str | None):
    ax.set_axis_off()
    ax.set_title(title, fontsize=13, fontweight="bold", color=INK, loc="left", pad=14)
    if subtitle:
        ax.text(0, 1.015, subtitle, transform=ax.transAxes, fontsize=9.5,
                color=MUTED, va="bottom")
    if source:
        ax.text(0, -0.03, source, transform=ax.transAxes, fontsize=8, color=MUTED, va="top")


def interactive_choropleth(gdf, column: str, legend_label: str, tooltips: list,
                           diverging: bool = False, height: int = 460,
                           value_format: str = ".1f"):
    """An Altair county map with hover tooltips.

    Altair ships with Streamlit, so this adds no dependency and works on Streamlit
    Community Cloud. The trade against matplotlib: the geometry is embedded in the
    page (hence the simplified display copy from `geo.display_geometry`), but the
    viewer can hover each county to read its name and value - which a PNG can never do.

    The same colour rules apply as for the static maps: a diverging scheme with a
    symmetric domain when zero is meaningful, a single-hue scheme otherwise.
    """
    import json

    import altair as alt

    if diverging:
        limit = float(gdf[column].abs().max())
        scale = alt.Scale(range=altair_range("diverging"),
                          domain=[-limit, limit], domainMid=0)
    else:
        scale = alt.Scale(range=altair_range("sequential"))

    # Hand Altair GeoJSON features rather than a GeoDataFrame: a geometry column
    # can't be converted to a table, and passing one makes Streamlit try. Going
    # through `to_json` also converts numpy types to plain JSON values.
    features = json.loads(gdf.to_json())["features"]
    source = alt.Data(values=features)

    encoded_tooltips = [
        alt.Tooltip(f"properties.{field}", type="quantitative", title=title,
                    format=value_format)
        if kind == "Q" else
        alt.Tooltip(f"properties.{field}", type="nominal", title=title)
        for field, kind, title in tooltips
    ]

    return (alt.Chart(source)
            .mark_geoshape(stroke="white", strokeWidth=0.6)
            .encode(color=alt.Color(f"properties.{column}:Q", scale=scale,
                                    legend=alt.Legend(title=legend_label)),
                    tooltip=encoded_tooltips)
            .project(type="mercator")
            .properties(width="container", height=height))


def choropleth(gdf, column: str, title: str, subtitle: str | None = None,
               source: str | None = None, diverging: bool = False,
               legend_label: str | None = None, out_path: Path | None = None,
               figsize=(9, 5.2)):
    """Render one county map. Returns the matplotlib figure."""
    fig, ax = plt.subplots(figsize=figsize, dpi=160)

    kwargs = {}
    if diverging:
        limit = float(gdf[column].abs().max())
        kwargs.update(cmap=DIVERGING, vmin=-limit, vmax=limit)
    else:
        kwargs.update(cmap=SEQUENTIAL)

    gdf.plot(column=column, ax=ax, linewidth=0.4, edgecolor=BOUNDARY,
             legend=True, missing_kwds={"color": NO_DATA, "edgecolor": BOUNDARY,
                                        "hatch": "////",
                                        "label": "no data"},
             legend_kwds={"shrink": 0.7, "label": legend_label or column,
                          "orientation": "vertical"},
             **kwargs)

    _style(ax, title, subtitle, source)
    fig.tight_layout()

    if out_path:
        out_path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(out_path, bbox_inches="tight", facecolor=SURFACE)
        plt.close(fig)
    return fig
