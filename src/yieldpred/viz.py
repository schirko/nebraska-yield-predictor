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
from matplotlib.colors import LinearSegmentedColormap

DIVERGING = LinearSegmentedColormap.from_list(
    "blue_grey_orange", ["#2166AC", "#7FA8CE", "#E8E8E8", "#E8A85C", "#B35806"])
SEQUENTIAL = LinearSegmentedColormap.from_list(
    "greens", ["#F2F7F2", "#C6E0C2", "#8CC084", "#4E9A51", "#1E6B33"])

INK = "#1A1A1A"
MUTED = "#6B6B6B"
BOUNDARY = "#FFFFFF"


def _style(ax, title: str, subtitle: str | None, source: str | None):
    ax.set_axis_off()
    ax.set_title(title, fontsize=13, fontweight="bold", color=INK, loc="left", pad=14)
    if subtitle:
        ax.text(0, 1.015, subtitle, transform=ax.transAxes, fontsize=9.5,
                color=MUTED, va="bottom")
    if source:
        ax.text(0, -0.03, source, transform=ax.transAxes, fontsize=8, color=MUTED, va="top")


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
             legend=True, missing_kwds={"color": "#F0F0F0", "edgecolor": BOUNDARY,
                                        "label": "no data"},
             legend_kwds={"shrink": 0.7, "label": legend_label or column,
                          "orientation": "vertical"},
             **kwargs)

    _style(ax, title, subtitle, source)
    fig.tight_layout()

    if out_path:
        out_path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(out_path, bbox_inches="tight", facecolor="white")
        plt.close(fig)
    return fig
