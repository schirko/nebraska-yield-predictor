"""Charts with years along the bottom.

Streamlit's st.line_chart and st.bar_chart read a year index as an ordinary number and
label the axis "2,000  2,005  2,010". These draw the same charts with Altair so the years
read as years, and the series take the validated Sky & Soil colours from theme.py (the
built-in charts used Streamlit's own palette instead).

    st.altair_chart(year_chart(frame, y_label="bu/acre"), width="stretch")

`frame` has one row per year (the index) and one column per series; the column names are
the legend labels.
"""

from __future__ import annotations

import altair as alt
import pandas as pd

from yieldpred.theme import series_colors


def _long(frame: pd.DataFrame) -> tuple[pd.DataFrame, list[str]]:
    series = [str(c) for c in frame.columns]
    wide = frame.copy()
    wide.columns = series
    wide.index = pd.Index(wide.index.astype(int), name="year")
    long = wide.reset_index().melt(id_vars="year", var_name="series", value_name="value").dropna()
    return long, series


def year_chart(frame: pd.DataFrame, y_label: str | None = None, kind: str = "line",
               height: int = 280) -> alt.Chart:
    """A line (or bar) chart of `frame`'s columns by year, with plain year labels."""
    long, series = _long(frame)
    color = alt.Color("series:N", title=None,
                      scale=alt.Scale(domain=series, range=series_colors(len(series))),
                      legend=alt.Legend(orient="bottom") if len(series) > 1 else None)
    tooltip = [alt.Tooltip("year:O", title="Year"), alt.Tooltip("series:N", title="Series"),
               alt.Tooltip("value:Q", title=y_label or "Value", format=",.1f")]
    y = alt.Y("value:Q", title=y_label, scale=alt.Scale(zero=kind == "bar"))
    if kind == "bar":
        # One bar per year: an ordered axis, labelled every fifth year so phones stay readable.
        x = alt.X("year:O", title=None,
                  axis=alt.Axis(labelAngle=0, labelExpr="datum.value % 5 == 0 ? datum.label : ''"))
        chart = alt.Chart(long).mark_bar().encode(x=x, y=y, color=color, tooltip=tooltip)
    else:
        x = alt.X("year:Q", title=None, axis=alt.Axis(format="d", tickMinStep=1),
                  scale=alt.Scale(zero=False, nice=False))
        chart = alt.Chart(long).mark_line(strokeWidth=2).encode(x=x, y=y, color=color, tooltip=tooltip)
    return chart.properties(height=height)
