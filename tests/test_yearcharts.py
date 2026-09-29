"""Year axes read as years ("2005"), not numbers ("2,005")."""

import pandas as pd

from yieldpred.theme import CATEGORICAL
from yieldpred.yearcharts import year_chart

FRAME = pd.DataFrame({"Actual": [150.0, 160.0, None], "Predicted": [148.0, 158.0, 170.0]},
                     index=pd.Index([2000, 2001, 2002], name="year"))


def test_line_chart_labels_years_without_thousands_separators():
    spec = year_chart(FRAME, y_label="bu/acre").to_dict()
    assert spec["encoding"]["x"]["axis"]["format"] == "d"
    assert spec["encoding"]["x"]["field"] == "year"


def test_series_use_the_validated_palette_in_column_order():
    spec = year_chart(FRAME).to_dict()
    scale = spec["encoding"]["color"]["scale"]
    assert scale["domain"] == ["Actual", "Predicted"] and scale["range"] == CATEGORICAL[:2]


def test_missing_values_are_dropped_not_drawn_as_zero():
    chart = year_chart(FRAME)
    assert len(chart.data) == 5  # 6 cells, one missing


def test_bar_chart_labels_every_fifth_year():
    spec = year_chart(FRAME[["Predicted"]], kind="bar").to_dict()
    assert spec["mark"]["type"] == "bar" and "% 5 == 0" in spec["encoding"]["x"]["axis"]["labelExpr"]
    assert "legend" in spec["encoding"]["color"] and spec["encoding"]["color"]["legend"] is None
