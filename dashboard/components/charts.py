"""
charts.py
---------
Visualization functions for FairML.
"""

import pandas as pd
import altair as alt


def plot_fairness_bar(metric_dict: dict) -> alt.Chart:
    """
    Plot a bar chart of fairness metrics across protected groups.

    Args:
        metric_dict (dict): Mapping group -> metric

    Returns:
        alt.Chart: Altair bar chart.
    """
    df = pd.DataFrame(list(metric_dict.items()), columns=["Group", "Metric"])

    chart = (
        alt.Chart(df)
        .mark_bar(color="orange")
        .encode(x="Group:N", y="Metric:Q")
        .properties(
            title="Fairness Metric Across Groups",
            width=600,
            height=400
        )
    )

    return chart
