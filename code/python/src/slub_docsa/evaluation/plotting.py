"""Generates plots for evaluation."""

# pylint: disable=dangerous-default-value

import math
from typing import cast, Any, List, Tuple

import numpy as np

import plotly.graph_objects as go
import plotly.express as px

from plotly.subplots import make_subplots


def _get_marker_color(i, colorlist=px.colors.qualitative.Plotly):
    return colorlist[i % len(colorlist)]


def score_matrix_box_plot(
    score_matrix: np.ndarray,
    model_names: List[str],
    score_names: List[str],
    score_ranges: List[Tuple[float, float]],
    columns: int = 1
) -> Any:
    """Return box plot for score matrix."""
    _, n_scores, _ = score_matrix.shape

    fig = cast(Any, make_subplots(
        rows=math.ceil(n_scores/columns),
        cols=columns,
        subplot_titles=score_names,
    ))
    for i in range(n_scores):
        for j, model_name in enumerate(model_names):
            box = cast(Any, go.Box)(
                name=model_name,
                y=score_matrix[j, i, :],
                showlegend=i <= 0,
                legendgroup=j,
                marker_color=_get_marker_color(j)
            )
            fig.add_trace(box, row=math.floor(i/columns)+1, col=(i % columns)+1)

    for i in range(n_scores):
        fig.update_layout(
            {"yaxis" + str(i+1): {"range": score_ranges[i]}}
        )

    return fig
