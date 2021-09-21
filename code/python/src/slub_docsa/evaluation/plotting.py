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


def precision_recall_plot(
    score_matrix: np.ndarray,
    model_names: List[str],
) -> Any:
    """Return a precision recall scatter plot."""
    n_models, _, _ = score_matrix.shape

    fig = cast(Any, go.Figure)()

    for i in range(n_models):
        fig.add_trace(
            cast(Any, go.Scatter)(
                x=score_matrix[i, 0, :],
                y=score_matrix[i, 1, :],
                name=model_names[i],
                mode="markers"
            )
        )

    fig.update_layout(
        xaxis_title="top3 precision micro",
        yaxis_title="top3 recall micro",
        xaxis_range=[0, 1],
        yaxis_range=[0, 1],
    )

    return fig


def _calculate_score_histogram_bin(values, score_range):
    """Calculate histogram bins from values and optional score range."""
    non_nan_values = values[np.logical_not(np.isnan(values))]
    bin_min = np.min(non_nan_values) if score_range[0] is None else score_range[0]
    bin_max = np.max(non_nan_values) if score_range[1] is None else score_range[1]
    bin_size = (bin_max - bin_min) / 20.0

    return dict(
        start=bin_min,
        end=bin_max+bin_size,
        size=bin_size,
    )


def per_subject_score_histograms_plot(
    score_matrix: np.ndarray,
    model_names: List[str],
    score_names: List[str],
    score_ranges: List[Tuple[float, float]],
) -> Any:
    """Return a plot of score histograms showing score distribution for over all subjects individually."""
    n_models, n_scores, _, _ = score_matrix.shape

    if n_scores != len(score_names):
        raise ValueError("number of columns in score matrix does not match score names list")

    if n_models != len(model_names):
        raise ValueError("number of rows in score matrix does not match model names list")

    fig = cast(Any, make_subplots(
        rows=n_scores,
        cols=n_models,
        subplot_titles=[mn + "<br>" + sn for sn in score_names for mn in model_names],
    ))

    bin_ranges = []

    for i, score_name in enumerate(score_names):
        bin_dict = _calculate_score_histogram_bin(score_matrix[:, i, :, :].flatten(), score_ranges[i])
        bin_ranges.append([bin_dict["start"], bin_dict["end"]])

        for j, model_name in enumerate(model_names):
            values = score_matrix[j, i, :, :].flatten()

            fig.add_trace(
                cast(Any, go.Histogram)(
                    x=values[np.logical_not(np.isnan(values))],
                    xbins=bin_dict,
                    autobinx=False,
                    marker_color=_get_marker_color(j),
                    name=f"{model_name} - {score_name}",
                ),
                row=i+1,
                col=j+1
            )

    for i in range(n_scores):
        for j in range(n_models):
            fig.update_layout(
                {"xaxis" + str(i * n_models + j + 1): {"range": bin_ranges[i]}}
            )

    fig.update_layout(showlegend=False)

    return fig
