"""Generates plots for evaluation."""

# pylint: disable=dangerous-default-value, too-many-arguments, too-many-locals

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


def score_matrices_box_plot(
    score_matrices: List[np.ndarray],
    dataset_names: List[str],
    model_names: List[str],
    score_names: List[str],
    score_ranges: List[Tuple[float, float]],
    columns: int = 1
) -> Any:
    """Return box plot for multiple score matrix evaluated for different datasets."""
    n_models, n_scores, n_splits = score_matrices[0].shape

    fig = cast(Any, make_subplots(
        rows=math.ceil(n_scores/columns),
        cols=columns,
        subplot_titles=score_names,
        horizontal_spacing=0.1/columns,
        vertical_spacing=0.1/columns
    ))

    for i in range(n_scores):
        for k, dataset_name in enumerate(dataset_names):
            x_values = np.repeat(model_names, n_splits)
            y_values = np.hstack([score_matrices[k][j, i, :] for j in range(n_models)])
            box = cast(Any, go.Box)(
                name=dataset_name,
                x=x_values,
                y=y_values,
                offsetgroup=dataset_name,
                showlegend=i <= 0,
                legendgroup=k,
                marker_color=_get_marker_color(k),
                line={"width": 3},
            )
            fig.add_trace(box, row=math.floor(i/columns)+1, col=(i % columns)+1)

    fig.update_layout(
        boxmode='group'
    )

    for i in range(n_scores):
        # set y axis range for every plot
        fig.update_layout({
                "yaxis" + str(i+1): {
                    "range": score_ranges[i],
                    # "title": score_names[i],
                    # "title_font": {
                    #     "size": 12,
                    # }
                },
            }
        )
        # hide x axis tick labels for all plots but the last row
        if i < n_scores - columns:
            fig.update_layout({
                "xaxis" + str(i+1): {
                    "showticklabels": False,
                }
            })
        # set plot title font size
        fig.layout.annotations[i].font.size = 12

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


def per_subject_precision_recall_vs_samples_plot(
    score_matrix: np.ndarray,
    model_names: List[str],
) -> Any:
    """Return plot showing precision vs number of test examples per subject."""
    n_models, _, _, _ = score_matrix.shape

    fig = cast(Any, make_subplots(
        rows=1,
        cols=2,
        subplot_titles=["precision vs. # test samples", "recall vs. # test samples"],
    ))

    for i in range(n_models):
        for j in [0, 1]:
            fig.add_trace(
                cast(Any, go.Scatter)(
                    x=score_matrix[i, 0, :, :].flatten(),
                    y=score_matrix[i, j+1, :, :].flatten(),
                    name=model_names[i],
                    showlegend=j <= 0,
                    legendgroup=i,
                    marker_color=_get_marker_color(i),
                    mode="markers"
                ),
                row=1, col=j+1,
            )

    fig.update_layout(
        xaxis1_title="# test examples",
        yaxis1_title="t=0.5 precision",
        xaxis2_title="# test examples",
        yaxis2_title="t=0.5 recall",
        yaxis1_range=[0, 1],
        yaxis2_range=[0, 1],
    )

    return fig


def _calculate_score_histogram_bin(values, score_range):
    """Calculate histogram bins from values and optional score range."""
    non_nan_values = values[np.logical_not(np.isnan(values))]  # type: ignore
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
                    x=values[np.logical_not(np.isnan(values))],  # type: ignore
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


def ann_training_history_plot(
    training_loss: List[float],
    test_loss: List[float],
    train_best_threshold_f1_score: List[float],
    test_best_threshold_f1_score: List[float],
    train_top3_f1_score: List[float],
    test_top3_f1_score: List[float],
) -> Any:
    """Return a plot of scores that were recorded for each epoch while training a neural network."""
    fig = cast(Any, make_subplots(
        rows=1,
        cols=2,
        subplot_titles=["loss", "f1 scores"],
    ))

    epochs = list(range(len(training_loss)))

    fig.add_trace(
        cast(Any, go.Scatter)(
            x=epochs,
            y=training_loss,
            mode="lines",
            name="training loss",
            line=dict(color=_get_marker_color(0), width=3),
        ), row=1, col=1,
    )

    fig.add_trace(
        cast(Any, go.Scatter)(
            x=epochs,
            y=test_loss,
            mode="lines",
            name="test loss",
            line=dict(color=_get_marker_color(1), width=3),
         ), row=1, col=1,
    )

    fig.add_trace(
        cast(Any, go.Scatter)(
            x=epochs,
            y=train_best_threshold_f1_score,
            mode="lines",
            name="train t=best f1_score",
            line=dict(color=_get_marker_color(0), width=3),
        ), row=1, col=2,
    )

    fig.add_trace(
        cast(Any, go.Scatter)(
            x=epochs,
            y=test_best_threshold_f1_score,
            mode="lines",
            name="test t=best f1_score",
            line=dict(color=_get_marker_color(1), width=3),
        ), row=1, col=2,
    )

    fig.add_trace(
        cast(Any, go.Scatter)(
            x=epochs,
            y=train_top3_f1_score,
            mode="lines",
            name="train top3 f1_score",
            line=dict(color=_get_marker_color(0), width=3, dash='dot'),
        ), row=1, col=2,
    )

    fig.add_trace(
        cast(Any, go.Scatter)(
            x=epochs,
            y=test_top3_f1_score,
            mode="lines",
            name="test top3 f1_score",
            line=dict(color=_get_marker_color(1), width=3, dash='dot'),
         ), row=1, col=2,
    )

    max_loss = np.max([np.max(training_loss), np.max(test_loss)])

    fig.update_layout({"yaxis1": {"range": [0.0, max_loss], "title": "loss"}})
    fig.update_layout({"xaxis1": {"range": [0.0, len(training_loss)-1], "title": "epoch"}})
    fig.update_layout({"yaxis2": {"range": [0.0, 1.0], "title": "f1 score (micro)"}})
    fig.update_layout({"xaxis2": {"range": [0.0, len(training_loss)-1], "title": "epoch"}})

    return fig


def write_multiple_figure_formats(
    figure: Any,
    filepath: str,
):
    """Write a plotly figure as a html, pdf and jpg file."""
    figure.write_html(
        f"{filepath}.html",
        include_plotlyjs="cdn",
    )
    figure.write_image(
        f"{filepath}.pdf",
        width=1600, height=900
    )
    figure.write_image(
        f"{filepath}.jpg",
        width=1600, height=900
    )
