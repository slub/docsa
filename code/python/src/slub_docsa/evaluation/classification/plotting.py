"""Methods to generate various plots for evaluation."""

# pylint: disable=dangerous-default-value, too-many-arguments, too-many-locals

import math
import os

from typing import Optional, Sequence, cast, Any, Tuple

import numpy as np

import plotly.graph_objects as go
import plotly.express as px

from plotly.subplots import make_subplots


def _get_marker_color(i, colorlist=px.colors.qualitative.Plotly):
    return colorlist[i % len(colorlist)]


def score_matrix_box_plot(
    score_matrix: np.ndarray,
    model_names: Sequence[str],
    score_names: Sequence[str],
    score_ranges: Sequence[Tuple[Optional[float], Optional[float]]],
    columns: int = 1
) -> Any:
    """Return box plot visualizing the overall score matrix.

    Shows one plot for each score. Each model is represented by a box plot visualizing all scores for each
    cross-validation split.

    Parameters
    ----------
    score_matrix: numpy.ndarray
        the score matrix of shape `(n_splits, len(models), len(scores))`
    model_names: List[str]
        the labels for each model
    score_names: List[str]
        the labels for each score function
    score_ranges: List[Tuple[float, float]]
        min-max ranges for the y-axis of each score
    columns: int = 1
        the number of columns for the grid layout that is used to show each plot for each score function

    Returns
    -------
    plotly.Figure
        the plotly figure that can be save to a file, etc.
    """
    score_matrix = np.array(score_matrix)
    _, _, n_scores = score_matrix.shape

    fig = cast(Any, make_subplots(
        rows=math.ceil(n_scores / columns),
        cols=columns,
        subplot_titles=score_names,
    ))
    for i in range(n_scores):
        for j, model_name in enumerate(model_names):
            box = cast(Any, go.Box)(
                name=model_name,
                y=score_matrix[:, j, i],
                showlegend=i <= 0,
                legendgroup=j,
                marker_color=_get_marker_color(j)
            )
            fig.add_trace(box, row=math.floor(i / columns) + 1, col=(i % columns) + 1)

    for i in range(n_scores):
        fig.update_layout(
            {"yaxis" + str(i + 1): {"range": score_ranges[i]}}
        )

    return fig


def score_matrices_box_plot(
    score_matrices: Sequence[np.ndarray],
    dataset_names: Sequence[str],
    model_names: Sequence[str],
    score_names: Sequence[str],
    score_ranges: Sequence[Tuple[Optional[float], Optional[float]]],
    columns: int = 1
) -> Any:
    """Return box plot for multiple score matrices evaluated for different datasets.

    Shows one plot for each score function. Datasets are represented by boxes of different colors. Models are sorted on
    the x-axis.

    Parameters
    ----------
    score_matrices: List[np.ndarray]
        list of score matrices of shape `(n_splits, len(models), len(scores))` for each dataset
    dataset_names: List[str]
        the labels for each dataset
    model_names: List[str]
        the labels for each model
    score_names: List[str]
        the labels for each score
    score_ranges: List[Tuple[float, float]]
        min-max ranges for the y-axis of each score
    columns: int = 1
        the number of columns for the grid layout that is used to show each plot for each score function

    Returns
    -------
    plotly.Figure
        the plotly figure that can be save to a file, etc.
    """
    n_splits, n_models, n_scores = score_matrices[0].shape

    fig = cast(Any, make_subplots(
        rows=math.ceil(n_scores / columns),
        cols=columns,
        subplot_titles=score_names,
        horizontal_spacing=0.1 / columns,
        vertical_spacing=0.1 / columns
    ))

    for i in range(n_scores):
        for k, dataset_name in enumerate(dataset_names):
            x_values = np.repeat(model_names, n_splits)
            y_values = np.hstack([score_matrices[k][:, j, i] for j in range(n_models)])
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
            fig.add_trace(box, row=math.floor(i / columns) + 1, col=(i % columns) + 1)

    fig.update_layout(
        boxmode='group'
    )

    for i in range(n_scores):
        # set y axis range for every plot
        fig.update_layout({
            "yaxis" + str(i + 1): {
                "range": score_ranges[i],
                # "title": score_names[i],
                # "title_font": {
                #     "size": 12,
                # }
            },
        })
        # hide x axis tick labels for all plots but the last row
        if i < n_scores - columns:
            fig.update_layout({
                "xaxis" + str(i + 1): {
                    "showticklabels": False,
                }
            })
        # set plot title font size
        fig.layout.annotations[i].font.size = 12

    return fig


def precision_recall_plot(
    score_matrix: np.ndarray,
    model_names: Sequence[str],
) -> Any:
    """Return a precision recall scatter plot.

    Parameters
    ----------
    score_matrix: numpy.ndarray
        the score matrix that contains both precision (first index) and recall (second index) in a matrix of shape
        `(n_splits, len(models), 2)`
    model_names: List[str]
        the list of model labels

    Returns
    -------
    plotly.Figure
        the plotly figure that can be save to a file, etc.
    """
    _, n_models, _ = score_matrix.shape

    fig = cast(Any, go.Figure)()

    for i in range(n_models):
        fig.add_trace(
            cast(Any, go.Scatter)(
                x=score_matrix[:, i, 0],
                y=score_matrix[:, i, 1],
                name=model_names[i],
                mode="markers"
            )
        )

    fig.update_layout(
        xaxis_title="precision micro",
        yaxis_title="recall micro",
        xaxis_range=[0, 1],
        yaxis_range=[0, 1],
    )

    return fig


def per_subject_precision_recall_vs_samples_plot(
    score_matrix: np.ndarray,
    model_names: Sequence[str],
) -> Any:
    """Return plot showing precision vs. number of test examples per subject.

    Parameters
    ----------
    score_matrix: numpy.ndarray
        the per-subject score matrix that contains both the number of test samples (first index), recall (second index)
        and precision (third index) for every subject in a matrix of shape `(n_splits, len(models), 3, len(subjects))`
    model_names: List[str]
        the list of model labels

    Returns
    -------
    plotly.Figure
        the plotly figure that can be save to a file, etc.
    """
    _, n_models, _, _ = score_matrix.shape

    fig = cast(Any, make_subplots(
        rows=1,
        cols=2,
        subplot_titles=["precision vs. # test samples", "recall vs. # test samples"],
    ))

    for i in range(n_models):
        for j in [0, 1]:
            fig.add_trace(
                cast(Any, go.Scatter)(
                    x=score_matrix[:, i, 0, :].flatten(),
                    y=score_matrix[:, i, j + 1, :].flatten(),
                    name=model_names[i],
                    showlegend=j <= 0,
                    legendgroup=i,
                    marker_color=_get_marker_color(i),
                    mode="markers"
                ),
                row=1, col=j + 1,
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
        end=bin_max + bin_size,
        size=bin_size,
    )


def per_subject_score_histograms_plot(
    score_matrix: np.ndarray,
    model_names: Sequence[str],
    score_names: Sequence[str],
    score_ranges: Sequence[Tuple[Optional[float], Optional[float]]],
) -> Any:
    """Return a plot of score histograms illustrating the score distribution for all subjects individually.

    Parameters
    ----------
    score_matrix: numpy.ndarray
        the per-subject score matrix that contains all scores for every subject in a matrix of shape
        `(n_splits, len(models), len(scores_functions), len(subjects))`
    model_names: List[str]
        the list of model labels
    score_names: List[str]
        the list of score labels
    score_ranges: List[Tuple[float, float]]
        min-max ranges for each score

    Returns
    -------
    plotly.Figure
        the plotly figure that can be save to a file, etc.
    """
    _, n_models, n_scores, _ = score_matrix.shape

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
        bin_dict = _calculate_score_histogram_bin(score_matrix[:, :, i, :].flatten(), score_ranges[i])
        bin_ranges.append([bin_dict["start"], bin_dict["end"]])

        for j, model_name in enumerate(model_names):
            values = score_matrix[:, j, i, :].flatten()

            fig.add_trace(
                cast(Any, go.Histogram)(
                    x=values[np.logical_not(np.isnan(values))],  # type: ignore
                    xbins=bin_dict,
                    autobinx=False,
                    marker_color=_get_marker_color(j),
                    name=f"{model_name} - {score_name}",
                ),
                row=i + 1,
                col=j + 1
            )

    for i in range(n_scores):
        for j in range(n_models):
            fig.update_layout(
                {"xaxis" + str(i * n_models + j + 1): {"range": bin_ranges[i]}}
            )

    fig.update_layout(showlegend=False)

    return fig


def ann_training_history_plot(
    training_scores: Sequence[Tuple[float, float, float]],
    test_scores: Sequence[Tuple[float, float, float]],
) -> Any:
    """Return a illustration of scores that were recorded for each epoch while training a neural network.

    Parameters
    ----------
    training_loss: Sequence[Tuple[float, float, float]]
        the training loss, t=0.1 f1 score and t=0.5 f1 score over all epochs
    test_loss: Sequence[Tuple[float, float, float]]
        the test loss, t=0.1 f1 score and t=0.5 f1 score over all epochs

    Returns
    -------
    plotly.Figure
        the plotly figure that can be save to a file, etc.
    """
    fig = cast(Any, make_subplots(
        rows=1,
        cols=2,
        subplot_titles=["loss", "f1 scores"],
    ))

    epochs = list(range(len(training_scores)))

    for i, name in enumerate(["loss", "t=0.1 f1", "t=0.5 f1"]):
        fig.add_trace(
            cast(Any, go.Scatter)(
                x=epochs,
                y=[score[i] for score in training_scores],
                mode="lines",
                name="training " + name,
                line=dict(color=_get_marker_color(0), width=3, dash='dot' if i == 2 else None),
            ), row=1, col=1 if i == 0 else 2,
        )

        fig.add_trace(
            cast(Any, go.Scatter)(
                x=epochs,
                y=[score[i] for score in test_scores],
                mode="lines",
                name="test " + name,
                line=dict(color=_get_marker_color(1), width=3, dash='dot' if i == 2 else None),
            ), row=1, col=1 if i == 0 else 2,
        )

    max_loss = np.max([np.max([score[0] for score in training_scores]), np.max([score[0] for score in test_scores])])

    fig.update_layout({"yaxis1": {"range": [0.0, max_loss], "title": "loss"}})
    fig.update_layout({"xaxis1": {"range": [0.0, len(training_scores) - 1], "title": "epoch"}})
    fig.update_layout({"yaxis2": {"range": [0.0, 1.0], "title": "f1 score (micro)"}})
    fig.update_layout({"xaxis2": {"range": [0.0, len(training_scores) - 1], "title": "epoch"}})

    return fig


def write_multiple_figure_formats(
    figure: Any,
    filepath: str,
):
    """Write a plotly figure as a html, pdf and jpg file.

    Can be used to export any of the plotly figures returned by the plotting methods of this module, e.g.,
    `score_matrix_box_plot`.

    Parameters
    ----------
    figure: plotly.Figure
        the plotly figure that is being exported
    filepath: str
        the filepath where the figure is being saved to; an appropriate file extension is added (html, pdf, jpg) for
        each of the three export formats

    Returns
    -------
    None
    """
    if os.path.dirname(filepath):
        os.makedirs(os.path.dirname(filepath), exist_ok=True)

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
