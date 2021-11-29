"""Common plots that are can be printed as part of an evaluation experiment."""

import os
from typing import List, NamedTuple, Sequence

import numpy as np

from slub_docsa.evaluation.plotting import score_matrices_box_plot, write_multiple_figure_formats
from slub_docsa.evaluation.plotting import per_subject_precision_recall_vs_samples_plot
from slub_docsa.evaluation.plotting import precision_recall_plot, score_matrix_box_plot
from slub_docsa.evaluation.plotting import per_subject_score_histograms_plot
from slub_docsa.experiments.common.scores import DefaultScoreLists


class DefaultScoreMatrixDatasetResult(NamedTuple):
    """Stores evaluation result matrices as well as model and score info."""

    dataset_name: str
    model_names: List[str]
    overall_score_matrix: np.ndarray
    per_class_score_matrix: np.ndarray
    overall_score_lists: DefaultScoreLists
    per_class_score_lists: DefaultScoreLists


DefaultScoreMatrixResult = Sequence[DefaultScoreMatrixDatasetResult]


def write_default_classification_plots(
    evaluation_result: DefaultScoreMatrixResult,
    plot_directory: str,
    filename_suffix: str,
):
    """Write all default plots to a file."""
    write_multiple_score_matrix_box_plot(
        evaluation_result,
        os.path.join(plot_directory, f"overall_score_plot_{filename_suffix}"),
    )

    for dataset_result in evaluation_result:
        prefix = f"{dataset_result.dataset_name}"

        write_precision_recall_plot(
            dataset_result,
            os.path.join(plot_directory, f"{prefix}_precision_recall_plot_{filename_suffix}"),
        )

        write_per_subject_precision_recall_vs_samples_plot(
            dataset_result,
            os.path.join(plot_directory, f"{prefix}_per_subject_precision_recall_vs_samples_plot_{filename_suffix}"),
        )

        write_score_matrix_box_plot(
            dataset_result,
            os.path.join(plot_directory, f"{prefix}_score_plot_{filename_suffix}"),
        )

        write_per_subject_score_histograms_plot(
            dataset_result,
            os.path.join(plot_directory, f"{prefix}_per_subject_score_{filename_suffix}"),
        )


def write_multiple_score_matrix_box_plot(
    evaluation_result: DefaultScoreMatrixResult,
    plot_filepath: str
):
    """Generate the score matrix box plot comparing multiple datasets and write it as html file."""
    score_matrices = [er.overall_score_matrix for er in evaluation_result]
    dataset_names = [er.dataset_name for er in evaluation_result]
    model_names = evaluation_result[0].model_names
    score_names = evaluation_result[0].overall_score_lists.names
    score_ranges = evaluation_result[0].overall_score_lists.ranges

    # generate figure
    figure = score_matrices_box_plot(
        score_matrices,
        dataset_names,
        model_names,
        score_names,
        score_ranges,
        columns=2
    )
    write_multiple_figure_formats(figure, plot_filepath)


def write_score_matrix_box_plot(
    evaluation_result: DefaultScoreMatrixDatasetResult,
    plot_filepath: str
):
    """Generate the score matrix box plot from evaluation results and write it as html file."""
    # generate figure
    figure = score_matrix_box_plot(
        evaluation_result.overall_score_matrix,
        evaluation_result.model_names,
        evaluation_result.overall_score_lists.names,
        evaluation_result.overall_score_lists.ranges,
        columns=2
    )
    write_multiple_figure_formats(figure, plot_filepath)


def write_precision_recall_plot(
    evaluation_result: DefaultScoreMatrixDatasetResult,
    plot_filepath: str
):
    """Generate the precision recall plot from evaluation results and write it as html file."""
    score_names = evaluation_result.overall_score_lists.names
    if "top3 precision micro" not in score_names or "top3 recall micro" not in score_names:
        raise ValueError("score matrix needs to contain top3 precision/recall micro scores")

    precision_idx = score_names.index("top3 precision micro")
    recall_idx = score_names.index("top3 recall micro")

    figure = precision_recall_plot(
        evaluation_result.overall_score_matrix[:, [precision_idx, recall_idx], :],
        evaluation_result.model_names,
    )
    write_multiple_figure_formats(figure, plot_filepath)


def write_per_subject_score_histograms_plot(
    evaluation_result: DefaultScoreMatrixDatasetResult,
    plot_filepath: str
):
    """Generate the subject score histograms plot from evaluation results and write it as html file."""
    figure = per_subject_score_histograms_plot(
        evaluation_result.per_class_score_matrix,
        evaluation_result.model_names,
        evaluation_result.per_class_score_lists.names,
        evaluation_result.per_class_score_lists.ranges,
    )
    write_multiple_figure_formats(figure, plot_filepath)


def write_per_subject_precision_recall_vs_samples_plot(
    evaluation_result: DefaultScoreMatrixDatasetResult,
    plot_filepath: str
):
    """Generate the per subject precision vs samples plot from evaluation results and write it as html file."""
    score_names = evaluation_result.per_class_score_lists.names
    if "t=0.5 precision" not in score_names or "t=0.5 recall" not in score_names \
            or "# test samples" not in score_names:
        raise ValueError("score matrix needs to contain t=0.5 precision, recall and # test examples scores")

    precision_idx = score_names.index("t=0.5 precision")
    recall_idx = score_names.index("t=0.5 recall")
    samples_idx = score_names.index("# test samples")

    figure = per_subject_precision_recall_vs_samples_plot(
        evaluation_result.per_class_score_matrix[:, [samples_idx, precision_idx, recall_idx], :, :],
        evaluation_result.model_names,
    )
    write_multiple_figure_formats(figure, plot_filepath)