"""Clustering experiments for qucosa data."""

# pylint: disable=invalid-name

import logging
import os
from typing import Optional

import numpy as np

from slub_docsa.common.paths import get_figures_dir
from slub_docsa.data.preprocess.subject import prune_subject_targets_to_level, subject_label_breadcrumb
from slub_docsa.evaluation.incidence import membership_matrix_to_crisp_cluster_assignments, unique_subject_order
from slub_docsa.evaluation.plotting import cluster_distribution_by_subject_plot, subject_distribution_by_cluster_plot
from slub_docsa.evaluation.plotting import write_multiple_figure_formats
from slub_docsa.experiments.qucosa.models import default_qucosa_named_clustering_models_tuple_list
from slub_docsa.experiments.qucosa.datasets import qucosa_named_datasets

logger = logging.getLogger(__name__)


def load_qucosa_cluster_model_by_name(model_name, n_subjects):
    """Return a single model retrieving it by its name."""
    for name, model in default_qucosa_named_clustering_models_tuple_list(n_subjects):
        if name == model_name:
            return model()
    raise ValueError(f"model with name '{model_name}' not known")


def qucosa_experiments_cluster_one(
    dataset_name: str,
    model_name: str,
    max_documents: Optional[int] = None,
):
    """Run clustering experiment evaluating single dataset."""
    _, dataset, subject_hierarchy = list(qucosa_named_datasets([dataset_name]))[0]

    if max_documents is not None:
        sampled_idx = np.random.choice(range(len(dataset.documents)), size=max_documents, replace=False)
        documents = [dataset.documents[i] for i in sampled_idx]
        subject_targets = [dataset.subjects[i] for i in sampled_idx]
        subject_targets = prune_subject_targets_to_level(2, subject_targets, subject_hierarchy)
    else:
        documents = dataset.documents
        subject_targets = dataset.subjects

    subject_order = unique_subject_order(subject_targets)
    model = load_qucosa_cluster_model_by_name(model_name, len(subject_order))

    model.fit(documents)
    membership = model.predict(documents)
    cluster_assignments = membership_matrix_to_crisp_cluster_assignments(membership)

    document_labels = [d.title for d in documents]
    subject_labels = {
        s: subject_label_breadcrumb(subject_hierarchy[s], subject_hierarchy) for s in subject_order
    }

    fig = subject_distribution_by_cluster_plot(
        cluster_assignments,
        document_labels,
        subject_targets,
        subject_labels,
    )

    write_multiple_figure_formats(fig, os.path.join(get_figures_dir(), "qucosa/subject_distribution_by_cluster"))

    fig = cluster_distribution_by_subject_plot(
        cluster_assignments,
        document_labels,
        subject_targets,
        subject_labels,
    )

    write_multiple_figure_formats(fig, os.path.join(get_figures_dir(), "qucosa/cluster_distribution_by_subject"))


if __name__ == "__main__":
    logging.basicConfig(level=logging.DEBUG)

    qucosa_experiments_cluster_one(
        dataset_name="qucosa_de_fulltexts_langid_ddc",
        model_name="tfidf_10k_kMeans_c=20",
        max_documents=5000
    )
