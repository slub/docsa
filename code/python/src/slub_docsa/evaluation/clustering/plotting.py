"""Methods to generate various plots for evaluating clustering algorithms."""

# pylint: disable=dangerous-default-value, too-many-arguments, too-many-locals

from typing import Iterable, Mapping, Sequence

import numpy as np

import plotly.express as px

from slub_docsa.data.preprocess.subject import count_number_of_samples_by_subjects
from slub_docsa.evaluation.classification.incidence import unique_subject_order


def subject_distribution_by_cluster_plot(
    cluster_assignments: Sequence[int],
    document_labels: Sequence[str],
    subject_targets: Sequence[Iterable[str]],
    subject_labels: Mapping[str, str],
):
    """Return plot showing the distribution of subjects for each cluster in a sunburst chart.

    Parameters
    ----------
    cluster_assignments: Sequence[int]
        the list of cluster assignments for each document
    document_labels: Sequence[str]
        a label for each document (list must match ordering of `cluster_assignments`)
    subject_targets: Sequence[Iterable[str]]
        the list of subject annotations for each document (list must match ordering of `cluster_assignments`)
    subject_labels: Mapping[str, str]
        a dictionary containing a label for each subject indexed by its URI

    Returns
    -------
    plotly.Figure
        the sunburst chart showing the distribution of subjects for each cluster
    """
    n_clusters = np.max(cluster_assignments) + 1

    sb_ids = []
    sb_labels = []
    sb_parents = []
    sb_values = []
    sb_hovers = []

    for i in range(n_clusters):
        cluster_document_idx = np.where(np.array(cluster_assignments) == i)[0]
        cluster_subject_targets = [t for j, t in enumerate(subject_targets) if j in cluster_document_idx]
        subject_counts = count_number_of_samples_by_subjects(cluster_subject_targets)

        for subject_uri, count in subject_counts.items():
            with_subject_idx = [j for j in cluster_document_idx if subject_uri in subject_targets[j]]
            with_subject_labels = [l for j, l in enumerate(document_labels) if j in with_subject_idx]
            sb_ids.append(f"Cluster {i+1} " + subject_uri)
            sb_labels.append(subject_labels[subject_uri])
            sb_parents.append(f"Cluster {i+1}")
            sb_values.append(count)
            sb_hovers.append("<br>".join([label[:120] for label in with_subject_labels[:20]]))

        sb_ids.append(f"Cluster {i+1}")
        sb_labels.append(f"Cluster {i+1}")
        sb_parents.append("")
        sb_values.append(np.sum(list(subject_counts.values())))
        sb_hovers.append("")

    data = {
        "ids": sb_ids,
        "labels": sb_labels,
        "parents": sb_parents,
        "values": sb_values,
        "hovers": sb_hovers,
    }

    return px.sunburst(
        data,
        ids="ids",
        names="labels",
        parents="parents",
        values="values",
        hover_name="hovers",
        branchvalues="total",
        maxdepth=2
    )


def cluster_distribution_by_subject_plot(
    cluster_assignments: Sequence[int],
    document_labels: Sequence[str],
    subject_targets: Sequence[Iterable[str]],
    subject_labels: Mapping[str, str],
):
    """Return plot showing the distribution of clusters for each subject in a sunburst chart.

    Parameters
    ----------
    cluster_assignments: Sequence[int]
        the list of cluster assignments for each document
    document_labels: Sequence[str]
        a label for each document (list must match ordering of `cluster_assignments`)
    subject_targets: Sequence[Iterable[str]]
        the list of subject annotations for each document (list must match ordering of `cluster_assignments`)
    subject_labels: Mapping[str, str]
        a dictionary containing a label for each subject indexed by its URI

    Returns
    -------
    plotly.Figure
        the sunburst chart showing the distribution of clusters for each subject
    """
    subject_order = unique_subject_order(subject_targets)

    sb_ids = []
    sb_labels = []
    sb_parents = []
    sb_values = []
    sb_hovers = []

    for subject_uri in subject_order:
        subject_documents_idx = [i for i in range(len(subject_targets)) if subject_uri in subject_targets[i]]
        subject_clusters = {cluster_assignments[i] for i in subject_documents_idx}

        subject_total_count = 0

        for cluster_id in subject_clusters:
            docs_of_cluster_idx = [i for i in subject_documents_idx if cluster_assignments[i] == cluster_id]
            docs_of_cluster_labels = [document_labels[i] for i in docs_of_cluster_idx]

            sb_ids.append(subject_uri + f"Cluster f{cluster_id + 1}")
            sb_labels.append(f"Cluster {cluster_id + 1}")
            sb_parents.append(subject_uri)
            sb_values.append(len(docs_of_cluster_idx))
            sb_hovers.append("<br>".join([label[:120] for label in docs_of_cluster_labels[:20]]))

            subject_total_count += len(docs_of_cluster_idx)

        sb_ids.append(subject_uri)
        sb_labels.append(subject_labels[subject_uri])
        sb_parents.append("")
        sb_values.append(subject_total_count)
        sb_hovers.append("")

    data = {
        "ids": sb_ids,
        "labels": sb_labels,
        "parents": sb_parents,
        "values": sb_values,
        "hovers": sb_hovers,
    }

    return px.sunburst(
        data,
        ids="ids",
        names="labels",
        parents="parents",
        values="values",
        hover_name="hovers",
        branchvalues="total",
        maxdepth=2
    )
