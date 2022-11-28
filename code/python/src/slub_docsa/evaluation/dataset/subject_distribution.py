"""Generates a plot illustrating the subject distribution as a sunburst chart."""

# pylint: disable=invalid-name

import logging
from typing import Iterator, Mapping

import plotly.express as px
from slub_docsa.common.subject import SubjectHierarchy
from slub_docsa.common.sample import Sample
from slub_docsa.data.preprocess.subject import subject_ancestors_list, subject_label_breadcrumb_as_string

logger = logging.getLogger(__name__)


def _get_parent_uri_from_subject(
    subject_hierarchy: SubjectHierarchy,
    subject_uri: str,
):
    """Return parent notation of subject or empty string for sunburst chart."""
    parent_uri = subject_hierarchy.subject_parent(subject_uri)
    if parent_uri is not None and parent_uri in subject_hierarchy:
        return parent_uri
    return ""


def number_of_documents_by_subjects(
    subject_hierarchy: SubjectHierarchy,
    samples: Iterator[Sample],
) -> Mapping[str, float]:
    """Count the number of documents for each subject."""
    logger.debug("count subject occurances in qucosa")
    counts: Mapping[str, float] = {
        "uri://no_value": 0.0,
        "uri://not_found": 0.0,
    }
    for sample in samples:
        for subject_uri in sample.subjects:
            fraction = 1.0 / len(sample.subjects)
            if subject_uri in subject_hierarchy:
                counts[subject_uri] = counts.get(subject_uri, 0.0) + fraction
            else:
                counts["uri://not_found"] = counts.get("uri://not_found", 0.0) + fraction
        if not sample.subjects:
            counts["uri://no_value"] = counts.get("uri://no_value", 0.0) + 1.0
    return counts


def generate_subject_sunburst(
    subject_hierarchy: SubjectHierarchy,
    lang_code: str,
    samples: Iterator[Sample],
    max_depth: int = 3,
    use_breadcrumb: bool = True,
):
    """Generate a sunburst chart for qucosa visualizing the RVK class distribution."""
    subject_counts = number_of_documents_by_subjects(subject_hierarchy, samples)

    custom_labels = {
        "uri://no_value": "no value",
        "uri://not_found": "not found"
    }

    def _subject_sunburst_dict(ancestor):
        label = subject_hierarchy.subject_labels(ancestor).get(lang_code, "not available")
        if use_breadcrumb:
            hover = " | ".join(subject_label_breadcrumb_as_string(ancestor, lang_code, subject_hierarchy))
        else:
            hover = label
        return {
            "uri": ancestor,
            "label": label,
            "parent": _get_parent_uri_from_subject(subject_hierarchy, ancestor),
            "count": 0,
            "hover": hover
        }

    logger.debug("apply counts to all ancestor subjects")
    sunburst_by_subject_uri = {}
    for subject_uri, count in subject_counts.items():
        # in case notation is no value or not found
        if subject_uri in custom_labels:
            custom_label = custom_labels[subject_uri]
            if subject_uri not in sunburst_by_subject_uri:
                sunburst_by_subject_uri[subject_uri] = {
                    "uri": subject_uri,
                    "label": custom_label,
                    "parent": "",
                    "count": 0,
                    "hover": custom_label,
                }
            sunburst_by_subject_uri[subject_uri]["count"] += count
        elif subject_uri in subject_hierarchy:
            # iterate through ancestors, including subject itself
            for ancestor in subject_ancestors_list(subject_uri, subject_hierarchy):
                # create and add counts for ancestors
                if ancestor not in sunburst_by_subject_uri:
                    sunburst_by_subject_uri[ancestor] = _subject_sunburst_dict(ancestor)
                sunburst_by_subject_uri[ancestor]["count"] += count

    sunburst_list = list(sunburst_by_subject_uri.values())

    logger.debug("compile data for sunburst chart")
    data = {
        "uri": list(map(lambda d: d["uri"], sunburst_list)),
        "label": list(map(lambda d: d["label"], sunburst_list)),
        "parent": list(map(lambda d: d["parent"], sunburst_list)),
        "count": list(map(lambda d: d["count"], sunburst_list)),
        "hover": list(map(lambda d: d["hover"], sunburst_list)),
    }

    logger.debug("generate sunburst chart")
    return px.sunburst(
        data,
        ids="uri",
        names="label",
        parents="parent",
        values="count",
        hover_name="hover",
        branchvalues="total",
        maxdepth=max_depth
    )
