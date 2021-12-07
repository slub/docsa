"""Generates a plot illustrating the subject distribution of the Qucosa dataset."""

# pylint: disable=invalid-name

import os
import logging
from typing import Callable, List, Mapping

import plotly.express as px
from slub_docsa.common.subject import SubjectHierarchyType, SubjectNodeType
from slub_docsa.data.load.ddc import get_generic_ddc_subject_hierarchy
from slub_docsa.data.preprocess.subject import subject_ancestors_list, subject_label_breadcrumb

from slub_docsa.data.load.qucosa import QucosaJsonDocument, read_qucosa_documents_from_directory
from slub_docsa.data.load.qucosa import _get_rvk_subjects_from_qucosa_metadata, _get_ddc_subjects_from_qucosa_metadata
from slub_docsa.data.load.rvk import get_rvk_subject_store

logger = logging.getLogger(__name__)


def _get_parent_uri_from_subject(
    subject_hierarchy: SubjectHierarchyType[SubjectNodeType],
    subject: SubjectNodeType
):
    """Return parent notation of subject or empty string for sunburst chart."""
    if subject.parent_uri is not None and subject.parent_uri in subject_hierarchy:
        return subject.parent_uri
    return ""


def qucosa_number_of_documents_by_subjects(
    subject_hierarchy: SubjectHierarchyType[SubjectNodeType],
    get_subject_from_qucosa_doc: Callable[[QucosaJsonDocument], List[str]],
) -> Mapping[str, float]:
    """Count the number of Qucosa documents for each RVK subject."""
    logger.debug("count subject occurances in qucosa")
    qucosa_subjects: Mapping[str, float] = {
        "uri://no_value": 0.0,
        "uri://not_found": 0.0,
    }
    for doc in read_qucosa_documents_from_directory():
        subject_uri_list = get_subject_from_qucosa_doc(doc)
        for subject_uri in subject_uri_list:
            fraction = 1.0 / len(subject_uri_list)
            if subject_uri in subject_hierarchy:
                qucosa_subjects[subject_uri] = qucosa_subjects.get(subject_uri, 0.0) + fraction
            else:
                qucosa_subjects["uri://not_found"] = qucosa_subjects.get("uri://not_found", 0.0) + fraction
        if not subject_uri_list:
            qucosa_subjects["uri://no_value"] = qucosa_subjects.get("uri://no_value", 0.0) + 1.0
    return qucosa_subjects


def generate_qucosa_subject_sunburst(
    subject_hierarchy: SubjectHierarchyType[SubjectNodeType],
    get_subject_from_qucosa_doc: Callable[[QucosaJsonDocument], List[str]],
):
    """Generate a sunburst chart for qucosa visualizing the RVK class distribution."""
    qucosa_subjects = qucosa_number_of_documents_by_subjects(subject_hierarchy, get_subject_from_qucosa_doc)

    custom_labels = {
        "uri://no_value": "no value",
        "uri://not_found": "not found"
    }

    logger.debug("apply counts to all ancestor subjects")
    sunburst_by_subject_uri = {}
    for subject_uri, count in qucosa_subjects.items():
        # in case notation is no value or not found
        if subject_uri in custom_labels:
            custom_label = custom_labels[subject_uri]
            sunburst_by_subject_uri[subject_uri] = sunburst_by_subject_uri.get(custom_label, {
                "uri": subject_uri,
                "label": custom_label,
                "parent": "",
                "count": 0,
                "hover": custom_label,
            })
            sunburst_by_subject_uri[subject_uri]["count"] += count
        elif subject_uri in subject_hierarchy:
            subject_node = subject_hierarchy[subject_uri]
            ancestors = subject_ancestors_list(subject_node, subject_hierarchy)

            # iterate through ancestors, but not subject itself
            for ancestor in ancestors:
                # create and add counts for ancestors
                sunburst_by_subject_uri[ancestor.uri] = sunburst_by_subject_uri.get(ancestor.uri, {
                    "uri": ancestor.uri,
                    "label": ancestor.label,
                    "parent": _get_parent_uri_from_subject(subject_hierarchy, ancestor),
                    "count": 0,
                    "hover": subject_label_breadcrumb(ancestor, subject_hierarchy)
                })
                sunburst_by_subject_uri[ancestor.uri]["count"] += count

    sunburst_list = list(sunburst_by_subject_uri.values())

    data = {
        "uri": list(map(lambda d: d["uri"], sunburst_list)),
        "label": list(map(lambda d: d["label"], sunburst_list)),
        "parent": list(map(lambda d: d["parent"], sunburst_list)),
        "count": list(map(lambda d: d["count"], sunburst_list)),
        "hover": list(map(lambda d: d["hover"], sunburst_list)),
    }

    return px.sunburst(
        data,
        ids="uri",
        names="label",
        parents="parent",
        values="count",
        hover_name="hover",
        branchvalues="total",
        maxdepth=3
    )


if __name__ == "__main__":
    logging.basicConfig(level=logging.DEBUG)

    from slub_docsa.common.paths import get_figures_dir

    # subject_schema = "rvk"
    subject_schema = "ddc"

    subject_store = {
        "rvk": get_rvk_subject_store(),
        "ddc": get_generic_ddc_subject_hierarchy(),
    }[subject_schema]

    qucosa_subject_getter = {
        "rvk": _get_rvk_subjects_from_qucosa_metadata,
        "ddc": _get_ddc_subjects_from_qucosa_metadata,
    }[subject_schema]

    generate_qucosa_subject_sunburst(subject_store, qucosa_subject_getter).write_html(
        os.path.join(get_figures_dir(), f"qucosa/{subject_schema}_distribution.html"),
        include_plotlyjs="cdn",
    )

    logger.info("done plotting")

    subjects = qucosa_number_of_documents_by_subjects(subject_store, qucosa_subject_getter)
    qucosa_doc_count = sum(1 for x in read_qucosa_documents_from_directory())

    count_no_value = subjects["uri://no_value"]
    count_not_found = subjects["uri://not_found"]

    print(f"qucosa has {qucosa_doc_count} documents")
    print(f"qucosa uses only {len(subjects) - 2} of in total {len(subject_store)} unique {subject_schema} subjects")
    print(f"qucosa has {count_no_value} documents containing no subject annotation")
    print(f"qucosa has {count_not_found} documents with invalid annotations")
