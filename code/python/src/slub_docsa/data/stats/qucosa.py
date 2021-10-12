"""Generates various statistics and figures of Qucosa as part of an exploratory data analysis."""

import os
import logging
from typing import Mapping

import plotly.express as px
from slub_docsa.common.subject import SubjectHierarchyType
from slub_docsa.data.preprocess.subject import subject_ancestors_list

from slub_docsa.data.load.qucosa import _get_rvk_notations_from_qucosa_metadata, read_qucosa_documents_from_directory
from slub_docsa.data.load.rvk import RvkSubjectNode, get_rvk_subject_store, rvk_notation_to_uri

logger = logging.getLogger(__name__)


def _get_parent_notation_from_subject(
    rvk_subject_hierarchy: SubjectHierarchyType[RvkSubjectNode],
    rvk_subject: RvkSubjectNode
):
    """Return parent notation of subject or empty string for sunburst chart."""
    if rvk_subject.parent_uri is not None and rvk_subject.parent_uri in rvk_subject_hierarchy:
        return rvk_subject_hierarchy[rvk_subject.parent_uri].notation
    return ""


def qucosa_number_of_documents_by_rvk_subjects(
    rvk_subject_store: SubjectHierarchyType[RvkSubjectNode]
) -> Mapping[str, float]:
    """Count the number of Qucosa documents for each RVK subject."""
    logger.debug("count rvk subject occurances in Qucosa")
    qucosa_rvk_subjects: Mapping[str, float] = {}
    for doc in read_qucosa_documents_from_directory():
        notations = _get_rvk_notations_from_qucosa_metadata(doc)
        for notation in notations:
            subject_uri = rvk_notation_to_uri(notation)
            fraction = 1.0 / len(notations)
            if subject_uri in rvk_subject_store:
                qucosa_rvk_subjects[subject_uri] = qucosa_rvk_subjects.get(subject_uri, 0) + fraction
            else:
                qucosa_rvk_subjects["uri://not_found"] = qucosa_rvk_subjects.get("uri://not_found", 0) + fraction
        if not notations:
            qucosa_rvk_subjects["uri://no_value"] = qucosa_rvk_subjects.get("uri://no_value", 0) + 1

    return qucosa_rvk_subjects


def generate_qucosa_rvk_sunburst():
    """Generate a sunburst chart for qucosa visualizing the RVK class distribution."""
    rvk_subject_store = get_rvk_subject_store()
    qucosa_rvk_subjects = qucosa_number_of_documents_by_rvk_subjects(rvk_subject_store)

    custom_notations = {
        "uri://no_value": "no value",
        "uri://not_found": "not found"
    }

    logger.debug("apply counts to all ancestor subjects")
    sunburst_by_notation = {}
    for subject_uri, count in qucosa_rvk_subjects.items():

        # in case notation is no value or not found
        if subject_uri in custom_notations:
            notation = custom_notations[subject_uri]
            sunburst_by_notation[notation] = sunburst_by_notation.get(notation, {
                "notation": notation,
                "parent": "",
                "count": 0,
                "label": notation,
            })
            sunburst_by_notation[notation]["count"] += count
        elif subject_uri in rvk_subject_store:
            rvk_subject_node = rvk_subject_store[subject_uri]
            ancestors = subject_ancestors_list(rvk_subject_node, rvk_subject_store)

            # iterate through ancestors, but not subject itself
            for ancestor in ancestors:
                notation = ancestor.notation
                label = ancestor.label
                parent_notation = _get_parent_notation_from_subject(rvk_subject_store, ancestor)
                # create and add counts for ancestors
                sunburst_by_notation[notation] = sunburst_by_notation.get(notation, {
                    "notation": notation,
                    "parent": parent_notation,
                    "count": 0,
                    "label": label
                })
                sunburst_by_notation[notation]["count"] += count

    sunburst_list = list(sunburst_by_notation.values())

    data = {
        "notation": list(map(lambda d: d["notation"], sunburst_list)),
        "parent": list(map(lambda d: d["parent"], sunburst_list)),
        "count": list(map(lambda d: d["count"], sunburst_list)),
        "label": list(map(lambda d: d["label"], sunburst_list)),
    }

    return px.sunburst(
        data,
        names="notation",
        parents="parent",
        values="count",
        hover_name="label",
        branchvalues="total",
        maxdepth=3
    )


if __name__ == "__main__":
    logging.basicConfig(level=logging.DEBUG)

    from slub_docsa.common.paths import FIGURES_DIR

    rvk_store = get_rvk_subject_store()
    rvk_subjects = qucosa_number_of_documents_by_rvk_subjects(rvk_store)
    qucosa_doc_count = sum(1 for x in read_qucosa_documents_from_directory())

    count_no_value = rvk_subjects["uri://no_value"]
    count_not_found = rvk_subjects["uri://not_found"]

    print(f"qucosa has {qucosa_doc_count} documents")
    print(f"qucosa uses only {len(rvk_subjects) - 2} of in total {len(rvk_store)} unique RVK subjects")
    print(f"qucosa has {count_no_value} documents containing no RVK subject annotation")
    print(f"qucosa has {count_not_found} documents with invalid RVK annotations")

    generate_qucosa_rvk_sunburst().write_html(
        os.path.join(FIGURES_DIR, "qucosa_rvk_distribution.html"),
        include_plotlyjs="cdn",
    )
