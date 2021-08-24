"""Generates various statistics and figures of Qucosa as part of an exploratory data analysis."""

import os
import logging

import plotly.express as px
from slub_docsa.data.common.subject import SubjectHierarchyType, get_subject_ancestors_list

from slub_docsa.data.load.qucosa import get_rvk_notations_from_qucosa_metadata, read_qucosa_metadata
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


def generate_qucosa_rvk_sunburst():
    """Generate a sunburst chart for qucosa visualizing the RVK class distribution."""
    rvk_subject_store = get_rvk_subject_store()

    logger.debug("count rvk subject occurances in Qucosa")
    qucosa_rvk_notations = {}
    for doc in read_qucosa_metadata():
        notations = get_rvk_notations_from_qucosa_metadata(doc)
        for notation in notations:
            subject_uri = rvk_notation_to_uri(notation)
            if subject_uri in rvk_subject_store:
                qucosa_rvk_notations[notation] = qucosa_rvk_notations.get(notation, 0) + 1.0 / len(notations)
            else:
                qucosa_rvk_notations["not found"] = qucosa_rvk_notations.get("not found", 0) + 1.0 / len(notations)
        if not notations:
            qucosa_rvk_notations["no value"] = qucosa_rvk_notations.get("no value", 0) + 1

    logger.debug("apply counts to all ancestor subjects")
    sunburst_by_notation = {}
    for notation, count in qucosa_rvk_notations.items():
        subject_uri = rvk_notation_to_uri(notation)

        # in case notation is no value or not found
        if notation in ("no value", "not found"):
            sunburst_by_notation[notation] = sunburst_by_notation.get(notation, {
                "notation": notation,
                "parent": "",
                "count": 0,
                "label": notation,
            })
            sunburst_by_notation[notation]["count"] += count
        elif subject_uri in rvk_subject_store:
            rvk_subject = rvk_subject_store[subject_uri]
            ancestors = get_subject_ancestors_list(rvk_subject_store, rvk_subject)

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

    from slub_docsa.common import FIGURES_DIR

    generate_qucosa_rvk_sunburst().write_html(os.path.join(FIGURES_DIR, "qucosa_rvk_distribution.html"))
