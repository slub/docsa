"""Generates various statistics and figures of Qucosa as part of an exploratory data analysis."""

import os
import logging
from typing import List

import plotly.express as px

from slub_docsa.data.load.qucosa import get_rvk_notations_from_qucosa_metadata, read_qucosa_metadata
from slub_docsa.data.load.rvk import RvkClass, load_rvk_classes_indexed_by_notation

logger = logging.getLogger(__name__)


def _load_extended_rvk_classes():
    """Load RVK classes indexed by notation and adds two artificial classes for visualization purposes."""
    rvk_classes = load_rvk_classes_indexed_by_notation()

    # add artificial not found class
    rvk_not_found_class: RvkClass = {
        "notation": "not found",
        "label": "not found",
        "uri": "",
        "ancestors": [],
    }
    rvk_classes["not found"] = rvk_not_found_class

    # add artificial no value class
    rvk_no_value_class: RvkClass = {
        "notation": "no value",
        "label": "no value",
        "uri": "",
        "ancestors": [],
    }
    rvk_classes["no value"] = rvk_no_value_class

    return rvk_classes


def generate_qucosa_rvk_sunburst():
    """Generate a sunburst chart for qucosa visualizing the RVK class distribution."""
    logger.debug("load rvk classes by notation")
    rvk_indexed_by_notation = _load_extended_rvk_classes()

    logger.debug("count rvk class occurances in Qucosa")
    qucosa_rvk_notations = {}
    for doc in read_qucosa_metadata():
        rvk_notations = get_rvk_notations_from_qucosa_metadata(doc)
        for notation in rvk_notations:
            fraction = 1.0 / len(rvk_notations)
            if notation in rvk_indexed_by_notation:
                qucosa_rvk_notations[notation] = qucosa_rvk_notations.get(notation, 0) + fraction
            else:
                qucosa_rvk_notations["not found"] = qucosa_rvk_notations.get("not found", 0) + fraction
        if not rvk_notations:
            qucosa_rvk_notations["no value"] = qucosa_rvk_notations.get("no value", 0) + 1

    logger.debug("apply counts to ancestor classes as well")
    sunburst_by_notation = {}
    for notation, count in qucosa_rvk_notations.items():
        rvk_cls = rvk_indexed_by_notation[notation]

        # only count current class if not too deep in class hierarchy
        sunburst_by_notation[notation] = sunburst_by_notation.get(notation, {
            "notation": notation,
            "parent": "" if not rvk_cls["ancestors"] else rvk_cls["ancestors"][-1]["notation"],
            "count": 0,
            "label": rvk_cls["label"],
        })
        sunburst_by_notation[notation]["count"] += count

        if rvk_cls["ancestors"]:
            # class has parents
            ancestors: List = rvk_cls["ancestors"]
            # iterate through ancestors
            for i, ancestor in enumerate(ancestors):
                parent_notation = "" if i < 1 else ancestors[i-1]["notation"]
                # create and add counts for ancestors
                sunburst_by_notation[ancestor["notation"]] = sunburst_by_notation.get(ancestor["notation"], {
                    "notation": ancestor["notation"],
                    "parent": parent_notation,
                    "count": 0,
                    "label": ancestor["label"]
                })
                sunburst_by_notation[ancestor["notation"]]["count"] += count

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
