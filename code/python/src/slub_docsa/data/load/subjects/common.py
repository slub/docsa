"""Common methods when loading subject hierarchies."""

from typing import Literal, Union
from slub_docsa.common.subject import SubjectHierarchy
from slub_docsa.data.load.subjects.rvk import load_rvk_subject_hierarchy_from_sqlite
from slub_docsa.data.load.subjects.jskos import load_jskos_subject_hierarchy_from_sqlite


def subject_hierarchy_by_subject_schema(
    schema: Union[Literal["rvk"], Literal["ddc"], Literal["bk"]],
) -> SubjectHierarchy:
    """Return either rvk, ddc or bk subject hierarchy depending on the requested subject schema.

    Downloads and prepares subject hierarchies using their default parameters, if they do no exist yet.

    Parameters
    ----------
    schema : str
        the schema, either "rvk", "ddc" or "bk"

    Returns
    -------
    SubjectHierarchy
        the corresponding subject hierarchy
    """
    return {
        "rvk": load_rvk_subject_hierarchy_from_sqlite,
        "ddc": lambda: load_jskos_subject_hierarchy_from_sqlite("ddc"),
        "bk": lambda: load_jskos_subject_hierarchy_from_sqlite("bk"),
    }[schema]()
