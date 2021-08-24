"""Model of a subject and subject hierarchy."""

# pylint: disable=too-few-public-methods

from typing import Mapping, TypeVar, List, Optional


class Subject:
    """A subject consisting of both an URI and a label."""

    uri: str
    """The URI for this subject."""

    label: str
    """The label of this subject."""

    def __init__(self, uri: str, label: str):
        """Initialize new subject with URI and label."""
        self.uri = uri
        self.label = label

    def __str__(self):
        """Return simple string identifying subject by its URI."""
        return f"<{self.__class__.__name__} uri=\"{self.uri}\">"


class SubjectNode(Subject):
    """A subject node extending a subject with a parent URI."""

    parent_uri: Optional[str]
    """The URI of the parent subject (None = root)."""

    def __init__(self, uri: str, label: str, parent_uri: Optional[str]):
        """Initialize new subject with URI and label."""
        super().__init__(uri, label)
        self.parent_uri = parent_uri


SubjectNodeType = TypeVar('SubjectNodeType', bound=SubjectNode)
"""Generic Type for SubjectNode"""

SubjectHierarchyType = Mapping[str, SubjectNodeType]
"""A dictionary mapping a subject uri to its subject node."""


def get_subject_ancestors_list(
    subject_hierarchy: SubjectHierarchyType[SubjectNodeType],
    subject: SubjectNodeType
) -> List[SubjectNodeType]:
    """Return the list of ancestors for a subject node in a subject hierarchy."""
    ancestors: List[SubjectNodeType] = []
    next_subject = subject

    while next_subject is not None:
        ancestors.insert(0, next_subject)
        if next_subject.parent_uri is not None:
            next_subject = subject_hierarchy.get(next_subject.parent_uri, None)
        else:
            next_subject = None

    return ancestors


def get_subject_label_breadcrumb(
    subject_hierarchy: SubjectHierarchyType[SubjectNodeType],
    subject: SubjectNodeType
) -> str:
    """Return a label breadcrumb as a string describing the subject hierarchy path from root to this subject."""
    subject_path = get_subject_ancestors_list(subject_hierarchy, subject)
    return " | ".join(map(lambda s: s.label, subject_path))
