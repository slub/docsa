"""Model of a subject and subject hierarchy."""

# pylint: disable=too-few-public-methods

from typing import Iterable, Mapping, Sequence, TypeVar, Optional


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

SubjectUriList = Iterable[str]
"""A subject list of subjects uris."""

SubjectTargets = Sequence[SubjectUriList]
"""An ordered list of subject uri lists, each corresponding to the target classes assigned to a document."""
