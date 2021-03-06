"""Base classes modeling a subject and subject hierarchy.

Note: Type aliases `SubjectHierarchy`, `SubjectTargets` and `SubjectUriList`
are not correctly described in API documentation due to [issue in pdoc](https://github.com/pdoc3/pdoc/issues/229).
"""

# pylint: disable=too-few-public-methods

from typing import Iterable, List, Mapping, Sequence, Tuple, Optional


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

    def __repr__(self):
        """Return simple string identifying subject node by its URI and parent URI."""
        return f"<{self.__class__.__name__} uri=\"{self.uri}\" parent_uri=\"{self.parent_uri}\">"


SubjectHierarchy = Mapping[str, SubjectNode]
"""A dictionary mapping a subject uri to its subject node."""

SubjectUriList = Iterable[str]
"""A subject list of subjects uris."""

SubjectTargets = Sequence[SubjectUriList]
"""An ordered list of subject uri lists, each corresponding to the target classes assigned to a document.
See `slub_docsa.common.dataset.Dataset`.
"""


def print_subject_hierarchy(subject_hierarchy: SubjectHierarchy):
    """Print a subject hierarchy to `stdout` using simple indentation."""
    root_subjects_uri = [s.uri for s in subject_hierarchy.values() if s.parent_uri is None]

    for root_subject_uri in root_subjects_uri:
        subject_backlog: List[Tuple[int, str]] = [(0, root_subject_uri)]
        while subject_backlog:
            current_level, current_subject_uri = subject_backlog.pop(0)
            print("    " * current_level, current_subject_uri)

            # find and add children for processing
            children_subjects_uri = [s.uri for s in subject_hierarchy.values() if s.parent_uri == current_subject_uri]
            for children_subject_uri in children_subjects_uri:
                subject_backlog.insert(0, (current_level + 1, children_subject_uri))
