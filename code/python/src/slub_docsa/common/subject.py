"""Base classes modeling a subject and subject hierarchy.

Note: Type aliases `SubjectHierarchy`, `SubjectTargets` and `SubjectUriList`
are not correctly described in API documentation due to [issue in pdoc](https://github.com/pdoc3/pdoc/issues/229).
"""

# pylint: disable=too-few-public-methods

from typing import Iterable, Iterator, List, Mapping, NamedTuple, Sequence, Tuple, Optional


class SubjectHierarchy(Iterable):
    """A subject hierarchy."""

    def root_subjects(self) -> Iterable[str]:
        """Return a list of root subjects.

        Returns
        -------
        Iterable[str]
            the list of URIs of subjects that have no parent subject
        """
        raise NotImplementedError()

    def subject_labels(self, subject_uri: str) -> Mapping[str, str]:
        """Return the labels mapping for a subject.

        Parameters
        ----------
        subject_uri: str
            the uri of the subject whose label mapping is requested

        Returns
        -------
        Mapping[str, str]
            the mapping from ISO 639-1 language codes to labels for this subject;
            an empty mapping for valid subjects with unknown labels

        Raises
        ------
        LookupError
            if the subject with this uri is not available in this subject hierarchy
        """
        raise NotImplementedError()

    def subject_parent(self, subject_uri: str) -> Optional[str]:
        """Return the parent of the subject or None if the subject does not have a parent.

        Parameters
        ----------
        subject_uri: str
            the uri of the subject whose parent is requested

        Returns
        -------
        Optional[str]
            the uri of the parent subject or None if the requested subject does not have a parent

        Raises
        ------
        LookupError
            if the subject with this uri is not available in this subject hierarchy
        """
        raise NotImplementedError()

    def subject_children(self, subject_uri: str) -> Iterable[str]:
        """Return the children of the subject.

        Parameters
        ----------
        subject_uri: str
            the uri of the subject whose children is requested

        Returns
        -------
        Iterable[str]
            the list of URIs of children subjects or an empty list if the requested subject does not have any children

        Raises
        ------
        LookupError
            if the subject with this uri is not available in this subject hierarchy
        """
        raise NotImplementedError()

    def subject_notation(self, subject_uri: str) -> Optional[str]:
        """Return notation for a subject.

        Parameters
        ----------
        subject_uri: str
            the uri of the subject whose notation is requested

        Returns
        -------
        Optional[str]
            the notation of the requested subject, or None if the subject does not have a notation

        Raises
        ------
        LookupError
            if the subject with this uri is not available in this subject hierarchy
        """
        raise NotImplementedError()

    def __contains__(self, subject_uri: str) -> bool:
        """Return true if the subject_uri is a valid subject in this subject hierarchy.

        Parameters
        ----------
        subject_uri: str
            the uri of the subject to be checked

        Returns
        -------
        bool
            whether the subject_uri is a valid subject in this subject hierarchy.
        """
        raise NotImplementedError()

    def __iter__(self) -> Iterator[str]:
        """Return an iterator over all subject uris of this subject hierarchy.

        The order in which subjects are iterated is not guaranteed.
        """
        raise NotImplementedError()


class SimpleSubjectHierarchy(SubjectHierarchy):
    """A naive implementation for a subject hierarchy."""

    def __init__(
        self,
        root_subjects: Iterable[str],
        subject_labels: Mapping[str, Mapping[str, str]],
        subject_parent: Mapping[str, str],
        subject_children: Mapping[str, Iterable[str]],
        subject_notation: Mapping[str, str],
    ):
        """Initialize subject hierarchy with required information about subjects."""
        self._root_subjects = root_subjects
        self._subject_labels = subject_labels
        self._subject_parent = subject_parent
        self._subject_children = subject_children
        self._subject_notation = subject_notation

    def root_subjects(self) -> Iterable[str]:
        """Return the list of root subjects."""
        return self._root_subjects

    def subject_labels(self, subject_uri: str) -> Mapping[str, str]:
        """Return the label mapping for a subject."""
        return self._subject_labels[subject_uri]

    def subject_parent(self, subject_uri: str) -> Optional[str]:
        """Return the parent of a subject or None if the subject does not have a parent."""
        return self._subject_parent[subject_uri]

    def subject_children(self, subject_uri: str) -> Iterable[str]:
        """Return a list of children for a subject."""
        return self._subject_children[subject_uri]

    def subject_notation(self, subject_uri: str) -> Optional[str]:
        """Return notation for a subject."""
        return self._subject_notation[subject_uri]

    def __contains__(self, subject_uri: str) -> bool:
        """Return true if the subject_uri is a valid subject in this subject hierarchy."""
        return subject_uri in self._subject_labels

    def __iter__(self) -> Iterator[str]:
        """Return an iterator over all subject uris."""
        return iter(self._subject_labels.keys())


class SubjectTuple(NamedTuple):
    """Tuple representation of all information about a subject."""

    subject_uri: str
    """The uri of the subject."""

    labels: Mapping[str, str]
    """The mapping from ISO 639-1 language codes to labels for the subject."""

    parent_uri: Optional[str]
    """The uri of the parent subject or None if the subject does not have a parent."""

    notation: Optional[str]
    """The notation for a subject or None, if the subject does not have a notation."""


SubjectUriList = Iterable[str]
"""A subject list of subjects uris."""

SubjectTargets = Sequence[SubjectUriList]
"""An ordered list of subject uri lists, each corresponding to the target classes assigned to a document.
See `slub_docsa.common.dataset.Dataset`.
"""


def print_subject_hierarchy(lang_code: str, subject_hierarchy: SubjectHierarchy, depth: int = 0):
    """Print a subject hierarchy to `stdout` using simple indentation."""
    for root_subject_uri in sorted(subject_hierarchy.root_subjects()):
        subject_backlog: List[Tuple[int, str]] = [(0, root_subject_uri)]
        while subject_backlog:
            current_level, current_subject_uri = subject_backlog.pop(0)

            # skip subjects at certain depth
            if current_level >= depth > 0:
                continue

            current_label = subject_hierarchy.subject_labels(current_subject_uri).get(lang_code, "label not available")
            print("    " * current_level, current_subject_uri, current_label)

            # find and add children for processing
            for children_subject_uri in reversed(sorted(subject_hierarchy.subject_children(current_subject_uri))):
                subject_backlog.insert(0, (current_level + 1, children_subject_uri))
