"""Persistent storage for subject hierarchies."""

import logging
import os

from typing import Callable, Iterable, Iterator, Mapping, Optional

from sqlitedict import SqliteDict
from slub_docsa.common.subject import SubjectHierarchy

logger = logging.getLogger(__name__)


class SqliteSubjectIterable(Iterable[str]):  # pylint: disable=too-few-public-methods
    """Iterable implementation for root subjects."""

    def __init__(self, subject_store: SqliteDict):
        """Init iterable with subject store."""
        self.subject_store = subject_store
        self.count = int(subject_store["count"])

    def __iter__(self):
        """Iterate over root subjects."""
        for i in range(self.count):
            yield self.subject_store[str(i)]


class SqliteSubjectHierarchy(SubjectHierarchy):
    """Subject hierarchy that is loaded from Sqlite file."""

    TABLE_FOR_ROOT_SUBJECTS = "root_subjects"
    TABLE_FOR_SUBJECT_LABELS = "subject_labels"
    TABLE_FOR_SUBJECT_PARENT = "subject_parent"
    TABLE_FOR_SUBJECT_CHILDREN = "subject_children"

    def __init__(self, filepath: str):
        """Init sqlite subject hierarchy."""
        self.root_subjects_store = SqliteDict(filepath, tablename=self.TABLE_FOR_ROOT_SUBJECTS, flag="r")
        self.labels_store = SqliteDict(filepath, tablename=self.TABLE_FOR_SUBJECT_LABELS, flag="r")
        self.parent_store = SqliteDict(filepath, tablename=self.TABLE_FOR_SUBJECT_PARENT, flag="r")
        self.children_store = SqliteDict(filepath, tablename=self.TABLE_FOR_SUBJECT_CHILDREN, flag="r")

    def root_subjects(self) -> Iterable[str]:
        """Return a list of root subjects."""
        return SqliteSubjectIterable(self.root_subjects_store)

    def subject_labels(self, subject_uri: str) -> Mapping[str, str]:
        """Return the labels mapping for a subject."""
        return self.labels_store[subject_uri]

    def subject_parent(self, subject_uri: str) -> Optional[str]:
        """Return the parent of the subject."""
        return self.parent_store[subject_uri]

    def subject_children(self, subject_uri: str) -> Optional[Iterable[str]]:
        """Return the children of the subject."""
        return self.children_store[subject_uri]

    def __contains__(self, subject_uri: str) -> bool:
        """Return true if the subject_uri is a valid subject in this subject hierarchy."""
        return subject_uri in self.labels_store

    def __iter__(self) -> Iterator[str]:
        """Return an iterator over all subject uris of this hierarchy."""
        return self.labels_store.keys()

    @staticmethod
    def save(
        subject_hierarchy: SubjectHierarchy,
        filepath: str,
    ):
        """Save a subject hierarchy to an sqlite file using multiple tables."""
        logger.info("save subject hierarchy as sqlite file")
        os.makedirs(os.path.dirname(filepath), exist_ok=True)

        cls = SqliteSubjectHierarchy

        tasks = [
            (cls.TABLE_FOR_SUBJECT_LABELS, subject_hierarchy.subject_labels),
            (cls.TABLE_FOR_SUBJECT_PARENT, subject_hierarchy.subject_parent),
            (cls.TABLE_FOR_SUBJECT_CHILDREN, subject_hierarchy.subject_children),
        ]

        for tablename, apply_func in tasks:
            store = SqliteDict(filepath, tablename=tablename, flag="w", autocommit=False)
            for subject_uri in subject_hierarchy:
                store[subject_uri] = apply_func(subject_uri)
            store.commit()
            store.close()

        # save root subjects
        store = SqliteDict(filepath, tablename=cls.TABLE_FOR_ROOT_SUBJECTS, flag="w", autocommit=False)
        count = 0
        for i, subject_uri in enumerate(subject_hierarchy.root_subjects()):
            store[str(i)] = subject_uri
            count += 1
        store["count"] = count
        store.commit()
        store.close()


def load_persisted_subject_hierarchy_from_lazy_subject_generator(
    lazy_subject_hierarchy: Callable[[], SubjectHierarchy],
    filepath: str,
) -> SubjectHierarchy:
    """Load a subject hierarchy if it was stored before, or otherwise store it using the provided subject generator.

    Parameters
    ----------
    lazy_subject_hierarchy: Callable[[], SubjectHierarchy]
        a function that returns a subjects hierarchy, which is then saved in case it was not saved before
    filepath: str
        the path to the database file

    Returns
    -------
    SubjectHierarchy
        the subject hierarchy as loaded from the persisted database
    """
    if not os.path.exists(filepath):
        SqliteSubjectHierarchy.save(lazy_subject_hierarchy(), filepath)
    return SqliteSubjectHierarchy(filepath)
