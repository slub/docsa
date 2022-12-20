"""Persistent storage for subject hierarchies."""

import logging
import os
import pickle  # nosec

from typing import Callable, Iterable, Iterator, Mapping, Optional, Sequence

from sqlitedict import SqliteDict
from slub_docsa.common.subject import SubjectHierarchy, SubjectTargets
from slub_docsa.common.paths import get_cache_dir
from slub_docsa.evaluation.classification.incidence import unique_subject_order

logger = logging.getLogger(__name__)


class SqliteSubjectIterable(Iterable[str]):  # pylint: disable=too-few-public-methods
    """Iterable implementation that is used to load a simple list of subjects from an `sqlitedict`.

    The keys are integers representing the order in the list of subjects. The amount of subjects is stored in the
    key "count".
    """

    def __init__(self, subject_store: SqliteDict):
        """Init iterable with subject store."""
        self.subject_store = subject_store
        self.count = int(subject_store["count"])

    def __iter__(self):
        """Iterate over the list of subjects."""
        for i in range(self.count):
            yield self.subject_store[str(i)]


class SqliteSubjectHierarchy(SubjectHierarchy):
    """Subject hierarchy that is loaded from Sqlite file."""

    TABLE_FOR_ROOT_SUBJECTS = "root_subjects"
    TABLE_FOR_SUBJECT_LABELS = "subject_labels"
    TABLE_FOR_SUBJECT_PARENT = "subject_parent"
    TABLE_FOR_SUBJECT_CHILDREN = "subject_children"
    TABLE_FOR_SUBJECT_NOTATION = "subject_notation"

    def __init__(self, filepath: str, preload_contains: bool = False):
        """Load sqlite subject hierarchy.

        Parameters
        ----------
        filepath : str
            the filepath to the sqlite database file
        preload_contains : bool, optional
            whether to load all available subject URIs into memory in order to improve the performance of the
            `__contains__` operation, by default False
        """
        self.root_subjects_store = SqliteDict(filepath, tablename=self.TABLE_FOR_ROOT_SUBJECTS, flag="r")
        self.labels_store = SqliteDict(filepath, tablename=self.TABLE_FOR_SUBJECT_LABELS, flag="r")
        self.parent_store = SqliteDict(filepath, tablename=self.TABLE_FOR_SUBJECT_PARENT, flag="r")
        self.children_store = SqliteDict(filepath, tablename=self.TABLE_FOR_SUBJECT_CHILDREN, flag="r")
        self.notation_store = SqliteDict(filepath, tablename=self.TABLE_FOR_SUBJECT_NOTATION, flag="r")
        self.contains = set(self) if preload_contains else self.labels_store

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

    def subject_notation(self, subject_uri: str) -> Optional[str]:
        """Return RVK notation for a subject."""
        return self.notation_store[subject_uri]

    def __contains__(self, subject_uri: str) -> bool:
        """Return true if the subject_uri is a valid subject in this subject hierarchy."""
        return subject_uri in self.contains

    def __iter__(self) -> Iterator[str]:
        """Return an iterator over all subject uris of this hierarchy."""
        return self.labels_store.keys()

    @staticmethod
    def save(
        subject_hierarchy: SubjectHierarchy,
        filepath: str,
    ):
        """Save a subject hierarchy to an sqlite file using multiple tables.

        Parameters
        ----------
        subject_hierarchy : SubjectHierarchy
            the subject hierarchy to be saved in an sqlite database
        filepath : str
            the filepath to the sqlite database file that is generated
        """
        logger.info("save subject hierarchy as sqlite file")
        os.makedirs(os.path.dirname(filepath), exist_ok=True)

        cls = SqliteSubjectHierarchy

        tasks = [
            (cls.TABLE_FOR_SUBJECT_LABELS, subject_hierarchy.subject_labels),
            (cls.TABLE_FOR_SUBJECT_PARENT, subject_hierarchy.subject_parent),
            (cls.TABLE_FOR_SUBJECT_CHILDREN, subject_hierarchy.subject_children),
            (cls.TABLE_FOR_SUBJECT_NOTATION, subject_hierarchy.subject_notation),
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


def cached_unique_subject_order(
    dataset_name: str,
    subject_targets: SubjectTargets,
    cache_directory: Optional[str] = None
) -> Sequence[str]:
    """Load or generate a unique subject order from cache to improve performance.

    Parameters
    ----------
    dataset_name : str
        the dataset name that is used to identify a cached unique subject order
    subject_targets : SubjectTargets
        the subject targets that are used to generate the unique subject order in case it was not cached yet
    cache_directory : Optional[str], optional
        the directory where unique subject order are saved to, if None, the directory
        `SLUB_DOCSA_CACHE_DIR/subject_order` is used

    Returns
    -------
    Sequence[str]
        the loaded or generated unique subject order for a dataset
    """
    if cache_directory is None:
        cache_directory = os.path.join(get_cache_dir(), "subject_order")
    os.makedirs(cache_directory, exist_ok=True)

    filepath = os.path.join(cache_directory, f"{dataset_name}.pickle")
    if os.path.exists(filepath):
        # load subject order
        logger.debug("load unique subject order from cache file for dataset '%s'", dataset_name)
        with open(filepath, "rb") as file:
            subject_order = pickle.load(file)  # nosec
        return subject_order

    # generate and save suject order
    logger.debug("generate and save unique subject order for dataset '%s'", dataset_name)
    subject_order = unique_subject_order(subject_targets)
    with open(filepath, "wb") as file:
        pickle.dump(subject_order, file)

    return subject_order
