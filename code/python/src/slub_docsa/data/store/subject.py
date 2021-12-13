"""Persistent storage for subject hierarchies."""

import logging
import os

from typing import Callable, Iterable

from sqlitedict import SqliteDict
from slub_docsa.common.subject import SubjectHierarchy, SubjectNode
from slub_docsa.data.preprocess.subject import subject_label_breadcrumb

logger = logging.getLogger(__name__)


class SubjectHierarchySqliteStore(SqliteDict):
    """Stores a subject hierarchy in a python dbm database."""

    def __init__(
        self,
        filepath: str,
        read_only: bool = True,
        autocommit: bool = True,
    ):
        """Initialize store with a filepath.

        Parameters
        ----------
        filepath: str
            the path to the database file
        read_only: bool = True
            whether to open the database in read-only mode
        """
        flag = "r" if read_only else "c"
        super().__init__(filename=filepath, tablename="subject_nodes", flag=flag, autocommit=autocommit)


def load_persisted_subject_hierarchy_from_lazy_subject_generator(
    lazy_subject_generator: Callable[[], Iterable[SubjectNode]],
    filepath: str,
) -> SubjectHierarchy:
    """Load a subject hierarchy if it was stored before, or otherwise store it using the provided subject generator.

    Parameters
    ----------
    lazy_subject_generator: Callable[[], Iterable[SubjectNodeType]]
        a function that returns an iterator over subjects, which are stored in case the database does not exist yet
    filepath: str
        the path to the database file

    Returns
    -------
    SubjectHierarchy
        the subject hierarchy as loaded from the persisted database
    """
    if not os.path.exists(filepath):
        store = SubjectHierarchySqliteStore(filepath, read_only=False, autocommit=False)

        for i, subject_node in enumerate(lazy_subject_generator()):
            store[subject_node.uri] = subject_node
            if i % 10000 == 0:
                store.commit()
                logger.debug("added %d subjects to store so far", i)

        store.commit()
        store.close()

    return SubjectHierarchySqliteStore(filepath)


if __name__ == "__main__":
    import tempfile

    with tempfile.TemporaryDirectory(suffix="_subject_store") as td:
        s = SubjectHierarchySqliteStore(os.path.join(td, "data"), read_only=False)

        subject1 = SubjectNode("uri://subject1", "subject 1", None)
        subject2 = SubjectNode("uri://subject2", "subject 2", "uri://subject1")

        s[subject1.uri] = subject1
        s[subject2.uri] = subject2

        print(subject_label_breadcrumb(s[subject2.uri], s))
