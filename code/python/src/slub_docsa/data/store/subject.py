"""Persistent storage for subject hierarchies."""

import dbm
import pickle  # nosec
import logging
import os

from typing import Callable, Generic, Iterable, cast

from slub_docsa.common.subject import SubjectHierarchyType, SubjectNodeType, SubjectNode
from slub_docsa.data.preprocess.subject import subject_label_breadcrumb

logger = logging.getLogger(__name__)


class SubjectHierarchyDbmStore(Generic[SubjectNodeType], SubjectHierarchyType[SubjectNodeType]):
    """Stores a subject hierarchy in a python dbm database."""

    def __init__(self, filepath: str, read_only: bool = True):
        """Initialize store with a filepath.

        Parameters
        ----------
        filepath: str
            the path to the database file
        read_only: bool = True
            whether to open the database in read-only mode
        """
        flag = "r" if read_only else "c"
        self.store = dbm.open(filepath, flag)

    # pylint: disable=no-self-use
    def subject_node_to_bytes(self, subject_node: SubjectNodeType) -> bytes:
        """Convert subject node to bytes for storage in dbm database."""
        return pickle.dumps(subject_node)  # nosec

    # pylint: disable=no-self-use
    def bytes_to_subject_node(self, data: bytes) -> SubjectNodeType:
        """Convert bytes data to subject node when reading from dbm database."""
        return pickle.loads(data)  # nosec

    def __getitem__(self, uri: str) -> SubjectNodeType:
        """Retrieve subject node from dbm database."""
        if self.store is None:
            raise RuntimeError("dbm store already closed")
        return self.bytes_to_subject_node(self.store.__getitem__(uri))

    def __setitem__(self, uri: str, subject_node: SubjectNodeType):
        """Save subject node to dbm database."""
        if self.store is None:
            raise RuntimeError("dbm store already closed")
        self.store.__setitem__(uri, self.subject_node_to_bytes(subject_node))

    def __iter__(self):
        """Iterate over all subject nodes in dbm database."""
        if self.store is None:
            raise RuntimeError("dbm store already closed")
        for uri in self.store.keys():
            yield cast(bytes, uri).decode()

    def __len__(self) -> int:
        """Return number of subject nodes in dbm database."""
        if self.store is None:
            raise RuntimeError("dbm store already closed")
        return self.store.__len__()

    def close(self):
        """Close dbm database."""
        if self.store:
            self.store.close()
            self.store = None


def load_persisted_subject_hierarchy_from_lazy_subject_generator(
    lazy_subject_generator: Callable[[], Iterable[SubjectNodeType]],
    filepath: str,
) -> SubjectHierarchyType[SubjectNodeType]:
    """Load a subject hierarchy if it was stored before, or otherwise store it using the provided subject generator.

    Parameters
    ----------
    lazy_subject_generator: Callable[[], Iterable[SubjectNodeType]]
        a function that returns an iterator over subjects, which are stored in case the database does not exist yet
    filepath: str
        the path to the database file

    Returns
    -------
    SubjectHierarchyType[SubjectNodeType]
        the subject hierarchy as loaded from the persisted database
    """
    if not os.path.exists(filepath):
        store = SubjectHierarchyDbmStore[SubjectNodeType](filepath, read_only=False)

        for i, subject_node in enumerate(lazy_subject_generator()):
            store[subject_node.uri] = subject_node
            if i % 10000 == 0:
                logger.debug("added %d subjects to store so far", i)

        store.close()

    return SubjectHierarchyDbmStore[SubjectNodeType](filepath, read_only=True)


if __name__ == "__main__":
    import tempfile

    with tempfile.TemporaryDirectory(suffix="_subject_store") as td:
        s = SubjectHierarchyDbmStore[SubjectNode](os.path.join(td, "data"))

        subject1 = SubjectNode("uri://subject1", "subject 1", None)
        subject2 = SubjectNode("uri://subject2", "subject 2", "uri://subject1")

        s[subject1.uri] = subject1
        s[subject2.uri] = subject2

        print(subject_label_breadcrumb(s[subject2.uri], s))
