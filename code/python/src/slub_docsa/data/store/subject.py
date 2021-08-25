"""Persitent storage for subject hierarchies."""

import dbm
import pickle  # nosec

from typing import Generic

from slub_docsa.data.common.subject import SubjectHierarchyType, SubjectNodeType, SubjectNode
from slub_docsa.data.common.subject import get_subject_label_breadcrumb


class SubjectHierarchyDbmStore(Generic[SubjectNodeType], SubjectHierarchyType[SubjectNodeType]):
    """Stores a subject hierarchy in a python dbm database."""

    def __init__(self, filepath, read_only=True):
        """Initialize store with a filepath."""
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
        return self.bytes_to_subject_node(self.store.__getitem__(uri))

    def __setitem__(self, uri: str, subject_node: SubjectNodeType):
        """Save subject node to dbm database."""
        return self.store.__setitem__(uri, self.subject_node_to_bytes(subject_node))

    def __iter__(self):
        """Iterate over all subject nodes in dbm database."""
        for uri in self.store.keys():
            yield uri

    def __len__(self):
        """Return number of subject nodes in dbm database."""
        return self.store.__len__()

    def close(self):
        """Close dbm database."""
        if self.store:
            self.store.close()
            self.store = None


if __name__ == "__main__":
    import os
    import tempfile

    with tempfile.TemporaryDirectory(suffix="_subject_store") as td:
        store = SubjectHierarchyDbmStore[SubjectNode](os.path.join(td, "data"))

        subject1 = SubjectNode("uri://subject1", "subject 1", None)
        subject2 = SubjectNode("uri://subject2", "subject 2", "uri://subject1")

        store[subject1.uri] = subject1
        store[subject2.uri] = subject2

        print(get_subject_label_breadcrumb(store, store[subject2.uri]))
