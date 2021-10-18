"""Persistent storage of datasets for fast random access of samples."""

import dbm.gnu as dbm
import pickle  # nosec
from typing import Iterable, Sequence, Tuple

from slub_docsa.common.dataset import Dataset
from slub_docsa.common.document import Document
from slub_docsa.common.subject import SubjectUriList


class _DatasetDbmStoreSequence(Sequence):

    def __init__(self, store, index: int, populate_mode: bool):
        self.store = store
        self.idx = index
        self.populate_mode = populate_mode
        self.closed = False

    def __getitem__(self, key):
        if self.populate_mode:
            raise ValueError("can not access data in populate mode")
        if self.closed:
            raise ValueError("can not access data of closed store")
        return pickle.loads(self.store[str(key)])[self.idx]

    def __len__(self):
        if self.populate_mode:
            raise ValueError("can not access data in populate mode")
        if self.closed:
            raise ValueError("can not access data of closed store")
        return len(self.store)

    def __iter__(self):
        if self.populate_mode:
            raise ValueError("can not access data in populate mode")
        if self.closed:
            raise ValueError("can not access data of closed store")
        for i in range(self.__len__()):
            yield self.__getitem__(i)

    def close(self):
        """Mark store as closed."""
        self.closed = True


class DatasetDbmStore(Dataset):
    """Persistent storage for a dataset using the python dbm interface."""

    documents: _DatasetDbmStoreSequence
    subjects: _DatasetDbmStoreSequence

    def __init__(self, filepath: str, populate_mode: bool = False, batch_size: int = 1000):
        """Create a new dbm store for a dataset."""
        self.populate_mode = populate_mode
        self.batch_size = batch_size
        self.store = dbm.open(filepath, "cf" if populate_mode else "r")
        self.documents = _DatasetDbmStoreSequence(self.store, 0, populate_mode)
        self.subjects = _DatasetDbmStoreSequence(self.store, 1, populate_mode)

    def populate(self, samples: Iterable[Tuple[Document, SubjectUriList]]):
        """Populate the store with documents and subjects."""
        if not self.populate_mode:
            raise ValueError("can not populate store in read only mode")
        if self.store is None:
            raise ValueError("can not populate already closed store")
        for i, sample in enumerate(samples):
            self.store[str(i)] = pickle.dumps(sample)  # nosec
            if i % self.batch_size == 0:
                # sync data to disc every x-th sample
                self.store.sync()

    def close(self):
        """Close dbm database."""
        if self.store:
            self.store.sync()
            self.documents.close()
            self.subjects.close()
            self.store.close()
            self.store = None
