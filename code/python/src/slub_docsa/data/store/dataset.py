"""Persistent storage of datasets for fast random access of both documents and subject annotations."""

import os
import logging

from typing import Callable, Iterator, Sequence

from sqlitedict import SqliteDict

from slub_docsa.common.dataset import Dataset
from slub_docsa.common.document import Document
from slub_docsa.common.sample import Sample
from slub_docsa.common.subject import SubjectUriList

logger = logging.getLogger(__name__)


class _DatasetSqliteStoreSequence(Sequence):
    """Allows to access documents and subjects via sequence interface in a dbm database."""

    def __init__(self, store, populate_mode: bool, idx):
        self.store = store
        self.idx = idx
        self.populate_mode = populate_mode
        self.closed = False

    def __getitem__(self, key):
        if self.populate_mode:
            raise ValueError("can not access data in populate mode")
        if self.closed:
            raise ValueError("can not access data of closed store")
        return self.store[str(key)][self.idx]

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


class DatasetSqliteStore(Dataset):
    """Persistent storage for a dataset as a sqlite database.

    This naive implementation stores documents and subject annotations via `pickle` in a sqlite database.

    .. note::

        No checks are performed to guarantee anything, e.g. continuity, or that data is not overwritten.

    Examples
    --------
    Create a new dataset store and populate a list of documents and subject annotations.

    >>> store = DatasetSqliteStore("/tmp/dataset.sqlite", populate_mode=True)
    >>> store.populate(iter([(Document(uri="doc1", title="Test Document"), ["subject1"])]))
    >>> store.close()

    Re-open the database in read-only mode when using it as dataset reference.

    >>> dataset = DatasetSqliteStore("/tmp/dataset.sqlite", populate_mode=False)
    >>> dataset.documents[0]
    <slub_docsa.common.document.Document object at 0x7fc3ffe4fa10>
    """

    documents: Sequence[Document]
    """Provides access to documents as sequence interface, loading documents from dbm database."""

    subjects: Sequence[SubjectUriList]
    """Provides access to subject annotations as sequence interface, loading them from the dbm database."""

    def __init__(self, filepath: str, populate_mode: bool = False, batch_size: int = 1000):
        """Initialize new sqlite database.

        Parameters
        ----------
        filepath: str
            Path to file that is loaded or created as sqlite database

        populate_mode: bool
            If true, allows to call `populate` method, which stores documents and subjects in batch mode.
            If false, database is loaded in read-only mode.

        batch_size: int = 1000
            Number of samples until disc synchronization is triggered.
        """
        self.populate_mode = populate_mode
        self.batch_size = batch_size
        self.store = SqliteDict(filepath, "dataset", autocommit=not populate_mode, flag="w" if populate_mode else "r")
        self._documents = _DatasetSqliteStoreSequence(self.store, self.populate_mode, 0)
        self._subjects = _DatasetSqliteStoreSequence(self.store, self.populate_mode, 1)
        self.documents = self._documents
        self.subjects = self._subjects

    def populate(self, samples: Iterator[Sample]):
        """Populate the database from an iterator over samples (documents and subjects).

        Stores samples one by one, but synchronizes with disc in batches.
        Overwrites samples if called multiple times on the same database instance.
        """
        if not self.populate_mode:
            raise ValueError("can not populate store in read only mode")
        if self.store is None:
            raise ValueError("can not populate already closed store")
        for i, sample in enumerate(samples):
            logger.debug("write example %d", i)
            self.store[str(i)] = sample
            if i % self.batch_size == 0:
                # sync data to disc every x-th sample
                self.store.commit(True)

    def close(self):
        """Close sqlite database. Reads and writes are no longer possible and will result in an exception."""
        if self.store:
            if self.populate_mode:
                self.store.commit()
            self._documents.close()
            self._subjects.close()
            self.store.close()
            self.store = None


def load_persisted_dataset_from_lazy_sample_iterator(
    lazy_sample_iterator: Callable[[], Iterator[Sample]],
    filepath: str
) -> Dataset:
    """Return dataset from persistent dbm store, or use sample iterator to populate and return a new dbm store."""
    if not os.path.exists(filepath):
        store = DatasetSqliteStore(filepath, populate_mode=True)
        store.populate(lazy_sample_iterator())
        store.close()
    return DatasetSqliteStore(filepath)
