"""Test script to use Annif backend model via import."""

# pylint: disable=too-few-public-methods

import logging
import os
from typing import Iterable

from annif.corpus import Subject, Document, SubjectIndex
from annif.backend import get_backend
from annif.analyzer.analyzer import Analyzer
from annif.analyzer.snowball import SnowballAnalyzer

from slub_docsa.data.load.nltk import download_nltk
from slub_docsa.common import ANNIF_DIR

logger = logging.getLogger(__name__)


class CustomAnnifVocabulary:
    """A custom Annif vocabulary, which does not support exposing subjects as an RDFlib graph yet."""

    subjects: SubjectIndex

    def __init__(self, subject_index):
        """Set the subject index."""
        self.subjects = subject_index

    def as_graph(self):
        """Return an RDFlib grahh of subjects."""
        raise NotImplementedError()


class CustomAnnifProject:
    """A custom Annif project that allows to customize the datadir, subject index and analyzer."""

    datadir: str
    subjects: SubjectIndex
    vocab: CustomAnnifVocabulary
    analyzer: Analyzer

    def __init__(self, datadir, subject_corpus, analyzer):
        """Set the datadir, subject index and analyzer."""
        self.datadir = datadir
        self.subjects = SubjectIndex(subject_corpus)
        self.vocab = CustomAnnifVocabulary(subject_index=self.subjects)
        self.analyzer = analyzer


class CustomAnnifSubjectCorpus:
    """A custom subject corpus, which simply allows to iterate over all subjects."""

    subjects: Iterable[Subject]

    def __init__(self, subjects: Iterable[Subject]):
        """Set the list of subjects."""
        self.subjects = subjects


class CustomAnnifDocumentCorpus:
    """A custom Annif document corpus, which simply allows to iterate over all documets."""

    documents: Iterable[Document]

    def __init__(self, documents: Iterable[Document]):
        """Set the document iterable."""
        self.documents = documents

    def is_empty(self):
        """Check whether the iterable of documents is empty."""
        try:
            return next(self.documents.__iter__()) is None
        except StopIteration:
            return True


if __name__ == "__main__":
    logging.basicConfig(level=logging.DEBUG)

    download_nltk("punkt")

    # define subjects
    subject_list = [
        Subject(uri="uri://subject1", label="subject 1", notation="S 1", text="subect 1 description text"),
        Subject(uri="uri://subject2", label="subject 2", notation="S 2", text="subect 2 description text"),
        Subject(uri="uri://subject3", label="subject 3", notation="S 3", text="subect 3 description text"),
    ]
    subject_vocab = CustomAnnifSubjectCorpus(subject_list)

    # define corpus
    document_list = [
        Document(text="magdeburg is a great city in saxoxy anhalt", uris=[subject_list[0].uri], labels=None),
        Document(text="music makes people happy", uris=[subject_list[1].uri], labels=None),
        Document(text="music makes people sad", uris=[subject_list[2].uri], labels=None),
    ]
    document_corpus = CustomAnnifDocumentCorpus(document_list)

    # initialize analyzer
    snowball_analyzer = SnowballAnalyzer("english")

    # setup project
    logger.debug("creating annif project")
    project = CustomAnnifProject(
        datadir=os.path.join(ANNIF_DIR, "testproject"),
        subject_corpus=subject_vocab,
        analyzer=snowball_analyzer
    )

    # MODEL_IDENTIFIER = "vw_multi"  # works, requires pip install
    # MODEL_IDENTIFIER = "yake"  # does not work, requires skos attribute of subject corpus
    # MODEL_IDENTIFIER = "stwfsa"  # does not work, requires subjects as rdflib graph
    # MODEL_IDENTIFIER = "mllm"  # does not work, requires subjects as rdflib graph
    # MODEL_IDENTIFIER = "omikuji"  # works, requires pip install
    # MODEL_IDENTIFIER = "svc"  # works
    # MODEL_IDENTIFIER = "fasttext"  # works, requires pip install
    MODEL_IDENTIFIER = "tfidf"  # works

    logger.debug("creating model backend")
    model_type = get_backend(MODEL_IDENTIFIER)
    model = model_type(backend_id=MODEL_IDENTIFIER, config_params={'limit': 10}, project=project)

    logger.debug("call train on model")
    model.train(document_corpus, params={"language": "english"})

    logger.debug("call suggest on model")
    results = model.suggest("""magdeburg makes people happy""")
    for result in results.as_list(project.subjects):
        print(result)
