"""Test script to use Annif backend model via import."""

# pylint: disable=too-few-public-methods, consider-using-with

import os
import logging
import tempfile
from typing import Iterable, Sequence

import numpy as np

from annif.corpus import Subject as AnnifSubject
from annif.corpus import Document as AnnifDocument
from annif.corpus import SubjectIndex as AnnifSubjectIndex
from annif.backend import get_backend
from annif.analyzer.analyzer import Analyzer
from annif.analyzer.snowball import SnowballAnalyzer

from slub_docsa.common.model import Model
from slub_docsa.common.document import Document
from slub_docsa.data.load.nltk import download_nltk
from slub_docsa.evaluation.incidence import subject_list_from_incidence_matrix

logger = logging.getLogger(__name__)


class _CustomAnnifVocabulary:
    """A custom Annif vocabulary, which does not support exposing subjects as an RDFlib graph yet."""

    subjects: AnnifSubjectIndex

    def __init__(self, subject_index):
        """Set the subject index."""
        self.subjects = subject_index

    def as_graph(self):
        """Return an RDFlib grahh of subjects."""
        raise NotImplementedError()


class _CustomAnnifProject:
    """A custom Annif project that allows to customize the datadir, subject index and analyzer."""

    datadir: str
    subjects: AnnifSubjectIndex
    vocab: _CustomAnnifVocabulary
    analyzer: Analyzer

    def __init__(self, datadir, subject_corpus, analyzer):
        """Set the datadir, subject index and analyzer."""
        self.datadir = datadir
        self.subjects = AnnifSubjectIndex(subject_corpus)
        self.vocab = _CustomAnnifVocabulary(subject_index=self.subjects)
        self.analyzer = analyzer


class _CustomAnnifSubjectCorpus:
    """A custom subject corpus, which simply allows to iterate over all subjects."""

    subjects: Iterable[AnnifSubject]

    def __init__(self, subjects: Iterable[AnnifSubject]):
        """Set the list of subjects."""
        self.subjects = subjects


class _CustomAnnifDocumentCorpus:
    """A custom Annif document corpus, which simply allows to iterate over all documets."""

    documents: Iterable[AnnifDocument]

    def __init__(self, documents: Iterable[AnnifDocument]):
        """Set the document iterable."""
        self.documents = documents

    def is_empty(self):
        """Check whether the iterable of documents is empty."""
        try:
            return next(self.documents.__iter__()) is None
        except StopIteration:
            return True


# MODEL_IDENTIFIER = "vw_multi"  # works, requires pip install
# MODEL_IDENTIFIER = "yake"  # does not work, requires skos attribute of subject corpus
# MODEL_IDENTIFIER = "stwfsa"  # does not work, requires subjects as rdflib graph
# MODEL_IDENTIFIER = "mllm"  # does not work, requires subjects as rdflib graph
# MODEL_IDENTIFIER = "omikuji"  # works, requires pip install
# MODEL_IDENTIFIER = "svc"  # works
# MODEL_IDENTIFIER = "fasttext"  # works, requires pip install


class AnnifModel(Model):
    """Interfaces with Annif to train various models and allow predictions."""

    def __init__(self, model_type: str, data_dir: str = None):
        """Initialize model with a Annif model type identifier and data directory.

        Parameters
        ----------
        model_type: str
            The Annif model identifier, e.g., tfidf, fasttext, svc, omikuji, etc.
        data_dir: str | None
            The directory Annif will store temporary files, e.g., trained models.
            If it is None, a temporary directory is created and deleted as soon as the model instance is deleted.
        """
        self.model_type = model_type
        self.data_dir = data_dir
        self.temporary_directory = None
        self.analyzer = None
        self.n_unique_subject = None
        self.project = None
        self.model = None

        self._init_analyzer()
        self._init_data_dir()

    def _init_analyzer(self):
        download_nltk("punkt")
        self.analyzer = SnowballAnalyzer("english")

    def _init_data_dir(self):
        if self.data_dir is None:
            self.temporary_directory = tempfile.TemporaryDirectory()
            self.data_dir = self.temporary_directory.name
        elif not os.path.exists(self.data_dir):
            raise ValueError("data directory %s does not exist" % self.data_dir)

    def fit(self, train_documents: Sequence[Document], train_targets: np.ndarray):
        """Train an Annif model with a sequence of documents and a subject incidence matrix.

        Parameters
        ----------
        train_documents: Sequence[Document]
            The sequence of documents that is used for training a model.
        train_targets: numpy.ndarray
            The incidence matrix describing which document of `train_documents` belongs to which subjects. The matrix
            has to have a shape of (n_docs, n_subjects).

        Returns
        -------
        Model
            self
        """
        self.n_unique_subject = int(train_targets.shape[1])
        logger.debug("there are %d unique subjects", self.n_unique_subject)

        # define subjects
        numbered_subjects = [str(i) for i in range(self.n_unique_subject)]
        train_subject_list = subject_list_from_incidence_matrix(train_targets, numbered_subjects)
        annif_subject_list = [
            AnnifSubject(uri=uri, label=uri, notation=None, text=None) for uri in numbered_subjects
        ]
        subject_vocab = _CustomAnnifSubjectCorpus(annif_subject_list)

        # define corpus
        annif_document_list = [
            AnnifDocument(text=d.title, uris=train_subject_list[i], labels=None) for i, d in enumerate(train_documents)
        ]
        document_corpus = _CustomAnnifDocumentCorpus(annif_document_list)

        # setup project
        logger.debug("annif: creating project")
        self.project = _CustomAnnifProject(
            datadir=self.data_dir,
            subject_corpus=subject_vocab,
            analyzer=self.analyzer
        )

        model_type = get_backend(self.model_type)
        self.model = model_type(
            backend_id=self.model_type,
            config_params={'limit': len(annif_subject_list)},
            project=self.project
        )

        logger.debug("annif: call train on model")
        self.model.train(document_corpus, params={"language": "english"})
        return self

    def predict_proba(self, test_documents: Sequence[Document]) -> Iterable[Iterable[str]]:
        """Predict subject probabilities using a trained Annif model.

        Parameters
        ----------
        test_documents: Sequence[Document]
            The test sequence of documents that are supposed to be evaluated.

        Returns
        -------
        numpy.ndarray
            The matrix of subject probabilities with a shape of (n_docs, n_subjects). The column order has to match
            the order that was provided as `train_targets` to the `fit` method.
        """
        if self.model is None or self.project is None or self.n_unique_subject is None:
            raise RuntimeError("project and model is not available, call fit before predict!")

        probabilities = np.empty((len(test_documents), self.n_unique_subject))
        probabilities[:, :] = np.nan

        for i, doc in enumerate(test_documents):

            results = self.model.suggest(doc.title)
            annif_score_vector = results.as_vector(self.project.subjects)

            for j in range(self.n_unique_subject):
                idx = int(self.project.subjects[j][0])
                probabilities[i, idx] = annif_score_vector[j]

        return probabilities

    def __del__(self):
        """Delete temporary directory for Annif model if it was created before."""
        if self.temporary_directory is not None:
            self.temporary_directory.cleanup()


if __name__ == "__main__":

    from slub_docsa.evaluation.data import get_static_mini_dataset
    from slub_docsa.evaluation.incidence import subject_incidence_matrix_from_list, unique_subject_list

    logging.basicConfig(level=logging.DEBUG)

    dataset = get_static_mini_dataset()
    model = AnnifModel("tfidf")

    subject_order = unique_subject_list(dataset.subjects)
    incidence_matrix = subject_incidence_matrix_from_list(dataset.subjects, subject_order)
    model.fit(dataset.documents, incidence_matrix)

    probabilties = model.predict_proba([Document(uri="test", title="boring document title")])

    print(probabilties)
