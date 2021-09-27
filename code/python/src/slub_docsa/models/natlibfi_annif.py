"""Annif model implementation.

This module provides an Annif model interface such that Annif models can be used with this library.
"""

# pylint: disable=too-few-public-methods, consider-using-with, too-many-instance-attributes, too-many-arguments

import os
import logging
import tempfile
import time
from typing import Iterable, Mapping, Optional, Sequence, Any

import numpy as np

from annif.corpus import Subject as AnnifSubject
from annif.corpus import Document as AnnifDocument
from annif.corpus import SubjectIndex as AnnifSubjectIndex
from annif.backend import get_backend
from annif.analyzer.analyzer import Analyzer
from annif.analyzer.snowball import SnowballAnalyzer

import rdflib
from rdflib.namespace import SKOS

from slub_docsa.common.model import Model
from slub_docsa.common.document import Document
from slub_docsa.common.subject import SubjectHierarchyType, SubjectNodeType
from slub_docsa.data.preprocess.document import document_as_concatenated_string
from slub_docsa.data.preprocess.skos import subject_hierarchy_to_skos_graph
from slub_docsa.data.load.nltk import download_nltk
from slub_docsa.evaluation.incidence import subject_targets_from_incidence_matrix

logger = logging.getLogger(__name__)


class _CustomAnnifVocabulary:
    """A custom Annif vocabulary, which does not support exposing subjects as an RDFlib graph yet."""

    subject_skos_graph: Optional[rdflib.Graph]
    subjects: AnnifSubjectIndex
    skos: Any

    def __init__(self, subject_index, subject_corpus, subject_skos_graph: rdflib.Graph = None):
        """Set the subject index."""
        self.subjects = subject_index
        self.skos = subject_corpus
        self.subject_skos_graph = subject_skos_graph

    def as_graph(self):
        """Return an RDFlib graph of subjects."""
        logger.debug("annif vocabulary subject skos as graph called")
        # if self.subject_skos_graph is not None:
        #     print(self.subject_skos_graph.serialize(format="turtle"))
        return self.subject_skos_graph


class _CustomAnnifProject:
    """A custom Annif project that allows to customize the datadir, subject index and analyzer."""

    datadir: str
    subjects: AnnifSubjectIndex
    vocab: _CustomAnnifVocabulary
    analyzer: Analyzer

    def __init__(self, datadir, subject_corpus, analyzer, subject_skos_graph: rdflib.Graph = None):
        """Set the datadir, subject index and analyzer."""
        self.datadir = datadir
        self.subjects = AnnifSubjectIndex(subject_corpus)
        self.vocab = _CustomAnnifVocabulary(
            subject_index=self.subjects,
            subject_corpus=subject_corpus,
            subject_skos_graph=subject_skos_graph
        )
        self.analyzer = analyzer


class _CustomAnnifSubjectCorpus:
    """A custom subject corpus, which simply allows to iterate over all subjects."""

    subjects: Iterable[AnnifSubject]
    concepts: Iterable[str]

    subjects_by_uri: Mapping[str, AnnifSubject]

    def __init__(self, subjects: Iterable[AnnifSubject]):
        """Set the list of subjects."""
        self.subjects = subjects
        self.concepts = [s.uri for s in subjects]
        self.subjects_by_uri = {s.uri: s for s in subjects}

    def get_concept_labels(self, concept, _label_types, _language):
        """Return a list of labels for each subject."""
        if concept in self.subjects_by_uri:
            return [self.subjects_by_uri[concept].label]
        return []


class _CustomAnnifDocumentCorpus:
    """A custom Annif document corpus, which simply allows to iterate over all documets."""

    documents: Sequence[AnnifDocument]

    def __init__(self, documents: Sequence[AnnifDocument]):
        """Set the document iterable."""
        self.documents = documents

    def is_empty(self):
        """Check whether the iterable of documents is empty."""
        try:
            return next(self.documents.__iter__()) is None
        except StopIteration:
            return True


class AnnifModel(Model):
    """Interfaces with Annif to train various models and allow predictions."""

    def __init__(
        self,
        model_type: str,
        language: str,
        subject_order: Sequence[str] = None,
        subject_hierarchy: SubjectHierarchyType[SubjectNodeType] = None,
        data_dir: str = None
    ):
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
        self.language = language
        self.subject_hierarchy = subject_hierarchy
        self.subject_order = subject_order
        self.subject_skos_graph = None
        self.data_dir = data_dir
        self.temporary_directory = None
        self.analyzer = None
        self.n_unique_subject = None
        self.project = None
        self.model = None

        self._init_analyzer()
        self._init_data_dir()
        self._init_subject_skos_graph()

    def _init_analyzer(self):
        download_nltk("punkt")
        self.analyzer = SnowballAnalyzer(self.language)

    def _init_data_dir(self):
        if self.data_dir is None:
            self.temporary_directory = tempfile.TemporaryDirectory()
            self.data_dir = self.temporary_directory.name
        elif not os.path.exists(self.data_dir):
            raise ValueError("data directory %s does not exist" % self.data_dir)

    def _init_subject_skos_graph(self):
        if self.subject_hierarchy is not None and self.subject_order is not None:
            self.subject_skos_graph = subject_hierarchy_to_skos_graph(
                subject_hierarchy=self.subject_hierarchy,
                language=self.language,
                mandatory_subject_list=self.subject_order,
            )
        elif self.model_type in ["yake", "stwfsa", "mllm"]:
            logger.error("Annif model %s requires that subject hierarchy is provided", self.model_type)

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
        if len(train_documents) != train_targets.shape[0]:
            raise ValueError("train documents size %d does not match incidence matrix shape %s" % (
                len(train_documents), str(train_targets.shape)
            ))

        if self.subject_order is None:
            # there is no info on which columns in train_targets are what subjets, generate numbered subjects
            self.n_unique_subject = int(train_targets.shape[1])
            logger.debug("there are %d unique subjects", self.n_unique_subject)

            # define subjects
            numbered_subjects = [str(i) for i in range(self.n_unique_subject)]
            train_subject_targets = subject_targets_from_incidence_matrix(train_targets, numbered_subjects)
            annif_subject_list = [
                AnnifSubject(uri=uri, label=uri, notation=None, text=None) for uri in numbered_subjects
            ]
        else:
            # use uri from subject order to identify columns in train_targets
            self.n_unique_subject = len(self.subject_order)
            train_subject_targets = subject_targets_from_incidence_matrix(train_targets, self.subject_order)

            if self.subject_hierarchy is None:
                # create Annif subjects from subject order only (label info via subject hierarchy not available)
                annif_subject_list = [
                   AnnifSubject(uri=uri, label=uri, notation=None, text=None) for uri in self.subject_order
                ]
            else:
                # create Annif subjects with label info from subject hierarchy
                logger.debug("create annif subject list from subject order and subject hierarchy")
                annif_subject_list = [
                    AnnifSubject(
                        uri=uri,
                        label=self.subject_hierarchy[uri].label,
                        notation=None,
                        text=None
                    ) for uri in self.subject_order
                ]

        subject_vocab = _CustomAnnifSubjectCorpus(annif_subject_list)

        # define corpus
        annif_document_list = [
            AnnifDocument(
                text=document_as_concatenated_string(d),
                uris=train_subject_targets[i],
                labels=None
            )
            for i, d in enumerate(train_documents)
        ]
        document_corpus = _CustomAnnifDocumentCorpus(annif_document_list)

        # setup project
        logger.debug("annif: creating project")
        self.project = _CustomAnnifProject(
            datadir=self.data_dir,
            subject_corpus=subject_vocab,
            analyzer=self.analyzer,
            subject_skos_graph=self.subject_skos_graph
        )

        model_type = get_backend(self.model_type)
        self.model = model_type(
            backend_id=self.model_type,
            config_params={
                "limit": len(annif_subject_list),
                "language": self.language,
            },
            project=self.project
        )

        logger.debug("annif: call train on model with %d documents", len(document_corpus.documents))
        if self.model_type != "yake":
            self.model.train(document_corpus, params={
                "language": self.language,
                "concept_type_uri": SKOS.Concept,
                "thesaurus_relation_type_uri": SKOS.broader,
                "thesaurus_relation_is_specialisation": False,
            })
        return self

    def predict_proba(self, test_documents: Sequence[Document]) -> np.ndarray:
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

        last_info_time = time.time()

        for i, doc in enumerate(test_documents):

            results = self.model.suggest(document_as_concatenated_string(doc))
            annif_score_vector = results.as_vector(self.project.subjects)

            for j in range(self.n_unique_subject):
                if self.subject_order is None:
                    idx = int(self.project.subjects[j][0])
                else:
                    idx = j
                probabilities[i, idx] = annif_score_vector[j]

            if time.time() - last_info_time > 5:
                last_info_time = time.time()
                logger.info("predicted %d out of %d samples", i+1, len(test_documents))

        if np.min(probabilities) < 0.0:
            raise RuntimeError("some probabilities below 0.0")

        if np.max(probabilities) > 1.0:
            raise RuntimeError("some probabilities above 1.0")

        if np.isnan(np.sum(probabilities)):
            raise RuntimeError("some probabilities are nan")

        return probabilities

    def __del__(self):
        """Delete temporary directory for Annif model if it was created before."""
        if self.temporary_directory is not None:
            self.temporary_directory.cleanup()


if __name__ == "__main__":

    from slub_docsa.data.artificial.simple import get_static_mini_dataset
    from slub_docsa.evaluation.incidence import subject_incidence_matrix_from_targets, unique_subject_order

    logging.basicConfig(level=logging.DEBUG)

    dataset = get_static_mini_dataset()
    model = AnnifModel("tfidf", "english")

    my_subject_order = unique_subject_order(dataset.subjects)
    incidence_matrix = subject_incidence_matrix_from_targets(dataset.subjects, my_subject_order)
    model.fit(dataset.documents, incidence_matrix)

    probabilties = model.predict_proba([Document(uri="test", title="boring document title")])

    print(probabilties)
