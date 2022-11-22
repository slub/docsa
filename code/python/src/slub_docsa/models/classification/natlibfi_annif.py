"""Annif model implementation.

This module provides an Annif model interface such that Annif models can be used with this library.
"""

# pylint: disable=too-few-public-methods, consider-using-with, too-many-instance-attributes, too-many-arguments

import os
import logging
import tempfile
import time
import json

from typing import Iterable, Mapping, Optional, Sequence, Any, cast
from distutils.dir_util import copy_tree

import numpy as np

from annif.corpus import Subject as AnnifSubject
from annif.corpus import Document as AnnifDocument
from annif.corpus import SubjectIndex as AnnifSubjectIndex
from annif.backend import get_backend
from annif.analyzer.analyzer import Analyzer
from annif.analyzer.snowball import SnowballAnalyzer
from annif.suggestion import VectorSuggestionResult

import rdflib
from rdflib.namespace import SKOS

from slub_docsa.common.model import PersistableClassificationModel
from slub_docsa.common.document import Document
from slub_docsa.common.subject import SubjectHierarchy
from slub_docsa.data.preprocess.document import document_as_concatenated_string
from slub_docsa.data.preprocess.skos import subject_hierarchy_to_skos_graph
from slub_docsa.evaluation.incidence import subject_idx_from_incidence_matrix
from slub_docsa.data.load.nltk import download_nltk

logger = logging.getLogger(__name__)

SNOWBALL_LANGUAGES_FROM_CODE = {
    "de": "german",
    "en": "english"
}


class _CustomAnnifVocabulary:
    """A custom Annif vocabulary, which does not support exposing subjects as an RDFlib graph yet."""

    subject_skos_graph: Optional[rdflib.Graph]
    subjects: AnnifSubjectIndex
    skos: Any

    def __init__(self, subject_index, subject_corpus, subject_skos_graph: Optional[rdflib.Graph] = None):
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

    def __init__(self, datadir, subject_corpus, analyzer, subject_skos_graph: Optional[rdflib.Graph] = None):
        """Set the datadir, subject index and analyzer."""
        self.datadir = datadir
        self.subjects = AnnifSubjectIndex()
        self.subjects.load_subjects(subject_corpus)
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
    languages: Iterable[str]
    subjects_by_uri: Mapping[str, AnnifSubject]

    def __init__(self, subjects: Iterable[AnnifSubject], languages: Iterable[str]):
        """Set the list of subjects."""
        self.subjects = subjects
        self.concepts = [s.uri for s in subjects]
        self.subjects_by_uri = {s.uri: s for s in subjects}
        self.languages = languages

    def get_concept_labels(self, concept, _label_types, _language):
        """Return a list of labels for each subject."""
        if concept in self.subjects_by_uri:
            return [self.subjects_by_uri[concept].labels[_language]]
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
            return next(iter(self.documents)) is None
        except StopIteration:
            return True


class AnnifModel(PersistableClassificationModel):
    """Interfaces with Annif to train various models and allow predictions."""

    def __init__(
        self,
        model_type: str,
        lang_code: str,
        subject_order: Optional[Sequence[str]] = None,
        subject_hierarchy: Optional[SubjectHierarchy] = None,
        data_dir: Optional[str] = None,
        max_document_length: int = 10000
    ):
        """Initialize model with a Annif model type identifier and data directory.

        Parameters
        ----------
        model_type: str
            The Annif model identifier, e.g., tfidf, fasttext, svc, omikuji, etc.
        data_dir: str | None
            The directory Annif will store temporary files, e.g., trained models.
            If it is None, a temporary directory is created and deleted as soon as the model instance is deleted.
        max_document_length: int | None
            Reduces the document to a maximum length of this many characters if set
        """
        self.model_type = model_type
        self.lang_code = lang_code
        self.subject_hierarchy = subject_hierarchy
        self.subject_order = subject_order
        self.subject_skos_graph = None
        self.data_dir = data_dir
        self.max_document_length = max_document_length
        self.temporary_directory = None
        self.analyzer = None
        self.n_unique_subjects: int = None
        self.project = None
        self.model = None
        self.model_params: Mapping[str, Any] = None

        self._init_analyzer()
        self._init_data_dir()
        self._init_subject_skos_graph()

    def __str__(self):
        """Return string describing the Annif model and its parameters used for predictions."""
        return f"<AnnifModel type='{self.model_type}' lang_code='{self.lang_code}' " \
            + f"max_document_length={self.max_document_length}>"

    def _init_analyzer(self):
        download_nltk("punkt")
        self.analyzer = SnowballAnalyzer(SNOWBALL_LANGUAGES_FROM_CODE[self.lang_code])

    def _init_data_dir(self):
        if self.data_dir is None:
            self.temporary_directory = tempfile.TemporaryDirectory()
            self.data_dir = self.temporary_directory.name
        elif not os.path.exists(self.data_dir):
            raise ValueError(f"data directory {self.data_dir} does not exist")

    def _init_subject_skos_graph(self):
        if self.subject_hierarchy is not None and self.subject_order is not None:
            self.subject_skos_graph = subject_hierarchy_to_skos_graph(
                subject_hierarchy=self.subject_hierarchy,
                lang_code=self.lang_code,
                mandatory_subject_list=self.subject_order,
            )
        elif self.model_type in ["yake", "stwfsa", "mllm"]:
            raise ValueError(f"annif model '{self.model_type}' requires that subject hierarchy is provided")

    def _init_subject_vocab(self):
        if self.subject_order is None:
            # there is no info on which columns in train_targets are what subjets, generate numbered subjects
            numbered_subjects = [str(i) for i in range(self.n_unique_subjects)]
            annif_subject_list = [
                AnnifSubject(uri=uri, labels={self.lang_code: uri}, notation=None) for uri in numbered_subjects
            ]
        else:
            # use uri from subject order to identify columns in train_targets
            if self.subject_hierarchy is None:
                # create Annif subjects from subject order only (label info via subject hierarchy not available)
                annif_subject_list = [
                    AnnifSubject(uri=uri, labels={self.lang_code: uri}, notation=None) for uri in self.subject_order
                ]
            else:
                # create Annif subjects with label info from subject hierarchy
                logger.debug("create annif subject list from subject order and subject hierarchy")
                annif_subject_list = [
                    AnnifSubject(
                        uri=uri,
                        labels={self.lang_code: self.subject_hierarchy.subject_labels(uri).get(self.lang_code)},
                        notation=None
                    ) for uri in self.subject_order
                ]

        subject_vocab = _CustomAnnifSubjectCorpus(annif_subject_list, languages=[self.lang_code])

        return subject_vocab, annif_subject_list

    def _init_project(self, subject_vocab):
        """Initialize Annif project."""
        logger.debug("annif: creating project")
        self.project = _CustomAnnifProject(
            datadir=self.data_dir,
            subject_corpus=subject_vocab,
            analyzer=self.analyzer,
            subject_skos_graph=self.subject_skos_graph
        )

    def _init_model(self, annif_subject_list):
        """Initialize Annif model to be used for training or prediction."""
        model_type = get_backend(self.model_type)
        self.model = model_type(
            backend_id=self.model_type,
            config_params={
                "limit": len(annif_subject_list),
                "language": self.lang_code,
            },
            project=self.project
        )

        self.model_params = {
            "language": self.lang_code,
        }

        if self.model_type == "fasttext":
            self.model_params.update({
                "dim": 100,
                "lr": 0.25,
                "epoch": 5,
                "loss": "hs",
                "chunksize": 24
            })

        if self.model_type == "stwfsa":
            self.model_params.update({
                "concept_type_uri": SKOS.Concept,
                "thesaurus_relation_type_uri": SKOS.broader,
                "thesaurus_relation_is_specialisation": False,
            })

    def fit(
        self,
        train_documents: Sequence[Document],
        train_targets: np.ndarray,
        validation_documents: Optional[Sequence[Document]] = None,
        validation_targets: Optional[np.ndarray] = None,
    ):
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
            raise ValueError(
                f"train documents size {len(train_documents)} does not match "
                + f"incidence matrix shape {str(train_targets.shape)}"
            )

        train_idxs_targets = subject_idx_from_incidence_matrix(train_targets)

        # define corpus
        annif_document_list = [
            AnnifDocument(
                text=document_as_concatenated_string(d, max_length=self.max_document_length),
                subject_set=train_idxs_targets[i]
            )
            for i, d in enumerate(train_documents)
        ]
        document_corpus = _CustomAnnifDocumentCorpus(annif_document_list)

        self.n_unique_subjects = int(train_targets.shape[1])
        logger.debug("there are %d unique subjects", self.n_unique_subjects)

        subject_vocab, annif_subject_list = self._init_subject_vocab()
        self._init_project(subject_vocab)
        self._init_model(annif_subject_list)

        if self.model_type != "yake":
            logger.debug("annif: call train on model with %d documents", len(document_corpus.documents))
            self.model.train(document_corpus, params=self.model_params)
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
        if self.model is None or self.project is None or self.n_unique_subjects is None:
            raise RuntimeError("project and model is not available, call fit before predict!")

        probabilities = np.empty((len(test_documents), self.n_unique_subjects))
        probabilities[:, :] = np.nan

        last_info_time = time.time()

        for i, doc in enumerate(test_documents):

            results = self.model.suggest(document_as_concatenated_string(doc, max_length=self.max_document_length))
            results = cast(VectorSuggestionResult, results)
            annif_score_vector = results.as_vector(len(self.project.subjects))

            for j in range(self.n_unique_subjects):
                if self.subject_order is None:
                    idx = int(self.project.subjects[j][0])
                else:
                    idx = j
                probabilities[i, idx] = annif_score_vector[j]

            if time.time() - last_info_time > 5:
                last_info_time = time.time()
                logger.info("predicted %d out of %d samples", i + 1, len(test_documents))

        if np.min(probabilities) < 0.0:
            raise RuntimeError("some probabilities below 0.0")

        if np.max(probabilities) > 1.0:
            raise RuntimeError("some probabilities above 1.0")

        if np.isnan(np.sum(probabilities)):  # type: ignore
            raise RuntimeError("some probabilities are nan")

        return probabilities

    def save(self, persist_dir: str):
        """Save annif model to a directory."""
        if self.model is None or self.project is None or self.n_unique_subjects is None:
            raise RuntimeError("model has not been trained yet, call fit before saving!")

        if self.model_type in ["yake", "stwfsa", "mllm"]:
            raise ValueError(f"annif model '{self.model_type}' can not be saved (yet)")

        # save data directory contents
        copy_tree(self.data_dir, os.path.join(persist_dir, "annif"))

        # save required properties
        annif_parameters = {
            "lang_code": self.lang_code,
            "model_type": self.model_type,
            "max_document_length": self.max_document_length,
            "n_unique_subjects": self.n_unique_subjects,
        }

        with open(os.path.join(persist_dir, "annif_model_parameters.json"), "wt", encoding="utf8") as file:
            json.dump(annif_parameters, file)

    def load(self, persist_dir: str):
        """Load annif model from a directory."""
        # set data dir
        self.data_dir = os.path.join(persist_dir, "annif")

        # load parameters
        with open(os.path.join(persist_dir, "annif_model_parameters.json"), "rt", encoding="utf8") as file:
            json_data = json.load(file)
            self.lang_code = json_data["lang_code"]
            self.model_type = json_data["model_type"]
            self.max_document_length = int(json_data["max_document_length"])
            self.n_unique_subjects = int(json_data["n_unique_subjects"])

        # initialize Annif interface
        subject_vocab, annif_subject_list = self._init_subject_vocab()
        self._init_project(subject_vocab)
        self._init_model(annif_subject_list)

    def __del__(self):
        """Delete temporary directory for Annif model if it was created before."""
        if self.temporary_directory is not None:
            self.temporary_directory.cleanup()


if __name__ == "__main__":

    from slub_docsa.data.artificial.simple import get_static_mini_dataset
    from slub_docsa.evaluation.incidence import subject_incidence_matrix_from_targets, unique_subject_order

    logging.basicConfig(level=logging.DEBUG)

    with tempfile.TemporaryDirectory() as directory:
        dataset = get_static_mini_dataset()
        model = AnnifModel("omikuji", "en")

        my_subject_order = unique_subject_order(dataset.subjects)
        incidence_matrix = subject_incidence_matrix_from_targets(dataset.subjects, my_subject_order)
        model.fit(dataset.documents, incidence_matrix)

        probabilties = model.predict_proba([Document(uri="test", title="boring document title")])
        print(probabilties)

        # save and reload model
        model.save(directory)
        model = AnnifModel("omikuji", "en")
        model.load(directory)
        probabilties = model.predict_proba([Document(uri="test", title="boring document title")])
        print(probabilties)
