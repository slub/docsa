"""Setup classification models for REST service."""

import logging

from typing import Sequence, Optional, Mapping, Tuple

import numpy as np

from slub_docsa.common.document import Document
from slub_docsa.common.model import ClassificationModel
from slub_docsa.serve.common import ClassificationPrediction, ClassificationResult, ClassificationModelsRestService
from slub_docsa.serve.common import PublishedClassificationModel, PublishedClassificationModelInfo
from slub_docsa.serve.common import ModelNotFoundException
from slub_docsa.serve.models.classification.classic import get_classic_classification_models_map
from slub_docsa.serve.store.models import find_stored_classification_model_infos, load_published_classification_model


logger = logging.getLogger(__name__)


class PartialModelInfosRestService(ClassificationModelsRestService):
    """Abstract implementation of a Rest service."""

    def _get_model_infos_dict(self) -> Mapping[str, PublishedClassificationModelInfo]:
        """Return dictionary containing model information indexed by their id.

        Returns
        -------
        Mapping[str, PublishedClassificationModelInfo]
            a dictionary from model_id to the information about published model
        """
        raise NotImplementedError()

    def find_models(
        self,
        languages: Optional[Sequence[str]] = None,
        schema_id: Optional[str] = None,
        tags: Optional[Sequence[str]] = None,
    ) -> Sequence[str]:
        """List available models matching certain criteria like supported languages."""
        model_ids = []
        for model_id, info in self._get_model_infos_dict().items():

            # skip models that do not support requested languages
            if languages is not None and any(lang_code not in info.supported_languages for lang_code in languages):
                continue

            # skip models that do not match requested schema_id
            if schema_id is not None and schema_id != info.schema_id:
                continue

            # skip models that are not labelled with tags
            if tags is not None and any(tag not in info.tags for tag in tags):
                continue

            model_ids.append(model_id)
        return model_ids

    def model_info(self, model_id: str) -> PublishedClassificationModelInfo:
        """Return information describing a classification model."""
        if model_id not in self._get_model_infos_dict():
            raise ModelNotFoundException(model_id)
        return self._get_model_infos_dict()[model_id]


class PartialAllModelsRestService(ClassificationModelsRestService):
    """An abstract implementation of a REST service that serves already loaded classification models."""

    def _get_models_dict(self) -> Mapping[str, PublishedClassificationModel]:
        """Return a mapping from model_id to already loaded classification models.

        Returns
        -------
        Mapping[str, ClassificationModel]
            the map from model_id to already loaded classification models

        Raises
        ------
        NotImplementedError
            Overwrite abstract method in subclass
        """
        raise NotImplementedError()

    def classify(
        self,
        model_id: str,
        documents: Sequence[Document],
        limit: int = 10,
        threshold: float = 0.0
    ) -> Sequence[Sequence[ClassificationResult]]:
        """Perform classification for a list of documents."""
        if model_id not in self._get_models_dict():
            raise ModelNotFoundException(model_id)

        pulished_model = self._get_models_dict()[model_id]

        return classify_with_limit_and_threshold(
            pulished_model.model,
            documents,
            limit,
            threshold
        )

    def classify_and_describe(
        self,
        model_id: str,
        documents: Sequence[Document],
        limit: int = 10,
        threshold: float = 0.0
    ) -> Sequence[ClassificationResult]:
        """Perform classification for a list of documents and provide detailed classification results."""
        predictions = self.classify(model_id, documents, limit, threshold)
        pulished_model = self._get_models_dict()[model_id]

        return [
            ClassificationResult(document_uri=document.uri, predictions=[
                ClassificationPrediction(score=s, subject_uri=pulished_model.subject_order[i])
                for s, i in prediction
            ]) for document, prediction in zip(documents, predictions)
        ]

    def subjects(self, model_id: str) -> Sequence[str]:
        """Return subjects supported by a model."""
        if model_id not in self._get_models_dict():
            raise ModelNotFoundException(model_id)
        return self._get_models_dict()[model_id].subject_order


class SingleStoredModelRestService(PartialModelInfosRestService):
    """REST implementation that only loads a single model at a time for classification."""

    def __init__(self, directory: str):
        """Init."""
        logger.info("load one model at a time")
        self.model_types = get_classic_classification_models_map()
        self.model_infos = find_stored_classification_model_infos(directory)
        self.loaded_model: Optional[PublishedClassificationModel] = None
        logger.info("discovered %d models %s", len(self.model_infos), str(list(self.model_infos.keys())))

    def _get_model_infos_dict(self) -> Mapping[str, PublishedClassificationModelInfo]:
        return {model_id: info.info for model_id, info in self.model_infos.items()}

    def _load_model(self, model_id):
        if model_id not in self.model_infos:
            raise ModelNotFoundException(model_id)

        if self.loaded_model is None or model_id != self.loaded_model.info.model_id:
            self.loaded_model = load_published_classification_model(
                self.model_infos[model_id].directory, self.model_types
            )

    def classify(
        self,
        model_id: str,
        documents: Sequence[Document],
        limit: int = 10,
        threshold: float = 0.0
    ) -> Sequence[Sequence[Tuple[float, int]]]:
        """Perform classification for a list of documents and return tuples of score and subject order id."""
        self._load_model(model_id)

        return classify_with_limit_and_threshold(
            self.loaded_model.model,
            documents,
            limit,
            threshold
        )

    def classify_and_describe(
        self,
        model_id: str,
        documents: Sequence[Document],
        limit: int = 10,
        threshold: float = 0.0
    ) -> Sequence[ClassificationResult]:
        """Perform classification for a list of documents and provide detailed classification results."""
        predictions = self.classify(model_id, documents, limit, threshold)

        return [
            ClassificationResult(document_uri=document.uri, predictions=[
                ClassificationPrediction(score=s, subject_uri=self.loaded_model.subject_order[i])
                for s, i in prediction
            ]) for document, prediction in zip(documents, predictions)
        ]

    def subjects(self, model_id: str) -> Sequence[str]:
        """Return subjects supported by a model."""
        self._load_model(model_id)
        return self.loaded_model.subject_order


class AllStoredModelRestService(PartialAllModelsRestService, PartialModelInfosRestService):
    """A service implementation that pre-loads all stored models and serves them from memory."""

    def __init__(self, directory: str):
        """Init."""
        logger.info("load all stored models into memory")
        self.model_types = get_classic_classification_models_map()
        self.model_infos = find_stored_classification_model_infos(directory)
        self.models = {
            model_id: load_published_classification_model(info.directory, self.model_types)
            for model_id, info in self.model_infos.items()
        }
        logger.info("discovered %d models %s", len(self.model_infos), str(list(self.model_infos.keys())))

    def _get_model_infos_dict(self) -> Mapping[str, PublishedClassificationModelInfo]:
        return {model_id: info.info for model_id, info in self.model_infos.items()}

    def _get_models_dict(self) -> Mapping[str, PublishedClassificationModel]:
        return self.models


def classify_with_limit_and_threshold(
    model: ClassificationModel,
    documents: Sequence[Document],
    limit: int = 10,
    threshold: float = 0.0
) -> Sequence[Sequence[Tuple[float, int]]]:
    """Perform classification and compile results using certain limit and threshold."""
    # do actual classification
    probabilities = model.predict_proba(documents)
    limit = min(probabilities.shape[1], limit)
    if limit < probabilities.shape[1]:
        # find best results
        topk_indexes = np.argpartition(probabilities, -limit, axis=1)[:, -limit:]
    else:
        # return all results
        topk_indexes = np.tile(np.arange(limit), (probabilities.shape[0], 1))
    topk_probabilities = np.take_along_axis(probabilities, topk_indexes, axis=1)
    topk_sort_indexes = np.argsort(-topk_probabilities, axis=1)
    sorted_topk_indexed = np.take_along_axis(topk_indexes, topk_sort_indexes, axis=1)
    sorted_topk_probabilities = np.take_along_axis(topk_probabilities, topk_sort_indexes, axis=1)

    # compile and return results
    return [
        [(float(p), int(i)) for i, p in zip(indexes, probs) if p > threshold]
        for indexes, probs in zip(sorted_topk_indexed, sorted_topk_probabilities)
    ]
