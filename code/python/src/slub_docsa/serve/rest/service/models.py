"""Setup classification models for REST service."""

import logging

from typing import Sequence, Optional, Mapping, Tuple

import numpy as np

from slub_docsa.common.document import Document
from slub_docsa.common.model import ClassificationModel
from slub_docsa.serve.common import ClassificationPrediction, ClassificationResult, ClassificationModelsRestService
from slub_docsa.serve.common import PublishedClassificationModel, PublishedClassificationModelInfo
from slub_docsa.serve.common import ModelNotFoundException, SchemasRestService
from slub_docsa.serve.models.classification.classic import get_classic_classification_models_map
from slub_docsa.serve.store.models import find_stored_classification_model_infos, load_published_classification_model


logger = logging.getLogger(__name__)


class PartialModelClassificationRestService(ClassificationModelsRestService):
    """A partial implementation providing implementations for the `classify` and `classify_and_describe` methods."""

    def _get_schemas_service(self) -> SchemasRestService:
        """Return schema service such that labels and breadcrums can be added to classification results."""
        raise NotImplementedError()

    def _get_model(self, model_id) -> PublishedClassificationModel:
        """Load model such that it can be used for classifying documents."""
        raise NotImplementedError()

    def subjects(self, model_id: str) -> Sequence[str]:
        """Return subjects supported by a model."""
        return self._get_model(model_id).subject_order

    def classify(
        self,
        model_id: str,
        documents: Sequence[Document],
        limit: int = 10,
        threshold: float = 0.0
    ) -> Sequence[Sequence[ClassificationResult]]:
        """Perform classification for a list of documents."""
        return classify_with_limit_and_threshold(
            self._get_model(model_id).model,
            documents,
            limit,
            threshold
        )

    def classify_and_describe(
        self,
        model_id: str,
        documents: Sequence[Document],
        limit: int = 10,
        threshold: float = 0.0,
        subject_info: bool = True
    ) -> Sequence[ClassificationResult]:
        """Perform classification for a list of documents and provide detailed classification results."""
        predictions = self.classify(model_id, documents, limit, threshold)
        pulished_model = self._get_model(model_id)
        schema_id = pulished_model.info.schema_id
        schemas_service = self._get_schemas_service()

        def _transform_prediction(score, subject_idx):
            subject_uri = pulished_model.subject_order[subject_idx]
            return ClassificationPrediction(
                score=score,
                subject_uri=subject_uri,
                subject_info=None if not subject_info else schemas_service.subject_info(schema_id, subject_uri)
            )

        return [
            ClassificationResult(document_uri=document.uri, predictions=[
                _transform_prediction(s, i) for s, i in prediction
            ]) for document, prediction in zip(documents, predictions)
        ]


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


class SingleStoredModelRestService(PartialModelInfosRestService, PartialModelClassificationRestService):
    """REST implementation that only loads a single model at a time for classification."""

    def __init__(self, directory: str, schemas_service: SchemasRestService):
        """Init."""
        logger.info("load one model at a time")
        self.schemas_service = schemas_service
        self.model_types = get_classic_classification_models_map()
        self.model_infos = find_stored_classification_model_infos(directory)
        self.loaded_model: Optional[PublishedClassificationModel] = None
        logger.info("discovered %d models %s", len(self.model_infos), str(list(self.model_infos.keys())))

    def _get_schemas_service(self) -> SchemasRestService:
        return self.schemas_service

    def _get_model_infos_dict(self) -> Mapping[str, PublishedClassificationModelInfo]:
        return {model_id: info.info for model_id, info in self.model_infos.items()}

    def _get_model(self, model_id) -> PublishedClassificationModel:
        if model_id not in self.model_infos:
            raise ModelNotFoundException(model_id)

        if self.loaded_model is None or model_id != self.loaded_model.info.model_id:
            self.loaded_model = load_published_classification_model(
                self.model_infos[model_id].directory, self.model_types
            )
        return self.loaded_model


class AllStoredModelRestService(PartialModelInfosRestService, PartialModelClassificationRestService):
    """A service implementation that pre-loads all stored models and serves them from memory."""

    def __init__(self, directory: str, schemas_service: SchemasRestService):
        """Init."""
        logger.info("load all stored models into memory")
        self.schemas_service = schemas_service
        self.model_types = get_classic_classification_models_map()
        self.model_infos = find_stored_classification_model_infos(directory)
        self.models = {
            model_id: load_published_classification_model(info.directory, self.model_types)
            for model_id, info in self.model_infos.items()
        }
        logger.info("discovered %d models %s", len(self.model_infos), str(list(self.model_infos.keys())))

    def _get_schemas_service(self) -> SchemasRestService:
        return self.schemas_service

    def _get_model_infos_dict(self) -> Mapping[str, PublishedClassificationModelInfo]:
        return {model_id: info.info for model_id, info in self.model_infos.items()}

    def _get_model(self, model_id) -> PublishedClassificationModel:
        if model_id not in self.models:
            raise ModelNotFoundException(model_id)
        return self.models[model_id]


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
