"""Setup classification models for REST service."""

import logging

from typing import Sequence, Optional, Mapping

import numpy as np

from slub_docsa.common.document import Document
from slub_docsa.common.model import ClassificationModel
from slub_docsa.serve.common import ClassificationResult, ClassificationModelsRestService
from slub_docsa.serve.common import PublishedClassificationModel, PublishedClassificationModelInfo
from slub_docsa.serve.models.classification.classic import get_classic_classification_models_map
from slub_docsa.serve.store.models import find_stored_classification_model_infos, load_published_classification_model


logger = logging.getLogger(__name__)


class ModelNotFoundException(RuntimeError):
    """Exception stating that model with certain id could not be found."""

    def __init__(self, model_id: str):
        """Report custom error message.

        Parameters
        ----------
        model_id : str
            the id of the model that could not be found
        """
        super().__init__(f"model with id '{model_id}' could not be found")


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

    def classify(
        self,
        model_id: str,
        documents: Sequence[Document],
        limit: int = 10,
        threshold: float = 0.0
    ) -> Sequence[Sequence[ClassificationResult]]:
        """Perform classification for a list of documents."""
        raise NotImplementedError()


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
            pulished_model.subject_order,
            documents,
            limit,
            threshold
        )


class SingleStoredModelRestService(PartialModelInfosRestService):
    """REST implementation that only loads a single model at a time for classification."""

    def __init__(self, directory: str):
        """Init."""
        self.model_types = get_classic_classification_models_map()
        self.model_infos = find_stored_classification_model_infos(directory)
        self.loaded_model: Optional[PublishedClassificationModel] = None
        logger.info("discovered %d models %s", len(self.model_infos), str(list(self.model_infos.keys())))

    def _get_model_infos_dict(self) -> Mapping[str, PublishedClassificationModelInfo]:
        return {model_id: info.info for model_id, info in self.model_infos.items()}

    def classify(
        self,
        model_id: str,
        documents: Sequence[Document],
        limit: int = 10,
        threshold: float = 0.0
    ) -> Sequence[Sequence[ClassificationResult]]:
        """Perform classification for a list of documents."""
        if model_id not in self.model_infos:
            raise ModelNotFoundException(model_id)

        if self.loaded_model is None or model_id != self.loaded_model.info.model_id:
            self.loaded_model = load_published_classification_model(
                self.model_infos[model_id].directory, self.model_types
            )

        return classify_with_limit_and_threshold(
            self.loaded_model.model,
            self.loaded_model.subject_order,
            documents,
            limit,
            threshold
        )


class AllStoredModelRestService(PartialAllModelsRestService, PartialModelInfosRestService):
    """A service implementation that pre-loads all stored models and serves them from memory."""

    def __init__(self, directory: str):
        """Init."""
        self.model_types = get_classic_classification_models_map()
        self.model_infos = find_stored_classification_model_infos(directory)
        self.models = {
            model_id: load_published_classification_model(info.directory, self.model_types)
            for model_id, info in self.model_infos.items()
        }

    def _get_model_infos_dict(self) -> Mapping[str, PublishedClassificationModelInfo]:
        return {model_id: info.info for model_id, info in self.model_infos.items()}

    def _get_models_dict(self) -> Mapping[str, PublishedClassificationModel]:
        return self.models


def classify_with_limit_and_threshold(
    model: ClassificationModel,
    subject_order: Sequence[str],
    documents: Sequence[Document],
    limit: int = 10,
    threshold: float = 0.0
):
    """Perform classification and compile results using certain limit and threshold."""
    # do actual classification
    probabilities = model.predict_proba(documents)
    limit = min(probabilities.shape[1], limit)
    if limit < probabilities.shape[1]:
        # find best results
        topk_indexes = np.argpartition(probabilities, limit, axis=1)[:, -limit:]
    else:
        # return all results
        topk_indexes = np.tile(np.arange(limit), (probabilities.shape[0], 1))
    topk_probabilities = np.take_along_axis(probabilities, topk_indexes, axis=1)
    topk_sort_indexes = np.argsort(topk_probabilities, axis=1)
    sorted_topk_indexed = np.take_along_axis(topk_indexes, topk_sort_indexes, axis=1)
    sorted_topk_probabilities = np.take_along_axis(topk_probabilities, topk_sort_indexes, axis=1)

    # compile and return results
    return [
        [
            ClassificationResult(score=p, subject_uri=subject_order[i])
            for i, p in zip(indexes, probs) if p > threshold
        ] for indexes, probs in zip(sorted_topk_indexed, sorted_topk_probabilities)
    ]
