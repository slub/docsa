"""Methods to save and load models such that they can be served by the REST service."""

import os
import pickle  # nosec
import json
import logging

from typing import Callable, Mapping, NamedTuple, Sequence

from slub_docsa.common.model import PersistableClassificationModel


logger = logging.getLogger(__name__)


class PublishedClassificationModelInfo(NamedTuple):
    """Information about a published classification model."""

    model_id: str
    """A unique identifier for a specfic instance of a model."""

    model_type: str
    """The model type identifier that can be used to instanciate a new model."""

    schema_id: str
    """The identifier of the classification schema the model was trained for."""

    creation_date: str
    """The date the model was created (in format 'YYYY-MM-DD HH:MM:SS' in UTC time)."""

    supported_languages: Sequence[str]
    """The list of ISO 639-1 language codes of languages supported by this model."""

    description: str
    """A description of the model."""

    tags: Sequence[str]
    """A list of arbitrary tags associated with this model."""


class PublishedClassificationModel(NamedTuple):
    """Object that keeps track of a model, its descriptive information and the subject order."""

    info: PublishedClassificationModelInfo
    """the information about the model"""

    model: PersistableClassificationModel
    """the actual model itself"""

    subject_order: Sequence[str]
    """the subject order"""


def save_published_classification_model_info(
    directory: str,
    model_info: PublishedClassificationModelInfo
):
    """Save information about published classification model to directory."""
    with open(os.path.join(directory, "classification_model_info.json"), "wt", encoding="utf8") as file:
        json.dump({
            "model_id": model_info.model_id,
            "model_type": model_info.model_type,
            "schema_id": model_info.schema_id,
            "creation_date": model_info.creation_date,  # datetime.utcnow().strftime("%Y-%m-%d %H:%M:%S"),
            "supported_languages": model_info.supported_languages,
            "description": model_info.description,
            "tags": model_info.tags,
        }, file)


def load_published_classification_model_info(
    directory: str,
):
    """Load information about model from a directory."""
    with open(os.path.join(directory, "classification_model_info.json"), "rt", encoding="utf8") as file:
        data = json.load(file)
        return PublishedClassificationModelInfo(
            model_id=data["model_id"],
            model_type=data["model_type"],
            schema_id=data["schema_id"],
            creation_date=data["creation_date"],
            supported_languages=data["supported_languages"],
            description=data["description"],
            tags=data["tags"],
        )


def save_as_published_classification_model(
    directory: str,
    model: PersistableClassificationModel,
    subject_order: Sequence[str],
    model_info: PublishedClassificationModelInfo,
):
    """Store a model such that it can be loaded by the REST service."""
    # save internal model state
    model.save(os.path.join(directory, "model_state"))

    # save subject order
    with open(os.path.join(directory, "subject_order.pickle"), "wb") as file:
        pickle.dump(subject_order, file)

    save_published_classification_model_info(
        directory,
        model_info,
    )


def load_published_classification_model(
    directory: str,
    model_types: Mapping[str, Callable[[], PersistableClassificationModel]]
):
    """Load a model and its information such that it can be used by the REST service."""
    model_info = load_published_classification_model_info(directory)

    # instanciate model and load state
    model = model_types[model_info.model_type]()
    model.load(os.path.join(directory, "model_state"))

    logger.info("load subject order from disk at '%s'", directory)
    with open(os.path.join(directory, "subject_order.pickle"), "rb") as file:
        subject_order = pickle.load(file)  # nosec

    return PublishedClassificationModel(
        info=model_info,
        model=model,
        subject_order=subject_order
    )


def find_classification_model_directories(
    directory: str,
):
    """Index all subdirectories by their model ids read from the stored model information."""
    models_map = {}
    for name in os.listdir(directory):
        subdirectory = os.path.join(directory, name)
        if os.path.isdir(subdirectory):
            model_info = load_published_classification_model_info(subdirectory)
            models_map[model_info.model_id] = subdirectory
    return models_map
