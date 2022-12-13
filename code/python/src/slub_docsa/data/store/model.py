"""Methods to save and load models including meta information."""

import os
import pickle  # nosec
import json
import logging

from typing import Callable, Mapping, Sequence, NamedTuple

from slub_docsa.common.paths import get_serve_dir
from slub_docsa.common.model import PersistableClassificationModel
from slub_docsa.common.subject import SubjectHierarchy
from slub_docsa.serve.common import PublishedClassificationModelInfo, PublishedClassificationModel
from slub_docsa.serve.common import PublishedClassificationModelStatistics


logger = logging.getLogger(__name__)


def default_published_classification_models_directory():
    """Return default directory where to store a model."""
    return os.path.join(get_serve_dir(), "classification_models")


class StoredClassificationModelInfo(NamedTuple):
    """Remembers which model is stored in which directory."""

    directory: str
    """The directory of the model"""

    info: PublishedClassificationModelInfo
    """The information about the model"""


def save_published_classification_model_info(
    directory: str,
    model_info: PublishedClassificationModelInfo
):
    """Save information about published classification model to directory."""
    with open(os.path.join(directory, "classification_model_info.json"), "wt", encoding="utf8") as file:
        json.dump({
            "model_id": model_info.model_id,
            "model_type": model_info.model_type,
            "model_version": model_info.model_version,
            "schema_id": model_info.schema_id,
            "creation_date": model_info.creation_date,
            "supported_languages": model_info.supported_languages,
            "description": model_info.description,
            "tags": model_info.tags,
            "slub_docsa_version": model_info.slub_docsa_version,
            "statistics": {
                "number_of_training_samples": model_info.statistics.number_of_training_samples,
                "number_of_test_samples": model_info.statistics.number_of_test_samples,
                "scores": model_info.statistics.scores,
            }
        }, file, indent=4)


def load_published_classification_model_info(
    directory: str,
) -> PublishedClassificationModelInfo:
    """Load information about model from a directory."""
    with open(os.path.join(directory, "classification_model_info.json"), "rt", encoding="utf8") as file:
        data = json.load(file)
        return PublishedClassificationModelInfo(
            model_id=data["model_id"],
            model_type=data["model_type"],
            model_version=data.get("model_version"),
            schema_id=data["schema_id"],
            creation_date=data.get("creation_date"),
            supported_languages=data.get("supported_languages", []),
            description=data.get("description"),
            tags=data.get("tags", []),
            slub_docsa_version=data.get("slub_docsa_version"),
            statistics=PublishedClassificationModelStatistics(
                number_of_training_samples=data.get("statistics", {}).get("number_of_training_samples", -1),
                number_of_test_samples=data.get("statistics", {}).get("number_of_test_samples", -1),
                scores=data.get("statistics", {}).get("scores")
            )
        )


def save_as_published_classification_model(
    directory: str,
    model: PersistableClassificationModel,
    subject_order: Sequence[str],
    model_info: PublishedClassificationModelInfo,
):
    """Store a model such that it can be loaded by the REST service."""
    # save internal model state
    model_state_directory = os.path.join(directory, "model_state")
    os.makedirs(model_state_directory, exist_ok=True)
    model.save(model_state_directory)

    # save subject order
    with open(os.path.join(directory, "subject_order.pickle"), "wb") as file:
        pickle.dump(subject_order, file)

    save_published_classification_model_info(
        directory,
        model_info,
    )


def load_published_classification_model(
    directory: str,
    model_types: Mapping[str, Callable[[SubjectHierarchy, Sequence[str]], PersistableClassificationModel]],
    schema_generators: Mapping[str, Callable[[], SubjectHierarchy]],
):
    """Load a model and its information such that it can be used by the REST service."""
    model_info = load_published_classification_model_info(directory)

    logger.info("load subject order from disk at '%s'", directory)
    with open(os.path.join(directory, "subject_order.pickle"), "rb") as file:
        subject_order = pickle.load(file)  # nosec

    if model_info.schema_id not in schema_generators:
        raise ValueError(f"Cannot load model for unknown schema '{model_info.schema_id}'")
    logger.info("load subject hierarchy '%s' for model", model_info.schema_id)
    subject_hierarchy = schema_generators[model_info.schema_id]()

    # instanciate model and load state
    model = model_types[model_info.model_type](subject_hierarchy, subject_order)
    model.load(os.path.join(directory, "model_state"))

    return PublishedClassificationModel(
        info=model_info,
        model=model,
        subject_order=subject_order
    )


def find_stored_classification_model_infos(
    directory: str,
) -> Mapping[str, StoredClassificationModelInfo]:
    """Index all subdirectories by their model ids read from the stored model information."""
    models_map = {}
    for name in os.listdir(directory):
        subdirectory = os.path.join(directory, name)
        if os.path.isdir(subdirectory):
            try:
                model_info = load_published_classification_model_info(subdirectory)
                models_map[model_info.model_id] = StoredClassificationModelInfo(directory=subdirectory, info=model_info)
            except Exception as error:  # pylint: disable=broad-except
                logger.warning("error '%s' loading model from directory '%s'", str(error), subdirectory)
    return models_map
