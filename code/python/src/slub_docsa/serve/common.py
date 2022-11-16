"""The webapp context."""

from datetime import datetime

from typing import Sequence, Optional, NamedTuple

from slub_docsa.common.document import Document
from slub_docsa.common.model import PersistableClassificationModel


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


class ClassificationResult(NamedTuple):
    """A classification result consisting of a score and the predicted subject."""

    score: float
    """A certainty score for the prediction."""

    subject_uri: str
    """The URI of the predicted subject."""


class ClassificationModelsRestService:
    """The interface of the rest service dealing with classification models."""

    def find_models(
        self,
        languages: Optional[Sequence[str]] = None,
        schema_id: Optional[str] = None,
        tags: Optional[Sequence[str]] = None,
    ) -> Sequence[str]:
        """List available models matching certain criteria like supported languages."""
        raise NotImplementedError()

    def model_info(self, model_id: str) -> PublishedClassificationModelInfo:
        """Return information describing a classification model."""
        raise NotImplementedError()

    def classify(
        self,
        model_id: str,
        documents: Sequence[Document],
        limit: Optional[int] = None,
        threshold: Optional[float] = None
    ) -> Sequence[Sequence[ClassificationResult]]:
        """Perform classification for a list of documents."""
        raise NotImplementedError()

    def subjects(self, model_id: str) -> Sequence[str]:
        """Return subjects supported by a model."""
        raise NotImplementedError()


class SchemasRestService:
    """The interface of the REST service dealing with schema queries."""

    def find_schemas(self):
        """Return the list of avaialble schemas."""
        raise NotImplementedError()

    def schema_info(self, schema_id: str):
        """Return information about a specific schema."""
        raise NotImplementedError()

    def find_subjects(self, schema_id: str):
        """Return a list of available subjects for a specific schema."""
        raise NotImplementedError()

    def subject_info(self, schema_id: str, subject_uri: str):
        """Return information about a subject of a schema."""
        raise NotImplementedError()

    def subject_children(self, schema_id: str, subject_uri: str):
        """Return the list of children subjects for a subject of a schema."""
        raise NotImplementedError()


class LanguagesRestService:
    """The interface of the REST service dealing with language queries."""

    def find_languages(self):
        """Return the list of available languages."""
        raise NotImplementedError()

    def detect(self, documents: Sequence[Document]) -> Sequence[str]:
        """Detect the language of each document."""
        raise NotImplementedError()


class RestService:
    """Combines all rest service interfaces into one."""

    def get_classification_models_service(self) -> ClassificationModelsRestService:
        """Return classification models service implementation."""
        raise NotImplementedError()

    def get_schemas_service(self) -> SchemasRestService:
        """Return schemas service implementation."""
        raise NotImplementedError()

    def get_languages_service(self) -> LanguagesRestService:
        """Return language service implementation."""
        raise NotImplementedError()


class SimpleRestService(RestService):
    """Simple combination of all rest service implementations."""

    def __init__(self, classification_models_service, schemas_service, languages_services):
        """Initialize rest service by rememberin each individual sub-service."""
        self.classification_models_service = classification_models_service
        self.schemas_service = schemas_service
        self.languages_service = languages_services

    def get_classification_models_service(self) -> ClassificationModelsRestService:
        """Return classification models service implementation."""
        return self.classification_models_service

    def get_schemas_service(self) -> SchemasRestService:
        """Return schemas service implementation."""
        return self.schemas_service

    def get_languages_service(self) -> LanguagesRestService:
        """Return language service implementation."""
        return self.languages_service


def current_date_as_model_creation_date():
    """Return current date and time in format required by creation date property."""
    return datetime.utcnow().strftime("%Y-%m-%d %H:%M:%S")
