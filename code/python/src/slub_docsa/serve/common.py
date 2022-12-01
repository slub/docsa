"""The webapp context."""

# pylint: disable=too-many-arguments

from datetime import datetime

from typing import Sequence, Optional, NamedTuple, Tuple, Mapping

from slub_docsa.common.document import Document
from slub_docsa.common.model import PersistableClassificationModel


class PublishedClassificationModelStatistics(NamedTuple):
    """Statistics about a published classification model."""

    number_of_training_samples: int
    """The number of samples this model was trained on."""

    number_of_test_samples: int
    """The number of samples this model was evaluated on."""

    scores: Mapping[str, float]
    """Evaluation scores for this model."""


class PublishedClassificationModelInfo(NamedTuple):
    """Information about a published classification model."""

    model_id: str
    """A unique identifier for a specfic instance of a model."""

    model_type: str
    """The model type identifier that can be used to instanciate a new model."""

    model_version: Optional[str]
    """The version of the model."""

    schema_id: str
    """The identifier of the classification schema the model was trained for."""

    creation_date: Optional[str]
    """The date the model was created (in format 'YYYY-MM-DD HH:MM:SS' in UTC time)."""

    supported_languages: Sequence[str]
    """The list of ISO 639-1 language codes of languages supported by this model."""

    description: Optional[str]
    """A description of the model."""

    tags: Sequence[str]
    """A list of arbitrary tags associated with this model."""

    slub_docsa_version: Optional[str]
    """The version of the slub docsa python package that was used to create this model."""

    statistics: PublishedClassificationModelStatistics
    """Statistics about this model."""


class PublishedClassificationModel(NamedTuple):
    """Object that keeps track of a model, its descriptive information and the subject order."""

    info: PublishedClassificationModelInfo
    """the information about the model"""

    model: PersistableClassificationModel
    """the actual model itself"""

    subject_order: Sequence[str]
    """the subject order"""


class PublishedSubjectInfo(NamedTuple):
    """Information about a subject."""

    labels: Mapping[str, str]
    """A map of labels for this subject indexed by the ISO 639-1 language code of the label language."""

    ancestors: Sequence[str]
    """The list URIs of the ancestors of this subject"""

    breadcrumb: Sequence[Mapping[str, str]]
    """A list of mappings from the ISO 639-1 language code to humand readable labels for each ancestor subject."""

    children: Sequence[str]
    """The list of URIs of children subjects."""


class ClassificationPrediction(NamedTuple):
    """A classification prediction consisting of a score and the predicted subject."""

    score: float
    """A certainty score for the prediction."""

    subject_uri: str
    """The URI of the predicted subject."""

    subject_info: Optional[PublishedSubjectInfo]
    """The information about a subject."""


class ClassificationResult(NamedTuple):
    """A classification result for a specific document."""

    document_uri: str
    """The uri of the document that was classified."""

    predictions: Sequence[ClassificationPrediction]
    """The list of individual classification predictions including a score for each predicted subject."""


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
    ) -> Sequence[Sequence[Tuple[float, int]]]:
        """Perform classification for a list of documents and return tuples of score and subject order id."""
        raise NotImplementedError()

    def classify_and_describe(
        self,
        model_id: str,
        documents: Sequence[Document],
        limit: Optional[int] = None,
        threshold: Optional[float] = None,
        subject_info: bool = True
    ) -> Sequence[ClassificationResult]:
        """Perform classification for a list of documents and provide detailed classification results."""
        raise NotImplementedError()

    def subjects(self, model_id: str) -> Sequence[str]:
        """Return subjects supported by a model."""
        raise NotImplementedError()


class SchemasRestService:
    """The interface of the REST service dealing with schema queries."""

    def find_schemas(self) -> Sequence[str]:
        """Return the list of avaialble schemas."""
        raise NotImplementedError()

    def schema_info(self, schema_id: str):
        """Return information about a specific schema."""
        raise NotImplementedError()

    def find_subjects(self, schema_id: str, root_only: bool = True) -> Sequence[str]:
        """Return a list of available subjects for a specific schema."""
        raise NotImplementedError()

    def subject_info(self, schema_id: str, subject_uri: str) -> PublishedSubjectInfo:
        """Return information about a subject of a schema."""
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


class SchemaNotFoundException(RuntimeError):
    """Exception stating that schema with certain id could not be found."""

    def __init__(self, schema_id: str):
        """Report custom error message.

        Parameters
        ----------
        schema_id : str
            the id of the schema that could not be found
        """
        super().__init__(f"schema with id '{schema_id}' could not be found")


class SubjectNotFoundException(RuntimeError):
    """Exception stating that subject with certain URI could not be found."""

    def __init__(self, schema_id: str, subject_uri: str):
        """Report custom error message.

        Parameters
        ----------
        schema_id : str
            the id of the schema that is checked for the subject
        subject_uri : str
            the uri of the subject that could not be found
        """
        super().__init__(f"subject with uri '{subject_uri}' could not be found in schema with id '{schema_id}'")


def current_date_as_model_creation_date():
    """Return current date and time in format required by creation date property."""
    return datetime.utcnow().strftime("%Y-%m-%d %H:%M:%S")
