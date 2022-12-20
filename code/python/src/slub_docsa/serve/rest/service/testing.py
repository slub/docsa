"""REST service implementation for testing purposes."""

from typing import Mapping, Sequence

import numpy as np
import slub_docsa

from slub_docsa.models.classification.dummy import NihilisticModel, OptimisticModel
from slub_docsa.serve.common import ModelNotFoundException, PublishedClassificationModel
from slub_docsa.serve.common import PublishedSubjectInfo, SchemasRestService, PublishedClassificationModelInfo
from slub_docsa.serve.common import PublishedClassificationModelStatistics
from slub_docsa.serve.rest.service.models import PartialModelClassificationRestService, PartialModelInfosRestService


class MockupSchemasRestService(SchemasRestService):
    """Implementation providing mockup schema rest service for simple binary schema only."""

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
        if schema_id != "binary":
            raise ValueError(f"schema '{schema_id}' not supported")
        if subject_uri not in ["yes", "no"]:
            raise ValueError(f"subject_uri '{subject_uri}' not found in schema '{schema_id}'")
        return PublishedSubjectInfo(
            subject_uri=subject_uri,
            labels={"en": subject_uri},
            ancestors=[],
            children=[]
        )


class MockupClassificationModelsRestService(PartialModelInfosRestService, PartialModelClassificationRestService):
    """Implementation providing dummy models for testing purposes."""

    def __init__(self):
        """Init new testing rest service."""
        super().__init__()
        self.schemas_service = MockupSchemasRestService()
        self.models = {
            "nihilistic": PublishedClassificationModel(
                model=NihilisticModel().fit([None], np.zeros((1, 2))),
                subject_order=["yes", "no"],
                info=PublishedClassificationModelInfo(
                    model_id="nihilistic",
                    model_type="nihilistic",
                    model_version="v1",
                    schema_id="binary",
                    creation_date="2022-11-14 12:03:00",
                    supported_languages=["en"],
                    description="nihilistic model for testing",
                    tags=["testing", "nihilistic"],
                    slub_docsa_version=slub_docsa.__version__,
                    statistics=PublishedClassificationModelStatistics(
                        number_of_training_samples=1,
                        number_of_test_samples=1,
                        scores={
                            "some-score": 1.0
                        }
                    )
                ),
            ),
            "optimistic": PublishedClassificationModel(
                model=OptimisticModel().fit([None], np.zeros((1, 2))),
                subject_order=["yes", "no"],
                info=PublishedClassificationModelInfo(
                    model_id="optimistic",
                    model_type="optimistic",
                    model_version="v1",
                    schema_id="binary",
                    creation_date="2022-11-14 12:03:00",
                    supported_languages=["de"],
                    description="optimistic model for testing",
                    tags=["testing", "optimistic"],
                    slub_docsa_version=slub_docsa.__version__,
                    statistics=PublishedClassificationModelStatistics(
                        number_of_training_samples=1,
                        number_of_test_samples=1,
                        scores={
                            "some-score": 1.0
                        }
                    )
                ),
            ),
        }

    def _get_schemas_service(self) -> SchemasRestService:
        return self.schemas_service

    def _get_model(self, model_id) -> PublishedClassificationModel:
        if model_id not in self.models:
            raise ModelNotFoundException(model_id)
        return self.models[model_id]

    def _get_model_infos_dict(self) -> Mapping[str, PublishedClassificationModelInfo]:
        return {model_id: model.info for model_id, model in self.models.items()}
