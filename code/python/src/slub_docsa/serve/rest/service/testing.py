"""REST service implementation for testing purposes."""

from typing import Mapping

import numpy as np

from slub_docsa.models.classification.dummy import NihilisticModel, OptimisticModel
from slub_docsa.serve.common import PublishedClassificationModel, PublishedClassificationModelInfo
from slub_docsa.serve.rest.service.models import PartialModelInfosRestService, PartialAllModelsRestService


class MockupClassificationModelsRestService(PartialAllModelsRestService, PartialModelInfosRestService):
    """Implementation providing dummy models for testing purposes."""

    def __init__(self):
        """Init new testing rest service."""
        super().__init__()

        self.model_infos = {
            "nihilistic": PublishedClassificationModelInfo(
                model_id="nihilistic",
                model_type="nihilistic",
                schema_id="binary",
                creation_date="2022-11-14 12:03:00",
                supported_languages=["en"],
                description="nihilistic model for testing",
                tags=["testing", "nihilistic"]
            ),
            "optimistic": PublishedClassificationModelInfo(
                model_id="optimistic",
                model_type="optimistic",
                schema_id="binary",
                creation_date="2022-11-14 12:03:00",
                supported_languages=["de"],
                description="optimistic model for testing",
                tags=["testing", "optimistic"]
            )
        }

        self.models = {
            "nihilistic": PublishedClassificationModel(
                model=NihilisticModel().fit([None], np.zeros((1, 2))),
                subject_order=["yes", "no"],
                info=self.model_infos["nihilistic"],
            ),
            "optimistic": PublishedClassificationModel(
                model=OptimisticModel().fit([None], np.zeros((1, 2))),
                subject_order=["yes", "no"],
                info=self.model_infos["optimistic"],
            ),
        }

    def _get_models_dict(self) -> Mapping[str, PublishedClassificationModel]:
        return self.models

    def _get_model_infos_dict(self) -> Mapping[str, PublishedClassificationModelInfo]:
        return self.model_infos
