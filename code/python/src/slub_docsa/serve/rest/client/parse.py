"""Parse REST service respones."""

import json

from slub_docsa.serve.common import ClassificationPrediction, ClassificationResult, PublishedSubjectInfo


def parse_subject_info_from_json(data):
    """Parse information about a subject from json."""
    if data is None:
        return None
    return PublishedSubjectInfo(
        labels=data["labels"],
        breadcrumb=data["breadcrumb"],
        parent_subject_uri=data["parent_subject_uri"],
        children_subject_uris=data["children_subject_uris"]
    )


def parse_classification_results_from_json(data):
    """Parse the list of classification results from json."""
    return [
        ClassificationResult(
            document_uri=result["document_uri"],
            predictions=[
                ClassificationPrediction(
                    score=prediction["score"],
                    subject_uri=prediction["subject_uri"],
                    subject_info=parse_subject_info_from_json(prediction.get("subject_info")),
                )
                for prediction in result["predictions"]
            ]
        )
        for result in json.loads(data)
    ]
