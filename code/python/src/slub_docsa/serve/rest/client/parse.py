"""Parse REST service respones."""

import json

from slub_docsa.serve.common import ClassificationPrediction, ClassificationResult


def parse_classification_results_from_json(data):
    """Parse the list of classification results from json."""
    return [
        ClassificationResult(
            document_uri=result["document_uri"],
            predictions=[
                ClassificationPrediction(score=prediction["score"], subject_uri=prediction["subject_uri"])
                for prediction in result["predictions"]
            ]
        )
        for result in json.loads(data)
    ]
