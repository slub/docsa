"""Test models api of REST service."""

import json

from typing import Sequence

from flask.testing import FlaskClient

from slub_docsa.serve.common import ClassificationPrediction, ClassificationResult
from slub_docsa.serve.rest.client.parse import parse_classification_results_from_json

from .common import check_json_request_body_validation_error


def _assert_classification_results_match(
    results1: Sequence[ClassificationResult],
    results2: Sequence[ClassificationResult]
):
    """Compare two classification results for equality.

    Ignores potential ordering differences when the score is the same.
    """
    assert len(results1) == len(results2)
    for result1, result2 in zip(results1, results2):
        # check that document uris match
        assert result1.document_uri == result2.document_uri

        predictions1, predictions2 = result1.predictions, result2.predictions

        # check there are the same number of predictions
        assert len(predictions1) == len(predictions2)

        # check both results are about the same subjects
        assert {r.subject_uri for r in predictions1} == {r.subject_uri for r in predictions2}

        # check both results are ordered by score
        assert all(predictions1[i].score >= predictions1[i + 1].score for i in range(len(predictions1) - 1))
        assert all(predictions2[i].score >= predictions2[i + 1].score for i in range(len(predictions2) - 1))

        # check scores are the same
        for p1_ in predictions1:
            for p2_ in predictions2:
                if p1_.subject_uri == p2_.subject_uri:
                    assert p1_.score == p2_.score


def test_model_list_complete(rest_client: FlaskClient):
    """Check complete model list is returned."""
    response = rest_client.get("/v1/models")
    assert set(json.loads(response.data)) == set(["nihilistic", "optimistic"])


def test_model_list_empty_for_unknown_language(rest_client: FlaskClient):
    """Check no model is returned for unknown language."""
    response = rest_client.get("/v1/models?supported_languages=blub")
    assert json.loads(response.data) == []


def test_model_list_for_known_language(rest_client: FlaskClient):
    """Check models are filtered by language."""
    response = rest_client.get("/v1/models?supported_languages=de")
    assert json.loads(response.data) == ["optimistic"]


def test_model_list_empty_for_unknown_schema(rest_client: FlaskClient):
    """Check no model is returned for unknown schema."""
    response = rest_client.get("/v1/models?schema_id=blub")
    assert json.loads(response.data) == []


def test_model_list_for_known_schema(rest_client: FlaskClient):
    """Check models are filtered by known schema."""
    response = rest_client.get("/v1/models?schema_id=binary")
    assert set(json.loads(response.data)) == set(["nihilistic", "optimistic"])


def test_model_list_for_known_tags(rest_client: FlaskClient):
    """Check models are filtered by known tags."""
    response = rest_client.get("/v1/models?tags=nihilistic,testing")
    assert json.loads(response.data) == ["nihilistic"]


def test_model_list_for_unknown_tag(rest_client: FlaskClient):
    """Check no model is returned for unknown tag."""
    response = rest_client.get("/v1/models?tags=blub")
    assert json.loads(response.data) == []


def test_model_info_for_nihilistic_model(rest_client: FlaskClient):
    """Check model info is correct for nihilistic model."""
    response = rest_client.get("/v1/models/nihilistic")
    data = json.loads(response.data)
    assert data["model_id"] == "nihilistic"
    assert data["model_type"] == "nihilistic"
    assert data["supported_languages"] == ["en"]


def test_model_info_for_unknown_model(rest_client: FlaskClient):
    """Check 404 is returned for unknown model."""
    response = rest_client.get("/v1/models/unknown-model-id")
    assert response.status_code == 404
    assert json.loads(response.data)["type"] == "ModelNotFoundException"


def test_classify_nihilistic_model(rest_client: FlaskClient):
    """Check that nihistic model does not return any classification results."""
    body = [
        {"title": "Title of the first document"},
        {"title": "Title of the second document"},
        {"title": "Title of the third document"},
    ]
    response = rest_client.post(
        "/v1/models/nihilistic/classify",
        data=json.dumps(body),
        content_type="application/json"
    )
    assert json.loads(response.data) == [[], [], []]


def test_classify_and_describe_nihilistic_model(rest_client: FlaskClient):
    """Check that nihistic model does not return any classification results."""
    body = [
        {"title": "Title of the first document"},
        {"title": "Title of the second document"},
        {"title": "Title of the third document"},
    ]
    response = rest_client.post(
        "/v1/models/nihilistic/classify_and_describe",
        data=json.dumps(body),
        content_type="application/json"
    )
    assert json.loads(response.data) == [
        {"document_uri": "0", "predictions": []},
        {"document_uri": "1", "predictions": []},
        {"document_uri": "2", "predictions": []}
    ]


def test_classify_and_describe_optimistic_model(rest_client: FlaskClient):
    """Check that nihistic model does not return any classification results."""
    body = [
        {"uri": "document1", "title": "Title of the first document"},
        {"uri": "document2", "title": "Title of the second document", "abstract": "this is some abtract"},
    ]
    response = rest_client.post(
        "/v1/models/optimistic/classify_and_describe",
        data=json.dumps(body),
        content_type="application/json"
    )
    response_results = parse_classification_results_from_json(response.data)
    expected_results = [
        ClassificationResult(
            document_uri="document1",
            predictions=[
                ClassificationPrediction(score=1.0, subject_uri="yes"),
                ClassificationPrediction(score=1.0, subject_uri="no")
            ]
        ),
        ClassificationResult(
            document_uri="document2",
            predictions=[
                ClassificationPrediction(score=1.0, subject_uri="yes"),
                ClassificationPrediction(score=1.0, subject_uri="no")
            ]
        )
    ]
    _assert_classification_results_match(response_results, expected_results)


def test_classify_invalid_json_format(rest_client: FlaskClient):
    """Check classify with incorrect json payload returns 400."""
    check_json_request_body_validation_error(rest_client, "/v1/models/optimistic/classify")


def test_classify_unknown_model(rest_client: FlaskClient):
    """Check classify for unknown model returns 404."""
    body = [{"title": "Title of a document, not that it matters"}]
    response = rest_client.post(
        "/v1/models/unknown-model-id/classify",
        data=json.dumps(body),
        content_type="application/json"
    )
    assert response.status_code == 404
    assert json.loads(response.data)["type"] == "ModelNotFoundException"


def test_classify_too_large_request_body(rest_client: FlaskClient):
    """Check classify for too large request returns 413."""
    response = rest_client.post(
        "/v1/models/unknown-model-id/classify",
        data=bytes(32 * 1024 * 1024),
        content_type="application/json"
    )
    assert response.status_code == 413


def test_subjects_list_complete(rest_client: FlaskClient):
    """Check that list of subject supported by a model is complete."""
    response = rest_client.get("v1/models/nihilistic/subjects")
    assert set(json.loads(response.data)) == set(["yes", "no"])
