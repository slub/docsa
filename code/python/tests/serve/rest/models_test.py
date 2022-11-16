"""Test models api of REST service."""

import json

from typing import Sequence

from flask.testing import FlaskClient

from slub_docsa.serve.common import ClassificationResult

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
        # check there are the same number of results
        assert len(result1) == len(result2)

        # check both results are about the same subjects
        assert {r.subject_uri for r in result1} == {r.subject_uri for r in result2}

        # check both results are ordered by score
        assert all(result1[i].score >= result1[i + 1].score for i in range(len(result1) - 1))
        assert all(result2[i].score >= result2[i + 1].score for i in range(len(result2) - 1))

        # check scores are the same
        for r1_ in result1:
            for r2_ in result2:
                if r1_.subject_uri == r2_.subject_uri:
                    assert r1_.score == r2_.score


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


def test_classify_optimistic_model(rest_client: FlaskClient):
    """Check that nihistic model does not return any classification results."""
    body = [
        {"title": "Title of the first document"},
        {"title": "Title of the second document", "abstract": "this is some abtract"},
    ]
    response = rest_client.post(
        "/v1/models/optimistic/classify",
        data=json.dumps(body),
        content_type="application/json"
    )
    response_results = [
        [ClassificationResult(result["score"], result["subject_uri"]) for result in results]
        for results in json.loads(response.data)
    ]
    expected_results = [
        [
            ClassificationResult(score=1.0, subject_uri="yes"),
            ClassificationResult(score=1.0, subject_uri="no")
        ],
        [
            ClassificationResult(score=1.0, subject_uri="yes"),
            ClassificationResult(score=1.0, subject_uri="no")
        ]
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
