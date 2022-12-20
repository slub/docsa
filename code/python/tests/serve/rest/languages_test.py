"""Test language detection api of REST service."""

import json

from flask.testing import FlaskClient

from .common import check_json_request_body_validation_error


def test_language_list_complete(rest_client: FlaskClient):
    """Check that supported languages are returned."""
    response = rest_client.get("/v1/languages")
    assert set(json.loads(response.data)) == set(["de", "en"])


def test_language_detect_multiple_documents(rest_client: FlaskClient):
    """Check that german text is detected as german."""
    body = [
        {"title": "Wie was warum, das ist die Frage"},
        {"title": "This is an english sentence"},
        {"title": "Linguistique comparée et typologie des langues romanes"}
    ]
    response = rest_client.post(
        "/v1/languages/detect",
        data=json.dumps(body),
        content_type="application/json"
    )
    assert json.loads(response.data) == ["de", "en", "fr"]


def test_language_detect_invalid_json_format(rest_client: FlaskClient):
    """Check invalid json request for language detection returns 400."""
    check_json_request_body_validation_error(rest_client, "/v1/languages/detect")
