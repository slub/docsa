"""Test language detection api of REST service."""

import json

from flask.testing import FlaskClient


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
