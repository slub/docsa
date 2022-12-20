"""Common methods when testing the REST service."""

import json


def check_json_request_body_validation_error(rest_client, url):
    """Check that invalid json request body returns 400."""
    body = {"something": "something"}
    response = rest_client.post(url, data=json.dumps(body), content_type="application/json")
    data = json.loads(response.data)
    assert response.status_code == 400  # nosec B101
    assert data["title"] == "Bad Request"  # nosec B101
    assert data["type"] == "about:blank"  # nosec B101
    assert "detail" in data and len(data["detail"]) > 0  # nosec B101
