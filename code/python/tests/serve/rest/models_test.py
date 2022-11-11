"""Test models api of REST service."""


def test_model_list(rest_client):
    """Check model list is returned."""
    response = rest_client.get("/v1/models")
    assert response.data.decode("utf8") == "not implemented yet"
