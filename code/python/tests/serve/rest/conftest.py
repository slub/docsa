"""Fixtures for testing REST api."""

from typing import cast

import pytest

from flask import Flask
from flask.testing import FlaskClient
from slub_docsa.serve.app import create_webapp
from slub_docsa.serve.common import SimpleRestService
from slub_docsa.serve.rest.service.languages import LangidLanguagesRestService
from slub_docsa.serve.rest.service.testing import MockupClassificationModelsRestService


@pytest.fixture(scope="package")
def rest_service():
    """Create a rest service with dummy models for testing purposes."""
    return SimpleRestService(
        MockupClassificationModelsRestService(),
        None,
        LangidLanguagesRestService()
    )


@pytest.fixture(scope="package")
def rest_client(rest_service) -> FlaskClient:  # pylint: disable=redefined-outer-name
    """Start the webapp and return a simple REST client."""
    return cast(Flask, create_webapp(rest_service).app).test_client()
