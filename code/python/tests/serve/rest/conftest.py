"""Fixtures for testing REST api."""

from typing import cast

import pytest

from flask import Flask
from flask.testing import FlaskClient
from slub_docsa.serve.app import create_webapp


@pytest.fixture(scope="package", autouse=True)
def rest_client() -> FlaskClient:
    """Start the webapp and return a simple REST client."""
    return cast(Flask, create_webapp().app).test_client()
