"""Additional web app routes."""

from flask import Blueprint
from flask.wrappers import Response

app = Blueprint("app", __name__)


@app.route("/")
def index():
    """Index page response."""
    return Response("index page of webapp", status=200, mimetype="text/plain")
