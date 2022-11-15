"""Handlers for various exceptions related to the REST service."""


def generate_connexion_exception_handler(title, status):
    """Genereate a connexion exception handler that reports exception details with specified status code."""

    def handle(exception):
        """Return exception details as json."""
        return {
            "type": type(exception).__name__,
            "title": title,
            "detail": str(exception),
            "status": status
        }, status

    return handle
