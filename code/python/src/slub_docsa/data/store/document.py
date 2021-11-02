"""Helper methods for caching data related to documents."""

import hashlib


def sha1_hash_from_text(text):
    """Return sha1 hex digest as string for text."""
    return hashlib.sha1(text.encode()).hexdigest()  # nosec
