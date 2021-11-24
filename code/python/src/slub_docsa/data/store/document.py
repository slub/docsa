"""Helper methods for caching data related to documents."""

import hashlib


def sha1_hash_from_text(text: str) -> str:
    """Return sha1 hex digest as string for text.

    Parameters
    ----------
    text: str
        The text to be hashed

    Returns
    -------
    str
        the hash of the text
    """
    return hashlib.sha1(text.encode()).hexdigest()  # nosec
