"""Base classification."""


def bk_notation_to_uri(notation):
    """Return uri for base classification notation.

    Parameters
    ----------
    notation: string
        the notation of a base classification subject, e.g. "01.24".

    Returns
    -------
    string
        the URI of the base classification subject, e.g. "https://gbv.de/basisklassifikation/01.24"
    """
    return f"https://gbv.de/basisklassifikation/{notation}"
