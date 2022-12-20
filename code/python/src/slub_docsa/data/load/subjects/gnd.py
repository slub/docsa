"""GND.

Download from: https://lobid.org/gnd/search?format=jsonl
"""


def gnd_notation_to_uri(notation):
    """Return uri for gnd notation.

    Parameters
    ----------
    notation: string
        the notation of a GND subject, e.g. "1183720181".

    Returns
    -------
    string
        the URI of the GND subject, e.g. "https://d-nb.info/gnd/1183720181"
    """
    return f"https://d-nb.info/gnd/{notation}"
