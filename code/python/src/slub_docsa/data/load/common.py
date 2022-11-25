"""Common methods used when loading data."""

import gzip
import tempfile
import urllib.request
import shutil

from typing import Iterator

USER_AGENT = \
    "Mozilla/5.0 (Windows NT 10.0; WOW64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/107.0.0.0 Safari/537.36"


def read_gzip_text_lines(
    filepath: str,
) -> Iterator[str]:
    """Read gzipped text file line by line.

    Parameters
    ----------
    filepath : str
        the filepath to the gzip compressed text file

    Yields
    ------
    str
        a line of the decompressed text file
    """
    with gzip.open(filepath, "rt", encoding="utf8") as file:
        for line in file:
            yield line


def download_file(url: str, filepath: str):
    """Download a file from an URL and save it to a filepath.

    In order to avoid partial failed downloads, the file is saved in a temporary file during download, and copied to
    the target file after the download finished succesfully.

    Parameters
    ----------
    url : str
        the URL that is used to download the file
    filepath : str
        the filepath where the file is saved

    Returns
    -------
    None
        when the file downloaded successfully
    """
    # define request with custom user agent
    request = urllib.request.Request(
        url=url,
        data=None,
        headers={
            "User-Agent": USER_AGENT
        }
    )

    with tempfile.NamedTemporaryFile(delete=True) as temp_file:
        with urllib.request.urlopen(request) as response:  # nosec
            shutil.copyfileobj(response, temp_file)
        temp_file.seek(0)
        with open(filepath, "wb") as file:
            shutil.copyfileobj(temp_file, file)
