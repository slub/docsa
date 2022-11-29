"""Common methods used when loading data."""

import gzip
import tempfile
import urllib.request
import shutil
import os
import logging

from typing import Iterator

logger = logging.getLogger(__name__)

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


def download_file(url: str, filepath: str, gzip_compress: bool = False):
    """Download a file from an URL and save it to a filepath.

    In order to avoid partial failed downloads, the file is saved in a temporary file during download, and copied to
    the target file after the download finished succesfully.

    Parameters
    ----------
    url : str
        the URL that is used to download the file
    filepath : str
        the filepath where the file is saved
    gzip_compress: bool = False
        whether to compress the downloaded file

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
            "User-Agent": USER_AGENT,
            "Accept-encoding": "gzip",
        }
    )
    directory = os.path.dirname(filepath)
    suffix = "." + os.path.basename(filepath)
    with tempfile.NamedTemporaryFile(delete=True, dir=directory, suffix=suffix, mode="w+b") as temp_file:
        with urllib.request.urlopen(request) as response:  # nosec
            already_compressed = response.info().get("Content-Encoding") == "gzip"
            if (not gzip_compress and not already_compressed) or (gzip_compress and already_compressed):
                # just copy content (either uncompressed, or already compressed)
                shutil.copyfileobj(response, temp_file)
            if not gzip_compress and already_compressed:
                # uncompress content during download
                with gzip.GzipFile(fileobj=response) as uncompressed_data:
                    shutil.copyfileobj(uncompressed_data, temp_file)
            if gzip_compress and not already_compressed:
                # compress content during download
                with gzip.GzipFile(mode="wb", fileobj=temp_file) as compressed_file:
                    shutil.copyfileobj(response, compressed_file)
        temp_file.seek(0)
        # copy file to actual target location after download is complete
        with open(filepath, "wb") as file:
            shutil.copyfileobj(temp_file, file)
