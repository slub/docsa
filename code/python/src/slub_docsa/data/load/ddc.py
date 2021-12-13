"""Methods to process Dewey Decimal Classification."""

# pylint: disable=too-many-return-statements

import logging
import os
import time

from typing import Iterable, Mapping, Optional, Sequence

import requests
from sqlitedict import SqliteDict

from slub_docsa.common.paths import get_cache_dir
from slub_docsa.common.subject import SubjectHierarchy, SubjectNode

logger = logging.getLogger(__name__)

COLIANA_DDC_API_URL = "https://coli-conc.gbv.de/coli-ana/app/analyze?notation="

DdcSubjectNode = SubjectNode


def ddc_key_to_uri(key: str):
    """Convert a ddc key to a uri.

    Parameters
    ----------
    key: str
        the ddc key, eg. `123.123`, `123` or `001`

    Returns
    -------
    str
        a URI build from the ddc key as string, e.g. `ddc:123.123`
    """
    return f"ddc:{key}"


def ddc_correct_short_keys(key: str):
    """Correct short ddc keys if they were not prefixed with zeros.

    Parameters
    ----------
    key: str
        a ddc key

    Returns
    -------
    str
        the corrected ddc key if correction was possible, otherwise the original input
    """
    if len(key) == 1:
        return "00" + key
    if len(key) == 2:
        return "0" + key
    return key


def is_valid_ddc_key(key: str) -> bool:
    """Check whether a ddc key follows a valid format.

    Parameters
    ----------
    key: str
        the ddc key to check for validity

    Returns
    -------
    bool
        true, if the key is correct
    """
    if "." in key:
        if len(key.split(".")) != 2:
            logger.debug("ddc key '%s' contains multiple dots", key)
            return False
        major, minor = key.split(".")
        if len(major) != 3:
            logger.debug("ddc key '%s' major part doesn't have exactly 3 digits", key)
            return False
        if not major.isnumeric():
            logger.debug("ddc key '%s' major part isn't numeric", key)
            return False
        if not minor.isnumeric():
            logger.debug("ddc key '%s' minor part isn't numeric", key)
            return False
        return True
    if len(key) != 3:
        logger.debug("ddc key '%s' doesn't have exactly 3 digits", key)
        return False
    if not key.isnumeric():
        logger.debug("ddc key '%s' isn't numeric", key)
        return False
    return True


def is_valid_ddc_uri(uri: str) -> bool:
    """Check whether a string is a valid ddc uri.

    Parameters
    ----------
    uri: str
        the string to check

    Returns
    -------
    bool
        true if the string is a correct ddc uri
    """
    if not uri.startswith("ddc:"):
        return False
    return is_valid_ddc_key(uri[4:])


def ddc_key_from_uri(uri: str):
    """Extract the ddc key from a ddc uri.

    Parameters
    ----------
    uri: str
        the uri to extract the key from

    Returns
    -------
    str
        the ddc key extracted from the ddc uri
    """
    return uri[4:]


def ddc_parent_from_uri(uri: str):
    """Return the parent ddc uri of a ddc uri by reducing its key one step.

    Parameters
    ----------
    uri: str
        the ddc uri, whose parent is calculcated

    Returns
    -------
    str
        the parent ddc uri of the provided ddc uri
    """
    key = ddc_key_from_uri(uri)
    if "." in key:
        major, minor = key.split(".")
        if len(minor) == 1:
            return ddc_key_to_uri(major)
        return ddc_key_to_uri(major + "." + minor[:-1])
    if key[1:] == "00":
        return None
    if key[2] == "0":
        return ddc_key_to_uri(key[0] + "00")
    return ddc_key_to_uri(key[:2] + "0")


def extend_ddc_subject_list_with_ancestors(subjects: Sequence[str]):
    """Return an extended list of ddc subjects that also contains all ancestor subjects."""
    extended_set = set(subjects)
    to_be_checked = set(subjects)

    while len(to_be_checked) > 0:
        subject = to_be_checked.pop()
        parent = ddc_parent_from_uri(subject)
        if parent is not None and parent not in extended_set:
            extended_set.add(parent)
            to_be_checked.add(parent)

    return list(extended_set)


def ddc_notation_from_uri_via_coliana(
    uri: str,
) -> Optional[str]:
    """Return a ddc label a reported by the coli-ana web service.

    See: https://coli-conc.gbv.de/coli-ana/

    Parameters
    ----------
    uri: str
        the ddc uri as string

    Returns
    -------
    str | None
        the label of the matching ddc subject or None if not available
    """
    key = ddc_key_from_uri(uri)

    logger.debug("do ddc label request to coli-ana for ddc uri: %s", uri)
    response = requests.get(COLIANA_DDC_API_URL + key)

    if not response or not response.ok:
        logger.info("ddc request to coli-ana failed")
        return None

    json = response.json()
    member_list = json[0]["memberList"]
    for member in member_list:
        if member is not None and key in member["notation"]:
            label = member["prefLabel"]["de"]
            logger.debug("ddc label for %s is %s", uri, label)
            return label
    return None


class SimpleDdcHierarchy(SubjectHierarchy):
    """A simple subject hierarchy implementation that is not stored."""

    def __init__(
        self,
        subject_uris: Sequence[str] = None,
        subject_labels: Optional[Mapping[str, str]] = None,
    ):
        """Initialize unlabled ddc hierarchy.

        Parameters
        ----------
        subject_uris: Optional[Sequence[str]] = None
            if set, this list is used for iterating all available ddc keys instead of assuming that only 1000 major
            keys exist
        subject_labels: Optional[Mapping[str, str]] = None
            an optional mapping of subject uris to labels
        """
        self.subject_uris = subject_uris
        self.subject_labels = subject_labels

    def __len__(self):
        """Return the size of the provided available ddc subjects or a static length of 1000."""
        if self.subject_uris is not None:
            return len(self.subject_uris)
        raise ValueError("length not available if list of ddc subjects was not provided")

    def __getitem__(self, uri: str) -> SubjectNode:
        """Return a subject hierarchy node for the given ddc uri.

        Parameters
        ----------
        uri: str
            the ddc uri

        Returns
        -------
        SubjectNode
            a subject node describing this ddc subject
        """
        if is_valid_ddc_uri(uri):
            label = ddc_key_from_uri(uri)
            if self.subject_labels is not None and self.subject_labels.get(uri, None) is not None:
                label = self.subject_labels[uri]
            return SubjectNode(
                uri=uri,
                label=label,
                parent_uri=ddc_parent_from_uri(uri)
            )
        raise ValueError(f"uri {uri} is not a valid ddc uri")

    def __iter__(self) -> Iterable[str]:
        """Return an iterator over the major level ddc subjects."""
        if self.subject_uris is not None:
            return iter(self.subject_uris)
        raise ValueError("iteration not available if list of ddc subjects was not provided")

    def __contains__(self, k: str) -> bool:
        """Check whether a ddc uri is contained in the provided subject uris, or check if it is a valid ddc uri."""
        if self.subject_uris is not None:
            return k in self.subject_uris
        if is_valid_ddc_uri(k):
            return True
        return False

    def keys(self) -> Iterable[str]:
        """Return a list over all major level ddc uris."""
        return iter(self)

    def values(self) -> Iterable[SubjectNode]:
        """Return a list over all major level ddc subjects nodes."""
        return [self.__getitem__(k) for k in iter(self)]

    def get(self, k: str, default: Optional[SubjectNode] = None) -> Optional[SubjectNode]:
        """Return a ddc subject node for a given ddc uri."""
        if k in self:
            return self.__getitem__(k)
        return default

    def __eq__(self, other):
        """Check for instance equality."""
        return self == other

    def __ne__(self, other):
        """Check for instance inequality."""
        return self != other


class CachedColianaDdcLabels(SqliteDict):
    """Cache storing DDC labels retrieved from coli-ana.

    See `ddc_notation_from_uri_via_coliana` on how to retrieve labels from coli-ana.
    """

    def __init__(
        self,
        cache_filepath: Optional[str] = None,
        time_between_requests: float = 0.5,
    ):
        """Initialize a new cache.

        Parameters
        ----------
        cache_filepath: Optional[str] = None
            the path to the file that is used for caching DDC labels; if None, as default path is generated at
            `<cache_dir>/ddc/coliana_labels.sqlite`.
        time_between_requests: float = 0.5
            the minimum time between requests to not overflood coli-ana with too many requests
        """
        if cache_filepath is None:
            cache_filepath = os.path.join(get_cache_dir(), "ddc/coliana_labels.sqlite")

        os.makedirs(os.path.dirname(cache_filepath), exist_ok=True)
        super().__init__(filename=cache_filepath, tablename="ddc_labels_by_uri", flag="c", autocommit=True)

        self.time_between_requests = time_between_requests
        self.last_request = 0

    def __getitem__(self, key):
        """Return a label for a ddc uri either from cache or by retrieving it from coli-ana."""
        if not super().__contains__(key):
            time.sleep(max(self.time_between_requests - (time.time() - self.last_request), 0.0))
            super().__setitem__(key, ddc_notation_from_uri_via_coliana(key))
            self.last_request = time.time()
        return super().__getitem__(key)


def get_ddc_subject_store(
    subject_uris: Optional[Sequence[str]] = None,
    cache_filepath: Optional[str] = None,
) -> SubjectHierarchy:
    """Return an instance of the UnlabledDdcHierarchy."""
    subject_labels = CachedColianaDdcLabels(cache_filepath)
    return SimpleDdcHierarchy(subject_uris=subject_uris, subject_labels=subject_labels)
