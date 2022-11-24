"""Methods to process Dewey Decimal Classification."""

# pylint: disable=too-many-return-statements

import logging
import os
import time
import urllib.parse
import re

from typing import Callable, Iterable, Mapping, Optional, Sequence

import requests
from sqlitedict import SqliteDict

from slub_docsa.common.paths import get_cache_dir
from slub_docsa.common.subject import SubjectHierarchy, print_subject_hierarchy
from slub_docsa.data.preprocess.subject import children_map_from_subject_parent_map

logger = logging.getLogger(__name__)

COLIANA_DDC_API_URL = "https://coli-conc.gbv.de/coli-ana/app/analyze?notation="
COLICONC_DDC_NARROWER_API_URL = "https://coli-conc.gbv.de/api/narrower"


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


def _ddc_key_check_major_part(major):
    if len(major) > 3:
        logger.debug("ddc major part '%s' has more than 3 digits", major)
        return False
    if not major.isnumeric():
        logger.debug("ddc major part '%s' isn't numeric", major)
        return False
    return True


def _ddc_key_check_minor_part(minor):
    if not minor.isnumeric():
        logger.debug("ddc minor part '%s' isn't numeric", minor)
        return False
    return True


def _check_single_ddc_key(key):
    if "." in key:
        if len(key.split(".")) != 2:
            logger.debug("ddc key '%s' contains multiple dots", key)
            return False
        major, minor = key.split(".")
        return _ddc_key_check_major_part(major) and _ddc_key_check_minor_part(minor)
    return _ddc_key_check_major_part(key)


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
    if "-" in key:
        if len(key.split("-")) != 2:
            logger.debug("ddc key '%s' has multiple minus symbols")
            return False
        first, second = key.split("-")
        return _check_single_ddc_key(first) and _check_single_ddc_key(second)
    return _check_single_ddc_key(key)


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
    if len(key) > 1:
        return ddc_key_to_uri(key[:-1])
    return None


def ddc_root_subjects() -> Iterable[str]:
    """Return URIs of the 10 root DDC subjects."""
    return [ddc_key_to_uri(str(i)) for i in range(10)]


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


def ddc_label_from_uri_via_coliana(
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

    if len(key) == 1:
        key = key + "00"
    if len(key) == 2:
        key = key + "0"

    logger.debug("do ddc label request to coli-ana for ddc uri: %s", uri)
    response = requests.get(COLIANA_DDC_API_URL + key, timeout=10)

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


def ddc_children_from_uri_via_coliconc(uri: str) -> Iterable[str]:
    """Return the list of children DDC subjects via the coli-conv API."""
    # build request url
    key = ddc_key_from_uri(uri)
    dewey_info_uri = urllib.parse.quote("http://dewey.info/class/" + key + "/e23/")
    request_url = COLICONC_DDC_NARROWER_API_URL + f"?limit=10000&uri={dewey_info_uri}&language=en,de"

    # do request
    response = requests.get(request_url, timeout=10)
    json = response.json()

    # extract notation from dewey info uri
    uri_pattern = re.compile(r"http://dewey.info/class/([^/]+)/e23/")
    children_uris = [ddc_key_to_uri(uri_pattern.match(entry["uri"]).group(1)) for entry in json]

    # filter for invalid uris or the requested uri
    return [child_uri for child_uri in filter(is_valid_ddc_uri, children_uris) if child_uri != uri]


def cached_ddc_children_from_uri_via_coliconc(
    cache_filepath: Optional[str] = None,
    time_between_requests: float = 0.5,
) -> Iterable[str]:
    """Cache DDC children in an sqlite database to prevent excessive requests to coli-conv API."""
    if cache_filepath is None:
        cache_filepath = os.path.join(get_cache_dir(), "ddc/coliconv_children.sqlite")

    os.makedirs(os.path.dirname(cache_filepath), exist_ok=True)
    store = SqliteDict(filename=cache_filepath, tablename="ddc_children_by_uri", flag="c", autocommit=True)

    last_request = 0

    def retrieve_children(uri: str):
        nonlocal last_request

        if uri not in store:
            time.sleep(max(time_between_requests - (time.time() - last_request), 0.0))
            store[uri] = ddc_children_from_uri_via_coliconc(uri)
            last_request = time.time()

        return store[uri]

    return retrieve_children


class SimpleDdcSubjectHierarchy(SubjectHierarchy):
    """A simple subject hierarchy implementation that is not stored."""

    def __init__(
        self,
        subject_uris: Optional[Sequence[str]] = None,
        subject_labels: Optional[Mapping[str, Mapping[str, str]]] = None,
        get_ddc_children: Optional[Callable[[str], Iterable[str]]] = None,
    ):
        """Initialize unlabled ddc hierarchy.

        Parameters
        ----------
        subject_uris: Optional[Sequence[str]] = None
            if set, this list is used for iterating all available ddc keys instead of assuming that only 1000 major
            keys exist
        subject_labels: Optional[Mapping[str, str]] = None
            an optional mapping of subject uris to labels
        get_ddc_children: Optional[Callable[[str], Iterable[str]]] = None
            an optional method that is used to retrieve DDC children instead of the children that are calculated from
            the `subject_uris` argument; if both are missing, the list of DDC children can not be provided
        """
        self._subject_uris = subject_uris
        self._subject_labels = subject_labels
        self._get_ddc_children = get_ddc_children

        if subject_uris is not None:
            parent_map = {subject_uri: ddc_parent_from_uri(subject_uri) for subject_uri in subject_uris}
            self._subject_children = children_map_from_subject_parent_map(parent_map)
            self._root_subjects = [subject_uri for subject_uri, parent_uri in parent_map.items() if parent_uri is None]
        else:
            self._subject_children = None
            self._root_subjects = ddc_root_subjects()

    def subject_labels(self, subject_uri: str) -> Mapping[str, str]:
        """Return the labels mapping for a DDC subject.

        If not labels are provided, the DDC key is returned as english label.

        Parameters
        ----------
        subject_uri: str
            the uri of the subject whose label mapping is requested

        Returns
        -------
        Mapping[str, str]
            the mapping from ISO 639-1 language codes to labels for this subject;
            an empty mapping for valid subjects with unknown labels

        Raises
        ------
        LookupError
            if the subject with this uri is not available in this subject hierarchy
        """
        if not is_valid_ddc_uri(subject_uri):
            raise LookupError(f"uri {subject_uri} is not a valid ddc uri")
        if self._subject_labels is not None and self._subject_labels[subject_uri] is not None:
            return self._subject_labels[subject_uri]
        return {"en": ddc_key_from_uri(subject_uri)}

    def subject_parent(self, subject_uri: str) -> Optional[str]:
        """Return the parent of the subject or None if the subject does not have a parent.

        Parameters
        ----------
        subject_uri: str
            the uri of the subject whose parent is requested

        Returns
        -------
        Optional[str]
            the uri of the parent subject or None if the requested subject does not have a parent

        Raises
        ------
        LookupError
            if the subject with this uri is not available in this subject hierarchy
        """
        if not is_valid_ddc_uri(subject_uri):
            raise LookupError(f"uri {subject_uri} is not a valid ddc uri")
        return ddc_parent_from_uri(subject_uri)

    def subject_children(self, subject_uri: str) -> Iterable[str]:
        """Return the children of the DDC subject.

        If no list of subjects was provided, the children can not be calculated and a
        RuntimeError is raised.

        Parameters
        ----------
        subject_uri: str
            the uri of the subject whose children is requested

        Returns
        -------
        Iterable[str]
            the list of URIs of children subjects or an empty list if the requested subject does not have any children

        Raises
        ------
        LookupError
            if the subject with this uri is not available in this subject hierarchy
        RuntimeError
            if the list of DDC subjects was not provided, such that children can not be calculated
        """
        if not is_valid_ddc_uri(subject_uri):
            raise LookupError(f"uri {subject_uri} is not a valid ddc uri")
        if self._get_ddc_children is not None:
            return self._get_ddc_children(subject_uri)
        if self._subject_children is not None:
            return self._subject_children[subject_uri]
        raise RuntimeError("children can not be calculated if list of ddc subjects is not provided")

    def root_subjects(self) -> Iterable[str]:
        """Return a list of root DDC subjects.

        Returns
        -------
        Iterable[str]
            the list of URIs of DDC subjects that have no parent subject
        """
        return self._root_subjects

    def __iter__(self) -> Iterable[str]:
        """Return an iterator over all subject uris of this subject hierarchy.

        The order in which subjects are iterated is not guaranteed.
        """
        if self._subject_uris is not None:
            return iter(self._subject_uris)
        raise RuntimeError("iteration not available if list of ddc subjects was not provided")

    def __contains__(self, subject_uri: str) -> bool:
        """Check whether a DDC uri is contained in the provided subject uris, or check if it is a valid ddc uri.

        Parameters
        ----------
        subject_uri: str
            the uri of the subject to be checked

        Returns
        -------
        bool
            whether the subject_uri is contained in the list of provided DDC subjects or a valid DDC subject
        """
        if self._subject_uris is not None:
            return subject_uri in self._subject_uris
        if is_valid_ddc_uri(subject_uri):
            return True
        return False


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

    def __getitem__(self, subject_uri) -> Mapping[str, str]:
        """Return a label for a ddc uri either from cache or by retrieving it from coli-ana."""
        if not super().__contains__(subject_uri):
            time.sleep(max(self.time_between_requests - (time.time() - self.last_request), 0.0))
            super().__setitem__(subject_uri, ddc_label_from_uri_via_coliana(subject_uri))
            self.last_request = time.time()

        label = super().__getitem__(subject_uri)
        if label is not None:
            return {"de": label}
        return {"en": ddc_key_from_uri(subject_uri)}


def load_ddc_subject_hierarchy(
    subject_uris: Optional[Sequence[str]] = None,
    labels_cache_filepath: Optional[str] = None,
    children_cache_filepath: Optional[str] = None,
) -> SubjectHierarchy:
    """Return an instance of the UnlabledDdcHierarchy."""
    subject_labels = CachedColianaDdcLabels(labels_cache_filepath)
    get_ddc_children = cached_ddc_children_from_uri_via_coliconc(children_cache_filepath)
    return SimpleDdcSubjectHierarchy(subject_uris, subject_labels, get_ddc_children)


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    ddc_subject_hierarchy = load_ddc_subject_hierarchy(ddc_root_subjects())
    print_subject_hierarchy("de", ddc_subject_hierarchy)

    # print(ddc_children_from_uri_via_coliconc(ddc_key_to_uri("700")))
