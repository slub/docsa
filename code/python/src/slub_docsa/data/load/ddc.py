"""Methods to process Dewey Decimal Classification."""

# pylint: disable=too-many-return-statements

import logging

from typing import Iterable, Optional
from slub_docsa.common.subject import SubjectHierarchyType, SubjectNode

logger = logging.getLogger(__name__)


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


class UnlabeledDdcHierarchy(SubjectHierarchyType[SubjectNode]):
    """A simple subject hierarchy implementation without any ddc labels."""

    def __len__(self):
        """Return a static length of 1000, which howerver only corresponds to the major level ddc subjects."""
        return 1000

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
            return SubjectNode(uri=uri, label=ddc_key_from_uri(uri), parent_uri=ddc_parent_from_uri(uri))
        raise ValueError(f"uri {uri} is not a valid ddc uri")

    def __iter__(self) -> Iterable[str]:
        """Return an iterator over the major level ddc subjects."""
        for i in range(1000):
            yield ddc_key_to_uri(ddc_correct_short_keys(str(i)))

    def __contains__(self, k: str) -> bool:
        """Check whether a ddc uri is a valid uri, and therefore, contained in this hierarchy."""
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


def get_ddc_subject_hierarchy():
    """Return an instance of the UnlabledDdcHierarchy."""
    return UnlabeledDdcHierarchy()
