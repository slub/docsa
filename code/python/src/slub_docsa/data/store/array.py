"""Helper methods to store numpy arrays."""

import io
from typing import cast

import numpy as np


def numpy_array_to_bytes(array: np.ndarray) -> bytes:
    """Convert numpy array to bytes.

    Parameters
    ----------
    array: numpy.ndarray
        numpy array that is converted to bytes

    Returns
    -------
    bytes
        the array as bytes
    """
    buffer = io.BytesIO()
    np.save(buffer, array)
    buffer.seek(0)
    return buffer.read()


def bytes_to_numpy_array(data: bytes) -> np.ndarray:
    """Read numpy array from bytes.

    Parameters
    ----------
    data: bytes
        the bytes that are converted to a numpy array

    Returns
    -------
    numpy.ndarray
        the converted numpy array
    """
    buffer = io.BytesIO(data)
    buffer.seek(0)
    return cast(np.ndarray, np.load(buffer))
