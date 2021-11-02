"""Helper methods to store numpy arrays."""

import io
from typing import cast

import numpy as np


def numpy_array_to_bytes(array):
    """Convert numpy array to bytes."""
    buffer = io.BytesIO()
    np.save(buffer, array)
    buffer.seek(0)
    return buffer.read()


def bytes_to_numpy_array(data):
    """Read numpy array from bytes."""
    buffer = io.BytesIO(data)
    buffer.seek(0)
    return cast(np.ndarray, np.load(buffer))
