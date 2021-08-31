"""Provides methods that decide which subjects are chosen based on their class probabilities."""

import numpy as np


def threshold_decision(probabilities, threshold):
    """Select subjects based on a threshold over probabilities."""
    return np.where(probabilities > threshold, 1, 0)
