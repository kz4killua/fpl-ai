import numpy as np


def compare_arrays(a, b):
    """Returns True if two arrays are almost equal."""
    return np.allclose(a, b, atol=0.001)