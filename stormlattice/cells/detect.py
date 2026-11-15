"""Step 2: storm mask and storm objects for one radar frame.

A storm pixel has composite reflectivity at or above a threshold (35 dBZ in settings.yaml).
Storm objects are groups of touching storm pixels (scipy.ndimage.label). The labels and
features use the mask; the lattice moves and kills the objects. No tracking between frames.
"""

from __future__ import annotations

import numpy as np


def storm_mask(dbz: np.ndarray, threshold_dbz: float) -> np.ndarray:
    """True where reflectivity is at or above the threshold. NaN (no radar coverage) is False."""
    raise NotImplementedError


def label_objects(mask: np.ndarray, min_pixels: int) -> tuple[np.ndarray, int]:
    """Group touching storm pixels into objects, dropping objects smaller than `min_pixels`.

    Returns (label image, number of objects). In the label image 0 = no storm and
    1..n = the object's id.
    """
    raise NotImplementedError
