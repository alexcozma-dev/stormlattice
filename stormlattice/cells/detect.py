"""Step 2: storm-cell detection.

Threshold composite reflectivity, group connected pixels with scipy.ndimage.label,
drop small objects, and compute per-cell features.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass
class Cell:
    cell_id: int
    centroid_yx: tuple[float, float]   # grid coordinates
    area_km2: float
    max_dbz: float
    max_vil: float
    echo_top_km: float
    flashes_5min: int


def detect_cells(
    reflectivity: np.ndarray,
    vil: np.ndarray,
    echo_top: np.ndarray,
    flashes: np.ndarray,
    threshold_dbz: float,
    min_area_km2: float,
    cell_area_km2: float,
) -> tuple[np.ndarray, list[Cell]]:
    """Return (label image, list of Cell) for one radar frame."""
    raise NotImplementedError
