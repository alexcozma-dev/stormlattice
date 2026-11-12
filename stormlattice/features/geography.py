"""Step 4: static geography features on the 2 km grid (computed once, cached as .npz).

Sea-breeze physics: land heats faster than water, onshore flow develops, fronts from the
Atlantic and Gulf coasts move inland and collide over the peninsula, lifting moist air.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np

FEATURE_NAMES = [
    "dist_atlantic_km",
    "dist_gulf_km",
    "coast_land_frac_20km",   # curvature proxy
    "water_frac_5km",
    "water_frac_20km",
    "urban_frac",
    "forest_frac",
    "wetland_frac",
    "elevation_m",
    "relief_m",
]


def build(cache_path: Path) -> dict[str, np.ndarray]:
    """Compute every static feature (distance transforms, land-cover fractions, terrain) and cache."""
    raise NotImplementedError


def load(cache_path: Path) -> dict[str, np.ndarray]:
    """Load cached static features."""
    with np.load(cache_path) as data:
        return {name: data[name] for name in FEATURE_NAMES}
