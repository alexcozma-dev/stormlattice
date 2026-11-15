"""Step 4: static geography features on the 0.02 deg grid (computed once, cached as .npz).

Built from Natural Earth coastline and lake shapes.

Sea-breeze physics: land heats faster than water, onshore flow develops, fronts from the
Atlantic and Gulf coasts move inland and collide over the peninsula, lifting moist air.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np

FEATURE_NAMES = [
    "dist_atlantic_km",
    "dist_gulf_km",
    "water_frac_5km",
    "water_frac_20km",
    "land_mask",
]


def build(cache_path: Path) -> dict[str, np.ndarray]:
    """Compute every static feature (coast distances, water fractions, land mask) and cache."""
    raise NotImplementedError


def load(cache_path: Path) -> dict[str, np.ndarray]:
    """Load cached static features."""
    with np.load(cache_path) as data:
        return {name: data[name] for name in FEATURE_NAMES}
