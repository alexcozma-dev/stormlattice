"""Step 1: environment and lightning inputs.

- HRRR (3 km, hourly): steering wind (850-300 hPa mean), CAPE, precipitable water,
  low-level convergence, onshore wind component on each coast.
- GOES-East GLM: lightning flash points, binned onto the 2 km grid.
"""

from __future__ import annotations

from datetime import datetime

import xarray as xr


def hrrr_fields(valid_time: datetime) -> xr.Dataset:
    """Return HRRR fields on the 2 km Florida grid for the analysis nearest `valid_time`."""
    raise NotImplementedError


def glm_flash_counts(start: datetime, end: datetime) -> xr.DataArray:
    """Count GLM flashes per 2 km cell between `start` and `end`."""
    raise NotImplementedError
