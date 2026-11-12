"""Step 1: MRMS radar mosaic from the NOAA bucket on AWS (anonymous access).

Products of interest (CONUS, ~1 km, 2-minute cadence):
    - MergedReflectivityQCComposite   composite reflectivity, dBZ
    - VIL                             vertically integrated liquid
    - EchoTop_18                      18 dBZ echo-top height
"""

from __future__ import annotations

from datetime import datetime
from pathlib import Path

import xarray as xr


def latest_key(product: str = "MergedReflectivityQCComposite_00.50") -> str:
    """Return the S3 key of the newest file for `product`."""
    raise NotImplementedError


def download(key: str, dest: Path) -> Path:
    """Download one gzipped GRIB2 file and decompress it. Returns the local path."""
    raise NotImplementedError


def load_florida(path: Path) -> xr.DataArray:
    """Open a GRIB2 file with cfgrib, crop to the Florida box and regrid to the 2 km grid."""
    raise NotImplementedError


def frame_time(key: str) -> datetime:
    """Parse the valid time (UTC) from an MRMS file name. Shown in the UI as the data timestamp."""
    raise NotImplementedError
