"""
Step 1: MRMS radar mosaic from the NOAA bucket on AWS (anonymous access).

Composite reflectivity, CONUS, ~1km (0.01 deg), one frame every ~2 minutes.
File names end in UTC timestamp, e.g. ..._20261112-180339.grib2.gz
"""

from __future__ import annotations

import gzip
import shutil
from datetime import datetime, timezone
from pathlib import Path

import s3fs
import xarray as xr

from stormlattice.config import settings

BUCKET = "noaa-mrms-pds"
REFLECTIVITY = "MergedReflectivityQCComposite_00.50"

#MRMS flag values (not real measurements)
MISSING = -999.0        #no radar coverage -> unknown (NaN)
NO_ECHO = -99.0         #covered, but nothing there -> clear air
NO_ECHO_FLOOR = -32.0   #what clear air becomes: just below the weakest real echo

def _filesystem() -> s3fs.S3FileSystem:
    """Anonymous connection to S3 (NOAA's buckets are public)."""
    return s3fs.S3FileSystem(anon=True)

def frame_time(key: str) -> datetime:
    """Valid time (UTC) parsed from an MRMS file name or S3 key."""
    name = key.rsplit("/", 1)[-1]               # drop all folders
    stamp = name.split("_")[-1].split(".")[0]   #  '20261112-180839'
    return datetime.strptime(stamp, "%Y%m%d-%H%M%S").replace(tzinfo=timezone.utc)

def latest_key(product: str = REFLECTIVITY, fs=None) -> str:
    """
    S3 key of the newest file for `product`.

    Walks backwards through the day folders, because just after 00 UTC the newest
    day's folder can exist but still be empty.
    """
    fs = fs or _filesystem()
    days = fs.ls(f"{BUCKET}/CONUS/{product}")
    for day in reversed(days):
        files = fs.ls(day)
        if files:
            return files[-1]    # names sort by time, so the last one is newest
    raise FileNotFoundError(f"No MRMS files found for {product}")

def download(key: str, dest_dir: Path, fs=None) -> Path:
    """
    Download one .grib2.gz file and decompress it. Returns the .grib2 path.

    Skips work that's already done, and writes to a temporary '.part' file firs so
    interrupted run never leaves a half-written ,grib2 that looks complete.
    """
    dest_dir.mkdir(parents=True, exist_ok=True)
    gz_path = dest_dir / key.rsplit("/", 1)[-1]
    grib_path = gz_path.with_suffix("")         # drop '.gz'
    if grib_path.exists():
        return grib_path
    if not gz_path.exists():
        (fs or _filesystem()).get(key, str(gz_path))

    part = grib_path.with_name(grib_path.name + ".part")
    with gzip.open(gz_path, "rb") as src, open(part, "wb") as dst:
        shutil.copyfileobj(src, dst)
    part.replace(grib_path)     # rename only once it's complete
    gz_path.unlink()            # keep just the decompressed file
    return grib_path

def load_florida(path: Path) -> xr.DataArray:
    """
    Open a GRIB2 frame, crop to the Florida box from settings.yaml, and clean the flags.

    Returns reflectivity in dBZ with longitudes in -180..180. No coverage -> NaN,
    clean air -> NO_ECHO_FLOOR. (Regridding to 0.02 deg cells (grid.py) comes later. )
    """
    box = settings()["grid"]
    ds = xr.open_dataset(path, engine="cfgrib", backend_kwargs={"indexpath": ""})
    da = ds[next(iter(ds.data_vars))]      # the only variable ('unknown')

    # Latitude is stored north -> south, so the slice runs max -> min.
    fl = da.sel(latitude=slice(box["lat_max"], box["lat_min"]),
                longitude=slice(box["lon_min"] % 360, box["lon_max"] % 360))
    fl = fl.where(fl != MISSING)
    fl = fl.where(fl != NO_ECHO, NO_ECHO_FLOOR)
    fl = fl.assign_coords(longitude=fl.longitude - 360)

    fl = fl.load()      # read into memory before closing the file
    ds.close()
    fl.name = "reflectivity"
    fl.attrs = {"units": "dBZ"}
    return fl
