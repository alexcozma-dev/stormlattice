"""Tests for the MRMS helpers that don't need the network."""
import os
from datetime import datetime, timedelta, timezone

import pytest

from stormlattice import mrms
from stormlattice.mrms import frame_time, latest_key


def test_frame_time_from_key():
    key = ("noaa-mrms-pds/CONUS/MergedReflectivityQCComposite_00.50/20261112/"
           "MRMS_MergedReflectivityQCComposite_00.50_20261112-180839.grib2.gz")
    assert frame_time(key) == datetime(2026, 11, 12, 18, 8, 39, tzinfo=timezone.utc)

def test_frame_time_from_decompressed_name():
    name = "MRMS_MergedReflectivityQCComposite_00.50_20261112-180839.grib2"
    assert frame_time(name).hour == 18

class FakeFS:
    """Stands in for S3 in tests: a dict of folder -> list of entries."""

    def __init__(self, tree):
        self.tree = tree

    def ls(self, path):
        return self.tree.get(path, [])

ROOT = "noaa-mrms-pds/CONUS/P"

def test_latest_key_picks_newest_file():
    fs = FakeFS({ROOT: [f"{ROOT}/20261111", f"{ROOT}/20261112"],
                 f"{ROOT}/20261112": [f"{ROOT}/20261112/a_20261112-000039.grib2.gz",
                                      f"{ROOT}/20261112/a_20261112-180839.grib2.gz"]})
    assert latest_key("P", fs=fs).endswith("180839.grib2.gz")

def test_latest_key_skips_empty_new_day():
    # just after midnight UTC: today's folder exists but has no files yet
    fs = FakeFS({ROOT: [f"{ROOT}/20261111", f"{ROOT}/20261112"],
                 f"{ROOT}/20261111": [f"{ROOT}/20261111/a_20261111-235839.grib2.gz"],
                 f"{ROOT}/20261112": []})
    assert latest_key("P", fs=fs).endswith("20261111-235839.grib2.gz")

def test_latest_key_raises_when_nothing_exists():
    with pytest.raises(FileNotFoundError):
        latest_key("P", fs=FakeFS({ROOT: []}))

# Live tests hit NOAA's S3 bucket, so they're off by default.
# Run then with: STORM_NETWORK_TESTS=1 pytest -s
network = pytest.mark.skipif(os.environ.get("STORM_NETWORK_TESTS") != "1",
                             reason="set STORM_NETWORK_TESTS=1 to run live S3 tests")

@network
def test_latest_frame_is_recent():
    key = mrms.latest_key()
    age = datetime.now(timezone.utc) - frame_time(key)
    print(f"newest frame: {key.rsplit('/', 1)[-1]}, age: {age}")
    assert timedelta(0) <= age < timedelta(minutes=30)      # recent, not in the future

@network
def test_download_and_load_florida(tmp_path):
    path = mrms.download(mrms.latest_key(), tmp_path)
    fl = mrms.load_florida(path)
    print("shape:", fl.shape, "min:", float(fl.min()), "max:", float(fl.max()))
    assert fl.shape == (648, 752)                   # the Florida box
    assert float(fl.min()) >= mrms.NO_ECHO_FLOOR    # no -999/-99 flags left
    assert fl.longitude.min() < -87 and fl.longitude.max() > -80.1  # -180..180 convention