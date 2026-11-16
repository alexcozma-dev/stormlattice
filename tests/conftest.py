"""Helpers shared by more than one test file: a fake S3 and fake MRMS keys."""

from __future__ import annotations

FOLDER = "noaa-mrms-pds/CONUS/MergedReflectivityQCComposite_00.50"


def key(date, hhmmss):
    """A fake MRMS key with the real naming pattern."""
    return f"{FOLDER}/{date}/MRMS_MergedReflectivityQCComposite_00.50_{date}-{hhmmss}.grib2.gz"


class FakeFS:
    """Stands in for S3 in tests: a dict of folder -> list of entries."""

    def __init__(self, tree):
        self.tree = tree

    def ls(self, path):
        return self.tree.get(path, [])
