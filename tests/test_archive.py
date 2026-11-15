from stormlattice import archive

FOLDER = "noaa-mrms-pds/CONUS/MergedReflectivityQCComposite_00.50"


def key(date, hhmmss):
    """A fake MRMS key with the real naming pattern."""
    return f"{FOLDER}/{date}/MRMS_MergedReflectivityQCComposite_00.50_{date}-{hhmmss}.grib2.gz"


class FakeFS:
    def __init__(self, tree):
        self.tree = tree

    def ls(self, path):
        return self.tree.get(path, [])


def test_storm_day_reads_both_utc_folders():
    fs = FakeFS({
        f"{FOLDER}/20240715": [key("20240715", "095838"), key("20240715", "100038"),
                               key("20240715", "235838")],
        f"{FOLDER}/20240716": [key("20240716", "000038"), key("20240716", "095838"),
                               key("20240716", "100038")],
    })
    keys = archive.frame_keys("20240715", 10, fs=fs)
    times = [ k.rsplit("_", 1)[-1][:15] for k in keys ]
    assert times == ["20240715-100038", "20240715-235838", "20240716-000038", "20240716-095838"]

def test_slot_times():
    t = archive.slot_times("20240715", 10, 10)
    assert len(t) == 144
    assert t[0].hour == 10 and t[0].tzinfo is not None
    assert t[-1].strftime("%Y%m%d-%H%M") == "20240716-0950"
