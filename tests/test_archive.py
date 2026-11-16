from datetime import datetime, timedelta, timezone

import numpy as np
import pytest
from conftest import FOLDER, FakeFS, key

from stormlattice import archive, mrms

# One fake storm day for the fetch_day tests: a frame 38 s after every 10-minute mark, except
# slot 3 (no frame at all) and slot 5 (its only frame is 5 min late, outside the 4 min limit).
DAY = "20240715"
ARGS = {"start_hour_utc": 10, "cadence_min": 10, "max_offset_min": 4}
MARKS = [ datetime(2024, 7, 15, 10, tzinfo=timezone.utc) + timedelta(minutes=10 * i) for i in range(144) ]
EMPTY = [3, 5]


def storm_day_fs():
    """A fake S3 holding the frames of storm day 20240715, in its two UTC date folders."""
    stamps = [ m + timedelta(seconds=38) for i, m in enumerate(MARKS) if i not in EMPTY ]
    stamps.append(MARKS[5] + timedelta(minutes=5))      # 10:55:00, 5 min from both 10:50 and 11:00
    tree = {}
    for t in sorted(stamps):
        date = t.strftime("%Y%m%d")
        tree.setdefault(f"{FOLDER}/{date}", []).append(key(date, t.strftime("%H%M%S")))
    return FakeFS(tree)

def hhmm_cells(key, grib_dir, fs=None):
    """Stands in for archive.load_cells: a 2 x 3 grid filled with the frame's HHMM, e.g. 1430.0."""
    t = mrms.frame_time(key)
    return np.full((2, 3), t.hour * 100 + t.minute, dtype=np.float32)


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
    assert t[0].hour == 10 and t[0].utcoffset() == timedelta(0)
    assert t[-1].strftime("%Y%m%d-%H%M") == "20240716-0950"

def test_fetch_day_round_trip(tmp_path, monkeypatch):
    monkeypatch.setattr(archive, "load_cells", hhmm_cells)
    frames_dir = tmp_path / "frames"
    frames_dir.mkdir()
    path = archive.fetch_day(DAY, frames_dir, tmp_path / "grib", fs=storm_day_fs(), **ARGS)
    dbz, times = archive.load_day(path)

    assert dbz.shape == (144, 2, 3) and dbz.dtype == np.float32
    assert [ i for i in range(144) if np.isnan(dbz[i]).all() ] == EMPTY
    assert [ p.name for p in frames_dir.iterdir() ] == ["20240715.npz"]     # no .part file left

    # an empty slot keeps the mark's time, a real one the frame's own time
    assert times == [ m if i in EMPTY else m + timedelta(seconds=38) for i, m in enumerate(MARKS) ]
    assert [ t.utcoffset() for t in times ] == [timedelta(0)] * 144         # UTC, not just aware
    for i, m in enumerate(MARKS):
        if i not in EMPTY:
            assert (dbz[i] == m.hour * 100 + m.minute).all()                # each frame in its own slot

def test_fetch_day_skips_day_on_disk(tmp_path, monkeypatch):
    def no_download(key, grib_dir, fs=None):
        raise AssertionError("load_cells was called for a day already on disk")
    monkeypatch.setattr(archive, "load_cells", no_download)
    out = tmp_path / "20240715.npz"
    out.write_bytes(b"already here")
    assert archive.fetch_day(DAY, tmp_path, tmp_path / "grib", fs=storm_day_fs(), **ARGS) == out
    assert out.read_bytes() == b"already here"      # not overwritten

def test_failed_fetch_leaves_no_npz(tmp_path, monkeypatch):
    def fails_at_noon(key, grib_dir, fs=None):
        if mrms.frame_time(key).hour == 12:         # two hours into the storm day
            raise OSError("connection lost")
        return hhmm_cells(key, grib_dir)
    monkeypatch.setattr(archive, "load_cells", fails_at_noon)
    with pytest.raises(OSError):
        archive.fetch_day(DAY, tmp_path, tmp_path / "grib", fs=storm_day_fs(), **ARGS)
    assert list(tmp_path.iterdir()) == []           # no .npz that looks complete, and no .part

def test_fetch_day_creates_frames_dir(tmp_path, monkeypatch):
    monkeypatch.setattr(archive, "load_cells", hhmm_cells)
    frames_dir = tmp_path / "data" / "frames"       # does not exist yet
    path = archive.fetch_day(DAY, frames_dir, tmp_path / "grib", fs=storm_day_fs(), **ARGS)
    assert path == frames_dir / "20240715.npz" and path.exists()
