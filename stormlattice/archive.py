"""
Radar archive: fetch one storm day from the MRMS bucket and store it as one .npz file.

A storm day runs 10:00 UTC to 10:00 UTC the next day and is named by its starting UTC date
(YYYYMMDD). Florida's evening storms cross 00 UTC, so a file per UTC date would cut one storm
cycle in two. A storm day therefore reads from two S3 date folders.

A storm day has one slot per 10-minute mark (144 slots). For each mark: take the nearest MRMS
frame, but only if it is within a maximum offset of the mark (archive.max_frame_offset_min in
settings.yaml, 4 minutes). Download it, crop to the Florida box, coarsen to 0.02 deg cells
(grid.py), then delete the GRIB. Only the cropped arrays are kept:

    data/frames/<YYYYMMDD>.npz
        dbz          (t, ny, nx) float32   NaN = no radar coverage, -32 = clear air
        times_unix   (t,) int64            seconds since 1970 UTC: the frame's own time

If no frame is close enough to a mark, the slot is kept anyway: its dbz is all NaN and its
times_unix is the 10-minute mark itself. A missing frame then shows up as NaN, not as a gap,
and every storm day has the same shape.

Fetching is resumable: a day whose file exists is skipped, and the file is written as .part and
renamed only when complete (same idea as mrms.download).

To write: the frame keys for a storm day, fetch_day, load_day.
"""

from __future__ import annotations

from datetime import datetime, timedelta, timezone
from pathlib import Path

import numpy as np

from stormlattice import grid, mrms


def day_start(day, start_hour_utc):
    """When storm day `day` ("YYYYMMDD") begins, as a timezone-aware UTC datetime."""
    return datetime.strptime(day, "%Y%m%d").replace(hour=start_hour_utc, tzinfo=timezone.utc)

def slot_times(day, start_hour_utc, cadence_min):
    """
    The 10-minute marks of one storm day, as timezone-aware UTC datetimes.

    `day` is the storm day's name, "YYYYMMDD". The day runs from start_hour_utc on that date to the same hour the next day, so it has 24 * 60 / cadence_min slots.
    """
    start = day_start(day, start_hour_utc)
    n_slots = 24 * 60 // cadence_min
    return [ start + timedelta(minutes=i * cadence_min) for i in range(n_slots) ]

def frame_keys(day, start_hour_utc, fs=None):
    """MRMS reflectivity keys whose frame time falls inside the storm day, oldest first."""
    fs = fs or mrms._filesystem()
    start = day_start(day, start_hour_utc)
    end = start + timedelta(days=1)
    keys = []
    for date in (start, end):       # a storm day spans two UTC date folders
        keys += fs.ls(f"{mrms.BUCKET}/CONUS/{mrms.REFLECTIVITY}/{date:%Y%m%d}")
    return [ k for k in keys if start <= mrms.frame_time(k) < end ]

def match_slots(slots, keys, max_offset_min):
    """The key of the frame nearest each slot time, or None if no frame is within max_offset_min."""
    times = [ mrms.frame_time(k) for k in keys ]
    limit = timedelta(minutes=max_offset_min)
    matched = []
    for slot in slots:
        offsets = [ abs(t - slot) for t in times ]      # timedelta from this slot to every frame
        i = offsets.index(min(offsets)) if offsets else None
        matched.append( keys[i] if i is not None and offsets[i] <= limit else None )
    return matched

def load_cells(key, grib_dir, fs=None):
    """Download one MRMS frame, crop it, coarsen to 0.02 deg cells, delete the GRIB: (ny, nx) float32."""
    grib_path = mrms.download(key, grib_dir, fs=fs)
    try:
        pixels = mrms.load_florida(grib_path).values        # (648, 752) dBZ; NaN = no radar coverage
    finally:
        grib_path.unlink(missing_ok=True)                   # runs even if loading failed
    return grid.block_max(pixels, grid.PIXELS_PER_CELL).astype(np.float32)

def fetch_day(day, frames_dir, grib_dir, *, start_hour_utc, cadence_min, max_offset_min, fs=None):
    """Fetch one storm day into frames_dir/<day>.npz and return the path. A day already on disk is skipped."""
    out = Path(frames_dir) / f"{day}.npz"
    if out.exists():
        return out
    slots = slot_times(day, start_hour_utc, cadence_min)
    keys = match_slots(slots, frame_keys(day, start_hour_utc, fs=fs), max_offset_min)
    frames = [ load_cells(k, grib_dir, fs=fs) if k else None for k in keys ]

    real = [ f for f in frames if f is not None ]
    if not real:
        raise ValueError(f"storm day {day}: no MRMS frames found")
    gap = np.full_like(real[0], np.nan)         # an all-NaN frame for empty slots
    dbz = np.stack([ f if f is not None else gap for f in frames ])
    times = [ mrms.frame_time(k) if k else s for s, k in zip(slots, keys) ]
    times_unix = np.array([ int(t.timestamp()) for t in times ], dtype=np.int64)

    part = out.with_name(out.name + ".part")
    with open(part, "wb") as f:         # pass a file, so NumPy can't add ".npz" to the name
        np.savez_compressed(f, dbz=dbz, times_unix=times_unix)
    part.replace(out)                   # appears under its real name only when complete
    return out

def load_day(path):
    """One storm day: dbz (t, ny, nx) float32 and the frame times as UTC datetimes."""
    with np.load(path) as f:
        dbz = f["dbz"]
        times = [ datetime.fromtimestamp(int(t), tz=timezone.utc) for t in f["times_unix"] ]
    return dbz, times
