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

from stormlattice import mrms


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
