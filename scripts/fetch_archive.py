"""Fetch every storm day in config/days.yaml into data/frames/, in the order train, validation, test.

Resumable: a day already on disk is skipped, so after a crash or Ctrl+C just run it again.
One failed day is reported and skipped; it does not stop the batch.

Run from the repo root, hiding the harmless eccodes warnings and keeping a log:
    python -u scripts/fetch_archive.py 2>&1 | grep --line-buffered -v ECCODES | tee -a data/fetch.log
"""

from __future__ import annotations

import time
from pathlib import Path

import yaml

from stormlattice import archive
from stormlattice.config import data_dir, settings

DAYS_FILE = Path(__file__).resolve().parents[1] / "config" / "days.yaml"
SPLITS = ("train", "validation", "test")


def main():
    cfg = settings()["archive"]
    days = yaml.safe_load(DAYS_FILE.read_text())
    todo = [(split, day) for split in SPLITS for day in days[split]]

    frames_dir = data_dir() / "frames"
    grib_dir = data_dir() / "grib"
    frames_dir.mkdir(parents=True, exist_ok=True)
    grib_dir.mkdir(parents=True, exist_ok=True)

    failed = []
    fetched = 0                    # days actually downloaded this run (skipped days don't count)
    t_start = time.time()
    for n, (split, day) in enumerate(todo, start=1):
        if (frames_dir / f"{day}.npz").exists():
            print(f"[{n}/{len(todo)}] {split} {day} already on disk", flush=True)
            continue
        t0 = time.time()
        try:
            archive.fetch_day(day, frames_dir, grib_dir,
                              start_hour_utc=cfg["storm_day_start_hour_utc"],
                              cadence_min=cfg["cadence_min"],
                              max_offset_min=cfg["max_frame_offset_min"])
        # Catch everything on purpose: one bad day must not stop on overnight batch.
        # Ctrl + C raises KeyboardInterrupt, which is not an Exception, so it still stops the run.
        except Exception as e:      # noqa: BLE001 
            failed.append(day)
            print(f"[{n}/{len(todo)}] {split} {day} FAILED: {e!r}", flush=True)
            continue
        fetched += 1
        per_day = (time.time() - t_start) / fetched
        eta_h = per_day * (len(todo) - n) / 3600
        print(f"[{n}/{len(todo)}] {split} {day} done in {time.time() - t0:.0f} s   "
              f"(about {eta_h:.1f} h left)", flush=True)

    print(f"Finished: {len(todo) - len(failed)} of {len(todo)} days on disk. Failed: {failed or 'none'}")


if __name__ == "__main__":
    main()
