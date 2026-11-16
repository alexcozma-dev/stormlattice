"""Build the replay files for one storm day, for the page in web/.

    python scripts/build_replay.py 20220601

Reads data/frames/<YYYYMMDD>.npz (archive.load_day) and writes:

    web/replay/<YYYYMMDD>/
      obs_<HHMM>.png    observed reflectivity, one per frame, named by the frame's own UTC time.
                        Transparent below 15 dBZ and where there is no radar coverage. Rows are
                        evenly spaced in Web Mercator y (overlay.py), so the page can place the
                        image by its bounds.
      index.json        mode, storm day, split, bounds, frame times and file names, colour
                        legend, corridor endpoints from settings.yaml
      briefing.json     a labelled placeholder until the LLM briefing exists
    web/replay/days.json   the storm days that have a replay folder

A slot with no radar frame (all NaN) gets no PNG; index.json lists it with "obs": null.

Test days: the script refuses a day in the test or replay list of config/days.yaml unless
--allow-test-day is given. Test days are looked at once, at the end. Develop on a train day.
"""

from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import yaml
from matplotlib import image as mpl_image

from stormlattice import archive, grid, mrms, overlay
from stormlattice.config import data_dir, settings

REPO_ROOT = Path(__file__).resolve().parents[1]
DAYS_FILE = REPO_ROOT / "config" / "days.yaml"
REPLAY_DIR = REPO_ROOT / "web" / "replay"
HELD_OUT = ("test", "replay")       # lists in days.yaml that must not be looked at early

PLACEHOLDER_BRIEFING = (
    "Placeholder text. No language model wrote this and it describes no real weather. "
    "Once the LLM is connected, this panel will show a short briefing in which every claim "
    "cites a source and every number comes from code."
)


def iso_utc(t: datetime) -> str:
    """A UTC datetime as text, e.g. 2022-06-01T18:00:38Z."""
    return t.strftime("%Y-%m-%dT%H:%M:%SZ")


def hex_colour(rgb) -> str:
    """An (r, g, b) triple of 0..255 as CSS text, e.g. #fdf802."""
    r, g, b = (int(c) for c in rgb)
    return f"#{r:02x}{g:02x}{b:02x}"


def split_of(day: str) -> str | None:
    """The list in config/days.yaml that holds `day`, or None if no list does."""
    days = yaml.safe_load(DAYS_FILE.read_text())
    for split, members in days.items():
        if day in (members or []):
            return split
    return None


def write_json(path: Path, content: dict) -> None:
    path.write_text(json.dumps(content, indent=1) + "\n")


def main():
    parser = argparse.ArgumentParser(description="Build web/replay/<day>/ from data/frames/<day>.npz")
    parser.add_argument("day", help="storm day, YYYYMMDD (UTC start date)")
    parser.add_argument("--allow-test-day", action="store_true",
                        help="build a day from the test or replay list (only once test days may be looked at)")
    args = parser.parse_args()
    day = args.day
    datetime.strptime(day, "%Y%m%d")    # raises ValueError on a day that is not YYYYMMDD

    split = split_of(day)
    if split in HELD_OUT and not args.allow_test_day:
        raise SystemExit(f"{day} is a {split} day. Test days are not looked at before the final run; "
                         "pass --allow-test-day when that time has come.")

    box = settings()["grid"]
    route = settings()["route"]
    dbz, times = archive.load_day(data_dir() / "frames" / f"{day}.npz")
    _, ny, nx = dbz.shape

    # The bounds written to index.json come from settings.yaml, the arrays from the archive.
    # If the box has changed since the day was fetched, the image would be placed wrongly.
    ny_box = round((box["lat_max"] - box["lat_min"]) / grid.CELL_DEG)
    nx_box = round((box["lon_max"] - box["lon_min"]) / grid.CELL_DEG)
    if (ny, nx) != (ny_box, nx_box):
        raise ValueError(f"{day}: frames are {ny} x {nx} cells, but the box in settings.yaml is "
                         f"{ny_box} x {nx_box}")

    out_dir = REPLAY_DIR / day
    out_dir.mkdir(parents=True, exist_ok=True)
    rows = overlay.mercator_rows(box["lat_min"], box["lat_max"], ny)    # same for every frame

    frames = []
    for frame, t in zip(dbz, times):
        if np.isnan(frame).all():       # the archive keeps a slot with no frame as all NaN
            frames.append({"time_utc": iso_utc(t), "obs": None})
            continue
        name = f"obs_{t:%H%M}.png"
        mpl_image.imsave(out_dir / name, overlay.reflectivity_rgba(frame[rows]),
                         pil_kwargs={"optimize": True})
        frames.append({"time_utc": iso_utc(t), "obs": name})

    names = [f["obs"] for f in frames if f["obs"]]
    if len(set(names)) != len(names):
        raise ValueError(f"{day}: two frames share one HHMM, so one PNG overwrote another")

    write_json(out_dir / "index.json", {
        "mode": "replay",
        "day": day,
        "split": split,
        "source": f"NOAA MRMS {mrms.REFLECTIVITY}, s3://{mrms.BUCKET}",
        "built_utc": iso_utc(datetime.now(timezone.utc)),
        "bounds": {k: box[k] for k in ("lat_min", "lat_max", "lon_min", "lon_max")},
        "legend": [{"dbz": int(level), "rgb": hex_colour(rgb)}
                   for level, rgb in zip(overlay.DBZ_LEVELS, overlay.LEVEL_RGB)],
        "route": {"suntrax_lonlat": route["suntrax_lonlat"], "mco_lonlat": route["mco_lonlat"]},
        "frames": frames,
    })
    write_json(out_dir / "briefing.json", {"placeholder": True, "text": PLACEHOLDER_BRIEFING})
    write_json(REPLAY_DIR / "days.json",
               {"days": sorted(p.parent.name for p in REPLAY_DIR.glob("*/index.json"))})

    size_mb = sum(p.stat().st_size for p in out_dir.iterdir()) / 1e6
    print(f"{day} ({split or 'not in days.yaml'}): {len(names)} PNGs, {len(frames) - len(names)} slots "
          f"without a frame, {rows.size} x {nx} px each, {size_mb:.1f} MB in {out_dir}")


if __name__ == "__main__":
    main()
