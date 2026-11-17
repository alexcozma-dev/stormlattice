"""Build the replay files for one storm day, for the page in web/.

    python scripts/build_replay.py 20220601

Reads data/frames/<YYYYMMDD>.npz (archive.load_day) and writes:

    web/replay/<YYYYMMDD>/
      obs_<HHMM>.png    observed reflectivity, one per frame, named by the frame's own UTC time.
                        Transparent below 15 dBZ and where there is no radar coverage. Rows are
                        evenly spaced in Web Mercator y (overlay.py), so the page can place the
                        image by its bounds.
      route_<HHMM>.json the route card for that issue time, one per slot (route.route_card).
                        "method" says what made it. A forecast method carries the route risk,
                        its green / amber / red level and the departure window (risk for each
                        departure in route.departures_min); "persistence" carries none of those.
                        Buffer and thresholds are risk.* in settings.yaml.
      index.json        mode, storm day, split, bounds, frame times and file names, colour
                        legend, corridor endpoints from settings.yaml
      briefing.json     a labelled placeholder until the LLM briefing exists
    web/replay/days.json   the storm days that have a replay folder

A slot with no radar frame (all NaN) gets no PNG; index.json lists it with "obs": null.

The route card is a BASELINE until the birth model and the lattice exist. It uses persistence:
the storm mask at the issue time (storms.threshold_dbz or more), held fixed for the whole flight.
That is one fixed picture of where storms are, not a forecast, so the file has "method":
"persistence", "level": "observed" and "storms_near_route": true or false. It has no risk
percentage, no green / amber / red and no departure window, and the page shows none. Every
route file says so in "label" and "note". A slot with no frame, or with no radar coverage (NaN)
within the buffer of the route, gets level "unavailable": no coverage is not clear air.

Test days: the script refuses a day in the test or replay list of config/days.yaml unless
--allow-test-day is given. Test days are looked at once, at the end. Develop on a train day.
"""

from __future__ import annotations

import argparse
import json
from datetime import datetime, timedelta, timezone
from pathlib import Path

import numpy as np
import yaml
from matplotlib import image as mpl_image

from stormlattice import archive, grid, mrms, overlay, route
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

BASELINE_LABEL = "baseline: persistence, not a forecast"


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


def persistence_route_card(frame: np.ndarray, t: datetime, corridor, cells: np.ndarray,
                           threshold_dbz: float, risk_cfg: dict, departures_min: list[int]) -> dict:
    """
    The route card for one issue time, from the persistence baseline: the content of route_<HHMM>.json.

    frame is the observed reflectivity (ny, nx) at the issue time t; corridor and cells come from
    route.sample_route and route.route_cells. risk_cfg is risk.* from settings.yaml and
    departures_min is route.departures_min: the departures, in minutes after t.
    """
    buffer_km = risk_cfg["buffer_km"]
    near = route.footprint(cells, buffer_km, frame.shape)
    usable = not np.isnan(frame[near]).any()        # also False for a slot with no frame at all

    # One frame, held fixed: T = 1, and arrival_steps with no gap limit always picks it.
    # NaN compares False, so it is never a storm cell.
    # TODO: use storms.storm_mask once it is written; this is the rule in its docstring.
    storm = (frame >= threshold_dbz)[None]

    def risk_at(offset_min):
        if not usable:
            return None
        steps = route.arrival_steps([t], t + timedelta(minutes=offset_min), corridor.flight_min)
        return route.route_risk(storm, cells, steps, buffer_km)

    # The window is computed the same way for every method; route.route_card decides what a
    # method may report. For "persistence" that is only whether storms are near the route now.
    # In a replay the issue time is the frame's own time, so the data is 0 minutes old.
    window = route.departure_window(risk_at, departures_min)
    card = route.route_card("persistence", t, window, risk_cfg["green_below"], risk_cfg["red_above"],
                            data_age_min=0.0, max_data_age_min=risk_cfg["max_data_age_min"])
    card.update({
        "label": BASELINE_LABEL,
        "note": (f"Persistence holds the {t:%H:%M} UTC storm mask fixed: it shows where storms are now, "
                 "not where they will be. It cannot see new storms form, so 'no' does not mean the "
                 "route will stay clear."),
        "storm_threshold_dbz": threshold_dbz,
        "buffer_km": buffer_km,
        "route_km": round(float(corridor.dist_km[-1]), 1),
        "flight_min": round(float(corridor.flight_min[-1]), 1),
    })
    return card


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
    route_cfg = settings()["route"]
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

    corridor = route.sample_route(route_cfg["suntrax_lonlat"], route_cfg["mco_lonlat"],
                                  route_cfg["cruise_speed_kmh"])
    cells = route.route_cells(corridor, box["lat_max"], box["lon_min"], (ny, nx))
    threshold_dbz = settings()["storms"]["threshold_dbz"]

    frames = []
    for frame, t in zip(dbz, times):
        # Every slot gets a route card, also one with no frame: there the card says "unavailable".
        route_name = f"route_{t:%H%M}.json"
        write_json(out_dir / route_name,
                   persistence_route_card(frame, t, corridor, cells, threshold_dbz, settings()["risk"],
                                          route_cfg["departures_min"]))
        if np.isnan(frame).all():       # the archive keeps a slot with no frame as all NaN
            frames.append({"time_utc": iso_utc(t), "obs": None, "route": route_name})
            continue
        name = f"obs_{t:%H%M}.png"
        mpl_image.imsave(out_dir / name, overlay.reflectivity_rgba(frame[rows]),
                         pil_kwargs={"optimize": True})
        frames.append({"time_utc": iso_utc(t), "obs": name, "route": route_name})

    names = [f["obs"] for f in frames if f["obs"]]
    route_names = [f["route"] for f in frames]
    if len(set(names)) != len(names) or len(set(route_names)) != len(route_names):
        raise ValueError(f"{day}: two frames share one HHMM, so one file overwrote another")

    write_json(out_dir / "index.json", {
        "mode": "replay",
        "day": day,
        "split": split,
        "source": f"NOAA MRMS {mrms.REFLECTIVITY}, s3://{mrms.BUCKET}",
        "built_utc": iso_utc(datetime.now(timezone.utc)),
        "bounds": {k: box[k] for k in ("lat_min", "lat_max", "lon_min", "lon_max")},
        "legend": [{"dbz": int(level), "rgb": hex_colour(rgb)}
                   for level, rgb in zip(overlay.DBZ_LEVELS, overlay.LEVEL_RGB)],
        "route": {"suntrax_lonlat": route_cfg["suntrax_lonlat"], "mco_lonlat": route_cfg["mco_lonlat"]},
        "frames": frames,
    })
    write_json(out_dir / "briefing.json", {"placeholder": True, "text": PLACEHOLDER_BRIEFING})
    write_json(REPLAY_DIR / "days.json",
               {"days": sorted(p.parent.name for p in REPLAY_DIR.glob("*/index.json"))})

    size_mb = sum(p.stat().st_size for p in out_dir.iterdir()) / 1e6
    print(f"{day} ({split or 'not in days.yaml'}): {len(names)} PNGs, {len(frames) - len(names)} slots "
          f"without a frame, {rows.size} x {nx} px each, {len(route_names)} route cards "
          f"({BASELINE_LABEL}), {size_mb:.1f} MB in {out_dir}")


if __name__ == "__main__":
    main()
