"""Early test 3, the buffer test: how often is the corridor blocked, for buffers of 5, 10 and 20 km?

    python scripts/check_buffer.py                      # the three default train days
    python scripts/check_buffer.py 20220615 20230705    # other train days
    python scripts/check_buffer.py --scan               # every train day on disk: the numbers to quote

Uses the OBSERVED radar, so this is what actually happened, not a forecast. For every afternoon
departure it flies the SunTrax -> MCO line (route.* in settings.yaml) and asks, for each buffer
in checks.buffer_test_km: was any route point, when the aircraft got there, within the buffer of
a storm cell (storms.threshold_dbz or more)? Each route point is checked against the radar frame
nearest to its arrival time, so one 15-minute flight uses two or three frames.

Observed radar is one outcome, not an ensemble: the route risk of each flight is 0 or 1. A flight
with risk 1 is called no-go here. The share of no-go departures per buffer is the number the
test is for: it shows what a larger buffer costs in lost departures.

Two more numbers per buffer say what a no-go costs the operator:

    waiting helps   of the no-go afternoon departures, the share with a go departure within
                    the window the route card offers: up to the last entry of
                    route.departures_min (40 minutes, the next four slots). This is hindsight from
                    observed radar: the most that a perfect departure-window forecast could offer.
                    The later departure may lie after the afternoon window.
    days            how many afternoons were no-go for every departure, go for every departure,
                    or mixed. A mixed afternoon is one where the time of departure matters.

Which numbers to quote: the --scan ones. The three default days were chosen BY a result of this
test (the first three train days with at least a quarter of afternoon departures no-go at 20 km),
so they are stormy afternoons by construction and overstate how often the corridor is blocked.
They are for reading the per-departure table. --scan covers every train day on disk, quiet days
included.

Departures leave at the time of each radar frame (every 10 minutes) from 12:00 to 18:00 local
solar time at the middle of the corridor, about 17:26 to 23:26 UTC. Local solar time is UTC plus
one hour per 15 degrees of longitude east.

A departure is "no data" and left out of the shares if a frame it needs is missing or has no
radar coverage (NaN) anywhere within the largest buffer of the route: no coverage is not clear air.
A no-go departure with no go in that window and a "no data" departure among them cannot
be told either way; it is counted as "no data" and left out of the waiting-helps share.

Only train days are accepted. Validation and test days are refused.
"""

from __future__ import annotations

import argparse
from datetime import datetime
from pathlib import Path

import numpy as np
import yaml

from stormlattice import archive, route
from stormlattice.config import data_dir, settings

REPO_ROOT = Path(__file__).resolve().parents[1]
DAYS_FILE = REPO_ROOT / "config" / "days.yaml"

# The first three train days by date on which at least a quarter of the afternoon departures were
# no-go at 20 km (--scan, 2026-11-16). That is a selection by result: quote the --scan numbers.
DEFAULT_DAYS = ["20220601", "20220603", "20220605"]

AFTERNOON_SOLAR_H = (12.0, 18.0)    # departures from 12:00 up to, not including, 18:00 local solar time


def solar_hour(t: datetime, lon_deg: float) -> float:
    """Local solar time at a longitude, in hours 0..24: UTC plus one hour per 15 degrees east."""
    return (t.hour + t.minute / 60 + t.second / 3600 + lon_deg / 15) % 24


def observed_flights(dbz, times, corridor, cells, mid_lon, buffers_km, threshold_dbz, max_gap_min, n_wait):
    """
    Route risk from observed radar for every afternoon departure of one storm day.

    Returns a list of (departure time, [risk per buffer], later). Each risk is 0.0 or 1.0, or
    None for every buffer if a frame the flight needs is missing or has NaN near the route.
    `later` holds the same [risk per buffer] list for each of the next n_wait slots, which is
    what the aircraft would have met had it waited.
    """
    # NaN (no radar coverage) compares False, so it is never a storm cell.
    # TODO: use storms.storm_mask once it is written; this is the rule in its docstring.
    storm = dbz >= threshold_dbz
    near = route.footprint(cells, max(buffers_km), dbz.shape[1:])
    usable = ~np.isnan(dbz[:, near]).any(axis=1)        # (T,) full coverage near the route

    def risks_at(slot):
        """[risk per buffer] for a departure at the frame time of `slot`."""
        if slot >= len(times):          # past the end of the storm day
            return [None] * len(buffers_km)
        steps = route.arrival_steps(times, times[slot], corridor.flight_min, max_gap_min)
        if steps.min() < 0 or not usable[steps].all():
            return [None] * len(buffers_km)
        return [route.route_risk(storm, cells, steps, buffer_km) for buffer_km in buffers_km]

    afternoon = [slot for slot, depart in enumerate(times)
                 if AFTERNOON_SOLAR_H[0] <= solar_hour(depart, mid_lon) < AFTERNOON_SOLAR_H[1]]
    if not afternoon:
        return []
    # Each slot is flown once, also the n_wait slots after the last afternoon departure.
    risks = {slot: risks_at(slot) for slot in range(afternoon[0], afternoon[-1] + n_wait + 1)}
    return [(times[slot], risks[slot], [risks[slot + k] for k in range(1, n_wait + 1)]) for slot in afternoon]


def no_go_shares(flights, n_buffers):
    """Flights with data, and for each buffer the share of them that were no-go (nan if there are none)."""
    scored = [risks for _, risks, _ in flights if risks[0] is not None]
    if not scored:
        return 0, [float("nan")] * n_buffers
    return len(scored), [sum(risks[i] for risks in scored) / len(scored) for i in range(n_buffers)]


def waiting_helps(flights, i):
    """
    For buffer number i: (no-go departures, those with a go departure among the later slots,
    those that cannot be told because a later departure has no data and none of them is a go).
    """
    n_no_go = n_helped = n_unknown = 0
    for _, risks, later in flights:
        if risks[i] != 1.0:
            continue
        n_no_go += 1
        if any(r[i] == 0.0 for r in later):
            n_helped += 1
        elif any(r[i] is None for r in later):
            n_unknown += 1
    return n_no_go, n_helped, n_unknown


def afternoon_kind(flights, i):
    """One day's afternoon at buffer number i: "all no-go", "all go", "mixed", or "no data"."""
    scored = {risks[i] for _, risks, _ in flights if risks[i] is not None}
    if not scored:
        return "no data"
    if len(scored) == 2:
        return "mixed"
    return "all no-go" if scored == {1.0} else "all go"


def main():
    parser = argparse.ArgumentParser(description="Buffer test on observed radar (train days only)")
    parser.add_argument("days", nargs="*", help=f"train storm days, YYYYMMDD (default: {' '.join(DEFAULT_DAYS)})")
    parser.add_argument("--scan", action="store_true",
                        help="print only the no-go shares, for every train day on disk")
    args = parser.parse_args()

    cfg = settings()
    box, rt = cfg["grid"], cfg["route"]
    buffers_km = cfg["checks"]["buffer_test_km"]
    threshold_dbz = cfg["storms"]["threshold_dbz"]
    # The furthest an arrival can be from its nearest frame when no slot is missing: half the
    # 10-minute cadence, plus the most a frame may sit off its 10-minute mark.
    max_gap_min = cfg["archive"]["cadence_min"] / 2 + cfg["archive"]["max_frame_offset_min"]
    # "Waiting helps" looks as far ahead as the route card's departure window. Slots, not clock
    # time: frame times jitter by a few seconds, so "40 minutes later" by the clock could miss
    # the fourth frame.
    wait_min = max(cfg["route"]["departures_min"])
    n_wait = wait_min // cfg["archive"]["cadence_min"]

    train_days = yaml.safe_load(DAYS_FILE.read_text())["train"]
    frames_dir = data_dir() / "frames"
    if args.scan:
        days = [d for d in train_days if (frames_dir / f"{d}.npz").exists()]
    else:
        days = args.days or DEFAULT_DAYS
        not_train = [d for d in days if d not in train_days]
        if not_train:
            raise SystemExit(f"{', '.join(not_train)}: not in the train list of config/days.yaml. "
                             "The buffer test runs on train days only.")

    corridor = route.sample_route(rt["suntrax_lonlat"], rt["mco_lonlat"], rt["cruise_speed_kmh"])
    mid_lon = (rt["suntrax_lonlat"][0] + rt["mco_lonlat"][0]) / 2

    print(f"Buffer test on observed radar. Storm cell: {threshold_dbz} dBZ or more. "
          f"SunTrax -> MCO: {corridor.dist_km[-1]:.1f} km, "
          f"{corridor.flight_min[-1]:.1f} min at {rt['cruise_speed_kmh']} km/h.")
    print(f"Departures at every radar frame from {AFTERNOON_SOLAR_H[0]:.0f}:00 to "
          f"{AFTERNOON_SOLAR_H[1]:.0f}:00 local solar time at the corridor.")
    print("Each flight either met a storm within the buffer (NO-GO) or did not (go). "
          "This is what happened, not a forecast.")
    buffer_heads = "".join(f"{f'{b} km':>8s}" for b in buffers_km)

    summary = []            # (day, flights with data, flights without, [share per buffer])
    every_flight = []
    kinds = []              # per day: [afternoon_kind per buffer]
    for day in days:
        dbz, times = archive.load_day(frames_dir / f"{day}.npz")
        cells = route.route_cells(corridor, box["lat_max"], box["lon_min"], dbz.shape[1:])
        flights = observed_flights(dbz, times, corridor, cells, mid_lon, buffers_km, threshold_dbz,
                                   max_gap_min, n_wait)
        every_flight += flights
        kinds.append([afternoon_kind(flights, i) for i in range(len(buffers_km))])
        n_scored, shares = no_go_shares(flights, len(buffers_km))
        summary.append((day, n_scored, len(flights) - n_scored, shares))

        if args.scan:
            continue
        print()
        print(f"{'day':9s} {'depart UTC':>10s} {'solar':>6s}{buffer_heads}")
        for depart, risks, _ in flights:
            hours = solar_hour(depart, mid_lon)
            solar = f"{int(hours):02d}:{int(hours % 1 * 60):02d}"
            words = ["no data" if r is None else "NO-GO" if r == 1.0 else "go" for r in risks]
            print(f"{day:9s} {depart:%H:%M}{'':5s} {solar:>6s}" + "".join(f"{w:>8s}" for w in words))

    print()
    print("Share of departures that would be no-go")
    print(f"{'day':9s} {'flights':>8s} {'no data':>8s}{buffer_heads}")
    for day, n_scored, n_missing, shares in summary:
        print(f"{day:9s} {n_scored:8d} {n_missing:8d}" + "".join(f"{s:8.0%}" for s in shares))
    n_scored, shares = no_go_shares(every_flight, len(buffers_km))
    print(f"{'all':9s} {n_scored:8d} {len(every_flight) - n_scored:8d}" + "".join(f"{s:8.0%}" for s in shares))

    print()
    print(f"Waiting helps, in hindsight: no-go departures with a go departure within the next {wait_min} min")
    print(f"{'buffer':9s} {'no-go':>8s} {'no data':>8s} {'helped':>8s} {'share':>8s}")
    for i, buffer_km in enumerate(buffers_km):
        n_no_go, n_helped, n_unknown = waiting_helps(every_flight, i)
        n_told = n_no_go - n_unknown
        share = f"{n_helped / n_told:8.0%}" if n_told else f"{'-':>8s}"
        print(f"{f'{buffer_km} km':9s} {n_no_go:8d} {n_unknown:8d} {n_helped:8d}{share}")

    print()
    print(f"Days by afternoon ({len(days)} days)")
    print(f"{'buffer':9s} {'all no-go':>10s} {'mixed':>8s} {'all go':>8s} {'no data':>8s}")
    for i, buffer_km in enumerate(buffers_km):
        count = [sum(day_kinds[i] == kind for day_kinds in kinds)
                 for kind in ("all no-go", "mixed", "all go", "no data")]
        print(f"{f'{buffer_km} km':9s} {count[0]:10d} {count[1]:8d} {count[2]:8d} {count[3]:8d}")


if __name__ == "__main__":
    main()
