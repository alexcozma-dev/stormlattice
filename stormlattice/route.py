"""Step 7: route risk from the ensemble, and the departure window.

Risk = fraction of ensemble members in which ANY route point, at its arrival time, lies within
`buffer_km` of an active storm. Counting members (not multiplying per-point probabilities)
respects the strong correlation between neighboring points.

The route is a straight line in lon/lat between two points, sampled every 1 km. Each point has a
flight time from the cruise speed, so a storm that forms near the far end only matters if it is
there when the aircraft is.

Units: distances in km, flight times and gaps in minutes, positions in degrees. Times are
timezone-aware UTC datetimes. Storm masks are bool arrays, True = storm cell: (T, ny, nx) for one
future (observed radar, persistence) or (T, M, ny, nx) for an ensemble of M members.

"Within buffer_km" is measured between cell centres, from the cell that holds the route point.
A route point can sit up to half a cell (about 1.5 km) from its cell centre, so a buffer is
good to about that distance.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timedelta

import numpy as np

from stormlattice import geography, grid

# Methods whose route card is a forecast: risk, level and departure window (route_card).
# The probabilistic-advection and neighbourhood benchmarks join when they exist.
FORECAST_METHODS = ("advection", "lattice")


@dataclass
class Route:
    lats: np.ndarray            # (P,) degrees, one entry per route point, start first
    lons: np.ndarray            # (P,) degrees
    dist_km: np.ndarray         # (P,) distance flown from the start
    flight_min: np.ndarray      # (P,) minutes after departure at which the aircraft is here


@dataclass
class Advisory:
    risk: float | None          # None = advisory unavailable
    level: str                  # "green", "amber", "red" or "unavailable"
    data_time_utc: str
    reason: str


def sample_route(start_lonlat, end_lonlat, cruise_speed_kmh: float, step_km: float = 1.0) -> Route:
    """
    The straight line from start to end, sampled every step_km, with the flight time to each point.

    Both ends are [lon, lat] in degrees, as settings.yaml stores them. The points sit at 0,
    step_km, 2 * step_km, ... from the start, and the end point itself is always the last one.
    """
    lon0, lat0 = start_lonlat
    lon1, lat1 = end_lonlat
    # Degrees -> km with grid.py's cell sizes at the route's middle latitude. Over the 61 km
    # corridor the east-west cell size changes by 0.2%, so one latitude is enough.
    mid_lat = (lat0 + lat1) / 2
    north_km = (lat1 - lat0) / grid.CELL_DEG * grid.dy_km(mid_lat)
    east_km = (lon1 - lon0) / grid.CELL_DEG * grid.dx_km(mid_lat)
    length_km = float(np.hypot(north_km, east_km))

    dist_km = np.append(np.arange(0.0, length_km, step_km), length_km)
    along = dist_km / length_km if length_km > 0 else np.zeros(1)       # 0 at the start, 1 at the end
    return Route(lats=lat0 + along * (lat1 - lat0),
                 lons=lon0 + along * (lon1 - lon0),
                 dist_km=dist_km,
                 flight_min=dist_km / cruise_speed_kmh * 60.0)


def route_cells(route: Route, lat_max: float, lon_min: float, shape: tuple[int, int]) -> np.ndarray:
    """
    The cell that holds each route point: (P, 2) int array of (row, column).

    lat_max and lon_min are the north and west edges of the grid and shape is its (ny, nx).
    Raises ValueError if a point lies outside the grid: there is no radar there to check.
    """
    # TODO: this lon/lat -> cell conversion (both lines) moves to the lon/lat <-> cell helper
    # planned for grid.py.
    rows = np.floor((lat_max - route.lats) / grid.CELL_DEG).astype(int)
    cols = np.floor((route.lons - lon_min) / grid.CELL_DEG).astype(int)
    ny, nx = shape
    if rows.min() < 0 or rows.max() >= ny or cols.min() < 0 or cols.max() >= nx:
        raise ValueError("the route leaves the grid")
    return np.stack([rows, cols], axis=1)


def arrival_steps(frame_times: list[datetime], depart: datetime, flight_min: np.ndarray,
                  max_gap_min: float | None = None) -> np.ndarray:
    """
    For each route point, the index of the frame nearest in time to the aircraft's arrival: (P,) int.

    frame_times are the UTC times of the T frames and depart is the departure time. Where the
    nearest frame is more than max_gap_min from the arrival the entry is -1 (no frame says what
    the weather is then). max_gap_min=None sets no limit; persistence uses that, because it holds
    one frame for the whole flight. An arrival exactly between two frames takes the earlier one.
    """
    frame_s = np.array([t.timestamp() for t in frame_times])
    arrival_s = depart.timestamp() + np.asarray(flight_min) * 60.0
    gap_s = np.abs(arrival_s[:, None] - frame_s[None, :])       # (P, T)
    steps = gap_s.argmin(axis=1)
    if max_gap_min is not None:
        steps[gap_s.min(axis=1) > max_gap_min * 60.0] = -1
    return steps


def footprint(cells: np.ndarray, buffer_km: float, shape: tuple[int, int]) -> np.ndarray:
    """Every cell within buffer_km of one of `cells`, a (P, 2) array of (row, column): (ny, nx) bool."""
    disk = geography.disk(buffer_km)        # TODO: move to grid.py
    ry, rx = disk.shape[0] // 2, disk.shape[1] // 2
    ny, nx = shape
    near = np.zeros(shape, dtype=bool)
    for r, c in cells:
        # Stamp the disk around this cell, cut off where it would leave the grid.
        r0, r1 = max(r - ry, 0), min(r + ry + 1, ny)
        c0, c1 = max(c - rx, 0), min(c + rx + 1, nx)
        near[r0:r1, c0:c1] |= disk[r0 - r + ry:r1 - r + ry, c0 - c + rx:c1 - c + rx]
    return near


def route_risk(states: np.ndarray, cells: np.ndarray, steps: np.ndarray, buffer_km: float) -> float:
    """
    Fraction of members whose flight meets a storm: 0.0 to 1.0.

    states: storm masks, (T, M, ny, nx) bool, or (T, ny, nx) for a single member, where the
            answer can only be 0.0 or 1.0.
    cells:  (P, 2) row and column of each route point (route_cells).
    steps:  (P,) index into T of the frame that applies when the aircraft reaches each point
            (arrival_steps). Raises ValueError on an index outside 0 .. T-1, such as the -1
            that arrival_steps gives where no frame is close enough.
    A member fails if any route point, in the frame of its arrival, is within buffer_km of a
    storm cell. A flight that fails at several points still counts once.
    """
    states = np.asarray(states, dtype=bool)
    if states.ndim == 3:
        states = states[:, None]            # one member
    if states.ndim != 4:
        raise ValueError(f"states must be (T, ny, nx) or (T, M, ny, nx), got shape {states.shape}")
    n_steps, n_members = states.shape[:2]
    if steps.min() < 0 or steps.max() >= n_steps:
        raise ValueError(f"steps must index the {n_steps} frames; got {steps.min()} .. {steps.max()}")

    failed = np.zeros(n_members, dtype=bool)
    for step in np.unique(steps):
        # The part of the route flown during this frame, widened by the buffer.
        near = footprint(cells[steps == step], buffer_km, states.shape[2:])
        failed |= states[step][:, near].any(axis=1)
    return float(failed.mean())


def classify(risk: float | None, green_below: float, red_above: float, data_age_min: float,
             max_data_age_min: float, data_time_utc: str) -> Advisory:
    """Turn a risk number into an advisory. Stale or missing data -> unavailable (treat as no-go)."""
    if risk is None or data_age_min > max_data_age_min:
        return Advisory(None, "unavailable", data_time_utc,
                        "Data missing or stale: advisory unavailable, treat as no-go.")
    if risk < green_below:
        level = "green"
    elif risk > red_above:
        level = "red"
    else:
        level = "amber"
    return Advisory(risk, level, data_time_utc, f"{risk:.0%} of ensemble members meet a storm.")


def departure_window(risk_fn, departures_min: list[int]) -> list[tuple[int, float]]:
    """Risk for each candidate departure offset (minutes from now)."""
    return [(d, risk_fn(d)) for d in departures_min]


def route_card(method: str, issue_time: datetime, window: list[tuple[int, float | None]],
               green_below: float, red_above: float, data_age_min: float, max_data_age_min: float) -> dict:
    """
    The part of a route card that depends on the method, as a dict that can be written as JSON.

    window is what departure_window returns: (minutes after issue_time, risk or None) per
    departure. Its first entry is the headline, "leave now".

    Stale or missing data comes first, for every method: level "unavailable" and no number.

    method "persistence" holds one observed frame fixed. That says where storms are, not where
    they will be, so it is reported as an observation: level "observed" and storms_near_route
    True or False. The dict has no "risk", no "thresholds" and no "departures", so nothing that
    reads it later can take "no storm now" for a green forecast.
        {"method", "issue_time_utc", "level", "storms_near_route"}          and "reason" if unavailable

    A method in FORECAST_METHODS gets the full card:
        {"method", "issue_time_utc", "risk", "level", "reason", "thresholds", "departures"}
    with one {"offset_min", "depart_utc", "risk", "level"} per departure.

    Any other method raises ValueError: a new method must be listed as a forecast on purpose.
    """
    if method != "persistence" and method not in FORECAST_METHODS:
        raise ValueError(f"unknown method {method!r}: use 'persistence' or one of {FORECAST_METHODS}")

    def utc_text(t):
        return t.strftime("%Y-%m-%dT%H:%M:%SZ")

    def advisory(risk):
        return classify(risk, green_below, red_above, data_age_min, max_data_age_min, utc_text(issue_time))

    now = advisory(window[0][1])
    card = {"method": method, "issue_time_utc": utc_text(issue_time)}
    if method == "persistence":
        if now.level == "unavailable":
            return {**card, "level": "unavailable", "storms_near_route": None, "reason": now.reason}
        # bool(): a plain Python bool for the JSON file, also if the risk is a NumPy float.
        return {**card, "level": "observed", "storms_near_route": bool(now.risk > 0)}

    departures = []
    for offset_min, risk in window:
        later = advisory(risk)          # stale data gives risk None here too, never a stale number
        departures.append({"offset_min": offset_min,
                           "depart_utc": utc_text(issue_time + timedelta(minutes=offset_min)),
                           "risk": later.risk, "level": later.level})
    return {**card, "risk": now.risk, "level": now.level, "reason": now.reason,
            "thresholds": {"green_below": green_below, "red_above": red_above},
            "departures": departures}
