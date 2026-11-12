"""Step 7: route risk from the ensemble, departure window, and reroute.

Risk = fraction of ensemble members in which ANY route point, at its arrival time, lies within
`buffer_km` of an active storm. Counting members (not multiplying per-point probabilities)
respects the strong correlation between neighboring points.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass
class Advisory:
    risk: float | None          # None = advisory unavailable
    level: str                  # "green", "amber", "red" or "unavailable"
    data_time_utc: str
    reason: str


def route_risk(states: np.ndarray, route_cells: np.ndarray, arrival_steps: np.ndarray, buffer_cells: int) -> float:
    """Fraction of members whose flight meets a storm. states: (T, M, ny, nx)."""
    raise NotImplementedError


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
