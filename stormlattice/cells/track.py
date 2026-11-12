"""Step 2: track cells between frames and label births and deaths.

1. Advect the previous frame's cells by the motion field (Step 3).
2. Match predicted to observed cells with the Hungarian algorithm
   (scipy.optimize.linear_sum_assignment) on centroid distance + overlap.
3. Unmatched new cell -> birth (unless it split from a neighbor).
   Unmatched old cell -> death (unless it merged into a neighbor).
"""

from __future__ import annotations

from dataclasses import dataclass, field

from .detect import Cell


@dataclass
class TrackEvents:
    matches: dict[int, int] = field(default_factory=dict)   # old id -> new id
    births: list[int] = field(default_factory=list)          # new ids
    deaths: list[int] = field(default_factory=list)          # old ids
    splits: list[tuple[int, int]] = field(default_factory=list)
    merges: list[tuple[int, int]] = field(default_factory=list)


def match_frames(previous: list[Cell], current: list[Cell], max_distance_km: float) -> TrackEvents:
    """Match cells across one time step and classify births, deaths, splits and merges."""
    raise NotImplementedError
