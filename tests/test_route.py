"""Tests for route sampling, arrival times and route risk, on tiny hand-made grids."""

from __future__ import annotations

from datetime import datetime, timedelta, timezone

import numpy as np
import pytest

from stormlattice import grid, route
from stormlattice.config import settings

# A tiny grid near the corridor: rows go south from 28.50 N, columns east from 82.00 W.
LAT_MAX, LON_MIN = 28.50, -82.00
SHAPE = (30, 30)
T0 = datetime(2022, 6, 1, 18, 0, tzinfo=timezone.utc)

# A route along row 10: one point in each of columns 5 .. 15.
ROW_CELLS = np.array([[10, c] for c in range(5, 16)])


def masks(n_frames, storms):
    """(T, ny, nx) storm masks, all clear except the listed (frame, row, column) cells."""
    m = np.zeros((n_frames,) + SHAPE, dtype=bool)
    for frame, r, c in storms:
        m[frame, r, c] = True
    return m


# --- sample_route ---

def test_route_due_north_is_sampled_every_km():
    # 0.10 deg of latitude = 5 cells north-south = 5 * 2.216 = 11.08 km
    r = route.sample_route([-81.5, 28.0], [-81.5, 28.1], cruise_speed_kmh=240)
    length_km = 5 * grid.dy_km(28.05)
    assert length_km == pytest.approx(11.08, abs=0.01)
    assert np.allclose(r.dist_km[:-1], np.arange(12))       # 0, 1, ..., 11 km, then the end point
    assert r.dist_km[-1] == pytest.approx(length_km)
    assert np.allclose(r.lons, -81.5)
    assert r.lats[0] == 28.0 and r.lats[-1] == pytest.approx(28.1)
    # 240 km/h is 4 km a minute
    assert np.allclose(r.flight_min[:3], [0.0, 0.25, 0.5])
    assert r.flight_min[-1] == pytest.approx(length_km / 4)


def test_route_due_east_uses_the_east_west_cell_size():
    # 0.10 deg of longitude = 5 cells east-west, which are narrower than they are tall
    r = route.sample_route([-81.6, 28.0], [-81.5, 28.0], cruise_speed_kmh=240)
    assert r.dist_km[-1] == pytest.approx(5 * grid.dx_km(28.0))
    assert r.dist_km[-1] < 5 * grid.DY_KM


def test_route_whose_ends_are_the_same_point():
    r = route.sample_route([-81.5, 28.0], [-81.5, 28.0], cruise_speed_kmh=240)
    assert r.dist_km.tolist() == [0.0] and r.flight_min.tolist() == [0.0]


def test_corridor_is_about_a_15_minute_flight():
    # the SunTrax and MCO values in settings.yaml; docs/design.md says "about 15 minutes"
    r = route.sample_route([-81.82897, 28.11423], [-81.31046, 28.42444], cruise_speed_kmh=241)
    assert r.dist_km[-1] == pytest.approx(61.4, abs=0.5)
    assert r.flight_min[-1] == pytest.approx(15.3, abs=0.2)
    assert len(r.dist_km) == 63                             # 0 .. 61 km and the end point


# --- route_cells ---

def test_route_cells_row_0_is_north_and_column_0_is_west():
    r = route.Route(lats=np.array([28.49, 28.47, 27.91]), lons=np.array([-81.99, -81.95, -81.41]),
                    dist_km=np.zeros(3), flight_min=np.zeros(3))
    cells = route.route_cells(r, LAT_MAX, LON_MIN, SHAPE)
    assert cells.tolist() == [[0, 0], [1, 2], [29, 29]]


def test_route_outside_the_grid_is_an_error():
    r = route.Route(lats=np.array([28.60]), lons=np.array([-81.99]), dist_km=np.zeros(1), flight_min=np.zeros(1))
    with pytest.raises(ValueError):
        route.route_cells(r, LAT_MAX, LON_MIN, SHAPE)       # 28.60 N is north of the grid


# --- arrival_steps ---

def test_arrival_takes_the_nearest_frame():
    frames = [T0, T0 + timedelta(minutes=10), T0 + timedelta(minutes=20)]
    flight_min = np.array([0.0, 4.0, 6.0, 14.0, 16.0])
    assert route.arrival_steps(frames, T0, flight_min).tolist() == [0, 0, 1, 1, 2]
    # Leaving 10 minutes later, the arrivals are at +10, +14, +16, +24 and +26 min. The last one
    # is 6 minutes past the final frame (+20), more than the 5 allowed.
    later = route.arrival_steps(frames, T0 + timedelta(minutes=10), flight_min, max_gap_min=5)
    assert later.tolist() == [1, 1, 2, 2, -1]


def test_arrival_exactly_between_two_frames_takes_the_earlier():
    frames = [T0, T0 + timedelta(minutes=10)]
    assert route.arrival_steps(frames, T0, np.array([5.0])).tolist() == [0]


def test_no_gap_limit_holds_one_frame_for_any_departure():
    # persistence: one frame, held fixed however late the aircraft leaves
    steps = route.arrival_steps([T0], T0 + timedelta(minutes=60), np.array([0.0, 15.0]))
    assert steps.tolist() == [0, 0]


# --- footprint ---

def test_footprint_of_one_cell_at_5_km():
    # Cells are 2.216 km tall and 1.978 km wide. Within 5 km of a cell centre: 2 cells north or
    # south (4.43 km), 2 east or west (3.96 km), 2 and 1 (4.85 or 4.54 km); not 2 and 2 (5.94 km)
    # and not 3 in any direction (5.93 km or more).
    near = route.footprint(np.array([[10, 10]]), buffer_km=5, shape=SHAPE)
    assert near[8, 10] and near[10, 12] and near[8, 11] and near[9, 12]
    assert not near[8, 12] and not near[7, 10] and not near[10, 13]
    assert near.sum() == 21                                 # the 5 x 5 square without its 4 corners


def test_footprint_is_cut_off_at_the_grid_edge():
    near = route.footprint(np.array([[0, 0]]), buffer_km=5, shape=SHAPE)
    assert near[0, 0] and near[2, 0] and near[0, 2] and not near[2, 2]
    assert near.sum() == 8                                  # one quarter of the disk, with its axes


# --- route_risk ---

def test_storm_three_cells_north_fails_at_10_km_but_not_at_5_km():
    # (7, 10) is 3 rows north of the route: 3 * 2.216 = 6.65 km
    states = masks(1, [(0, 7, 10)])
    steps = np.zeros(len(ROW_CELLS), dtype=int)
    assert route.route_risk(states, ROW_CELLS, steps, buffer_km=5) == 0.0
    assert route.route_risk(states, ROW_CELLS, steps, buffer_km=10) == 1.0


def test_clear_sky_is_zero_risk():
    steps = np.zeros(len(ROW_CELLS), dtype=int)
    assert route.route_risk(masks(1, []), ROW_CELLS, steps, buffer_km=20) == 0.0


def test_storm_only_counts_in_the_frame_the_aircraft_is_there():
    # The storm sits on the east end of the route (column 15), 10 columns = 19.8 km from the west
    # end, and exists only in frame 1.
    states = masks(2, [(1, 10, 15)])
    in_frame_0 = np.zeros(len(ROW_CELLS), dtype=int)
    assert route.route_risk(states, ROW_CELLS, in_frame_0, buffer_km=5) == 0.0     # flown before it forms
    east_half_in_frame_1 = np.array([0] * 6 + [1] * 5)
    assert route.route_risk(states, ROW_CELLS, east_half_in_frame_1, buffer_km=5) == 1.0
    # west end in frame 1, east end already passed in frame 0: the storm is 5 columns = 9.9 km
    # from the nearest cell flown in frame 1
    west_half_in_frame_1 = np.array([1] * 6 + [0] * 5)
    assert route.route_risk(states, ROW_CELLS, west_half_in_frame_1, buffer_km=5) == 0.0
    assert route.route_risk(states, ROW_CELLS, west_half_in_frame_1, buffer_km=10) == 1.0


def test_risk_is_the_fraction_of_members_that_fail():
    # 4 members: member 1 has a storm on the route, member 3 has two (still one failed flight)
    states = np.zeros((1, 4) + SHAPE, dtype=bool)           # (T, M, ny, nx)
    states[0, 1, 10, 8] = True
    states[0, 3, 10, 6] = True
    states[0, 3, 10, 14] = True
    steps = np.zeros(len(ROW_CELLS), dtype=int)
    assert route.route_risk(states, ROW_CELLS, steps, buffer_km=5) == 0.5


def test_a_step_with_no_frame_is_an_error_not_clear_sky():
    # -1 would otherwise read the LAST frame (NumPy counts negative indices from the end)
    steps = np.full(len(ROW_CELLS), -1)
    with pytest.raises(ValueError):
        route.route_risk(masks(2, []), ROW_CELLS, steps, buffer_km=5)


# --- departure window and classify ---

def test_departure_window_asks_once_per_departure():
    window = route.departure_window(lambda offset_min: offset_min / 100, [0, 10, 20])
    assert window == [(0, 0.0), (10, 0.1), (20, 0.2)]


def test_departure_window_end_to_end_storm_arrives_later():
    # Three frames 10 minutes apart; a storm sits on the route's east end from frame 1 on.
    # A 4-minute flight along the row: leaving at T0 it is flown in frame 0 (clear),
    # leaving 10 or 20 minutes later it meets the storm.
    frames = [T0, T0 + timedelta(minutes=10), T0 + timedelta(minutes=20)]
    states = masks(3, [(1, 10, 15), (2, 10, 15)])
    flight_min = np.linspace(0.0, 4.0, len(ROW_CELLS))

    def risk(offset_min):
        steps = route.arrival_steps(frames, T0 + timedelta(minutes=offset_min), flight_min, max_gap_min=5)
        return route.route_risk(states, ROW_CELLS, steps, buffer_km=5)

    assert route.departure_window(risk, [0, 10, 20]) == [(0, 0.0), (10, 1.0), (20, 1.0)]


def test_last_departure_lands_inside_the_forecast_horizon():
    # settings.yaml: the last departure (+40 min) plus the flight (15.3 min) must not need a
    # forecast beyond the longest lead time (60 min)
    s = settings()
    corridor = route.sample_route(s["route"]["suntrax_lonlat"], s["route"]["mco_lonlat"],
                                  s["route"]["cruise_speed_kmh"])
    last_arrival_min = max(s["route"]["departures_min"]) + corridor.flight_min[-1]
    assert last_arrival_min <= max(s["forecast"]["lead_times_min"])
    assert s["route"]["departures_min"][0] == 0             # the card's headline is "leave now"


def test_classify_thresholds_themselves_are_amber():
    # green is strictly below green_below, red strictly above red_above
    assert route.classify(0.10, 0.10, 0.30, 0, 10, "t").level == "amber"
    assert route.classify(0.30, 0.10, 0.30, 0, 10, "t").level == "amber"
    assert route.classify(0.0, 0.10, 0.30, 0, 10, "t").level == "green"
    assert route.classify(1.0, 0.10, 0.30, 0, 10, "t").level == "red"


def test_classify_without_a_risk_is_unavailable():
    advisory = route.classify(None, 0.10, 0.30, 0, 10, "t")
    assert advisory.level == "unavailable" and advisory.risk is None


# --- route_card ---

def card(method, window, data_age_min=0):
    return route.route_card(method, T0, window, green_below=0.10, red_above=0.30,
                            data_age_min=data_age_min, max_data_age_min=10)


def test_persistence_card_is_an_observation_never_a_green():
    # No storm near the route now. A forecast would call 0.0 "green"; persistence must not.
    clear = card("persistence", [(0, 0.0), (10, 0.0)])
    assert clear == {"method": "persistence", "issue_time_utc": "2022-06-01T18:00:00Z",
                     "level": "observed", "storms_near_route": False}
    stormy = card("persistence", [(0, np.float64(1.0)), (10, np.float64(1.0))])
    assert stormy["level"] == "observed" and stormy["storms_near_route"] is True    # a Python bool
    for c in (clear, stormy):
        assert "risk" not in c and "departures" not in c and "thresholds" not in c


def test_persistence_card_without_data_is_unavailable():
    missing = card("persistence", [(0, None), (10, None)])
    assert missing["level"] == "unavailable" and missing["storms_near_route"] is None
    assert "treat as no-go" in missing["reason"] and "risk" not in missing
    # a clear frame that is 15 minutes old (limit 10) is not "no storms": it is unavailable
    stale = card("persistence", [(0, 0.0)], data_age_min=15)
    assert stale["level"] == "unavailable" and stale["storms_near_route"] is None


def test_forecast_card_has_risk_level_and_departure_window():
    c = card("lattice", [(0, 0.05), (10, 0.20), (20, 0.45)])
    assert c["method"] == "lattice" and c["risk"] == 0.05 and c["level"] == "green"
    assert c["thresholds"] == {"green_below": 0.10, "red_above": 0.30}
    assert c["departures"] == [
        {"offset_min": 0, "depart_utc": "2022-06-01T18:00:00Z", "risk": 0.05, "level": "green"},
        {"offset_min": 10, "depart_utc": "2022-06-01T18:10:00Z", "risk": 0.20, "level": "amber"},
        {"offset_min": 20, "depart_utc": "2022-06-01T18:20:00Z", "risk": 0.45, "level": "red"},
    ]
    assert "storms_near_route" not in c


def test_stale_forecast_card_carries_no_numbers():
    c = card("lattice", [(0, 0.05), (10, 0.20)], data_age_min=15)
    assert c["risk"] is None and c["level"] == "unavailable"
    assert [d["risk"] for d in c["departures"]] == [None, None]
    assert [d["level"] for d in c["departures"]] == ["unavailable", "unavailable"]


def test_unknown_method_is_an_error_not_a_forecast():
    # a method must be listed in FORECAST_METHODS on purpose before it may show a risk and a level
    assert route.FORECAST_METHODS == ("advection", "lattice")
    with pytest.raises(ValueError, match="climatology"):
        card("climatology", [(0, 0.05)])
