"""Tests for the static geography features, on tiny boxes with made-up shapes (no Natural Earth)."""
import os

import numpy as np
import pytest
import shapely

from stormlattice import geography, grid
from stormlattice.config import settings

# A tiny box near the middle of Florida: `ny` rows down from 28.00 N, `nx` columns east from 82.00 W.
LAT_MAX, LON_MIN = 28.00, -82.00
NOTHING = shapely.Polygon()                         # an empty shape: no lakes, or no Gulf
EVERYWHERE = shapely.box(-90.0, 20.0, -70.0, 35.0)


def tiny_box(ny, nx):
    """lat_min, lat_max, lon_min, lon_max of a box of ny x nx cells."""
    return LAT_MAX - ny * grid.CELL_DEG, LAT_MAX, LON_MIN, LON_MIN + nx * grid.CELL_DEG


def test_cell_centres():
    lats, lons = geography.cell_centres(*tiny_box(2, 3))
    assert np.allclose(lats, [27.99, 27.97])            # row 0 is the northern row
    assert np.allclose(lons, [-81.99, -81.97, -81.95])


def test_land_strip_has_a_gulf_coast_and_an_atlantic_coast():
    # 5 x 11 cells. Land is columns 4, 5, 6 (centres -81.91, -81.89, -81.87): a peninsula running
    # north-south with sea on both sides. Everything west of its middle is the Gulf side.
    land = shapely.box(-81.92, 20.0, -81.86, 35.0)
    gulf = shapely.box(-90.0, 20.0, -81.89, 35.0)
    f = geography.compute(land, NOTHING, gulf, *tiny_box(5, 11))

    assert f["land_mask"].shape == (5, 11)
    assert f["land_mask"][:, 4:7].all() and f["land_mask"].sum() == 15
    assert np.all(f["dist_gulf_km"][:, 4] == 0)                             # west shore
    assert np.all(f["dist_atlantic_km"][:, 6] == 0)                         # east shore
    assert np.allclose(f["dist_gulf_km"][:, 6], 2 * grid.DX_KM)             # across the land
    assert np.allclose(f["dist_atlantic_km"][:, 4], 2 * grid.DX_KM)
    assert np.allclose(f["dist_gulf_km"][:, 0], 4 * grid.DX_KM)             # out at sea: still positive
    assert np.allclose(f["dist_atlantic_km"][:, 10], 4 * grid.DX_KM)


def test_north_south_distance_uses_dy_and_missing_coast_is_inf():
    # 7 x 3 cells. Land is rows 4, 5, 6; the sea to the north is Atlantic (there is no Gulf side).
    land = shapely.box(-90.0, 20.0, -70.0, LAT_MAX - 4 * grid.CELL_DEG)
    f = geography.compute(land, NOTHING, NOTHING, *tiny_box(7, 3))

    assert np.all(f["dist_atlantic_km"][4] == 0)
    assert np.allclose(f["dist_atlantic_km"][6], 2 * grid.DY_KM)
    assert np.allclose(f["dist_atlantic_km"][0], 4 * grid.DY_KM)
    assert np.all(np.isinf(f["dist_gulf_km"]))


def test_island_between_two_seas_is_on_both_coasts():
    # The Keys in miniature: 5 x 5 cells of sea with one land cell in the middle. The two rows
    # north of it are Gulf water, the rest is Atlantic water.
    island = shapely.box(-81.96, 27.94, -81.94, 27.96)                     # cell (2, 2)
    gulf = shapely.box(-90.0, 27.96, -70.0, 35.0)                           # rows 0 and 1
    f = geography.compute(island, NOTHING, gulf, *tiny_box(5, 5))

    assert f["land_mask"].sum() == 1 and f["land_mask"][2, 2]
    assert f["dist_gulf_km"][2, 2] == 0 and f["dist_atlantic_km"][2, 2] == 0
    assert np.isclose(f["dist_gulf_km"][0, 2], 2 * grid.DY_KM)
    assert np.isclose(f["dist_atlantic_km"][2, 0], 2 * grid.DX_KM)


def test_lake_is_water_but_its_shore_is_not_a_coast():
    # 9 x 9 cells of land with a one-cell lake in the middle.
    lake = shapely.box(-81.92, 27.90, -81.90, 27.92)                       # cell (4, 4)
    f = geography.compute(EVERYWHERE, lake, EVERYWHERE, *tiny_box(9, 9))

    assert not f["land_mask"][4, 4] and f["land_mask"].sum() == 80
    assert np.all(np.isinf(f["dist_atlantic_km"])) and np.all(np.isinf(f["dist_gulf_km"]))
    # The 5 km disk holds 21 cells (see test_disk), and one of them is the lake.
    assert f["water_frac_5km"][4, 4] == pytest.approx(1 / 21, abs=1e-3)
    assert f["water_frac_5km"][4, 5] == pytest.approx(1 / 21, abs=1e-3)   # the lake's neighbour sees it too
    assert f["water_frac_5km"][0, 0] == 0                                   # too far away


def test_disk():
    # Cells are 2.216 km north-south and 1.98 km east-west. Within 5 km of the middle cell:
    # 5 cells in its own row, 5 in each row next to it, 3 in each row two away: 21.
    d = geography.disk(5)
    assert d.shape == (5, 5)
    assert d.sum(axis=1).tolist() == [3, 5, 5, 5, 3]
    assert geography.disk(20).shape == (19, 21)         # 9 rows up and down, 10 columns each way


def test_water_fraction_at_a_straight_shore():
    # 9 x 9 cells: land is columns 0-3, sea is columns 4-8. Stand on the shore cell (4, 3).
    # Of the 21 cells within 5 km, the ones east of its column are sea: 2 + 2 + 2 + 1 + 1 = 8.
    land = shapely.box(-90.0, 20.0, LON_MIN + 4 * grid.CELL_DEG, 35.0)
    f = geography.compute(land, NOTHING, NOTHING, *tiny_box(9, 9))

    # Not exactly 8/21: rows differ in area by a few hundredths of a percent.
    assert f["water_frac_5km"][4, 3] == pytest.approx(8 / 21, abs=1e-3)
    assert f["water_frac_5km"][4, 0] == 0               # 3 columns inland: no sea within 5 km
    assert f["water_frac_5km"][4, 8] == 1               # out at sea, at the box edge


def test_all_sea_and_all_land():
    sea = geography.compute(NOTHING, NOTHING, NOTHING, *tiny_box(6, 6))
    assert not sea["land_mask"].any()
    assert np.all(sea["water_frac_5km"] == 1) and np.all(sea["water_frac_20km"] == 1)   # corners too

    ground = geography.compute(EVERYWHERE, NOTHING, NOTHING, *tiny_box(6, 6))
    assert ground["land_mask"].all()
    assert np.all(ground["water_frac_5km"] == 0) and np.all(ground["water_frac_20km"] == 0)


def test_gulf_side_polygon_splits_the_water_where_the_map_does():
    gulf = shapely.Polygon(geography.GULF_SIDE_LONLAT)
    assert gulf.is_valid                    # the outline does not cross itself

    def is_gulf(lat, lon):
        return bool(shapely.contains_xy(gulf, lon, lat))

    assert is_gulf(27.60, -82.80)           # mouth of Tampa Bay
    assert is_gulf(29.50, -85.50)           # off Apalachicola
    assert is_gulf(25.00, -80.75)           # Florida Bay
    assert is_gulf(24.80, -81.40)           # the backcountry north of Big Pine Key
    assert is_gulf(24.80, -82.90)           # north of the Dry Tortugas
    assert not is_gulf(28.40, -80.30)       # off Cape Canaveral
    assert not is_gulf(25.50, -80.25)       # Biscayne Bay
    assert not is_gulf(25.22, -80.37)       # Barnes Sound
    assert not is_gulf(24.60, -81.10)       # Hawk Channel, south of Marathon
    assert not is_gulf(24.30, -81.80)       # Straits of Florida, south of Key West


def test_build_writes_a_cache_that_load_reads(tmp_path, monkeypatch):
    # Stand in for Natural Earth: the land strip from the first test, and no lakes.
    shapes = {"land": shapely.box(-81.92, 20.0, -81.86, 35.0), "lakes": NOTHING}
    monkeypatch.setattr(geography, "natural_earth_shape", lambda name, *box: shapes[name])
    path = tmp_path / "artifacts" / "geography.npz"         # the folder does not exist yet

    built = geography.build(path, *tiny_box(5, 11))
    loaded = geography.load(path)

    assert list(loaded) == geography.FEATURE_NAMES
    for name in geography.FEATURE_NAMES:
        assert loaded[name].shape == (5, 11)
        assert np.array_equal(loaded[name], built[name])
    assert loaded["land_mask"].dtype == bool and loaded["dist_gulf_km"].dtype == np.float32
    with np.load(path) as data:
        assert np.allclose(data["box"], tiny_box(5, 11))


# The real build reads Natural Earth, which cartopy downloads on first use, so it is off by default.
# Run it with: STORM_NETWORK_TESTS=1 pytest -s
network = pytest.mark.skipif(os.environ.get("STORM_NETWORK_TESTS") != "1",
                             reason="set STORM_NETWORK_TESTS=1 to build from Natural Earth")


@network
def test_real_florida(tmp_path):
    box = settings()["grid"]
    f = geography.build(tmp_path / "geography.npz", box["lat_min"], box["lat_max"], box["lon_min"], box["lon_max"])

    def at(name, lat, lon):
        return f[name][int((box["lat_max"] - lat) / grid.CELL_DEG), int((lon - box["lon_min"]) / grid.CELL_DEG)]

    assert f["land_mask"].shape == (360, 456)
    orlando, tampa, key_west, okeechobee = (28.538, -81.379), (27.951, -82.457), (24.555, -81.780), (26.95, -80.83)
    print("Orlando to the Atlantic:", at("dist_atlantic_km", *orlando), "km;",
          "Tampa to the Gulf:", at("dist_gulf_km", *tampa), "km")
    assert 45 < at("dist_atlantic_km", *orlando) < 70
    assert at("dist_gulf_km", *orlando) > 100
    assert at("dist_gulf_km", *tampa) < 10
    assert at("dist_atlantic_km", *tampa) > 150
    assert at("dist_atlantic_km", *key_west) < 5 and at("dist_gulf_km", *key_west) < 5
    assert not at("land_mask", *okeechobee) and at("water_frac_5km", *okeechobee) == 1
