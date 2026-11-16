import numpy as np
import pytest

from stormlattice import grid
from stormlattice.config import settings


def test_single_storm_pixel_survives():
    # once 50 dBZ pixel in clear air must still be 50 after coursening, in the right cell
    a = np.full((4, 6), -32.0)          # 4 x 6 pixels of clear air -> 2 x 3 cells
    a[3, 2] = 50.0                     # bottow row, LEFT column of its 2 x 2 block
    out = grid.block_max(a, 2)
    assert out.shape == (2, 3)
    assert out[1, 1] == 50.0            # pixel (3, 2) lives in cell (3 //2, 2 // 2)
    assert (out == -32.0).sum() == 5    # the other five cells stay clear air

def test_nan_only_when_all_nan():
    a = np.array([[np.nan, 10.0, np.nan, np.nan],
                  [np.nan, np.nan, np.nan, np.nan]])
    out = grid.block_max(a, 2)      
    assert out[0, 0] == 10.0        # one covered pixel is enough
    assert np.isnan(out[0, 1])      # no covered pixel -> still no coverage

def test_shape_must_divide():
    with pytest.raises(ValueError):     # passes only if block_max raises ValueError
        grid.block_max(np.zeros((5, 4)), 2)

def test_row_lats_are_cell_centers():
    lats = grid.row_lats(31.0, 360)
    assert lats.shape == (360,)
    assert lats[0] == pytest.approx(30.99)      # half a cell south of the northern edge
    assert lats[-1] == pytest.approx(23.81)     # half a cell north of the southern edge

def test_cells_shrink_northward():
    area = grid.cell_area_km2(grid.row_lats(31.0, 360))
    assert area[0] / area[-1] == pytest.approx(0.938, abs=0.001)    # north / south: about 6% less

def test_dx_km_close_to_true_spacing_at_box_edges():
    # the error of using the one mid-latitude spacing, as a fraction of DX_KM (the docstring's 3.4%)
    for lat in (23.80, 31.0):
        assert abs(grid.dx_km(lat) - grid.DX_KM) / grid.DX_KM < 0.035

def test_mid_lat_is_middle_of_box():
    box = settings()["grid"]
    assert grid.MID_LAT == pytest.approx((box["lat_min"] + box["lat_max"]) / 2)
