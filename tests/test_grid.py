import numpy as np
import pytest

from stormlattice import grid


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
