"""Tests for the map overlay helpers: the Mercator row resample and the reflectivity colours."""

import numpy as np

from stormlattice import overlay


def test_mercator_y_known_values():
    # y = ln(tan(45 deg + lat / 2)). At 60 N that is ln(tan(75 deg)) = ln(2 + sqrt(3)) = 1.31696.
    assert overlay.mercator_y(0.0) == 0.0
    assert np.isclose(overlay.mercator_y(60.0), np.log(2 + np.sqrt(3)))
    assert np.isclose(overlay.mercator_lat(overlay.mercator_y(27.76)), 27.76)


def test_mercator_rows_hand_case():
    # Box 0 N to 60 N. Six source rows of 10 deg each: row 0 is 50-60 N, ..., row 5 is 0-10 N.
    # Four output rows, evenly spaced in Mercator y between y(60 N) = 1.31696 and y(0) = 0.
    # Each is 1.31696 / 4 = 0.32924 tall, so the centres and their latitudes atan(sinh(y)) are:
    #   y = 1.15234 -> 54.94 N -> source row 0   (50-60 N)
    #   y = 0.82310 -> 42.59 N -> source row 1   (40-50 N)
    #   y = 0.49386 -> 27.21 N -> source row 3   (20-30 N)
    #   y = 0.16462 ->  9.39 N -> source row 5   ( 0-10 N)
    # Rows evenly spaced in latitude would have centres 52.5, 37.5, 22.5 and 7.5 N instead.
    rows = overlay.mercator_rows(0.0, 60.0, ny=6, ny_out=4)
    assert rows.tolist() == [0, 1, 3, 5]


def test_row_count_uses_every_source_row():
    # Same box. The thinnest source row in Mercator y is the one at the equator, 0-10 N:
    # y(10 N) = asinh(tan(10 deg)) = 0.17543. The box is 1.31696 / 0.17543 = 7.51 of those tall,
    # so 8 output rows are needed before every source row is used at least once.
    assert overlay.mercator_row_count(0.0, 60.0, ny=6) == 8
    assert set(overlay.mercator_rows(0.0, 60.0, ny=6).tolist()) == {0, 1, 2, 3, 4, 5}
    assert set(overlay.mercator_rows(0.0, 60.0, ny=6, ny_out=4).tolist()) != {0, 1, 2, 3, 4, 5}


def test_florida_box_rows():
    rows = overlay.mercator_rows(24.52, 31.0, ny=324)       # an example box (the original one)
    assert rows[0] == 0 and rows[-1] == 323
    assert set(np.diff(rows).tolist()) == {0, 1}            # north to south, no source row skipped
    # Mercator stretches the north, so the middle of the image lies north of the middle latitude.
    assert rows[len(rows) // 2] < 162


def test_to_mercator_rows_copies_values():
    a = np.array([[0.0, 0.5], [1.0, 1.5], [2.0, 2.5], [3.0, 3.5], [4.0, np.nan], [5.0, 5.5]])
    out = overlay.to_mercator_rows(a, 0.0, 60.0, ny_out=4)
    assert out.shape == (4, 2)
    assert out[:, 0].tolist() == [0.0, 1.0, 3.0, 5.0]       # the hand case above, values unblended
    assert out[:, 1].tolist() == [0.5, 1.5, 3.5, 5.5]


def test_reflectivity_rgba():
    dbz = np.array([[np.nan, -32.0, 14.9],
                    [15.0, 37.0, 80.0]], dtype=np.float32)
    rgba = overlay.reflectivity_rgba(dbz)
    assert rgba.shape == (2, 3, 4) and rgba.dtype == np.uint8
    assert rgba[0].tolist() == [[0, 0, 0, 0]] * 3           # no coverage, clear air, under 15 dBZ
    assert rgba[1, 0].tolist() == [3, 0, 244, 255]          # 15 dBZ is the first step
    assert rgba[1, 1].tolist() == [253, 248, 2, 255]        # 37 dBZ is in the 35-40 step: a storm
    assert rgba[1, 2].tolist() == [253, 253, 253, 255]      # above 75 dBZ keeps the last colour
