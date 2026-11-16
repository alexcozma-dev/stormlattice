"""
The grid: MRMS's lat/lon grid coarsened to 0.02 deg cells (about 2 km), and 0.08 deg blocks.

MRMS composite reflectivity comes on a 0.01 deg lat/lon grid, rows north -> south. We keep
that grid and coarsen it; there is no reprojection.

    source pixel   0.01 deg                    720 x 912 over the Florida box in settings.yaml
    cell           0.02 deg = 2 x 2 pixels     360 x 456   (the grid everything else uses)
    block          0.08 deg = 4 x 4 cells       90 x 114    (birth labels, rates, lattice births)

Arrays are (ny, nx) with row 0 at the northern edge and column 0 at the western edge.

Cells are not square and not all the same size, because meridians converge:
    north-south   DY_KM = 2.216 km, the same everywhere
    east-west     dx_km(lat) = 0.02 deg * 111.32 km/deg * cos(lat): 2.04 km at 24.5 N, 1.91 km at 31 N
    area          DY_KM * dx_km(lat): 4.5 km2 in the south, 4.2 km2 in the north (6% less)

This module owns every km-to-pixel conversion. No other module multiplies by 2 or calls a cosine.

To write:
    DY_KM                 north-south cell size.
    dx_km(lat)            east-west cell size at a latitude.
    DX_KM                 one mid-latitude value (1.98 km) for kernels and distance transforms,
                          where a single spacing is needed. At most 3.4% off at the box edges.
    cell area per row     (ny,) array in km2. Use it wherever the plan says km2 or "fraction of
                          area"; a pixel count is not an area on this grid.
    2 x 2 block maximum   0.01 deg pixels -> cells, by reshape and max. NaN-safe: a cell is NaN
                          (no radar coverage) only if all four pixels are NaN.
    4 x 4 block pooling   cells -> blocks.
"""

from __future__ import annotations

import numpy as np

# --- Grid Structure: fixed, so it lives here and not in settings.yaml ---
PIXEL_DEG = 0.01    # one MRMS source pixel
CELL_DEG = 0.02     # one cell = 2 x 2 pixels (about 2km)
BLOCK_DEG = 0.08    # one block = 4 x 4 cells (about 8km)

# round() turns the float into an int, and absorbs floating-point error,
# like the box's 7.52 / 0.08 = 93.99999999999994 that the settings test hit
PIXELS_PER_CELL = round(CELL_DEG / PIXEL_DEG)
CELLS_PER_BLOCK = round(BLOCK_DEG / CELL_DEG)

# --- Earth model: the WGS84 ellipsoid (the shape behind GPS and MRMS cooridinates) ---
A_KM = 6378.137         # equitorial radius
E2 = 0.00669437999014   # eccentricity squared: how far the Earth is from a sphere

def dy_km(lat_deg):
    """North-south size of one cell (0.02 deg of latitude) at `lat_deg`, in km. Works on arrays."""
    s2 = np.sin(np.radians(lat_deg)) ** 2
    m = A_KM * (1 - E2) / (1 - E2 * s2) ** 1.5      # meridian radius of curvature
    return m * np.radians(CELL_DEG)

def dx_km(lat_deg):
    """East-west size of one cell (0.02 deg of longitude) at `lat_deg`, in km. Works on arrays."""
    s2 = np.sin(np.radians(lat_deg)) ** 2
    n = A_KM / np.sqrt(1 - E2 * s2)                 # prime-vertical radius of curvature
    return n * np.cos(np.radians(lat_deg)) * np.radians(CELL_DEG)

# --- One-number spacings, for kernels and distance transforms that need a single value ---
MID_LAT = 27.40             # middle of the box in settings.yaml: (23.80 + 31.0) / 2
DY_KM = dy_km(MID_LAT)      # varies only 0.1% across the box, so one value is exact enough
DX_KM = dx_km(MID_LAT)      # varies 6% across the box: use dx_km(lat) where that matters

def row_lats(lat_max, ny):
    """Latitude of the center of each row of cells, row 0 at the north edge: (ny,) array."""
    return lat_max - (np.arange(ny) + 0.5) * CELL_DEG

def cell_area_km2(lat_deg):
    """Area of one cell at `lat_deg`, in km2. Pass row_lats(...) to get one area per row."""
    return dy_km(lat_deg) * dx_km(lat_deg)

# --- Coursening: pixels -> cells, cells -> blocks ---
def block_max(a, k):
    """
    Maximum over each k x k block of a 2-D array: (ny, nx) -> (ny // k, nx // k).

    NaN-safe: np.fmax ignores NaN, so a block is NaN only if all of its values are NaN.
    """
    ny, nx = a.shape
    if ny % k or nx % k:
        raise ValueError(f"shape {a.shape} does not split into {k} x {k} blocks")
    blocks = a.reshape(ny // k, k, nx // k, k)      # axes: block row, row in block, block col, col in block
    return np.fmax.reduce(np.fmax.reduce(blocks, axis=3), axis=1)
