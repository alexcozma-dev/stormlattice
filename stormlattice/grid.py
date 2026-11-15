"""
The grid: MRMS's lat/lon grid coarsened to 0.02 deg cells (about 2 km), and 0.08 deg blocks.

MRMS composite reflectivity comes on a 0.01 deg lat/lon grid, rows north -> south. We keep
that grid and coarsen it; there is no reprojection.

    source pixel   0.01 deg                    648 x 752 over the Florida box in settings.yaml
    cell           0.02 deg = 2 x 2 pixels     324 x 376   (the grid everything else uses)
    block          0.08 deg = 4 x 4 cells       81 x 94    (birth labels, rates, lattice births)

Arrays are (ny, nx) with row 0 at the northern edge and column 0 at the western edge.

Cells are not square and not all the same size, because meridians converge:
    north-south   DY_KM = 2.216 km, the same everywhere
    east-west     dx_km(lat) = 0.02 deg * 111.32 km/deg * cos(lat): 2.03 km at 24.5 N, 1.91 km at 31 N
    area          DY_KM * dx_km(lat): 4.5 km2 in the south, 4.2 km2 in the north (6% less)

This module owns every km-to-pixel conversion. No other module multiplies by 2 or calls a cosine.

To write:
    DY_KM                 north-south cell size.
    dx_km(lat)            east-west cell size at a latitude.
    DX_KM                 one mid-latitude value (1.97 km) for kernels and distance transforms,
                          where a single spacing is needed. At most 3.2% off at the box edges.
    cell area per row     (ny,) array in km2. Use it wherever the plan says km2 or "fraction of
                          area"; a pixel count is not an area on this grid.
    2 x 2 block maximum   0.01 deg pixels -> cells, by reshape and max. NaN-safe: a cell is NaN
                          (no radar coverage) only if all four pixels are NaN.
    4 x 4 block pooling   cells -> blocks.
"""
