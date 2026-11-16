# %% Checkpoint A: grid structure constants
import numpy as np

from stormlattice import grid

print("PIXELS_PER_CELL:", grid.PIXELS_PER_CELL, type(grid.PIXELS_PER_CELL).__name__)
print("CELLS_PER_BLOCK:", grid.CELLS_PER_BLOCK)
print("box in blocks:  ", (31.0 - 23.80) / grid.BLOCK_DEG, (-78.40 + 87.52) / grid.BLOCK_DEG)

# %% Checkpoint B: cell size in km across the box
for lat in (23.80, 27.40, 31.0):
    dy, dx = grid.dy_km(lat), grid.dx_km(lat)
    print(f"lat {lat:5.2f}:  dy = {dy:.4f} km   dx = {dx:.4f} km   dy/dx = {dy / dx:.3f}")

# %% Checkpoint C: one-number spacings, row latitudes, cell areas
print(f"DY_KM = {grid.DY_KM:.4f} km   DX_KM = {grid.DX_KM:.4f} km")
lats = grid.row_lats(31.0, 360)
print("rows:", lats.shape, " first", round(lats[0], 3), " last", round (lats[-1], 3))
area = grid.cell_area_km2(lats)
print(f"area: north {area[0]:.3f}  south {area[-1]:.3f}  ratio {area[0] / area[-1]:.3f}")
print("total box area km2:", round(area.sum() * 456))

# %% Checkpoint D: block maximum on a small array you can check by hand
a = np.arange(16.0).reshape(4, 4)
print("a:\n", a)
print("block_max(a, 2):\n", grid.block_max(a, 2))
