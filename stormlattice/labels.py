"""
Birth labels on the grid, per 0.08 deg block and per lead time. No tracking.

A block is a birth at lead L (issue time t0) when all three hold:
    1. It holds storm cells (>= 35 dBZ) in the window [t0+L, t0+L+10 min]: at least 4 cells,
       persisting for two frames. (A cell is 0.02 deg; see grid.py.)
    2. It held none in the previous 20 minutes.
    3. It is not covered by the advected t0 storm mask, with a small tolerance so that motion
       errors do not become fake births.

So a birth is a storm that advection does not explain, which is also what the evaluation scores.
The label, the predicted rate and the lattice's birth footprint all use the same block.

The numbers (threshold, cells, frames, quiet window, tolerance) belong in settings.yaml.

To write: birth_labels.
"""
