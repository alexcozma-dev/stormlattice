"""
Survival table: how likely a storm object at t0 is to still be there at each lead time.

Measured from the archive, not modelled: advect each t0 storm object to lead L and take the
fraction of its advected footprint that is still >= 35 dBZ. Tabulated by storm size and lead.

The lattice reads a per-object hazard from this table, so each object dies as a whole. The
"advection + deaths only" row of the evaluation uses it too.

Storm size is an area in km2 (the sum of cell areas from grid.py), not a pixel count.

To write: build the table from training days, save and load it, look up the hazard for a
size and lead.
"""
