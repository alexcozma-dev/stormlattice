"""Step 6: the stochastic lattice, version 1.

Two states per grid cell: 0 = empty, 1 = active storm. No cold-pool state (deferred).

Rules, for each ensemble member (see "The lattice, version 1" in docs/design.md):
    1. Existing storms are objects: the connected components of the storm mask at t0.
       Each object moves by its cumulative displacement from t0. (Stepping a binary field
       forward one step at a time stalls in weak flow.)
    2. Each object dies as a whole, with a hazard from the survival table (by storm size
       and lead time). Deaths are never drawn pixel by pixel.
    3. Births switch on a whole 0.08 deg block (about 8 km) with the learned birth rate
       for that lead time.

Rates are frozen at t0 (Design A): one birth-rate map per lead, computed once per forecast.

Check: with these rules each cell's storm probability has a closed form from the cumulative
hazards. The ensemble mean must match it; that is the unit test for this module.

To write: step, run, closed_form_probability.
"""

from __future__ import annotations

import numpy as np

EMPTY, ACTIVE = 0, 1


def storm_probability(states: np.ndarray) -> np.ndarray:
    """Fraction of members with an active storm at each lead time and cell: (T, M, y, x) -> (T, y, x)."""
    return (states == ACTIVE).mean(axis=1)
