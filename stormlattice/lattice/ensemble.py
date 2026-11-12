"""Step 6: run the ensemble forward and summarize it."""

from __future__ import annotations

import numpy as np


def run(initial_state: np.ndarray, n_steps: int, members: int, seed: int, **step_inputs) -> np.ndarray:
    """Run `members` lattice realizations for `n_steps`. Returns (n_steps, M, ny, nx) states."""
    raise NotImplementedError


def storm_probability(states: np.ndarray) -> np.ndarray:
    """Fraction of members with an active storm at each lead time and cell: (T, M, y, x) -> (T, y, x)."""
    from .step import ACTIVE

    return (states == ACTIVE).mean(axis=1)
