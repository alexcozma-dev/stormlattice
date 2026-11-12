"""Step 6: the stochastic lattice (excitable-medium, contact-process model).

States per grid cell: 0 = empty, 1 = active storm, 2 = cold pool (refractory).

One step (dt = 5 min), vectorized over all ensemble members (arrays are (M, ny, nx)):
    1. Advect active cells and cold pools by the motion field.
    2. Die: active -> cold pool with probability p_death.
    3. Expand and recover: pools grow at the front speed; pool -> empty with prob dt / tau_cp.
    4. Be born: empty -> active with
           p = 1 - exp(-(lam_env + k2*[n_fronts >= 2] + k3*[n_fronts >= 3]) * dt)

Design A (MVP): lam_env from XGBoost once per forecast; triggering via the kappa terms.
Design B (research): re-query XGBoost every step on features recomputed from the simulated state.

Validation habit: with kappa = 0, the ensemble mean must match the mean-field PDEs in meanfield.py.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

EMPTY, ACTIVE, POOL = 0, 1, 2


@dataclass
class LatticeParams:
    dt_min: float
    cold_pool_lifetime_min: float
    kappa2_per_min: float
    kappa3_per_min: float


def count_fronts(pool: np.ndarray) -> np.ndarray:
    """Number of distinct cold pools whose edge is within one cell of each cell. (M, ny, nx) -> int."""
    raise NotImplementedError


def step(
    state: np.ndarray,
    motion: np.ndarray,
    lam_env: np.ndarray,
    p_death: np.ndarray,
    params: LatticeParams,
    rng: np.random.Generator,
) -> np.ndarray:
    """Advance the (M, ny, nx) lattice state by one step and return the new state."""
    raise NotImplementedError
