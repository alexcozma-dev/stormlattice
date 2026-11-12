"""Step 6 check: mean-field reaction-advection equations (triggering off).

    d(rho)/dt + u . grad(rho) = lam(x, t) (1 - rho - q) - mu rho
    d(q)/dt   + u . grad(q)   = mu rho - q / tau_cp

rho = storm fraction, q = cold-pool fraction. The lattice ensemble mean must reproduce this
when kappa2 = kappa3 = 0, the same check as matching the Brusselator solver to linear theory.
"""

from __future__ import annotations

import numpy as np


def integrate(rho0: np.ndarray, q0: np.ndarray, lam: np.ndarray, mu: np.ndarray,
              motion: np.ndarray, tau_cp_min: float, dt_min: float, n_steps: int) -> tuple[np.ndarray, np.ndarray]:
    """Integrate the mean-field system. Returns (rho, q) histories of shape (n_steps, ny, nx)."""
    raise NotImplementedError
