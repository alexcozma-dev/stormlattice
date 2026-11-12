"""Step 3: motion field and the advection baseline (Lagrangian persistence).

    dR/dt + u dR/dx + v dR/dy = 0   =>   R(x, t + tau) = R(x - alpha(tau), t)

Uses pysteps: motion.get_method("LK") for (u, v), extrapolation.semilagrangian to step forward.
This forecast is Baseline 1 in the evaluation; persistence is Baseline 0.
"""

from __future__ import annotations

import numpy as np


def motion_field(frames: np.ndarray) -> np.ndarray:
    """Estimate the (2, ny, nx) motion field in grid cells per step from a (t, ny, nx) stack."""
    raise NotImplementedError


def extrapolate(field: np.ndarray, motion: np.ndarray, n_steps: int) -> np.ndarray:
    """Advect `field` forward `n_steps` along `motion`. Returns (n_steps, ny, nx)."""
    raise NotImplementedError
