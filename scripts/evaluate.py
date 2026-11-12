"""Step 9: score forecasts on held-out test days and write the README results table.

Baselines: persistence, pysteps advection, climatology, no-geography XGBoost, lattice with kappa = 0.
Metrics:   CSI at 10/20/30/45/60 min (pixel and 4x4 pooled), Brier skill score vs. climatology,
           reliability diagram, birth-model ROC AUC, birth hit rate, spread-skill.
"""

from __future__ import annotations


def csi(hits: int, misses: int, false_alarms: int) -> float:
    """Critical success index = hits / (hits + misses + false alarms)."""
    denom = hits + misses + false_alarms
    return hits / denom if denom else float("nan")


def main() -> None:
    raise NotImplementedError


if __name__ == "__main__":
    main()
