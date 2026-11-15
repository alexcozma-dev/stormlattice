"""
Baseline forecasts for the evaluation table (see "Evaluation" in docs/design.md).

    persistence                the t0 storm mask, unchanged
    advection, probabilistic   the advected mask smoothed over a neighbourhood, threshold tuned
                               on validation days (a fair baseline for a probabilistic forecast)
    climatology                birth frequency by block and hour, from training days

Deterministic advection is in advection.py. The deaths-only row uses survival.py.

To write: one function per baseline.
"""
