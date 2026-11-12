"""Step 5: XGBoost birth and death models, calibration, and probability <-> rate conversion.

Birth model: unit = empty grid cell per time step; target = tracked birth within 4 km in 10 min.
Death model: unit = tracked storm cell; target = cell dies in the next 10 min.

Rules that make or break it:
    1. Split train/validation/test by whole days, never by frames.
    2. Handle class imbalance (births are rare), then recalibrate.
    3. Calibrate with isotonic regression on validation days; check a reliability diagram.
"""

from __future__ import annotations

import numpy as np


def probability_to_rate(p: np.ndarray, dt_train_min: float) -> np.ndarray:
    """Per-minute rate from a per-step probability, assuming Poisson events within a step.

        lambda = -ln(1 - p) / dt_train
    """
    p = np.clip(p, 0.0, 1.0 - 1e-12)
    return -np.log1p(-p) / dt_train_min


def rate_to_probability(rate: np.ndarray, dt_min: float) -> np.ndarray:
    """Per-step probability from a per-minute rate: p = 1 - exp(-lambda dt)."""
    return -np.expm1(-np.asarray(rate) * dt_min)


def train_birth_model(features, labels, groups):
    """Fit the birth classifier with day-grouped CV. Returns a fitted xgboost model."""
    raise NotImplementedError


def train_death_model(features, labels, groups):
    """Fit the death classifier with day-grouped CV. Returns a fitted xgboost model."""
    raise NotImplementedError


def calibrate(model, features_val, labels_val):
    """Fit isotonic calibration on validation days. Returns a callable raw-score -> probability."""
    raise NotImplementedError
