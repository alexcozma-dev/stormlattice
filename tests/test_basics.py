"""Smoke tests for the parts of the skeleton that already work."""

import numpy as np
from fastapi.testclient import TestClient

from scripts.evaluate import csi
from stormlattice.api.app import app
from stormlattice.config import settings
from stormlattice.llm.guardrails import ungrounded_numbers
from stormlattice.models.rates import probability_to_rate, rate_to_probability
from stormlattice.risk.route import classify


def test_settings_load():
    s = settings()
    assert s["grid"]["resolution_km"] == 2.0
    assert s["forecast"]["ensemble_members"] == 50


def test_health_endpoint():
    response = TestClient(app).get("/health")
    assert response.status_code == 200
    assert response.json()["status"] == "ok"


def test_rate_round_trip():
    p = np.array([0.0, 0.01, 0.1, 0.5])
    rate = probability_to_rate(p, dt_train_min=10)
    assert np.allclose(rate_to_probability(rate, dt_min=10), p)


def test_rate_rescales_step():
    # a 10% chance per 10 min is about a 5.1% chance per 5 min, not 5%
    rate = probability_to_rate(np.array([0.10]), dt_train_min=10)
    assert np.isclose(rate_to_probability(rate, dt_min=5)[0], 1 - np.sqrt(0.9))


def test_stale_data_is_unavailable():
    advisory = classify(0.02, 0.10, 0.30, data_age_min=15, max_data_age_min=10, data_time_utc="t")
    assert advisory.level == "unavailable" and advisory.risk is None


def test_risk_levels():
    assert classify(0.05, 0.10, 0.30, 1, 10, "t").level == "green"
    assert classify(0.20, 0.10, 0.30, 1, 10, "t").level == "amber"
    assert classify(0.45, 0.10, 0.30, 1, 10, "t").level == "red"


def test_guardrail_flags_invented_numbers():
    facts = "route risk 0.41, departure 15:10, SIGMET 12E valid to 16:55"
    assert ungrounded_numbers("Risk is 0.41 for a 15:10 departure.", facts) == set()
    assert ungrounded_numbers("Risk is 0.38.", facts) == {"0.38"}


def test_csi():
    assert csi(hits=3, misses=1, false_alarms=1) == 0.6
