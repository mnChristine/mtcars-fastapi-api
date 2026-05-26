"""
tests/test_api.py
-----------------
Automated tests for the MTCARS MPG Predictor API.

Run with:
    pytest tests/test_api.py -v
"""

import pytest
from fastapi.testclient import TestClient

from app.main import app

client = TestClient(app)


# ── /health ────────────────────────────────────────────────────────────────

def test_health_returns_200():
    response = client.get("/health")
    assert response.status_code == 200


def test_health_returns_ok_status():
    response = client.get("/health")
    data = response.json()
    assert data["status"] == "ok"


# ── /ready ─────────────────────────────────────────────────────────────────

def test_ready_returns_200_when_model_loaded():
    """Assumes model.pkl exists (run train_model.py first)."""
    response = client.get("/ready")
    # Either 200 (model loaded) or 503 (model missing) is acceptable,
    # but if the model IS present the response must be 200.
    if response.status_code == 200:
        data = response.json()
        assert data["status"] == "ready"
        assert data["model_loaded"] is True


# ── /predict ───────────────────────────────────────────────────────────────

def test_predict_valid_input():
    """A valid request should return 200 with a numeric predicted_mpg."""
    payload = {"wt": 2.62, "hp": 110}
    response = client.post("/predict", json=payload)
    # Skip if model not loaded
    if response.status_code == 503:
        pytest.skip("Model not loaded – run train_model.py first")
    assert response.status_code == 200
    data = response.json()
    assert "predicted_mpg" in data
    assert isinstance(data["predicted_mpg"], (int, float))


def test_predict_mpg_is_reasonable():
    """Predicted mpg for a typical car should be in a sane range (0–60)."""
    payload = {"wt": 3.0, "hp": 150}
    response = client.post("/predict", json=payload)
    if response.status_code == 503:
        pytest.skip("Model not loaded – run train_model.py first")
    assert response.status_code == 200
    mpg = response.json()["predicted_mpg"]
    assert 0 < mpg < 60, f"Predicted mpg {mpg} looks unreasonable"


def test_predict_returns_predictors_in_response():
    payload = {"wt": 2.62, "hp": 110}
    response = client.post("/predict", json=payload)
    if response.status_code == 503:
        pytest.skip("Model not loaded – run train_model.py first")
    data = response.json()
    assert data["predictors"]["wt"] == 2.62
    assert data["predictors"]["hp"] == 110


def test_predict_missing_field_returns_422():
    """Omitting a required field must return 422 Unprocessable Entity."""
    response = client.post("/predict", json={"wt": 2.62})  # missing hp
    assert response.status_code == 422


def test_predict_missing_both_fields_returns_422():
    response = client.post("/predict", json={})
    assert response.status_code == 422


def test_predict_invalid_type_returns_422():
    """Passing a string where float is expected must return 422."""
    response = client.post("/predict", json={"wt": "heavy", "hp": 110})
    assert response.status_code == 422


def test_predict_zero_wt_returns_422():
    """wt must be > 0."""
    response = client.post("/predict", json={"wt": 0, "hp": 110})
    assert response.status_code == 422


def test_predict_negative_hp_returns_422():
    """hp must be > 0."""
    response = client.post("/predict", json={"wt": 2.62, "hp": -50})
    assert response.status_code == 422


def test_predict_extra_fields_are_ignored():
    """Extra fields in the request body should not break the API."""
    payload = {"wt": 2.62, "hp": 110, "unknown_field": 99}
    response = client.post("/predict", json=payload)
    if response.status_code == 503:
        pytest.skip("Model not loaded – run train_model.py first")
    assert response.status_code == 200
