import pytest
from fastapi.testclient import TestClient
from app.main import app


@pytest.fixture(scope="session")
def client():
    with TestClient(app) as c:
        yield c

SAMPLE = {
    "estimated_days": 10,
    "item_count": 1,
    "total_price": 150.0,
    "freight_value": 18.5,
    "product_weight_g": 500.0,
    "product_volume": 3000.0,
    "payment_installments": 1.0,
    "same_state": 0,
    "purchase_dow": 2,
    "purchase_month": 6,
    "purchase_hour": 14,
    "payment_type": 1,
    "customer_state": 9,
}


def test_health(client):
    r = client.get("/health")
    assert r.status_code == 200
    assert r.json()["status"] == "ok"
    assert r.json()["features_count"] == 13


def test_features(client):
    r = client.get("/features")
    assert r.status_code == 200
    assert len(r.json()["features"]) == 13


def test_predict(client):
    r = client.post("/predict", json=SAMPLE)
    assert r.status_code == 200
    data = r.json()
    assert data["prediction"] in [0, 1]
    assert data["label"] in ["On Time", "Late"]
    assert 0 <= data["probability_late"] <= 1
    assert abs(data["probability_late"] + data["probability_on_time"] - 1.0) < 1e-4


def test_predict_validation_error(client):
    bad = {**SAMPLE, "estimated_days": 999}  # ge=1, le=90 ihlali
    r = client.post("/predict", json=bad)
    assert r.status_code == 422  # Unprocessable Entity


def test_explain(client):
    r = client.post("/explain", json=SAMPLE)
    assert r.status_code == 200
    data = r.json()
    assert "shap_values" in data
    assert "top_factor" in data
    assert len(data["shap_values"]) == 13
