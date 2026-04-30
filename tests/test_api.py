"""Tests unitaires pour l'API FastAPI."""

import pytest
from fastapi.testclient import TestClient

from backend.api import app

client = TestClient(app)

# Payload valide complet
VALID_PAYLOAD = {
    "age": 45,
    "tenure_months": 24,
    "monthly_logins": 15,
    "weekly_active_days": 4,
    "avg_session_time": 18.5,
    "features_used": 5,
    "usage_growth_rate": 0.05,
    "last_login_days_ago": 3,
    "monthly_fee": 30,
    "total_revenue": 720,
    "payment_failures": 1,
    "support_tickets": 2,
    "avg_resolution_time": 20.0,
    "csat_score": 4.0,
    "escalations": 0,
    "email_open_rate": 0.6,
    "marketing_click_rate": 0.3,
    "nps_score": 35,
    "referral_count": 1,
    "gender": "Male",
    "country": "USA",
    "city": "New York",
    "customer_segment": "Individual",
    "signup_channel": "Web",
    "contract_type": "Monthly",
    "payment_method": "Card",
    "discount_applied": "No",
    "price_increase_last_3m": "No",
    "complaint_type": None,
    "survey_response": "Satisfied",
}


# ---------------------------------------------------------------------------
# GET /
# ---------------------------------------------------------------------------


class TestRoot:
    def test_root_returns_200(self):
        resp = client.get("/")
        assert resp.status_code == 200

    def test_root_has_message(self):
        resp = client.get("/")
        assert "message" in resp.json()


# ---------------------------------------------------------------------------
# GET /health
# ---------------------------------------------------------------------------


class TestHealth:
    def test_health_returns_200(self):
        resp = client.get("/health")
        assert resp.status_code == 200

    def test_health_status_ok(self):
        resp = client.get("/health")
        data = resp.json()
        assert data["status"] in ("healthy", "unhealthy")

    def test_health_has_model_loaded(self):
        resp = client.get("/health")
        data = resp.json()
        assert "model_loaded" in data
        assert isinstance(data["model_loaded"], bool)


# ---------------------------------------------------------------------------
# POST /predict
# ---------------------------------------------------------------------------


class TestPredict:
    def test_predict_valid_payload(self):
        resp = client.post("/predict", json=VALID_PAYLOAD)
        assert resp.status_code == 200

    def test_predict_response_schema(self):
        resp = client.post("/predict", json=VALID_PAYLOAD)
        data = resp.json()
        assert "churn_prediction" in data
        assert "churn_probability" in data
        assert isinstance(data["churn_prediction"], int)
        assert isinstance(data["churn_probability"], float)

    def test_predict_with_complaint_type_null(self):
        payload = {**VALID_PAYLOAD, "complaint_type": None}
        resp = client.post("/predict", json=payload)
        assert resp.status_code == 200

    def test_predict_with_complaint_type_string(self):
        payload = {**VALID_PAYLOAD, "complaint_type": "Billing"}
        resp = client.post("/predict", json=payload)
        assert resp.status_code == 200

    def test_predict_missing_field_returns_422(self):
        payload = {**VALID_PAYLOAD}
        del payload["age"]
        resp = client.post("/predict", json=payload)
        assert resp.status_code == 422

    def test_predict_wrong_type_returns_422(self):
        payload = {**VALID_PAYLOAD, "age": "not_a_number"}
        resp = client.post("/predict", json=payload)
        assert resp.status_code == 422

    def test_predict_empty_body_returns_422(self):
        resp = client.post("/predict", json={})
        assert resp.status_code == 422

    def test_422_has_error_key(self):
        resp = client.post("/predict", json={})
        data = resp.json()
        assert "error" in data
        assert "details" in data


# ---------------------------------------------------------------------------
# Swagger docs
# ---------------------------------------------------------------------------


class TestDocs:
    def test_docs_available(self):
        resp = client.get("/docs")
        assert resp.status_code == 200

    def test_openapi_json_available(self):
        resp = client.get("/openapi.json")
        assert resp.status_code == 200
        data = resp.json()
        assert data["info"]["title"] == "Churn Prediction API"
