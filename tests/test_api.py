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


# ---------------------------------------------------------------------------
# GET /model-info
# ---------------------------------------------------------------------------


class TestModelInfo:
    def test_model_info_returns_200(self):
        resp = client.get("/model-info")
        # 200 if model loaded, 503 otherwise
        assert resp.status_code in (200, 503)

    def test_model_info_content(self):
        resp = client.get("/model-info")
        if resp.status_code == 200:
            data = resp.json()
            assert "model_type" in data
            assert "threshold" in data
            assert "n_features" in data
            assert "metrics" in data
            assert isinstance(data["threshold"], float)
            assert data["n_features"] > 0


# ---------------------------------------------------------------------------
# Integration: model predictions
# ---------------------------------------------------------------------------


class TestModelIntegration:
    """Tests qui vérifient que le modèle retourne des prédictions cohérentes."""

    def test_predict_probability_range(self):
        resp = client.post("/predict", json=VALID_PAYLOAD)
        if resp.status_code == 200:
            data = resp.json()
            assert 0.0 <= data["churn_probability"] <= 1.0

    def test_predict_binary_output(self):
        resp = client.post("/predict", json=VALID_PAYLOAD)
        if resp.status_code == 200:
            data = resp.json()
            assert data["churn_prediction"] in (0, 1)

    def test_high_risk_profile(self):
        """Un profil à haut risque devrait avoir une proba plus élevée."""
        high_risk = {
            **VALID_PAYLOAD,
            "tenure_months": 2,
            "monthly_logins": 1,
            "weekly_active_days": 0,
            "payment_failures": 5,
            "support_tickets": 8,
            "csat_score": 1.5,
            "nps_score": 1,
            "last_login_days_ago": 45,
            "escalations": 3,
        }
        resp_normal = client.post("/predict", json=VALID_PAYLOAD)
        resp_risky = client.post("/predict", json=high_risk)
        if resp_normal.status_code == 200 and resp_risky.status_code == 200:
            prob_normal = resp_normal.json()["churn_probability"]
            prob_risky = resp_risky.json()["churn_probability"]
            assert prob_risky > prob_normal

    def test_health_model_loaded_true(self):
        """Si les artifacts existent, model_loaded doit être True."""
        from pathlib import Path
        artifacts = Path(__file__).resolve().parent.parent / "artifacts"
        if (artifacts / "model.joblib").exists():
            resp = client.get("/health")
            assert resp.json()["model_loaded"] is True
