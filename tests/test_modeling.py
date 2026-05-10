"""Tests de modélisation — skippés en CI si artifacts absents."""

import json
from pathlib import Path

import joblib
import numpy as np
import pytest

from backend.data.preprocessing import load_and_split

ARTIFACTS_DIR = Path(__file__).resolve().parent.parent / "artifacts"
MODEL_PATH = ARTIFACTS_DIR / "model.joblib"
PREPROCESSOR_PATH = ARTIFACTS_DIR / "preprocessor.joblib"
THRESHOLD_PATH = ARTIFACTS_DIR / "threshold.json"

# Skip tout le fichier si les artifacts n'existent pas
pytestmark = pytest.mark.skipif(
    not MODEL_PATH.exists(),
    reason="Artifacts non disponibles (CI)",
)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture()
def model():
    return joblib.load(MODEL_PATH)


@pytest.fixture()
def preprocessor():
    return joblib.load(PREPROCESSOR_PATH)


@pytest.fixture()
def threshold():
    with open(THRESHOLD_PATH) as f:
        return json.load(f)["threshold"]


@pytest.fixture()
def test_data(preprocessor):
    """Retourne X_test transformé et y_test."""
    _, X_test, _, y_test = load_and_split("customer_churn.csv")
    X_test_t = preprocessor.transform(X_test)
    return X_test_t, y_test


# ---------------------------------------------------------------------------
# Chargement du modèle
# ---------------------------------------------------------------------------


class TestModelLoading:
    def test_model_loads(self, model):
        assert model is not None

    def test_model_has_predict(self, model):
        assert hasattr(model, "predict")

    def test_model_has_predict_proba(self, model):
        assert hasattr(model, "predict_proba")


# ---------------------------------------------------------------------------
# Tests end-to-end
# ---------------------------------------------------------------------------


class TestEndToEnd:
    """Test complet : CSV -> preprocessing -> predict -> résultat valide."""

    def test_pipeline_predict(self, model, test_data):
        X_test_t, y_test = test_data
        preds = model.predict(X_test_t)
        assert len(preds) == len(y_test)
        assert set(preds).issubset({0, 1})

    def test_predict_proba_range(self, model, test_data):
        X_test_t, _ = test_data
        probas = model.predict_proba(X_test_t)
        assert probas.shape[1] == 2
        assert np.all(probas >= 0) and np.all(probas <= 1)


# ---------------------------------------------------------------------------
# Tests du seuil de décision
# ---------------------------------------------------------------------------


class TestThreshold:
    def test_threshold_loads(self, threshold):
        assert isinstance(threshold, float)

    def test_threshold_in_valid_range(self, threshold):
        assert 0.0 < threshold < 1.0

    def test_threshold_applied_correctly(self, model, test_data, threshold):
        """Vérifie que le seuil produit des prédictions cohérentes."""
        X_test_t, _ = test_data
        probas = model.predict_proba(X_test_t)[:, 1]
        preds_with_threshold = (probas >= threshold).astype(int)

        # Au moins quelques clients doivent être flaggés avec le seuil bas
        assert preds_with_threshold.sum() > 0, "Aucun client flaggé avec le seuil — seuil trop haut ?"

        # Tous les flaggés doivent avoir une proba >= threshold
        assert np.all(probas[preds_with_threshold == 1] >= threshold)
