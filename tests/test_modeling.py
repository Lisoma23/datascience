"""Tests de modélisation — skippés en CI si artifacts absents."""

import pytest
import numpy as np
from pathlib import Path

ARTIFACTS_DIR = Path(__file__).resolve().parent.parent / "artifacts"
MODEL_PATH = ARTIFACTS_DIR / "model.joblib"
PREPROCESSOR_PATH = ARTIFACTS_DIR / "preprocessor.joblib"

# Skip tout le fichier si les artifacts n'existent pas
pytestmark = pytest.mark.skipif(
    not MODEL_PATH.exists(),
    reason="Artifacts non disponibles (CI)"
)


class TestModelLoading:
    def test_model_loads(self):
        import joblib
        model = joblib.load(MODEL_PATH)
        assert model is not None

    def test_model_has_predict(self):
        import joblib
        model = joblib.load(MODEL_PATH)
        assert hasattr(model, "predict")

    def test_model_has_predict_proba(self):
        import joblib
        model = joblib.load(MODEL_PATH)
        assert hasattr(model, "predict_proba")


class TestEndToEnd:
    """Test complet : CSV -> preprocessing -> predict -> résultat valide."""

    def test_pipeline_predict(self):
        import joblib
        from backend.data.preprocessing import load_and_split

        model = joblib.load(MODEL_PATH)
        _, X_test, _, y_test = load_and_split("customer_churn.csv")

        preprocessor = joblib.load(PREPROCESSOR_PATH)
        X_test_t = preprocessor.transform(X_test)

        preds = model.predict(X_test_t)
        assert len(preds) == len(y_test)
        assert set(preds).issubset({0, 1})

    def test_predict_proba_range(self):
        import joblib
        from backend.data.preprocessing import load_and_split

        model = joblib.load(MODEL_PATH)
        _, X_test, _, _ = load_and_split("customer_churn.csv")

        preprocessor = joblib.load(PREPROCESSOR_PATH)
        X_test_t = preprocessor.transform(X_test)

        probas = model.predict_proba(X_test_t)
        assert probas.shape[1] == 2
        assert np.all(probas >= 0) and np.all(probas <= 1)