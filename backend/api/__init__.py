"""API FastAPI — Churn Prediction."""

from __future__ import annotations

import json
import logging
from pathlib import Path

import joblib

import pandas as pd
from fastapi import FastAPI, Request
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse

from backend.api.schemas.client_input import ClientInput
from backend.api.schemas.prediction_output import PredictionOutput
from backend.data import check_data
from backend.data.preprocessing import add_engineered_features

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Chargement des artifacts
# ---------------------------------------------------------------------------

ARTIFACTS = Path(__file__).resolve().parent.parent.parent / "artifacts"

try:
    model = joblib.load(ARTIFACTS / "model.joblib")
    preprocessor = joblib.load(ARTIFACTS / "preprocessor.joblib")
    with open(ARTIFACTS / "threshold.json") as f:
        threshold = json.load(f)["threshold"]
    with open(ARTIFACTS / "feature_names.json") as f:
        feature_names = json.load(f)
    with open(ARTIFACTS / "metrics.json") as f:
        metrics = json.load(f)
    _model_loaded = True
except Exception as e:
    logger.warning(f"Artifacts non chargés : {e}")
    model = None
    preprocessor = None
    threshold = 0.5
    feature_names = []
    metrics = {}
    _model_loaded = False

# ---------------------------------------------------------------------------
# App
# ---------------------------------------------------------------------------

app = FastAPI(
    title="Churn Prediction API",
    description="API de prédiction du churn client",
    version="1.0.0",
)


# ---------------------------------------------------------------------------
# Error handlers
# ---------------------------------------------------------------------------


@app.exception_handler(RequestValidationError)
async def validation_exception_handler(request: Request, exc: RequestValidationError):
    return JSONResponse(
        status_code=422,
        content={"error": "Données invalides", "details": str(exc)},
    )


# ---------------------------------------------------------------------------
# Endpoints
# ---------------------------------------------------------------------------


@app.get("/")
def read_root():
    return {"message": "Churn Prediction API", "version": "1.0.0"}


@app.get("/health")
def health_check():
    data_check = check_data()
    if data_check is not None:
        status = "healthy"
    else:
        status = "unhealthy"
    return {
        "status": status,
        "model_loaded": _model_loaded,
    }


@app.get("/model-info")
def model_info():
    """Informations sur le modèle chargé."""
    if not _model_loaded:
        return JSONResponse(
            status_code=503,
            content={"error": "Modèle non chargé"},
        )
    return {
        "model_type": type(model).__name__,
        "threshold": threshold,
        "n_features": len(feature_names),
        "metrics": metrics,
    }


@app.post("/predict", response_model=PredictionOutput)
def predict(client: ClientInput):
    if not _model_loaded:
        return JSONResponse(
            status_code=503,
            content={"error": "Modèle non chargé — exécutez le notebook 03_modeling d'abord"},
        )

    # 1. Convertir en DataFrame (une ligne)
    df = pd.DataFrame([client.model_dump()])

    # 2. Feature engineering
    df = add_engineered_features(df)

    # 3. Preprocessing (transform seulement, pas fit)
    X = preprocessor.transform(df)

    # 4. Prédiction
    proba = float(model.predict_proba(X)[:, 1][0])
    prediction = int(proba >= threshold)

    return PredictionOutput(churn_prediction=prediction, churn_probability=round(proba, 4))
