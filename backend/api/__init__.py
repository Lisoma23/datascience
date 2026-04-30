"""API FastAPI — Churn Prediction."""

from fastapi import FastAPI, Request
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse

from backend.api.schemas.client_input import ClientInput
from backend.api.schemas.prediction_output import PredictionOutput
from backend.data import check_data

app = FastAPI(
    title="Churn Prediction API",
    description="API de prédiction du churn client",
    version="0.1.0",
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
    return {"message": "Churn Prediction API"}


@app.get("/health")
def health_check():
    data_check = check_data()
    if data_check is not None:
        status = "healthy"
    else:
        status = "unhealthy"
    return {
        "status": status,
        "model_loaded": False,  # deviendra True quand le modèle sera branché
    }


@app.post("/predict", response_model=PredictionOutput)
def predict(client: ClientInput):
    # TODO: brancher le vrai modèle (après Tâche B)
    # Pour l'instant, retourne une prédiction factice
    return PredictionOutput(churn_prediction=0, churn_probability=0.5)
