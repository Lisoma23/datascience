from fastapi import FastAPI
import pandas as pd

from backend.api.schemas.client_input import ClientInput
from backend.api.schemas.prediction_output import PredictionOutput 
from ..data import check_data

app = FastAPI(
    title="Churn Prediction API",
    description="API de prédiction du churn client",
    version="0.1.0",
)

@app.get("/")
def read_root():
    return {"Hello": "World"}

@app.get("/health")
def health_check():
    # Examine the data 
    data_check = check_data()
    if data_check is not None:
        return {"status": "healthy"}
    else:
        return {"status": "unhealthy"}
    
@app.post("/predict", response_model=PredictionOutput)
def predict(client: ClientInput):
    # TODO: brancher le vrai modèle (après Tâche B)
    # Pour l'instant, retourne une prédiction factice
    return PredictionOutput(churn_prediction=0, churn_probability=0.5)