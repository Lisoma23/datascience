from pydantic import BaseModel

class PredictionOutput(BaseModel):
    churn_probability: float
    churn_prediction: int 
    