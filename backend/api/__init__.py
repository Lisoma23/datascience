from fastapi import FastAPI
import pandas as pd 
from ..data import check_data

app = FastAPI()

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