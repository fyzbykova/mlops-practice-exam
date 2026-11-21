"""FastAPI-сервис для предсказания уровня ожирения.

Запуск локально:
    uvicorn app.app:app --host 0.0.0.0 --port 8000
В контейнере:
    uvicorn app:app --host 0.0.0.0 --port 8000
"""
import os

import joblib
import pandas as pd
from fastapi import FastAPI
from pydantic import BaseModel

ARTIFACTS_DIR = os.environ.get("ARTIFACTS_DIR", "artifacts")

app = FastAPI(title="Obesity Level Predictor")

model = joblib.load(os.path.join(ARTIFACTS_DIR, "model.joblib"))
label_encoder = joblib.load(os.path.join(ARTIFACTS_DIR, "label_encoder.joblib"))
feature_columns = joblib.load(os.path.join(ARTIFACTS_DIR, "feature_columns.joblib"))


class ObesityInput(BaseModel):
    Gender: str
    Age: float
    Height: float
    Weight: float
    family_history_with_overweight: str
    FAVC: str
    FCVC: float
    NCP: float
    CAEC: str
    SMOKE: str
    CH2O: float
    SCC: str
    FAF: float
    TUE: float
    CALC: str
    MTRANS: str


@app.post("/predict")
def predict(data: ObesityInput):
    row = pd.DataFrame([data.model_dump()])
    row = pd.get_dummies(row, drop_first=True)
    row = row.reindex(columns=feature_columns, fill_value=0)

    prediction = model.predict(row)[0]
    label = label_encoder.inverse_transform([prediction])[0]

    return {"prediction": label}


@app.get("/health")
def health():
    return {"status": "ok"}