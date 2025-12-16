from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import Any, Dict, Optional, List
from contextlib import asynccontextmanager
import pandas as pd
import mlflow
import mlflow.catboost
from datetime import datetime
import os
from catboost import CatBoostClassifier

LOG_PATH = "data/prod_data.csv"

import os

EXP_ID = os.getenv("EXP_ID", "921718720432123846")
MODEL_ID = os.getenv("MODEL_ID", "m-7b104c62922046008addf1296070033f")
LOCATION = os.getenv("LOCATION", "./mlruns")

MODEL_CB_PATH = f"{LOCATION}/{EXP_ID}/models/{MODEL_ID}/artifacts/model.cb"


# глобальные переменные для модели
model = None
feature_names = [
    "LOAN", "MORTDUE", "VALUE", "REASON", "JOB", "YOJ",
    "DEROG", "DELINQ", "CLAGE", "NINQ", "CLNO", "DEBTINC"
]


def log_prod_data(df: pd.DataFrame, probas):
    df_log = df.copy()
    df_log["prediction"] = probas
    df_log["timestamp"] = datetime.utcnow()

    header = not os.path.exists(LOG_PATH)
    df_log.to_csv(LOG_PATH, mode="a", header=header, index=False)


@asynccontextmanager
async def lifespan(app: FastAPI):
    global model
    print("Loading CatBoost model from:", MODEL_CB_PATH)

    model = CatBoostClassifier()
    model.load_model(MODEL_CB_PATH)
    yield

app = FastAPI(
    title="HMEQ BAD Prediction API",
    version="1.0",
    lifespan=lifespan
)


class PredictRequest(BaseModel):
    record: Dict[str, Any]
    threshold: Optional[float] = 0.5


class BatchPredictRequest(BaseModel):
    records: List[Dict[str, Any]]
    threshold: Optional[float] = 0.5


@app.get("/health")
def health():
    return {"status": "ok", "model_uri": MODEL_URI}


@app.post("/predict")
def predict(req: PredictRequest):
    if model is None:
        raise HTTPException(status_code=500, detail="Model not loaded")

    df = pd.DataFrame([req.record])

    missing = [c for c in feature_names if c not in df.columns]
    if missing:
        raise HTTPException(status_code=400, detail=f"Missing columns: {missing}")

    df = df[feature_names]

    try:
        probas = model.predict_proba(df)[:, 1]
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Prediction error: {e}")

    # логируем prod данные для Evidently
    log_prod_data(df, probas)

    proba = float(probas[0])
    threshold = float(req.threshold or 0.5)
    pred = int(proba >= threshold)

    return {
        "bad_proba": proba,
        "bad_pred": pred,
        "threshold": threshold
    }


@app.post("/predict_batch")
def predict_batch(req: BatchPredictRequest):
    if model is None:
        raise HTTPException(status_code=500, detail="Model not loaded")

    if not req.records:
        raise HTTPException(status_code=400, detail="Empty records list")

    df = pd.DataFrame(req.records)

    missing = [c for c in feature_names if c not in df.columns]
    if missing:
        raise HTTPException(status_code=400, detail=f"Missing columns: {missing}")

    df = df[feature_names]

    try:
        probas = model.predict_proba(df)[:, 1]
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Prediction error: {e}")

    # логируем prod данные для Evidently
    log_prod_data(df, probas)

    threshold = float(req.threshold or 0.5)

    results = [
        {
            "bad_proba": float(p),
            "bad_pred": int(p >= threshold)
        }
        for p in probas
    ]

    return {
        "count": len(results),
        "threshold": threshold,
        "results": results
    }


if __name__ == "__main__":
    import uvicorn
    uvicorn.run("app:app", host="0.0.0.0", port=8000, reload=True)
