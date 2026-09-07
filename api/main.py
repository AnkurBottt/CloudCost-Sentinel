from typing import Annotated

import joblib
import mlflow
import mlflow.xgboost
import numpy as np
import pandas as pd
import uvicorn
from fastapi import Depends, FastAPI, HTTPException
from pydantic import BaseModel
from sqlalchemy import select
from sqlalchemy.orm import Session

from api.database import Base, engine, get_db
from api.db_models import Prediction


# create the table if it does not exist yet
Base.metadata.create_all(bind=engine)

from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent.parent

MODEL_PATH = BASE_DIR / "models" / "cloudcost" / "xgb_optuna_model.pkl"

FEATURE_COLUMNS_PATH = (
    BASE_DIR / "models" / "cloudcost" / "xgb_optuna_feature_columns.pkl"
)

THRESHOLD_PATH = (
    BASE_DIR / "models" / "cloudcost" / "xgb_optuna_threshold.txt"
)

#model = mlflow.xgboost.load_model(str(MODEL_PATH))

feature_columns = joblib.load(FEATURE_COLUMNS_PATH)

with open(THRESHOLD_PATH, "r") as f:
    threshold = float(f.read())

# load once when the API starts, not again for every request
model = joblib.load(MODEL_PATH)

# these names are taken from the exact one-hot columns seen in training
VALID_PROVIDERS = {col.replace("ProviderName_", "") for col in feature_columns if col.startswith("ProviderName_")}
VALID_SERVICES = {col.replace("ServiceName_", "") for col in feature_columns if col.startswith("ServiceName_")}
VALID_REGIONS = {col.replace("RegionName_", "") for col in feature_columns if col.startswith("RegionName_")}

app = FastAPI()

# pydantic schemas for req nd resp validation
class PredictionRequest(BaseModel):
    date: str
    provider: str
    service: str
    region: str
    daily_cost: float
    usage_quantity: float
    resource_count: int


class PredictionResponse(BaseModel):
    probability: float
    threshold: float
    is_anomaly: int

@app.get("/health")
def health():
    return {"status": "ok"}


@app.post("/predict", response_model=PredictionResponse)
def predict(data: PredictionRequest, db: Annotated[Session, Depends(get_db)]):
    # pydantic checks the request shape/types, these checks make sure the
    # category names are ones the trained model actually knows
    if data.provider not in VALID_PROVIDERS:
        raise HTTPException(status_code=400, detail=f"Unknown provider: {data.provider}. Must be one of the trained providers.")
    if data.service not in VALID_SERVICES:
        raise HTTPException(status_code=400, detail=f"Unknown service: {data.service}. Must be one of the trained services.")
    if data.region not in VALID_REGIONS:
        raise HTTPException(status_code=400, detail=f"Unknown region: {data.region}. Must be one of the trained regions.")

    # get up to 7 older rows for the same provider/service/region
    # current date is excluded so it cannot become part of its own history
    stmt = (
        select(Prediction)
        .where(
            Prediction.provider == data.provider,
            Prediction.service == data.service,
            Prediction.region == data.region,
            Prediction.date < data.date,
        )
        .order_by(Prediction.date.desc())
        .limit(7)
    )
    history = db.execute(stmt).scalars().all()
    recent_costs = [row.daily_cost for row in history]  # newest -> oldest

    if recent_costs:
        previous_cost = recent_costs[0]
        rolling_7d_avg = sum(recent_costs) / len(recent_costs)
        rolling_7d_median = float(np.median(recent_costs))  # median is less affected by extreme vals
    else:
        # cold start: no older rows exist for this exact context yet
        previous_cost = data.daily_cost
        rolling_7d_avg = data.daily_cost
        rolling_7d_median = data.daily_cost

    cost_vs_7d_avg = data.daily_cost / rolling_7d_avg if rolling_7d_avg > 0 else 1.0
    cost_vs_7d_median = data.daily_cost / rolling_7d_median if rolling_7d_median > 0 else 1.0
    day_of_week = pd.to_datetime(data.date).dayofweek

    # build the same feature set that was used while training
    X = pd.DataFrame([{
        "daily_cost": data.daily_cost,
        "usage_quantity": data.usage_quantity,
        "resource_count": data.resource_count,
        "previous_cost": previous_cost,
        "rolling_7d_avg": rolling_7d_avg,
        "rolling_7d_median": rolling_7d_median,
        "cost_vs_7d_avg": cost_vs_7d_avg,
        "cost_vs_7d_median": cost_vs_7d_median,
        "day_of_week": day_of_week,
        "ProviderName": data.provider,
        "ServiceName": data.service,
        "RegionName": data.region,
    }])

    # strings -> one-hot numbers, then restore the exact train-time columns/order
    X = pd.get_dummies(X, columns=["ProviderName", "ServiceName", "RegionName"], dtype=int)
    X = X.reindex(columns=feature_columns, fill_value=0)

    # predict_proba gives probabilities for class 0 and class 1;
    # [:, 1][0] gives class-1(anomaly) probability for this one request
    probability = model.predict_proba(X)[:, 1][0]
    is_anomaly = int(probability >= threshold)

    # save this row after prediction so it can be history for future requests
    new_prediction = Prediction(
        date=data.date,
        provider=data.provider,
        service=data.service,
        region=data.region,
        daily_cost=data.daily_cost,
        usage_quantity=data.usage_quantity,
        resource_count=data.resource_count,
        probability=float(probability),
        is_anomaly=is_anomaly,
    )
    db.add(new_prediction)
    db.commit()

    return {
        "probability": float(probability),
        "threshold": threshold,
        "is_anomaly": is_anomaly,
    }


if __name__ == "__main__":
    uvicorn.run(app, host="127.0.0.1", port=8001)

# dev: uvicorn api.main:app --reload --port 8001
# swagger: http://127.0.0.1:8001/docs
