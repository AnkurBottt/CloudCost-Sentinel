# CloudCost Sentinel

**End-to-end cloud-cost anomaly detection with ML, MLOps experiment tracking, FastAPI serving, and Dockerized deployment.**

> The base billing records are anonymized FinOps FOCUS sample data (CC BY 4.0).
> Supervised anomaly labels are created through reproducible synthetic incident injection for benchmarking; results should not be interpreted as production performance on real incident labels.

## Overview

CloudCost Sentinel detects unusual cloud-spending behavior from FinOps FOCUS billing data. The project covers the full path from raw billing records to a containerized inference API:

- FOCUS billing data from AWS, Azure, GCP, and Oracle sample records
- Daily provider/service/region aggregation and objective data-quality filtering
- Reproducible synthetic anomaly injection
- Leakage-safe historical and rolling features
- Chronological train/validation/test splitting
- Logistic Regression, Neural Network, and XGBoost comparison
- Optuna hyperparameter tuning
- MLflow experiment tracking/model management
- Validation-selected decision threshold
- FastAPI inference with SQLAlchemy + SQLite logging
- Docker deployment with persistent database storage

## Dataset

| Property | Value |
|---|---|
| Source | FinOps Foundation FOCUS Sample Data |
| License | CC BY 4.0 |
| Raw input | 100,000 FOCUS billing records |
| Model-ready observations | 3,932 daily/context observations |
| Synthetic anomalies | 157 (~4%) |
| Split | Chronological 70/15/15 |
| Train | 2,716 rows / 96 anomalies |
| Validation | 615 rows / 31 anomalies |
| Test | 601 rows / 30 anomalies |

Raw records are aggregated by date, provider, service, and region. Dates with fewer than 10 positive-cost contexts are excluded as a data-quality rule because the sample contains a structurally sparse temporal tail for this modeling task.

## Pipeline

```text
FOCUS 100K billing sample
        ↓
Daily provider/service/region aggregation
        ↓
Data-quality filtering
        ↓
Synthetic anomaly injection (~4%)
        ↓
Leakage-safe feature engineering
        ↓
Chronological 70/15/15 split
        ↓
Logistic Regression ─┐
Neural Network ──────┼─→ comparison
XGBoost + Optuna ────┘
        ↓
Validation threshold selection
        ↓
Held-out test evaluation
        ↓
MLflow tracking / model management
        ↓
FastAPI + SQLAlchemy + SQLite
        ↓
Docker container + persistent volume
```

## Feature Engineering

Historical features are calculated within each provider/service/region context, with the current observation shifted out of rolling calculations to avoid leakage. Features include day of week, categorical context, previous cost, rolling 7-observation average/median, cost-relative-to-history features, usage quantity, and resource count.

## Model Results

| Model | Validation F1 | Test Precision | Test Recall | Test F1 |
|---|---:|---:|---:|---:|
| Logistic Regression | 0.4722 | 0.4151 | 0.7333 | 0.5301 |
| Neural Network | 0.5843 | 0.4565 | 0.7000 | 0.5526 |
| **Optuna-tuned XGBoost** | **0.8333** | **0.7778** | **0.9333** | **0.8485** |

Best XGBoost parameters: `n_estimators=52`, `max_depth=5`, `learning_rate=0.1356332575`.

The final decision threshold is **0.15**. It was selected using validation F1 and frozen before held-out test evaluation.

## Serving

FastAPI loads the packaged deployment artifacts directly:

```text
models/cloudcost/xgb_optuna_model.pkl
models/cloudcost/xgb_optuna_feature_columns.pkl
models/cloudcost/xgb_optuna_threshold.txt
```

Each request is transformed into the required feature vector, scored with `predict_proba()`, classified using the frozen threshold, and logged through SQLAlchemy into SQLite.

MLflow remains part of the training/experiment/model-management workflow; the serving container does **not** require a live MLflow server.

## Docker Quick Start

```bash
docker build -t cloudcost-sentinel:v1 .
docker volume create cloudcost-data
docker run --name cloudcost-api -p 8000:8000 -v cloudcost-data:/app/data cloudcost-sentinel:v1
```

PowerShell multiline equivalent:

```powershell
docker run --name cloudcost-api `
  -p 8000:8000 `
  -v cloudcost-data:/app/data `
  cloudcost-sentinel:v1
```

Open Swagger UI at `http://localhost:8000/docs`.

SQLite is stored at `/app/data/cloudcost.db`. The named volume `cloudcost-data` keeps prediction history when the container is deleted and recreated.

## Local Development

```bash
pip install -r requirements.txt
python -m src.data
python -m src.train
uvicorn api.main:app --reload --port 8000
```

> Training entry points should match the current `src/` implementation in the repository.

## Project Structure

```text
cloudCostReducn/
├── api/
│   ├── main.py
│   ├── database.py
│   └── db_models.py
├── data/
├── models/
│   └── cloudcost/
│       ├── xgb_optuna_model.pkl
│       ├── xgb_optuna_feature_columns.pkl
│       └── xgb_optuna_threshold.txt
├── src/
│   ├── data.py
│   ├── features.py
│   ├── evaluate.py
│   └── train.py
├── Dockerfile
├── .dockerignore
├── requirements.txt
└── start_mlflow.ps1
```

Generated/runtime files such as SQLite databases, MLflow tracking databases/artifacts, virtual environments, caches, and large raw data should not be committed.

## MLOps Design

- **Experiment tracking:** MLflow records parameters, metrics, and artifacts.
- **Optimization:** Optuna tunes XGBoost against validation F1.
- **Thresholding:** the decision threshold is selected only on validation data.
- **Deployment artifact:** model + feature schema + threshold are packaged for serving.
- **Containerization:** Docker packages the serving runtime.
- **Persistence:** a Docker volume keeps SQLite history independent of the container lifecycle.

## Limitations

- Labels are synthetic and provide a controlled benchmark, not real production incident ground truth.
- The FOCUS sample has temporal sparsity requiring a documented data-quality filter.
- Unknown categorical values at serving time are rejected.
- SQLite is suitable for this V1/demo workload, not high-concurrency production.
- Automated retraining, drift monitoring, and production cloud deployment are outside V1.

## Status

- [x] FOCUS 100K ingestion and aggregation
- [x] Data-quality filtering
- [x] Synthetic anomaly injection
- [x] Leakage-safe feature engineering
- [x] Chronological split
- [x] Logistic Regression + Neural Network comparison
- [x] Optuna-tuned XGBoost
- [x] MLflow tracking/model management
- [x] Validation-based threshold selection
- [x] FastAPI inference
- [x] SQLAlchemy + SQLite logging
- [x] Docker containerization
- [x] Persistent Docker volume

## Attribution

Dataset: **FinOps Foundation — FOCUS Sample Data**, licensed under **CC BY 4.0**.
