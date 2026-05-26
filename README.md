# MTCARS MPG Predictor — FastAPI + Podman

A production-ready REST API that predicts a car's fuel efficiency (**mpg**) from its weight and horsepower, trained on the classic [Motor Trend Car Road Tests (mtcars)](https://stat.ethz.ch/R-manual/R-devel/library/datasets/html/mtcars.html) dataset.

## Project Overview

This project demonstrates a complete ML deployment pipeline:

1. **Model training** – linear regression (scikit-learn) trained on `mtcars.csv`
2. **API serving** – FastAPI with Pydantic validation and structured logging
3. **Containerisation** – Dockerfile for Podman / Docker
4. **Deployment** – Google Cloud Run

---

## Model Description

| Property | Value |
|---|---|
| Algorithm | Ordinary Least Squares Linear Regression |
| Response variable | `mpg` (miles per gallon) |
| Predictors | `wt` (vehicle weight, 1000 lbs) · `hp` (gross horsepower) |
| Library | `scikit-learn` |
| Artifact | `models/model.pkl` (saved with pickle) |

Both `wt` and `hp` show strong negative correlation with `mpg` in the mtcars dataset: heavier and more powerful cars consume more fuel. Together they explain roughly 83 % of the variance in mpg (R² ≈ 0.83 on the full dataset).

---

## Repository Structure

```
mtcars-fastapi-api/
├── app/
│   ├── __init__.py
│   └── main.py           # FastAPI application
├── models/
│   └── model.pkl         # Trained model artifact
├── scripts/
│   └── train_model.py    # Training script
├── tests/
│   └── test_api.py       # Automated API tests
├── mtcars.csv            # Dataset
├── Dockerfile
├── .dockerignore
├── requirements.txt
└── README.md
```

---

## Local Setup

### Prerequisites

- Python 3.11+
- `pip` or `uv`

### 1 — Create a virtual environment

```bash
python -m venv .venv
source .venv/bin/activate      # macOS / Linux
# .venv\Scripts\activate       # Windows
```

Or with `uv`:

```bash
uv venv
source .venv/bin/activate
```

### 2 — Install dependencies

```bash
pip install -r requirements.txt
# or
uv pip install -r requirements.txt
```

### 3 — Train the model

```bash
python scripts/train_model.py
```

This reads `mtcars.csv`, fits a linear regression model, prints evaluation metrics, and saves `models/model.pkl`.

### 4 — Run the API locally

```bash
uvicorn app.main:app --host 0.0.0.0 --port 8080 --reload
```

The API is now available at <http://127.0.0.1:8080>.  
Interactive docs: <http://127.0.0.1:8080/docs>

---

## API Endpoints

### `GET /health`

Liveness check — always returns 200 while the process is running.

```bash
curl http://localhost:8080/health
# {"status":"ok"}
```

### `GET /ready`

Readiness check — returns 200 only when the model is loaded.

```bash
curl http://localhost:8080/ready
# {"status":"ready","model_loaded":true}
```

### `POST /predict`

Predict mpg from weight and horsepower.

**Request body**

| Field | Type | Constraint | Description |
|---|---|---|---|
| `wt` | float | > 0 | Vehicle weight in 1000 lbs |
| `hp` | float | > 0 | Gross horsepower |

**Example request**

```bash
curl -X POST "http://localhost:8080/predict" \
  -H "Content-Type: application/json" \
  -d '{"wt": 2.62, "hp": 110}'
```

**Example response**

```json
{
  "predicted_mpg": 23.17,
  "model_version": "1.0.0",
  "predictors": {
    "wt": 2.62,
    "hp": 110.0
  }
}
```

**Error responses**

| Status | Cause |
|---|---|
| 422 | Missing or invalid field (Pydantic validation) |
| 503 | Model not loaded |
| 500 | Internal prediction error |

---

## Running with Podman

### Build the image

```bash
podman build -t mtcars-api .
```

### Run the container

```bash
podman run --rm -p 127.0.0.1:8080:8080 mtcars-api
```

### Test it

```bash
curl -X POST "http://localhost:8080/predict" \
  -H "Content-Type: application/json" \
  -d '{"wt": 3.5, "hp": 180}'
```

---

## Running Tests

```bash
pytest tests/ -v
```

The test suite covers:

- `/health` liveness
- `/ready` readiness
- Valid `/predict` requests (reasonable output, response shape)
- Missing fields → 422
- Invalid types → 422
- Zero / negative values → 422

---

## Deployment to Google Cloud Run

### 1 — Authenticate and configure

```bash
gcloud auth login
gcloud config set project mtcars-fastapi-api
gcloud auth configure-docker
```

### 2 — Build and tag the image

```bash
podman build -t mtcars-api .
podman tag mtcars-api gcr.io/mtcars-fastapi-api/mtcars-api
```

### 3 — Push to Container Registry

```bash
podman push gcr.io/mtcars-fastapi-api/mtcars-api
```

### 4 — Deploy to Cloud Run

```bash
gcloud run deploy mtcars-api \
  --image gcr.io/mtcars-fastapi-api/mtcars-api \
  --platform managed \
  --region us-central1 \
  --allow-unauthenticated \
  --port 8080
```

### Deployed API URL

API root:

https://mtcars-api-790966735586.us-central1.run.app

Interactive docs:

https://mtcars-api-790966735586.us-central1.run.app/docs

---

## Environment Variables

| Variable | Default | Description |
|---|---|---|
| `MODEL_PATH` | `models/model.pkl` | Override model file location |

---

## Reproducibility Checklist

- [x] `mtcars.csv` included in the repo
- [x] Training script at `scripts/train_model.py` reproduces `models/model.pkl`
- [x] `requirements.txt` pins all dependencies
- [x] `Dockerfile` builds a self-contained image
- [x] `README.md` explains every step from clone to API call
