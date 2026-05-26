"""
app/main.py
-----------
FastAPI application that serves predictions from a linear regression model
trained on the mtcars dataset.

Endpoints:
  GET  /health   – liveness check
  GET  /ready    – readiness check (model loaded?)
  POST /predict  – predict mpg from wt and hp
"""

import logging
import os
import pathlib
from contextlib import asynccontextmanager

import pickle
import numpy as np
from fastapi import FastAPI, HTTPException
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field, field_validator

# ── Logging ────────────────────────────────────────────────────────────────
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger(__name__)

# ── Model path (override with MODEL_PATH env var for flexibility) ──────────
DEFAULT_MODEL_PATH = pathlib.Path(__file__).resolve().parent.parent / "models" / "model.pkl"
MODEL_PATH = pathlib.Path(os.getenv("MODEL_PATH", str(DEFAULT_MODEL_PATH)))

# ── Global model store ─────────────────────────────────────────────────────
_model = None


def load_model():
    """Load the model from disk into the global store."""
    global _model
    if not MODEL_PATH.exists():
        logger.error(f"Model file not found at {MODEL_PATH}")
        _model = None
        return
    try:
        with open(MODEL_PATH, "rb") as f:
            _model = pickle.load(f)
        logger.info(f"Model loaded successfully from {MODEL_PATH}")
    except Exception as exc:
        logger.exception(f"Failed to load model: {exc}")
        _model = None


# ── Lifespan (startup / shutdown) ──────────────────────────────────────────
@asynccontextmanager
async def lifespan(app: FastAPI):
    load_model()
    yield
    logger.info("Application shutting down")


# ── FastAPI app ────────────────────────────────────────────────────────────
app = FastAPI(
    title="MTCARS MPG Predictor",
    description=(
        "Predicts miles-per-gallon (mpg) for a car given its weight (wt) "
        "and horsepower (hp), using a linear regression model trained on "
        "the classic mtcars dataset."
    ),
    version="1.0.0",
    lifespan=lifespan,
)


# ── Pydantic schemas ────────────────────────────────────────────────────────
class PredictionRequest(BaseModel):
    wt: float = Field(
        ...,
        gt=0,
        description="Vehicle weight in 1000 lbs (e.g. 2.62 means 2620 lbs)",
        examples=[2.62],
    )
    hp: float = Field(
        ...,
        gt=0,
        description="Gross horsepower (e.g. 110)",
        examples=[110.0],
    )

    @field_validator("wt", "hp")
    @classmethod
    def must_be_positive(cls, v: float, info) -> float:
        if v <= 0:
            raise ValueError(f"{info.field_name} must be a positive number")
        return v


class PredictionResponse(BaseModel):
    predicted_mpg: float
    model_version: str = "1.0.0"
    predictors: dict


class HealthResponse(BaseModel):
    status: str


class ReadyResponse(BaseModel):
    status: str
    model_loaded: bool


# ── Endpoints ──────────────────────────────────────────────────────────────

@app.get(
    "/health",
    response_model=HealthResponse,
    summary="Liveness check",
    tags=["monitoring"],
)
def health() -> HealthResponse:
    """Returns 200 OK if the application process is running."""
    return HealthResponse(status="ok")


@app.get(
    "/ready",
    response_model=ReadyResponse,
    summary="Readiness check",
    tags=["monitoring"],
)
def ready() -> ReadyResponse:
    """
    Returns 200 if the model is loaded and ready to serve predictions.
    Returns 503 if the model is unavailable.
    """
    if _model is None:
        raise HTTPException(
            status_code=503,
            detail="Model is not loaded. Check that models/model.pkl exists.",
        )
    return ReadyResponse(status="ready", model_loaded=True)


@app.post(
    "/predict",
    response_model=PredictionResponse,
    summary="Predict MPG",
    tags=["prediction"],
)
def predict(request: PredictionRequest) -> PredictionResponse:
    """
    Predict miles per gallon (mpg) given vehicle weight and horsepower.

    - **wt**: weight in 1000 lbs (e.g. 2.62)
    - **hp**: gross horsepower (e.g. 110)
    """
    if _model is None:
        raise HTTPException(
            status_code=503,
            detail="Model is not loaded. Run scripts/train_model.py first.",
        )

    features = np.array([[request.wt, request.hp]])
    try:
        prediction: float = float(_model.predict(features)[0])
    except Exception as exc:
        logger.exception("Prediction failed")
        raise HTTPException(status_code=500, detail=f"Prediction error: {exc}")

    logger.info(f"Prediction: wt={request.wt}, hp={request.hp} → mpg={prediction:.2f}")

    return PredictionResponse(
        predicted_mpg=round(prediction, 2),
        predictors={"wt": request.wt, "hp": request.hp},
    )


# ── Load model at import time (ensures TestClient picks it up without lifespan) ─
load_model()

# ── Optional: run directly ──────────────────────────────────────────────────
if __name__ == "__main__":
    import uvicorn
    uvicorn.run("app.main:app", host="0.0.0.0", port=8080, reload=False)
