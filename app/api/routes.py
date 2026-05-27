"""
API route handlers.

Three concerns kept deliberately separate:
- /predict  → real-time classification
- /feedback → correction loop for continuous improvement
- /train    → model lifecycle management
- /evaluate → transparency endpoint (shows CV metrics, production readiness)
- /health   → operational readiness
"""

import json
import logging
from pathlib import Path

from fastapi import APIRouter, HTTPException, status

from app.ml.classifier import CONFIDENCE_THRESHOLD, InvoiceClassifier, train
from app.schemas.invoice import (
    EvaluationResponse,
    FeedbackRequest,
    FeedbackResponse,
    HealthResponse,
    PredictRequest,
    PredictResponse,
    TrainResponse,
)

logger = logging.getLogger(__name__)
router = APIRouter()

DATA_PATH = Path("data/training_data.json")
METRICS_PATH = Path("models/last_train_metrics.json")


# ── Health ────────────────────────────────────────────────────────────────────

@router.get(
    "/health",
    response_model=HealthResponse,
    tags=["System"],
    summary="Operational health check",
)
async def health():
    return HealthResponse(
        status="ok",
        model_loaded=InvoiceClassifier._pipeline is not None,
        version="2.0.0",
        confidence_threshold=CONFIDENCE_THRESHOLD,
    )


# ── Predict ───────────────────────────────────────────────────────────────────

@router.post(
    "/predict",
    response_model=PredictResponse,
    status_code=status.HTTP_200_OK,
    tags=["Classification"],
    summary="Classify an invoice description into an expense category",
    response_description=(
        "Predicted category with confidence score, human-review flag, "
        "full probability distribution, and GST/ITC guidance."
    ),
)
async def predict(payload: PredictRequest):
    """
    Accepts free-text invoice or expense descriptions and returns:

    - **category**: predicted expense bucket
    - **confidence**: model certainty (0–1)
    - **review_recommended**: `true` when confidence < 0.72 — safe to auto-approve above threshold
    - **scores**: full probability distribution for all categories
    - **gst**: ITC eligibility and HSN/SAC hint aligned to Indian GST law

    Designed for integration into GST filing and ERP expense routing workflows.
    """
    try:
        result = InvoiceClassifier.predict(payload.text)
        return PredictResponse(**result)
    except RuntimeError as exc:
        logger.error("Classifier not initialised: %s", exc)
        raise HTTPException(status_code=503, detail="Model not ready. Retry in a moment.")
    except Exception:
        logger.exception("Prediction error for text: %r", payload.text)
        raise HTTPException(status_code=500, detail="Internal classification error.")


# ── Feedback ──────────────────────────────────────────────────────────────────

@router.post(
    "/feedback",
    response_model=FeedbackResponse,
    status_code=status.HTTP_200_OK,
    tags=["Continuous Improvement"],
    summary="Submit a correction to improve future predictions",
)
async def feedback(payload: FeedbackRequest):
    """
    Records a corrected prediction into the training dataset.

    This is the feedback flywheel: every correction makes the next retrain
    more accurate. In production, call `/train` periodically after collecting
    a batch of corrections.
    """
    if not DATA_PATH.exists():
        raise HTTPException(status_code=404, detail="Training data file not found.")

    try:
        InvoiceClassifier.record_feedback(payload.text, payload.correct_category, DATA_PATH)
        with open(DATA_PATH) as f:
            total = len(json.load(f))
        return FeedbackResponse(
            message="Correction recorded. Run /train to apply.",
            recorded_text=payload.text,
            correct_category=payload.correct_category,
            total_training_samples=total,
        )
    except Exception:
        logger.exception("Feedback recording failed")
        raise HTTPException(status_code=500, detail="Failed to record feedback.")


# ── Train ─────────────────────────────────────────────────────────────────────

@router.post(
    "/train",
    response_model=TrainResponse,
    status_code=status.HTTP_200_OK,
    tags=["Model Management"],
    summary="Retrain the classifier on current training data",
)
async def retrain():
    """
    Triggers a full retrain on `data/training_data.json` — including any
    feedback corrections — and hot-reloads the model in-process.

    Returns CV F1 score and a `production_ready` flag (True when F1 >= 0.85).
    In a production pipeline, gate auto-deployment on this flag.
    """
    if not DATA_PATH.exists():
        raise HTTPException(status_code=404, detail="Training data not found.")
    try:
        metrics = train(DATA_PATH)
        InvoiceClassifier.load()
        return TrainResponse(message="Model retrained and hot-reloaded.", **metrics)
    except Exception:
        logger.exception("Training failed")
        raise HTTPException(status_code=500, detail="Training error. Check server logs.")


# ── Evaluate ──────────────────────────────────────────────────────────────────

@router.get(
    "/evaluate",
    response_model=EvaluationResponse,
    tags=["Model Management"],
    summary="Fetch the latest model evaluation metrics",
)
async def evaluate():
    """
    Returns CV F1 score, class list, sample count, and production readiness
    from the last training run — without retraining.

    Use this to expose model health to dashboards and alerting systems.
    """
    if not METRICS_PATH.exists():
        raise HTTPException(
            status_code=404,
            detail="No metrics found. Run /train first.",
        )
    with open(METRICS_PATH) as f:
        m = json.load(f)
    return EvaluationResponse(**m)
