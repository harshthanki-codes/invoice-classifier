import logging
from pathlib import Path

from fastapi import APIRouter, HTTPException, status

from app.ml.classifier import InvoiceClassifier, train
from app.schemas.invoice import HealthResponse, PredictRequest, PredictResponse, TrainResponse

logger = logging.getLogger(__name__)
router = APIRouter()


@router.get("/health", response_model=HealthResponse, tags=["System"])
async def health_check():
    return HealthResponse(
    status="ok",
    is_model_loaded=InvoiceClassifier._pipeline is not None,
    version="1.0.0",
)


@router.post(
    "/predict",
    response_model=PredictResponse,
    status_code=status.HTTP_200_OK,
    tags=["Classification"],
    summary="Classify an invoice description into an expense category",
)
async def predict(payload: PredictRequest):
    """
    Accepts a free-text invoice or expense description and returns the predicted
    expense category along with a confidence score and full probability distribution.
    """
    try:
        result = InvoiceClassifier.predict(payload.text)
        return PredictResponse(**result)
    except RuntimeError as e:
        logger.error("Classifier not ready: %s", e)
        raise HTTPException(status_code=503, detail="Model not loaded. Try again shortly.")
    except Exception as e:
        logger.exception("Unexpected error during prediction for text: %r", payload.text)
        raise HTTPException(status_code=500, detail="Internal classification error.")


@router.post(
    "/train",
    response_model=TrainResponse,
    status_code=status.HTTP_200_OK,
    tags=["Model Management"],
    summary="Retrain the classifier on the training dataset",
)
async def retrain():
    """
    Triggers a full retrain of the classifier using the data at data/training_data.json.
    Reloads the model in-memory after training completes. Useful after updating training data.
    """
    data_path = Path("data/training_data.json")
    if not data_path.exists():
        raise HTTPException(status_code=404, detail="Training data not found at data/training_data.json")

    try:
        metrics = train(data_path)
        InvoiceClassifier.load()
        return TrainResponse(message="Model retrained and reloaded successfully.", **metrics)
    except Exception as e:
        logger.exception("Training failed")
        raise HTTPException(status_code=500, detail=f"Training error: {str(e)}")
