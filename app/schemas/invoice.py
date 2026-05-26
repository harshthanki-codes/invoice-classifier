from typing import Dict, Optional
from pydantic import BaseModel, Field, field_validator


class PredictRequest(BaseModel):
    text: str = Field(..., min_length=3, max_length=1000, description="Invoice or expense description text")

    @field_validator("text")
    @classmethod
    def text_must_not_be_blank(cls, v: str) -> str:
        if not v.strip():
            raise ValueError("text must not be blank or whitespace")
        return v.strip()

    model_config = {
        "json_schema_extra": {
            "examples": [
                {"text": "AWS monthly cloud hosting bill"},
                {"text": "Blue Dart courier charges for warehouse delivery"},
            ]
        }
    }


class PredictResponse(BaseModel):
    category: str = Field(..., description="Predicted expense category")
    confidence: float = Field(..., ge=0.0, le=1.0, description="Model confidence score (0–1)")
    scores: Dict[str, float] = Field(..., description="Probability distribution across all categories")


class HealthResponse(BaseModel):
    status: str
    is_model_loaded: bool
    version: str


class TrainResponse(BaseModel):
    message: str
    cv_f1_mean: float
    cv_f1_std: float
    classes: list
    num_samples: int
