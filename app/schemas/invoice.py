"""
Pydantic v2 request/response contracts.

Every field is typed, documented, and range-constrained. This is the public
API surface — changes here are breaking changes and must be versioned.
"""

from typing import Dict, Optional, Union
from pydantic import BaseModel, Field, field_validator


class PredictRequest(BaseModel):
    text: str = Field(
        ...,
        min_length=3,
        max_length=1000,
        description="Free-text invoice or expense description",
        examples=["AWS monthly cloud hosting bill", "Blue Dart courier charges for warehouse delivery"],
    )

    @field_validator("text")
    @classmethod
    def text_not_blank(cls, v: str) -> str:
        if not v.strip():
            raise ValueError("text must not be blank or whitespace only")
        return v.strip()


class GSTInfo(BaseModel):
    itc_eligible: Union[bool, str, None] = Field(
        description="Whether input tax credit is claimable. 'partial' means category-dependent."
    )
    hsn_sac_hint: Optional[str] = Field(description="Likely HSN/SAC code range for this expense")
    note: Optional[str] = Field(description="GST treatment guidance for this category")


class PredictResponse(BaseModel):
    category: str = Field(description="Predicted expense category")
    confidence: float = Field(ge=0.0, le=1.0, description="Model confidence (0–1)")
    review_recommended: bool = Field(
        description="True when confidence < 0.72 — flag for human verification"
    )
    scores: Dict[str, float] = Field(description="Full probability distribution across all categories")
    gst: GSTInfo = Field(description="GST/ITC guidance for the predicted category")


class FeedbackRequest(BaseModel):
    text: str = Field(..., min_length=3, max_length=1000, description="Original invoice text")
    correct_category: str = Field(..., description="The correct expense category")

    @field_validator("correct_category")
    @classmethod
    def valid_category(cls, v: str) -> str:
        allowed = {"Logistics", "Office Supplies", "Cloud/Software", "Utilities", "Travel", "Inventory"}
        if v not in allowed:
            raise ValueError(f"correct_category must be one of: {sorted(allowed)}")
        return v


class FeedbackResponse(BaseModel):
    message: str
    recorded_text: str
    correct_category: str
    total_training_samples: int


class HealthResponse(BaseModel):
    model_config = {"protected_namespaces": ()}
    status: str
    model_loaded: bool
    version: str
    confidence_threshold: float


class TrainResponse(BaseModel):
    message: str
    cv_f1_mean: float
    cv_f1_std: float
    classes: list
    num_samples: int
    confidence_threshold: float
    production_ready: bool


class EvaluationResponse(BaseModel):
    cv_f1_mean: float
    cv_f1_std: float
    num_samples: int
    classes: list
    production_ready: bool
    confidence_threshold: float
