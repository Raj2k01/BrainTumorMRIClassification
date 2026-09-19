from typing import Dict
from typing import List

from pydantic import BaseModel
from pydantic import Field


class HealthResponse(BaseModel):
    status: str
    model_loaded: bool
    device: str
    environment: str


class ModelInfoResponse(BaseModel):
    model_name: str
    model_version: str
    framework: str
    classes: List[str]
    image_size: int
    device: str
    best_epoch: int
    best_validation_loss: float
    best_validation_macro_f1: float
    medical_diagnostic_use: bool


class PredictionResponse(BaseModel):
    filename: str
    predicted_class: str
    confidence: float = Field(
        ge=0.0,
        le=1.0
    )
    requires_review: bool
    confidence_threshold: float = Field(
        ge=0.0,
        le=1.0
    )
    class_probabilities: Dict[
        str,
        float
    ]
    disclaimer: str