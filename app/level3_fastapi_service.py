from typing import Literal

from fastapi import FastAPI
from pydantic import BaseModel, Field

from app.level1_pre_post_processing import run_pipeline


app = FastAPI(
    title="Realtime ML Serving Lab",
    description="Level 3: FastAPI model serving for fake camera events",
    version="0.1.0",
)


class CameraEvent(BaseModel):
    event_id: str
    camera_id: str
    location: str = "unknown"
    motion_score: float = Field(..., ge=0.0, le=1.0)
    brightness: float = Field(..., ge=0.0, le=1.0)
    object_size: float = Field(..., ge=0.0, le=1.0)


class PredictionResponse(BaseModel):
    event_id: str
    camera_id: str
    location: str
    label: str
    confidence: float
    action: Literal[
        "alert_customer",
        "escalate_for_review",
        "log_only",
        "suppress",
    ]
    reason: str


@app.get("/health")
def health_check() -> dict:
    return {"status": "ok"}


@app.post("/predict", response_model=PredictionResponse)
def predict(event: CameraEvent) -> dict:
    """
    Simulates a production inference endpoint.

    Flow:
    request payload
    -> validation by Pydantic
    -> preprocessing
    -> model prediction
    -> post-processing decision
    -> JSON response
    """

    result = run_pipeline(event.model_dump())
    return result