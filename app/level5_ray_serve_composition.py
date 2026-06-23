from typing import Any, Dict

import requests
from starlette.requests import Request

from ray import serve

from app.level1_pre_post_processing import (
    preprocess_event,
    model_predict,
    postprocess_decision,
)


@serve.deployment(num_replicas=1)
class Preprocessor:
    """
    Ray Serve deployment for preprocessing.

    In a real CV system, this could handle:
    - frame sampling
    - resizing
    - normalization
    - metadata enrichment
    """

    async def process(self, raw_event: Dict[str, Any]) -> Dict[str, Any]:
        return preprocess_event(raw_event)


@serve.deployment(num_replicas=1)
class Classifier:
    """
    Ray Serve deployment for model inference.

    In a real system, this would load a CV model and run inference.
    This deployment would likely need GPU resources.
    """

    async def predict(self, event: Dict[str, Any]) -> Dict[str, Any]:
        return model_predict(event)


@serve.deployment(num_replicas=1)
class DecisionService:
    """
    Ray Serve deployment for post-processing and product decisioning.

    This converts model output into an action:
    - alert_customer
    - escalate_for_review
    - log_only
    - suppress
    """

    async def decide(self, prediction: Dict[str, Any]) -> Dict[str, Any]:
        return postprocess_decision(prediction)


@serve.deployment(num_replicas=1)
class CameraEventPipeline:
    """
    Orchestrates the full pipeline by calling other Ray Serve deployments.
    """

    def __init__(self, preprocessor, classifier, decision_service):
        self.preprocessor = preprocessor
        self.classifier = classifier
        self.decision_service = decision_service

    async def __call__(self, request: Request) -> Dict[str, Any]:
        raw_event = await request.json()

        preprocessed = await self.preprocessor.process.remote(raw_event)
        prediction = await self.classifier.predict.remote(preprocessed)
        decision = await self.decision_service.decide.remote(prediction)

        return decision


app = CameraEventPipeline.bind(
    Preprocessor.bind(),
    Classifier.bind(),
    DecisionService.bind(),
)


if __name__ == "__main__":
    serve.run(app, route_prefix="/")

    print("Ray Serve composed pipeline is running at http://127.0.0.1:8000")
    print("Try POST http://127.0.0.1:8000/")

    sample_event = {
        "event_id": "evt-compose-001",
        "camera_id": "front-door",
        "location": "front_yard",
        "motion_score": 0.94,
        "brightness": 0.45,
        "object_size": 0.76,
    }

    response = requests.post("http://127.0.0.1:8000/", json=sample_event, timeout=10)
    print(response.json())