import asyncio
from typing import Any, Dict, List

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
    async def process(self, raw_event: Dict[str, Any]) -> Dict[str, Any]:
        return preprocess_event(raw_event)


@serve.deployment(num_replicas=1)
class BatchedClassifier:
    """
    Classifier with Ray Serve dynamic batching.

    Ray Serve can collect multiple incoming requests and send them
    to this method as a list.
    """

    @serve.batch(max_batch_size=4, batch_wait_timeout_s=0.1)
    async def predict_batch(
        self,
        events: List[Dict[str, Any]],
    ) -> List[Dict[str, Any]]:
        print(f"Processing batch of size: {len(events)}")

        predictions = []
        for event in events:
            prediction = model_predict(event)
            predictions.append(prediction)

        return predictions


@serve.deployment(num_replicas=1)
class DecisionService:
    async def decide(self, prediction: Dict[str, Any]) -> Dict[str, Any]:
        return postprocess_decision(prediction)


@serve.deployment(num_replicas=1)
class CameraEventPipeline:
    def __init__(self, preprocessor, classifier, decision_service):
        self.preprocessor = preprocessor
        self.classifier = classifier
        self.decision_service = decision_service

    async def __call__(self, request: Request) -> Dict[str, Any]:
        raw_event = await request.json()

        preprocessed = await self.preprocessor.process.remote(raw_event)

        # Even though each request calls this once, Ray Serve can batch
        # multiple concurrent calls to predict_batch().
        prediction = await self.classifier.predict_batch.remote(preprocessed)

        decision = await self.decision_service.decide.remote(prediction)

        return decision


app = CameraEventPipeline.bind(
    Preprocessor.bind(),
    BatchedClassifier.bind(),
    DecisionService.bind(),
)


def generate_event(event_id: int) -> Dict[str, Any]:
    return {
        "event_id": f"evt-batch-{event_id:03d}",
        "camera_id": "front-door" if event_id % 2 == 0 else "backyard",
        "location": "front_yard" if event_id % 2 == 0 else "backyard",
        "motion_score": 0.90 if event_id % 2 == 0 else 0.25,
        "brightness": 0.50,
        "object_size": 0.70 if event_id % 2 == 0 else 0.10,
    }


async def send_request(event: Dict[str, Any]) -> Dict[str, Any]:
    """
    Sends request in background thread so we can create concurrent HTTP calls.
    """

    def _post():
        response = requests.post(
            "http://127.0.0.1:8000/",
            json=event,
            timeout=10,
        )
        return response.json()

    return await asyncio.to_thread(_post)


async def run_concurrent_test() -> None:
    events = [generate_event(i) for i in range(1, 9)]

    tasks = [send_request(event) for event in events]
    results = await asyncio.gather(*tasks)

    print("\nResults:")
    for result in results:
        print(result)


if __name__ == "__main__":
    serve.run(app, route_prefix="/")

    print("Ray Serve batching pipeline is running at http://127.0.0.1:8000")
    print("Sending 8 concurrent requests...\n")

    asyncio.run(run_concurrent_test())