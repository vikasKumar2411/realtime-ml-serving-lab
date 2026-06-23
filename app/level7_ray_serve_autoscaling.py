import asyncio
import time
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
    async def process(self, raw_event: Dict[str, Any]) -> Dict[str, Any]:
        return preprocess_event(raw_event)


@serve.deployment(
    autoscaling_config={
        "min_replicas": 1,
        "max_replicas": 4,
        "target_ongoing_requests": 2,
        "upscale_delay_s": 1,
        "downscale_delay_s": 10,
    }
)
class AutoscalingClassifier:
    """
    Classifier with Ray Serve autoscaling.

    Ray Serve can add/remove replicas based on ongoing requests.
    """

    async def predict(self, event: Dict[str, Any]) -> Dict[str, Any]:
        # Simulate slower model inference so autoscaling has a reason to react.
        await asyncio.sleep(0.25)

        prediction = model_predict(event)
        return prediction


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
        prediction = await self.classifier.predict.remote(preprocessed)
        decision = await self.decision_service.decide.remote(prediction)

        return decision


app = CameraEventPipeline.bind(
    Preprocessor.bind(),
    AutoscalingClassifier.bind(),
    DecisionService.bind(),
)


def generate_event(event_id: int) -> Dict[str, Any]:
    return {
        "event_id": f"evt-scale-{event_id:03d}",
        "camera_id": "front-door" if event_id % 2 == 0 else "backyard",
        "location": "front_yard" if event_id % 2 == 0 else "backyard",
        "motion_score": 0.90 if event_id % 2 == 0 else 0.25,
        "brightness": 0.50,
        "object_size": 0.70 if event_id % 2 == 0 else 0.10,
    }


async def send_request(event: Dict[str, Any]) -> Dict[str, Any]:
    def _post():
        start = time.perf_counter()
        response = requests.post(
            "http://127.0.0.1:8000/",
            json=event,
            timeout=20,
        )
        latency_ms = round((time.perf_counter() - start) * 1000, 2)
        result = response.json()
        result["client_latency_ms"] = latency_ms
        return result

    return await asyncio.to_thread(_post)


async def run_load_test(total_requests: int = 30) -> None:
    events = [generate_event(i) for i in range(1, total_requests + 1)]

    print(f"Sending {total_requests} concurrent requests...\n")

    tasks = [send_request(event) for event in events]
    results = await asyncio.gather(*tasks)

    latencies = [r["client_latency_ms"] for r in results]
    avg_latency = round(sum(latencies) / len(latencies), 2)
    max_latency = round(max(latencies), 2)

    alert_count = sum(1 for r in results if r["action"] == "alert_customer")
    suppress_count = sum(1 for r in results if r["action"] == "suppress")

    print("\nSample results:")
    for result in results[:5]:
        print(result)

    print("\nLoad test summary:")
    print(
        {
            "total_requests": len(results),
            "avg_latency_ms": avg_latency,
            "max_latency_ms": max_latency,
            "alert_customer": alert_count,
            "suppress": suppress_count,
        }
    )


if __name__ == "__main__":
    serve.run(app, route_prefix="/")

    print("Ray Serve autoscaling pipeline is running at http://127.0.0.1:8000")
    print("Open Ray dashboard: http://127.0.0.1:8265")
    print("Watch AutoscalingClassifier replicas while load test runs.\n")

    asyncio.run(run_load_test(total_requests=30))