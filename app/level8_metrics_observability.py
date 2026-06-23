import asyncio
import statistics
import time
from collections import Counter
from typing import Any, Dict, List

import requests
from starlette.requests import Request
from starlette.responses import JSONResponse

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
class Classifier:
    async def predict(self, event: Dict[str, Any]) -> Dict[str, Any]:
        await asyncio.sleep(0.05)
        return model_predict(event)


@serve.deployment(num_replicas=1)
class DecisionService:
    async def decide(self, prediction: Dict[str, Any]) -> Dict[str, Any]:
        return postprocess_decision(prediction)


@serve.deployment(num_replicas=1)
class MetricsCollector:
    def __init__(self):
        self.records: List[Dict[str, Any]] = []

    async def record(self, record: Dict[str, Any]) -> None:
        self.records.append(record)

    async def summary(self) -> Dict[str, Any]:
        total = len(self.records)

        if total == 0:
            return {"total_requests": 0}

        latencies = [r["latency_ms"] for r in self.records]
        labels = Counter(r["label"] for r in self.records)
        actions = Counter(r["action"] for r in self.records)
        statuses = Counter(r["status"] for r in self.records)

        low_confidence_count = sum(
            1 for r in self.records if r.get("confidence", 1.0) < 0.75
        )

        if len(latencies) >= 20:
            p95_latency = statistics.quantiles(latencies, n=20)[18]
        else:
            p95_latency = max(latencies)

        return {
            "total_requests": total,
            "success_count": statuses.get("success", 0),
            "failure_count": statuses.get("failed", 0),
            "avg_latency_ms": round(sum(latencies) / total, 2),
            "p95_latency_ms": round(p95_latency, 2),
            "max_latency_ms": round(max(latencies), 2),
            "label_counts": dict(labels),
            "action_counts": dict(actions),
            "low_confidence_count": low_confidence_count,
        }


@serve.deployment(num_replicas=1)
class CameraEventApp:
    def __init__(self, preprocessor, classifier, decision_service, metrics):
        self.preprocessor = preprocessor
        self.classifier = classifier
        self.decision_service = decision_service
        self.metrics = metrics

    async def __call__(self, request: Request):
        path = request.url.path

        if path == "/metrics":
            summary = await self.metrics.summary.remote()
            return JSONResponse(summary)

        if path != "/predict":
            return JSONResponse(
                {"error": "Use POST /predict or GET /metrics"},
                status_code=404,
            )

        start = time.perf_counter()
        raw_event = await request.json()

        try:
            preprocessed = await self.preprocessor.process.remote(raw_event)
            prediction = await self.classifier.predict.remote(preprocessed)
            decision = await self.decision_service.decide.remote(prediction)

            latency_ms = round((time.perf_counter() - start) * 1000, 2)

            metric_record = {
                "event_id": decision["event_id"],
                "camera_id": decision["camera_id"],
                "label": decision["label"],
                "confidence": decision["confidence"],
                "action": decision["action"],
                "status": "success",
                "latency_ms": latency_ms,
            }

            await self.metrics.record.remote(metric_record)

            return {
                **decision,
                "latency_ms": latency_ms,
            }

        except Exception as exc:
            latency_ms = round((time.perf_counter() - start) * 1000, 2)

            metric_record = {
                "event_id": raw_event.get("event_id", "unknown"),
                "camera_id": raw_event.get("camera_id", "unknown"),
                "label": "unknown",
                "confidence": 0.0,
                "action": "failed",
                "status": "failed",
                "latency_ms": latency_ms,
                "error": str(exc),
            }

            await self.metrics.record.remote(metric_record)

            return JSONResponse(
                {
                    "event_id": raw_event.get("event_id", "unknown"),
                    "status": "failed",
                    "error": str(exc),
                    "latency_ms": latency_ms,
                },
                status_code=500,
            )


metrics_collector = MetricsCollector.bind()

app = CameraEventApp.bind(
    Preprocessor.bind(),
    Classifier.bind(),
    DecisionService.bind(),
    metrics_collector,
)


def generate_event(event_id: int) -> Dict[str, Any]:
    if event_id % 3 == 0:
        return {
            "event_id": f"evt-metrics-{event_id:03d}",
            "camera_id": "front-door",
            "location": "front_yard",
            "motion_score": 0.92,
            "brightness": 0.50,
            "object_size": 0.72,
        }

    if event_id % 3 == 1:
        return {
            "event_id": f"evt-metrics-{event_id:03d}",
            "camera_id": "backyard",
            "location": "backyard",
            "motion_score": 0.24,
            "brightness": 0.80,
            "object_size": 0.12,
        }

    return {
        "event_id": f"evt-metrics-{event_id:03d}",
        "camera_id": "garage",
        "location": "garage",
        "motion_score": 0.58,
        "brightness": 0.18,
        "object_size": 0.40,
    }


async def send_request(event: Dict[str, Any]) -> Dict[str, Any]:
    def _post():
        response = requests.post(
            "http://127.0.0.1:8000/predict",
            json=event,
            timeout=20,
        )
        return response.json()

    return await asyncio.to_thread(_post)


if __name__ == "__main__":
    serve.run(app, name="default", route_prefix="/")

    print("Ray Serve metrics pipeline is running.")
    print("Inference endpoint: http://127.0.0.1:8000/predict")
    print("Metrics endpoint:   http://127.0.0.1:8000/metrics")
    print()

    async def main():
        events = [generate_event(i) for i in range(1, 31)]

        print("Sending 30 requests...\n")

        tasks = [send_request(event) for event in events]
        results = await asyncio.gather(*tasks)

        print("Sample inference responses:")
        for result in results[:5]:
            print(result)

        metrics_response = requests.get("http://127.0.0.1:8000/metrics", timeout=10)

        print("\nMetrics summary:")
        print(metrics_response.json())

    asyncio.run(main())