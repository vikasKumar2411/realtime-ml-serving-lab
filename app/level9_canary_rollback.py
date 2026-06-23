import asyncio
import statistics
import time
from collections import Counter, defaultdict
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
class StableClassifierV1:
    """
    Stable model version.
    Uses the existing dummy model behavior.
    """

    async def predict(self, event: Dict[str, Any]) -> Dict[str, Any]:
        await asyncio.sleep(0.04)
        prediction = model_predict(event)
        prediction["model_version"] = "v1"
        return prediction


@serve.deployment(num_replicas=1)
class CanaryClassifierV2:
    """
    New canary model version.

    This intentionally behaves worse:
    - more likely to label medium/unknown motion as person
    - creates more customer alerts
    """

    async def predict(self, event: Dict[str, Any]) -> Dict[str, Any]:
        await asyncio.sleep(0.06)

        base_prediction = model_predict(event)

        # Intentional regression:
        # v2 over-alerts on unknown/background events.
        if base_prediction["label"] in {"unknown_motion", "background_motion"}:
            return {
                "event_id": event["event_id"],
                "camera_id": event["camera_id"],
                "location": event["location"],
                "label": "person",
                "confidence": 0.86,
                "model_version": "v2",
            }

        base_prediction["model_version"] = "v2"
        return base_prediction


@serve.deployment(num_replicas=1)
class CanaryRouter:
    """
    Routes a small percentage of traffic to v2.

    For deterministic behavior in this lab:
    - every 10th request goes to v2
    - all others go to v1
    """

    def __init__(self, stable_classifier, canary_classifier):
        self.stable_classifier = stable_classifier
        self.canary_classifier = canary_classifier

    async def predict(self, event: Dict[str, Any]) -> Dict[str, Any]:
        numeric_id = int(event["event_id"].split("-")[-1])

        if numeric_id % 10 == 0:
            return await self.canary_classifier.predict.remote(event)

        return await self.stable_classifier.predict.remote(event)


@serve.deployment(num_replicas=1)
class DecisionService:
    async def decide(self, prediction: Dict[str, Any]) -> Dict[str, Any]:
        decision = postprocess_decision(prediction)
        decision["model_version"] = prediction["model_version"]
        return decision


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

        by_version = defaultdict(list)
        for record in self.records:
            by_version[record["model_version"]].append(record)

        version_summary = {}

        for version, records in by_version.items():
            latencies = [r["latency_ms"] for r in records]
            actions = Counter(r["action"] for r in records)
            labels = Counter(r["label"] for r in records)

            alert_rate = actions.get("alert_customer", 0) / len(records)
            low_confidence_rate = (
                sum(1 for r in records if r["confidence"] < 0.75) / len(records)
            )

            p95_latency = (
                statistics.quantiles(latencies, n=20)[18]
                if len(latencies) >= 20
                else max(latencies)
            )

            version_summary[version] = {
                "requests": len(records),
                "avg_latency_ms": round(sum(latencies) / len(records), 2),
                "p95_latency_ms": round(p95_latency, 2),
                "label_counts": dict(labels),
                "action_counts": dict(actions),
                "alert_rate": round(alert_rate, 3),
                "low_confidence_rate": round(low_confidence_rate, 3),
            }

        rollback_decision = self._rollback_decision(version_summary)

        return {
            "total_requests": total,
            "version_summary": version_summary,
            "rollback_decision": rollback_decision,
        }

    def _rollback_decision(self, version_summary: Dict[str, Any]) -> Dict[str, Any]:
        if "v1" not in version_summary or "v2" not in version_summary:
            return {
                "decision": "insufficient_data",
                "reason": "Need both v1 and v2 traffic before deciding.",
            }

        v1 = version_summary["v1"]
        v2 = version_summary["v2"]

        # Simple canary policy:
        # rollback if v2 alert rate is much higher than v1.
        if v2["alert_rate"] > v1["alert_rate"] + 0.25:
            return {
                "decision": "rollback_v2",
                "reason": (
                    f"v2 alert rate {v2['alert_rate']} is much higher "
                    f"than v1 alert rate {v1['alert_rate']}."
                ),
            }

        if v2["p95_latency_ms"] > v1["p95_latency_ms"] * 1.5:
            return {
                "decision": "rollback_v2",
                "reason": (
                    f"v2 p95 latency {v2['p95_latency_ms']}ms is too high "
                    f"vs v1 p95 latency {v1['p95_latency_ms']}ms."
                ),
            }

        return {
            "decision": "promote_v2",
            "reason": "v2 metrics are within acceptable canary thresholds.",
        }


@serve.deployment(num_replicas=1)
class CanaryApp:
    def __init__(self, preprocessor, router, decision_service, metrics):
        self.preprocessor = preprocessor
        self.router = router
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
            prediction = await self.router.predict.remote(preprocessed)
            decision = await self.decision_service.decide.remote(prediction)

            latency_ms = round((time.perf_counter() - start) * 1000, 2)

            metric_record = {
                "event_id": decision["event_id"],
                "camera_id": decision["camera_id"],
                "label": decision["label"],
                "confidence": decision["confidence"],
                "action": decision["action"],
                "model_version": decision["model_version"],
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

            await self.metrics.record.remote(
                {
                    "event_id": raw_event.get("event_id", "unknown"),
                    "camera_id": raw_event.get("camera_id", "unknown"),
                    "label": "unknown",
                    "confidence": 0.0,
                    "action": "failed",
                    "model_version": "unknown",
                    "status": "failed",
                    "latency_ms": latency_ms,
                    "error": str(exc),
                }
            )

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

app = CanaryApp.bind(
    Preprocessor.bind(),
    CanaryRouter.bind(
        StableClassifierV1.bind(),
        CanaryClassifierV2.bind(),
    ),
    DecisionService.bind(),
    metrics_collector,
)


def generate_event(event_id: int) -> Dict[str, Any]:
    """
    Mix of event types:
    - background
    - unknown motion
    - person
    """

    if event_id % 3 == 0:
        return {
            "event_id": f"evt-canary-{event_id:03d}",
            "camera_id": "front-door",
            "location": "front_yard",
            "motion_score": 0.92,
            "brightness": 0.50,
            "object_size": 0.72,
        }

    if event_id % 3 == 1:
        return {
            "event_id": f"evt-canary-{event_id:03d}",
            "camera_id": "backyard",
            "location": "backyard",
            "motion_score": 0.24,
            "brightness": 0.80,
            "object_size": 0.12,
        }

    return {
        "event_id": f"evt-canary-{event_id:03d}",
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

    print("Ray Serve canary/rollback pipeline is running.")
    print("Inference endpoint: http://127.0.0.1:8000/predict")
    print("Metrics endpoint:   http://127.0.0.1:8000/metrics")
    print()

    async def main():
        events = [generate_event(i) for i in range(1, 101)]

        print("Sending 100 requests...")
        print("Every 10th request goes to canary model v2.\n")

        tasks = [send_request(event) for event in events]
        results = await asyncio.gather(*tasks)

        print("Sample inference responses:")
        for result in results[:12]:
            print(result)

        metrics_response = requests.get("http://127.0.0.1:8000/metrics", timeout=10)

        print("\nCanary metrics summary:")
        print(metrics_response.json())

    asyncio.run(main())