import queue
import random
import time
from typing import Dict, Any, List

from app.level1_pre_post_processing import run_pipeline


def generate_fake_event(event_id: int) -> Dict[str, Any]:
    camera_locations = ["front-door", "backyard", "garage", "driveway"]

    return {
        "event_id": f"evt-{event_id:03d}",
        "camera_id": random.choice(camera_locations),
        "location": random.choice(camera_locations),
        "motion_score": round(random.uniform(0.1, 0.99), 2),
        "brightness": round(random.uniform(0.05, 0.95), 2),
        "object_size": round(random.uniform(0.05, 0.90), 2),
    }


def enqueue_events(event_queue: queue.Queue, num_events: int) -> None:
    for i in range(1, num_events + 1):
        event = generate_fake_event(i)
        event_queue.put(event)


def process_event_queue(event_queue: queue.Queue) -> List[Dict[str, Any]]:
    results = []

    while not event_queue.empty():
        raw_event = event_queue.get()

        start = time.perf_counter()

        try:
            result = run_pipeline(raw_event)
            status = "success"
            error = None
        except Exception as exc:
            result = {
                "event_id": raw_event.get("event_id", "unknown"),
                "camera_id": raw_event.get("camera_id", "unknown"),
                "action": "failed",
            }
            status = "failed"
            error = str(exc)

        latency_ms = round((time.perf_counter() - start) * 1000, 3)

        result["status"] = status
        result["error"] = error
        result["latency_ms"] = latency_ms

        results.append(result)
        event_queue.task_done()

    return results


def summarize_results(results: List[Dict[str, Any]]) -> Dict[str, Any]:
    total = len(results)
    successful = sum(1 for r in results if r["status"] == "success")
    failed = total - successful

    alert_count = sum(1 for r in results if r.get("action") == "alert_customer")
    review_count = sum(1 for r in results if r.get("action") == "escalate_for_review")
    suppress_count = sum(1 for r in results if r.get("action") == "suppress")
    log_only_count = sum(1 for r in results if r.get("action") == "log_only")

    latencies = [r["latency_ms"] for r in results]
    avg_latency = round(sum(latencies) / total, 3) if total else 0

    return {
        "total_events": total,
        "successful": successful,
        "failed": failed,
        "alert_customer": alert_count,
        "escalate_for_review": review_count,
        "suppress": suppress_count,
        "log_only": log_only_count,
        "avg_latency_ms": avg_latency,
    }


if __name__ == "__main__":
    event_queue = queue.Queue()

    enqueue_events(event_queue, num_events=20)

    print(f"Initial queue depth: {event_queue.qsize()}")

    results = process_event_queue(event_queue)
    summary = summarize_results(results)

    print("\nSample results:")
    for row in results[:5]:
        print(row)

    print("\nSummary:")
    print(summary)