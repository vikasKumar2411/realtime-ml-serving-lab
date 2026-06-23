from typing import Any, Dict

import requests
from starlette.requests import Request

from ray import serve

from app.level1_pre_post_processing import run_pipeline


@serve.deployment(num_replicas=1)
class CameraEventClassifier:
    """
    Basic Ray Serve deployment.

    This class is the serving unit.
    Ray Serve can run one or more replicas of this deployment.
    """

    async def __call__(self, request: Request) -> Dict[str, Any]:
        payload = await request.json()
        result = run_pipeline(payload)
        return result


app = CameraEventClassifier.bind()


if __name__ == "__main__":
    serve.run(app, route_prefix="/")

    print("Ray Serve app is running at http://127.0.0.1:8000")
    print("Try POST http://127.0.0.1:8000/")

    sample_event = {
        "event_id": "evt-ray-001",
        "camera_id": "front-door",
        "location": "front_yard",
        "motion_score": 0.93,
        "brightness": 0.44,
        "object_size": 0.72,
    }

    response = requests.post("http://127.0.0.1:8000/", json=sample_event, timeout=10)
    print(response.json())