from typing import Dict, Any


def predict_event(event: Dict[str, Any]) -> Dict[str, Any]:
    """
    Toy ML inference function.

    In a real SimpliSafe-style system, this would be a CV model.
    Here we use simple rules to simulate model behavior.
    """

    motion_score = event.get("motion_score", 0.0)
    object_size = event.get("object_size", 0.0)
    brightness = event.get("brightness", 0.5)

    if motion_score > 0.75 and object_size > 0.45:
        label = "person"
        confidence = min(0.95, 0.60 + motion_score * 0.3 + object_size * 0.1)
    elif motion_score > 0.60 and object_size <= 0.45:
        label = "pet"
        confidence = min(0.90, 0.55 + motion_score * 0.25)
    elif motion_score > 0.40 and brightness < 0.25:
        label = "unknown_motion"
        confidence = 0.62
    else:
        label = "background_motion"
        confidence = 0.70

    return {
        "event_id": event["event_id"],
        "camera_id": event["camera_id"],
        "label": label,
        "confidence": round(confidence, 3),
    }


if __name__ == "__main__":
    sample_event = {
        "event_id": "evt-001",
        "camera_id": "front-door",
        "motion_score": 0.91,
        "brightness": 0.42,
        "object_size": 0.63,
    }

    prediction = predict_event(sample_event)
    print(prediction)