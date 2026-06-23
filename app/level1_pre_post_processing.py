from typing import Dict, Any


def preprocess_event(raw_event: Dict[str, Any]) -> Dict[str, Any]:
    """
    Simulates preprocessing before model inference.

    Real CV preprocessing could include:
    - frame sampling
    - image resizing
    - normalization
    - corrupt frame handling
    - metadata enrichment
    """

    return {
        "event_id": raw_event["event_id"],
        "camera_id": raw_event["camera_id"],
        "location": raw_event.get("location", "unknown"),
        "motion_score": float(raw_event.get("motion_score", 0.0)),
        "brightness": float(raw_event.get("brightness", 0.5)),
        "object_size": float(raw_event.get("object_size", 0.0)),
    }


def model_predict(event: Dict[str, Any]) -> Dict[str, Any]:
    """
    Simulates a computer vision model prediction.
    """

    motion_score = event["motion_score"]
    object_size = event["object_size"]
    brightness = event["brightness"]

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
        "location": event["location"],
        "label": label,
        "confidence": round(confidence, 3),
    }


def postprocess_decision(prediction: Dict[str, Any]) -> Dict[str, Any]:
    """
    Converts raw model prediction into product decision.

    This is important:
    model output != product action.
    """

    label = prediction["label"]
    confidence = prediction["confidence"]

    if label == "person" and confidence >= 0.85:
        action = "alert_customer"
        reason = "High-confidence person detected"
    elif label == "person" and confidence >= 0.65:
        action = "escalate_for_review"
        reason = "Medium-confidence person detected"
    elif label == "unknown_motion":
        action = "log_only"
        reason = "Unknown low-confidence motion"
    else:
        action = "suppress"
        reason = "Likely non-security event"

    return {
        **prediction,
        "action": action,
        "reason": reason,
    }


def run_pipeline(raw_event: Dict[str, Any]) -> Dict[str, Any]:
    preprocessed = preprocess_event(raw_event)
    prediction = model_predict(preprocessed)
    decision = postprocess_decision(prediction)
    return decision


if __name__ == "__main__":
    sample_event = {
        "event_id": "evt-002",
        "camera_id": "front-door",
        "location": "front_yard",
        "motion_score": 0.82,
        "brightness": 0.38,
        "object_size": 0.58,
    }

    result = run_pipeline(sample_event)
    print(result)