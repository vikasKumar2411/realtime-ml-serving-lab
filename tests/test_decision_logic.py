from app.level1_pre_post_processing import run_pipeline


def test_high_confidence_person_alerts_customer():
    event = {
        "event_id": "evt-test-001",
        "camera_id": "front-door",
        "location": "front_yard",
        "motion_score": 0.95,
        "brightness": 0.50,
        "object_size": 0.80,
    }

    result = run_pipeline(event)

    assert result["label"] == "person"
    assert result["action"] == "alert_customer"


def test_background_motion_is_suppressed():
    event = {
        "event_id": "evt-test-002",
        "camera_id": "backyard",
        "location": "backyard",
        "motion_score": 0.20,
        "brightness": 0.70,
        "object_size": 0.10,
    }

    result = run_pipeline(event)

    assert result["label"] == "background_motion"
    assert result["action"] == "suppress"