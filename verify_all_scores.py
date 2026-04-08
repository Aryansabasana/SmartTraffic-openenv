import math
import subprocess
import sys
import os

# Ensure the library is findable
sys.path.insert(0, os.getcwd())

from src.tasks import EasyTask, MediumTask, HardTask, to_open_unit_interval

def test_canonical_helper():
    print("Testing canonical helper...")
    EPS = 0.01
    test_cases = [
        (0.0, EPS),
        (1.0, 1.0 - EPS),
        (-1.0, EPS),
        (2.0, 1.0 - EPS),
        (float('nan'), 0.5),
        (float('inf'), 0.99),
        (float('-inf'), 0.01),
        (None, 0.5)
    ]
    for val, expected in test_cases:
        actual = to_open_unit_interval(val)
        assert 0.0 < actual < 1.0, f"FAILED: to_open_unit_interval({val}) -> {actual} (must be in (0,1))"
        assert math.isclose(actual, expected), f"FAILED: to_open_unit_interval({val}) -> {actual} (expected {expected})"
    print("  Helper PASS")

def test_tasks():
    print("Testing task evaluation...")
    tasks = [EasyTask(), MediumTask(), HardTask()]
    for t in tasks:
        t.reset()
        score = t.evaluate()
        assert 0.0 < score < 1.0, f"FAILED: {t.__class__.__name__}.evaluate() -> {score}"
        print(f"  {t.__class__.__name__} PASS")

def test_payload_sanitization():
    print("Testing API payload sanitization...")
    # Simulate server/app.py logic
    def sanitize(obj):
        from server.app import sanitize_score_payload
        return sanitize_score_payload(obj)

    sample_payload = {
        "reward": 100.5,
        "score": -5.0,
        "info": {
            "reward_trend_avg": 0.0,
            "metric_efficiency": 1.0,
            "raw_value": 0.0,
            "other": "text"
        },
        "list": [
            {"overall": 1.5},
            {"raw_reward": 500}
        ]
    }
    
    sanitized = sanitize(sample_payload)
    
    assert 0.0 < sanitized["reward"] < 1.0
    assert 0.0 < sanitized["score"] < 1.0
    assert 0.0 < sanitized["info"]["reward_trend_avg"] < 1.0
    assert 0.0 < sanitized["info"]["metric_efficiency"] < 1.0
    assert sanitized["info"]["raw_value"] == 0.0 
    assert sanitized["info"]["other"] == "text"
    assert 0.0 < sanitized["list"][0]["overall"] < 1.0
    assert sanitized["list"][1]["raw_reward"] == 500
    
    print("  Payload Sanitization PASS")

def test_environment_direct():
    print("Testing TrafficEnv.step() direct reward clamping...")
    from src.environment import TrafficEnv
    env = TrafficEnv({"arrival_rate": 10.0, "congestion_multiplier": 5.0}) 
    state = env.reset()
    
    for action in [0, 1, 2]:
        result = env.step(action)
        assert 0.0 < result.reward < 1.0, f"FAILED: TrafficEnv.step({action}) -> {result.reward} (source reward leak!)"
        assert 0.0 < result.info["reward_trend_avg"] < 1.0, f"FAILED: TrafficEnv.step({action}) info['reward_trend_avg'] leak!"
        
    print("  Environment Source PASS")

def test_inference_markers():
    print("Testing inference.py markers...")
    env = os.environ.copy()
    env["API_BASE_URL"] = "" 
    env["API_KEY"] = ""

    result = subprocess.run([sys.executable, "inference.py"], 
                            capture_output=True, text=True, env=env)
    
    output = result.stdout
    
    end_markers = [line for line in output.splitlines() if line.startswith("[END]")]
    assert len(end_markers) == 3, f"FAILED: Expected 3 [END] markers, found {len(end_markers)}"
    
    for marker in end_markers:
        parts = marker.split()
        score_part = [p for p in parts if p.startswith("score=")][0]
        score_val = float(score_part.split("=")[1])
        assert 0.0 < score_val < 1.0, f"FAILED: {marker} has out-of-range score"

    print("  Inference Markers PASS")

if __name__ == "__main__":
    try:
        test_canonical_helper()
        test_tasks()
        test_payload_sanitization()
        test_environment_direct()
        test_inference_markers()
        print("\nALL FORENSIC TESTS PASSED")
    except Exception as e:
        print(f"\nTESTS FAILED: {e}")
        sys.exit(1)
