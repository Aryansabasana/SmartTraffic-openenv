import math
import subprocess
import sys
import os
import re

# Ensure the library is findable
sys.path.insert(0, os.getcwd())

from src.tasks import EasyTask, MediumTask, HardTask, to_open_unit_interval

def is_score_like(key):
    k = str(key).lower()
    if k.startswith("raw_"):
        return False
    return any(token in k for token in ["score", "reward", "efficiency", "metric", "overall"])

def assert_recursive_safe(obj, path="root"):
    if isinstance(obj, dict):
        for k, v in obj.items():
            current_path = f"{path}.{k}"
            if is_score_like(k):
                if isinstance(v, (int, float)):
                    assert 0.0 < v < 1.0, f"FAILED: {current_path} -> {v} is out of strict (0,1) range!"
                elif v is None:
                    pass # None is ok if allowed by logic, but usually sanitized to 0.5
            assert_recursive_safe(v, current_path)
    elif isinstance(obj, (list, tuple)):
        for i, item in enumerate(obj):
            assert_recursive_safe(item, f"{path}[{i}]")

def test_canonical_helper():
    print("Testing canonical helper...")
    test_cases = [
        (0.0, 0.02), (1.0, 0.98), (-1.0, 0.02), (2.0, 0.98),
        (float('nan'), 0.5), (float('inf'), 0.5), (float('-inf'), 0.5), (None, 0.5)
    ]
    for val, expected in test_cases:
        actual = to_open_unit_interval(val)
        assert 0.0 < actual < 1.0
        assert math.isclose(actual, expected)
    print("  Helper PASS")

def test_modules_payloads():
    print("Testing module return payloads...")
    
    # Audit
    print("  Checking audit.py...")
    import audit
    s, sc, m = audit.run_single(seed=42)
    assert_recursive_safe({"seed": s, "scores": sc, "metrics": m}, "audit")
    
    # Stress Test
    print("  Checking stress_test.py...")
    import stress_test
    res = stress_test.run_single(seed=123)
    assert_recursive_safe(res, "stress_test")
    
    # Evaluate
    print("  Checking evaluate.py...")
    import evaluate
    res = evaluate.run_evaluation(base_seed=999, silent=True)
    assert_recursive_safe(res, "evaluate")
    
    # Dashboard
    print("  Checking dashboard.py...")
    import dashboard
    m1 = dashboard.run_agent("Random Agent", "Easy", 7)
    m2 = dashboard.compute_metrics(seed=777)
    assert_recursive_safe(m1, "dashboard.run_agent")
    assert_recursive_safe(m2, "dashboard.compute_metrics")
    
    print("  Module Payloads PASS")

def test_cli_outputs():
    print("Testing CLI outputs for forbidden strings (1.0, 0.0, etc. in score context)...")
    
    # We look for patterns like "Reward: 100.0" or "Score: 0.0" or "Score: 1.0"
    # Note: 0.0 and 1.0 are exactly forbidden.
    forbidden_patterns = [
        r"(score|reward|metric|efficiency|overall|grade|success_rate)[:\s=]+(0\.0+|1\.0+)\b",
        r"(score|reward|metric|efficiency|overall|grade|success_rate)[:\s=]+(-?\d+\.\d+)", # Catch any >1 or <0
    ]
    
    scripts = ["evaluate.py", "audit.py", "inference.py"]
    for script in scripts:
        print(f"  Checking {script} output...")
        env = os.environ.copy()
        env["API_BASE_URL"] = "" 
        env["API_KEY"] = ""
        
        # Run only 1 iteration if possible, but inference.py runs fixed 3
        result = subprocess.run([sys.executable, script], capture_output=True, text=True, env=env)
        output = result.stdout + result.stderr
        
        for line in output.splitlines():
            for pat in forbidden_patterns:
                match = re.search(pat, line, re.IGNORECASE)
                if match:
                    label = match.group(1)
                    val = float(match.group(2))
                    # If it's labeled with raw_, ignore it
                    if f"raw_{label.lower()}" in line.lower():
                        continue
                    # Check if val is in (0,1)
                    if not (0.0 < val < 1.0):
                        assert False, f"FAILED: {script} outputted unsafe value: '{line.strip()}' (matched {label}={val})"
    
    print("  CLI Outputs PASS")

def test_environment_direct():
    print("Testing TrafficEnv.step() direct reward clamping...")
    from src.environment import TrafficEnv
    env = TrafficEnv({"arrival_rate": 10.0}) 
    env.reset()
    for _ in range(5):
        result = env.step(1)
        assert 0.0 < result.reward < 1.0
        assert_recursive_safe(result.info, "env.info")
    print("  Environment Source PASS")

if __name__ == "__main__":
    try:
        test_canonical_helper()
        test_environment_direct()
        test_modules_payloads()
        test_cli_outputs()
        print("\nALL FORENSIC TESTS PASSED - SYSTEM IS VALIDATOR-SAFE")
    except Exception as e:
        print(f"\nTESTS FAILED: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
