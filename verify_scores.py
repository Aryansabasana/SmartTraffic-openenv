import sys
import os
import math

# Add project root to path
sys.path.append(os.getcwd())

from src.tasks import to_open_unit_interval, EasyTask

def test_normalization_helper():
    print("Testing to_open_unit_interval helper...")
    cases = [
        (0.0, "exactly zero"),
        (1.0, "exactly one"),
        (0.5, "middle"),
        (float('nan'), "NaN"),
        (float('inf'), "Positive Infinity"),
        (float('-inf'), "Negative Infinity"),
        (1e-10, "very small"),
        (1.0 + 1e-10, "just above one"),
        (-0.5, "negative"),
        (0.999999999, "near one"),
    ]
    
    all_pass = True
    for val, desc in cases:
        norm = to_open_unit_interval(val)
        is_valid = 0.0 < norm < 1.0
        status = "PASS" if is_valid else "FAIL"
        print(f"  [{status}] Input: {val:<15} ({desc:<18}) -> Output: {norm:.10f}")
        if not is_valid:
            all_pass = False
            
    return all_pass

def test_actual_evaluations():
    print("\nTesting actual Task evaluations...")
    task = EasyTask()
    
    # 1. Zero traffic
    task.reset(seed=42)
    task.env.north = 0; task.env.south = 0; task.env.east = 0; task.env.west = 0
    score_zero = task.evaluate()
    
    # 2. Extreme traffic
    task.reset(seed=42)
    task.env.north = 1000; task.env.south = 1000; task.env.east = 1000; task.env.west = 1000
    for _ in range(10): task.step(0) # All red, wait time increases
    score_extreme = task.evaluate()
    
    all_pass = True
    for score, desc in [(score_zero, "Zero Traffic"), (score_extreme, "Extreme Traffic")]:
        is_valid = 0.0 < score < 1.0
        status = "PASS" if is_valid else "FAIL"
        print(f"  [{status}] {desc:<18} -> Score: {score:.10f}")
        if not is_valid:
            all_pass = False
            
    return all_pass

if __name__ == "__main__":
    h_pass = test_normalization_helper()
    e_pass = test_actual_evaluations()
    
    if h_pass and e_pass:
        print("\n[SUCCESS] All scores are strictly within (0, 1).")
        sys.exit(0)
    else:
        print("\n[FAILURE] One or more scores are exactly 0.0 or 1.0.")
        sys.exit(1)
