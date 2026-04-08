import math
import sys
import os
import subprocess
import re

# Add project root to sys.path
sys.path.insert(0, os.path.abspath(os.path.dirname(__file__)))

from src.tasks import EasyTask, MediumTask, HardTask, to_open_unit_interval
from server.app import step_logic, reset_logic

def test_helper():
    print("--- Testing to_open_unit_interval ---")
    EPS = 0.01
    
    test_cases = [
        (0.0, EPS),
        (1.0, 1.0 - EPS),
        (-0.5, EPS),
        (1.5, 1.0 - EPS),
        (float('nan'), 0.5),
        (float('inf'), 1.0 - EPS),
        (float('-inf'), EPS),
        (0.5, 0.5),
        (0.995, 0.99),
        (0.005, 0.01),
    ]
    
    all_passed = True
    for val, expected in test_cases:
        res = to_open_unit_interval(val)
        # Use math.isclose for float comparison
        if not math.isclose(res, expected, rel_tol=1e-9):
            print(f"[FAIL] Input: {val} | Expected: {expected} | Got: {res}")
            all_passed = False
        else:
            print(f"[PASS] Input: {val} -> {res}")
            
    if all_passed:
        print("[OK] Helper tests passed.")
    return all_passed

def test_tasks():
    print("\n--- Testing Task Evaluations ---")
    tasks = [EasyTask(), MediumTask(), HardTask()]
    all_passed = True
    
    for task in tasks:
        # Test default evaluate
        task.reset(seed=42)
        score = task.evaluate()
        if not (0.0 < score < 1.0):
            print(f"[FAIL] {task.__class__.__name__} default score {score} out of range!")
            all_passed = False
        else:
            print(f"[PASS] {task.__class__.__name__} default score: {score:.6f}")
            
        # Test forced 0.0 internal state
        task.env.total_cleared = 0
        task.env.total_waiting_time = 1000000 # High wait
        task.env.north = 100 # High queue
        score = task.evaluate()
        if not (0.0 < score < 1.0):
            print(f"[FAIL] {task.__class__.__name__} crashed score {score} out of range!")
            all_passed = False
        else:
            print(f"[PASS] {task.__class__.__name__} forced-low score: {score:.6f}")
            
        # Test forced 1.0 internal state
        task.env.total_cleared = 100
        task.env.total_waiting_time = 0
        task.env.north = 0
        task.env.south = 0
        task.env.east = 0
        task.env.west = 0
        score = task.evaluate()
        if not (0.0 < score < 1.0):
            print(f"[FAIL] {task.__class__.__name__} perfect score {score} out of range!")
            all_passed = False
        else:
            print(f"[PASS] {task.__class__.__name__} forced-perfect score: {score:.6f}")

    if all_passed:
        print("[OK] Task evaluation range tests passed.")
    return all_passed

def test_inference_stdout():
    print("\n--- Testing inference.py stdout signals ---")
    try:
        # Run inference.py and capture stdout
        # Using a timeout to prevent hanging
        result = subprocess.run(
            [sys.executable, "inference.py"],
            capture_output=True,
            text=True,
            timeout=30,
            env={**os.environ, "API_BASE_URL": "", "API_KEY": ""} # Force local mode
        )
        
        stdout = result.stdout
        # Find all [END] signals
        end_signals = re.findall(r"\[END\] task=(\w+) score=([\d\.]+)", stdout)
        
        if len(end_signals) != 3:
            print(f"[FAIL] Expected 3 [END] signals, but found {len(end_signals)}!")
            print("STDOUT:", stdout)
            return False
            
        expected_tasks = {"Easy", "Medium", "Hard"}
        found_tasks = {t for t, _ in end_signals}
        if expected_tasks != found_tasks:
            print(f"[FAIL] Missing expected tasks. Expected: {expected_tasks}, Found: {found_tasks}")
            return False
            
        all_passed = True
        for task_name, score_str in end_signals:
            score = float(score_str)
            if not (0.0 < score < 1.0):
                print(f"[FAIL] Signal [END] task={task_name} score={score} out of range!")
                all_passed = False
            elif len(score_str.split('.')[-1]) < 6:
                print(f"[FAIL] Signal [END] task={task_name} score={score_str} has insufficient precision!")
                all_passed = False
            else:
                print(f"[PASS] Signal [END] task={task_name} score={score_str}")
                
        if all_passed:
            print("[OK] inference.py stdout verification passed.")
        return all_passed
        
    except Exception as e:
        print(f"[ERROR] during inference test: {e}")
        return False

def test_api_step_logic():
    print("\n--- Testing API step_logic ---")
    import asyncio
    
    async def run_test():
        await reset_logic("easy")
        res = await step_logic(0)
        
        reward = res["reward"]
        if not (0.0 < reward < 1.0):
            print(f"[FAIL] step_logic returned reward {reward} out of range!")
            return False
        
        raw_reward = res.get("raw_reward")
        if raw_reward is None:
            print("[FAIL] step_logic missing 'raw_reward' field!")
            return False
            
        print(f"[PASS] step_logic returned reward {reward:.6f} with raw_reward {raw_reward}")
        return True

    passed = asyncio.run(run_test())
    if passed:
        print("[OK] API step_logic validation passed.")
    return passed

if __name__ == "__main__":
    h_ok = test_helper()
    t_ok = test_tasks()
    i_ok = test_inference_stdout()
    a_ok = test_api_step_logic()
    
    if h_ok and t_ok and i_ok and a_ok:
        print("\nSUCCESS: ALL SCORING VALIDATION TESTS PASSED.")
        sys.exit(0)
    else:
        print("\nFAILURE: SOME TESTS FAILED.")
        sys.exit(1)
