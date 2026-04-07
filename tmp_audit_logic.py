import sys
import os
import random
import numpy as np

# Add project root to sys.path to allow imports
project_root = r"c:\Users\ARYAN\OneDrive\Desktop\OpenEnv"
sys.path.append(project_root)

from src.tasks import EasyTask, HardTask
from src.agent import DeterministicAgent

def test_efficiency_logic():
    print("--- Auditing Smart Traffic Scoring Logic ---")
    
    agent = DeterministicAgent()
    
    # 1. Test Low vs High Traffic on EasyTask
    # Logic: Low traffic should result in a higher efficiency score (Better wait times).
    print("\n[Test 1] Sensitivity to Traffic Volume")
    
    low_traffic_task = EasyTask()
    low_traffic_task.config['arrival_rate'] = 0.5 # Very low
    
    high_traffic_task = EasyTask()
    high_traffic_task.config['arrival_rate'] = 5.0 # Very high
    
    def run_sim(task, steps=100, seed=42):
        task.reset(seed=seed)
        total_r = 0
        for _ in range(steps):
            state = task.state()
            action = agent.get_action(state)
            res = task.step(action)
            total_r += res.reward
        return task.evaluate(), total_r

    score_low, reward_low = run_sim(low_traffic_task)
    score_high, reward_high = run_sim(high_traffic_task)
    
    print(f"  Low Traffic Case:  Score={score_low:.3f}, Total Reward={reward_low:.1f}")
    print(f"  High Traffic Case: Score={score_high:.3f}, Total Reward={reward_high:.1f}")
    
    if score_low > score_high:
        print("  OK Logic Check: Low traffic scenarios score higher (Better wait times).")
    else:
        print("  ERR Logic Error: High traffic scored higher/equal to low traffic.")

    # 2. Test Emergency Impact on HardTask
    # Logic: Handling emergencies properly boosts score.
    print("\n[Test 2] Sensitivity to Emergency Response")
    hard_task = HardTask()
    hard_task.config['emergency_prob'] = 1.0 # Force generation logic check
    
    # CASE A: Handle it
    hard_task.reset(seed=42)
    # Force an emergency state manually to be 100% sure of the trigger
    hard_task.env.emergency_present = True
    hard_task.env.emergency_direction_str = 'ns'
    hard_task.env.total_emergencies_generated = 1
    
    # Action 1 is Green-NS (Handles it)
    res_handled = hard_task.step(1)
    score_handled = hard_task.evaluate()
    
    # CASE B: Ignore it
    hard_task.reset(seed=42)
    hard_task.env.emergency_present = True
    hard_task.env.emergency_direction_str = 'ns'
    hard_task.env.total_emergencies_generated = 1
    
    # Action 2 is Green-EW (Ignores it)
    res_ignored = hard_task.step(2)
    score_ignored = hard_task.evaluate()
    
    print(f"  Emergency Handled: Reward={res_handled.reward:.1f}, Post-Step Score={score_handled:.3f}")
    print(f"  Emergency Ignored: Reward={res_ignored.reward:.1f}, Post-Step Score={score_ignored:.3f}")
    
    if res_handled.reward > res_ignored.reward and score_handled > score_ignored:
        print("  OK Logic Check: Handling emergencies properly boosts score/reward.")
    else:
        print("  ERR Logic Error: Emergency handling has insufficient impact.")

    # 3. Verify No Hardcoding (Stochasticity check)
    # Logic: Different seeds must produce different scores.
    print("\n[Test 3] Stochastic Variance (Anti-Hardcoding Check)")
    scores = []
    for s in range(5):
        hard_task.reset(seed=s*100)
        for _ in range(20): 
            hard_task.step(agent.get_action(hard_task.state()))
        scores.append(hard_task.evaluate())
    
    print(f"  Sample Scores: {[round(s, 4) for s in scores]}")
    if len(set(scores)) == len(scores):
        print("  OK Logic Check: Scores are dynamic and vary with environmental variance.")
    else:
        print("  ERR Logic Error: Detected repeating/static scores (potential hardcoding).")

    # 4. Efficiency Range Check
    # Logic: Score must be exactly between 0 and 1.
    print("\n[Test 4] Boundary Constraints (0.0 - 1.0)")
    # Test extreme wait times
    bad_task = EasyTask()
    bad_task.reset()
    bad_task.env.north = 100
    bad_task.env.south = 100
    bad_task.env.east = 100
    bad_task.env.west = 100
    # Simulate long wait without clearing
    for _ in range(50):
        bad_task.step(0) # All Red
    
    final_score = bad_task.evaluate()
    print(f"  Extreme Congestion Score: {final_score:.6f}")
    if 0.0 < final_score < 1.0:
        print("  OK Logic Check: Scores are strictly bounded (0.0, 1.0).")
    else:
        print(f"  ERR Logic Error: Score {final_score} is outside strict open interval (0, 1).")

if __name__ == "__main__":
    test_efficiency_logic()
