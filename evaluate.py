import random
import argparse
from src.tasks import EasyTask, MediumTask, HardTask, to_open_unit_interval
from src.agent import DeterministicAgent

def run_evaluation(base_seed=None, silent=False):
    if base_seed is None:
        base_seed = random.randint(1000, 99999)
        
    random.seed(base_seed)
    if not silent:
        print("==================================================")
        print(f"=== Smart Traffic Eval (Seed: {base_seed}) ===")
    
    agent = DeterministicAgent()
    tasks = {
        "Easy": EasyTask(),
        "Medium": MediumTask(),
        "Hard": HardTask()
    }
    
    results = {}
    total_score = 0.0
    
    for level, task in tasks.items():
        task_seed = base_seed + list(tasks.keys()).index(level) * 999
        
        state = task.reset(seed=task_seed)
        done = False
        steps = 0
        total_reward = 0.0
        
        while not done:
            action_idx = agent.get_action(state)
            result = task.step(action_idx)
            state = result.state
            reward = result.reward
            done = result.done
            total_reward += reward
            steps += 1
            
            if steps > 500:
                break
                
        score = task.evaluate()
        total_score += score
        results[level] = score
        info = result.info
        
        if not silent:
            print(f"[{level}] Steps: {steps} | Total Reward: {total_reward:.2f}")
            print(f"       Cleared: {info['total_cleared']} | Avg Wait/Car: {info['avg_waiting_time']:.1f} | Emg Handled: {info['emergencies_handled']}")
            print(f"       Final Level Score (0-1): {score:.3f}")
    
    avg_score = total_score / len(tasks)
    # Final safety clip for Overall score using centralized helper
    avg_score = to_open_unit_interval(avg_score)
    results["Overall"] = avg_score
    
    if not silent:
        print(f"==================================================")
        print(f"Overall Average Score: {avg_score:.4f} / 1.000")
        print(f"==================================================\n")
    
    return results

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Evaluate the Smart Traffic Agent")
    parser.add_argument("--seed", type=int, default=None, help="Fix the RNG seed for reproducible testing")
    args = parser.parse_args()
    
    run_evaluation(base_seed=args.seed)
