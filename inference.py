import json
import os
import sys
from src.tasks import EasyTask, MediumTask, HardTask
from src.agent import LLMAgent

def log_event(event_type, data):
    """Prints structured logs in the required format."""
    print(f"[{event_type}] {json.dumps(data)}")
    sys.stdout.flush()

def run_inference():
    # Initialize Agent (Requires ENV vars: API_BASE_URL, MODEL_NAME, HF_TOKEN)
    try:
        agent = LLMAgent()
    except Exception as e:
        print(f"CRITICAL: Failed to initialize LLM Agent. Check environment variables. Error: {e}")
        return

    tasks = {
        "Easy": EasyTask(),
        "Medium": MediumTask(),
        "Hard": HardTask()
    }

    for task_name, task in tasks.items():
        log_event("START", {"task": task_name})
        
        state = task.reset()
        done = False
        step_count = 0
        total_reward = 0.0
        
        # Max steps to ensure runtime < 20min
        max_steps = 100 
        
        while not done and step_count < max_steps:
            # 1. Get action from LLM
            action = agent.get_action(state)
            
            # 2. Step the environment
            result = task.step(action)
            state = result.state
            reward = result.reward
            done = result.done
            
            total_reward += reward
            step_count += 1
            
            # 3. Log step
            log_event("STEP", {
                "step": step_count,
                "action": action,
                "reward": round(reward, 4),
                "done": done
            })
            
        # 4. Final Evaluation
        final_score = task.evaluate()
        log_event("END", {
            "task": task_name,
            "steps": step_count,
            "total_reward": round(total_reward, 2),
            "final_score": round(final_score, 4)
        })

if __name__ == "__main__":
    run_inference()
