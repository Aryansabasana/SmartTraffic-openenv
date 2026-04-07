import os
import sys
from src.tasks import EasyTask, MediumTask, HardTask
from src.agent import LLMAgent

def emit_start(task_name):
    """Prints START marker for task."""
    print(f"[START] task={task_name}", flush=True)

def emit_step(step, reward):
    """Prints STEP marker with step count and reward."""
    print(f"[STEP] step={step} reward={reward:.4f}", flush=True)

def emit_end(task_name, score, steps):
    """Prints END marker with final evaluation results."""
    # Clamp score to (0, 1) range as requested
    score = max(0.0001, min(0.9999, score))
    print(f"[END] task={task_name} score={score:.4f} steps={steps}", flush=True)

def run_inference():
    # Initialize Agent (Requires ENV vars: API_BASE_URL, MODEL_NAME, HF_TOKEN)
    try:
        agent = LLMAgent()
    except Exception as e:
        # Standard error printing is fine for non-marker lines
        print(f"CRITICAL: Failed to initialize LLM Agent. Error: {e}", file=sys.stderr)
        return

    tasks = {
        "Easy": EasyTask(),
        "Medium": MediumTask(),
        "Hard": HardTask()
    }

    for task_name, task in tasks.items():
        emit_start(task_name)
        
        state = task.reset()
        done = False
        step_count = 0
        
        # Max steps to ensure runtime < 10min per task
        max_steps = 100 
        
        while not done and step_count < max_steps:
            # 1. Get action from LLM
            action = agent.get_action(state)
            
            # 2. Step the environment
            result = task.step(action)
            state = result.state
            reward = result.reward
            done = result.done
            
            step_count += 1
            
            # 3. Log step with requested syntax
            emit_step(step_count, reward)
            
        # 4. Final Evaluation
        final_score = task.evaluate()
        emit_end(task_name, final_score, step_count)

if __name__ == "__main__":
    run_inference()

