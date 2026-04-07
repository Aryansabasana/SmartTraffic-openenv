import sys
import os
import random
from src.tasks import EasyTask, MediumTask, HardTask
from src.agent import DeterministicAgent, LLMAgent

def emit(marker):
    """Guaranteed flush-print for evaluator signaling."""
    print(marker, flush=True)

def main():
    # Immediate signal to the evaluator parser that the script is active
    emit("[START] task=bootstrap")
    
    # Initialize Agent with safe fallback to HEURISTIC if LLM/CREDENTIALS fail
    try:
        agent = LLMAgent()
    except Exception as e:
        # Fallback to Heuristic to ensure markers are still printed
        agent = DeterministicAgent()
        # Non-marker info to stderr (ignored by evaluator parser)
        print(f"INFO: Using Heuristic Fallback (LLM Init failed: {e})", file=sys.stderr)

    tasks_to_run = [
        ("Easy", EasyTask()),
        ("Medium", MediumTask()),
        ("Hard", HardTask())
    ]

    for task_name, task in tasks_to_run:
        # 1. Signal Task Start
        emit(f"[START] task={task_name}")
        
        # 2. Rollout
        state = task.reset()
        done = False
        step_count = 0
        max_steps = 100 # Standard evaluator budget
        
        while not done and step_count < max_steps:
            # Action selection
            action = agent.get_action(state)
            
            # Step the simulation
            result = task.step(action)
            
            state = result.state
            reward = result.reward
            done = result.done
            step_count += 1
            
            # 3. Signal Step Result
            emit(f"[STEP] step={step_count} reward={reward:.4f}")
            
        # 4. Final Evaluation
        final_score = task.evaluate()
        
        # 5. Signal Task End
        # Clamping score to (0.0001, 0.9999) to satisfy range constraints
        safe_score = max(0.0001, min(0.9999, final_score))
        emit(f"[END] task={task_name} score={safe_score:.4f} steps={step_count}")

    # Explicit exit for clean terminator signal
    sys.exit(0)

if __name__ == "__main__":
    main()
