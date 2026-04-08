import sys
import os
import math
from src.tasks import EasyTask, MediumTask, HardTask, to_open_unit_interval
from src.agent import DeterministicAgent, LLMAgent

def emit(marker):
    print(marker, flush=True)


def main():
    emit("[START] task=bootstrap")
    api_url = os.environ.get("API_BASE_URL")
    api_key = os.environ.get("API_KEY") or os.environ.get("OPENAI_API_KEY")
    is_evaluator_mode = bool(api_url and api_key)
    try:
        if is_evaluator_mode:
            agent = LLMAgent()
        else:
            try:
                agent = LLMAgent()
            except Exception:
                agent = DeterministicAgent()
    except Exception as e:
        print(f"CRITICAL: {e}", file=sys.stderr)
        sys.exit(1)

    for task_name, task in [("Easy", EasyTask()), ("Medium", MediumTask()), ("Hard", HardTask())]:
        emit(f"[START] task={task_name}")
        state = task.reset()
        done = False
        step_count = 0
        success = False
        try:
            while not done and step_count < 100:
                action = agent.get_action(state)
                result = task.step(action)
                state = result.state
                done = result.done
                step_count += 1
                safe_reward = to_open_unit_interval(result.reward)
                done_str = "true" if done else "false"
                emit(f"[STEP] step={step_count} reward={safe_reward:.2f} done={done_str}")
            success = True
        except Exception as e:
            print(f"CRITICAL: {e}", file=sys.stderr)
        finally:
            try:
                final_score = to_open_unit_interval(task.evaluate())
            except Exception:
                final_score = 0.5
            success_str = "true" if success else "false"
            emit(f"[END] task={task_name} score={final_score:.6f} steps={step_count} success={success_str}")

    sys.exit(0)

if __name__ == "__main__":
    main()
