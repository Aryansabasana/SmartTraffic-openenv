from src.environment import TrafficEnv
from src.models import State, StepResult
from typing import Dict, Any, Optional
import math

EPS = 0.02

def hard_clamp(x):
    import math
    try:
        v = float(x)
        if math.isnan(v) or math.isinf(v):
            return 0.5
        return max(0.02, min(0.98, v))
    except Exception:
        return 0.5

def to_open_unit_interval(x: float) -> float:
    return hard_clamp(x)

def sanitize_score_payload(obj):
    if isinstance(obj, dict):
        out = {}
        for k, v in obj.items():
            if isinstance(v, (dict, list)):
                out[k] = sanitize_score_payload(v)
            else:
                if any(x in str(k).lower() for x in ["score", "reward", "efficiency", "metric", "overall"]) and not str(k).startswith("raw_"):
                    out[k] = hard_clamp(v)
                else:
                    out[k] = v
        return out
    elif isinstance(obj, list):
        return [sanitize_score_payload(x) for x in obj]
    else:
        return obj

class BaseTask:
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.env = TrafficEnv(config)

    def reset(self, seed=None):
        return self.env.reset(seed=seed)

    def step(self, action_type):
        return self.env.step(action_type)

    def state(self):
        return self.env.state()

    def evaluate(self) -> float:
        total_cleared = self.env.total_cleared
        remaining = max(0, self.env.north + self.env.south + self.env.east + self.env.west)
        total_arrived = total_cleared + remaining

        if total_arrived == 0 and total_cleared == 0:
            return 0.5

        arrival_rate = self.config.get("arrival_rate", 2.0)
        congestion = self.config.get("congestion_multiplier", 1.0)
        expected_arrived = self.env.max_time * arrival_rate * 4 * congestion
        max_possible = max(1.0, min(float(total_arrived), float(expected_arrived)))

        clear_raw = float(total_cleared) / max_possible
        clear_score = to_open_unit_interval(clear_raw)

        avg_wait = self.env.total_waiting_time / max(1, total_cleared)
        max_wait = 30.0
        wait_raw = max(0.0, 1.0 - (avg_wait / max_wait))
        wait_score = to_open_unit_interval(wait_raw)

        if self.config.get("emergency_prob", 0) > 0:
            total_emergencies = self.env.total_emergencies_generated
            handled = self.env.emergencies_handled
            em_raw = float(handled) / float(total_emergencies) if total_emergencies > 0 else 0.5
            em_score = to_open_unit_interval(em_raw)
            total = (0.5 * clear_score) + (0.3 * wait_score) + (0.2 * em_score)
        else:
            total = (0.625 * clear_score) + (0.375 * wait_score)

        return to_open_unit_interval(total)

class EasyTask(BaseTask):
    def __init__(self):
        super().__init__({"max_time": 100, "arrival_rate": 2.0, "congestion_multiplier": 1.0, "emergency_prob": 0.0})
    def evaluate(self):
        return to_open_unit_interval(super().evaluate())

class MediumTask(BaseTask):
    def __init__(self):
        super().__init__({"max_time": 200, "arrival_rate": 2.2, "congestion_multiplier": 1.5, "emergency_prob": 0.0})
    def evaluate(self):
        return to_open_unit_interval(super().evaluate())

class HardTask(BaseTask):
    def __init__(self):
        super().__init__({"max_time": 300, "arrival_rate": 2.0, "congestion_multiplier": 1.75, "emergency_prob": 0.08})
    def evaluate(self):
        return to_open_unit_interval(super().evaluate()) 