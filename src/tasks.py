from src.environment import TrafficEnv
from src.models import State, StepResult
from typing import Dict, Any, Optional

EPS = 0.01

def to_open_unit_interval(x: float) -> float:
    """
    Strictly maps any float x to the open unit interval (0, 1).
    Handles NaN, Inf, and ensures scores are never 0 or 1.
    """
    import math

    # Handle invalid numbers
    if x is None or math.isnan(x):
        return 0.5
    if math.isinf(x):
        return 1.0 - EPS if x > 0 else EPS

    # Clamp strictly inside (0,1)
    if x <= 0.0:
        return EPS
    if x >= 1.0:
        return 1.0 - EPS

    # Final safety clamp
    return max(EPS, min(1.0 - EPS, float(x)))

class BaseTask:
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.env = TrafficEnv(config)
        
    def reset(self, seed: Optional[int] = None) -> State:
        return self.env.reset(seed=seed)
        
    def step(self, action_type: int) -> StepResult:
        return self.env.step(action_type)
        
    def state(self) -> State:
        return self.env.state()
        
    def evaluate(self) -> float:
        total_arrived = self.env.total_cleared + max(0,
            self.env.north + self.env.south + self.env.east + self.env.west)

        if total_arrived == 0 and self.env.total_cleared == 0:
            return to_open_unit_interval(1.0) # High score for empty traffic

        # Accurately map max clearance logic to prevent low-traffic penalties & high-traffic inflation.
        expected_arrived = self.env.max_time * self.config.get("arrival_rate", 2.0) * 4 * self.config.get("congestion_multiplier", 1.0)
        max_possible = min(float(total_arrived), float(expected_arrived))
        clear_score = min(1.0, self.env.total_cleared / max(1.0, max_possible))

        avg_wait = self.env.total_waiting_time / max(1, self.env.total_cleared)
        max_wait = 30.0
        wait_score = max(0.0, 1.0 - (avg_wait / max_wait))

        if self.config.get("emergency_prob", 0) > 0:
            handled = self.env.emergencies_handled
            total_emergencies = self.env.total_emergencies_generated
            
            em_score = (handled / max(1, total_emergencies)) if total_emergencies > 0 else 1.0
            
            # User defined balanced score
            total = (0.5 * clear_score) + (0.3 * wait_score) + (0.2 * em_score)
        else:
            # Rescaled symmetrically if no emergency component applies
            total = (0.625 * clear_score) + (0.375 * wait_score)

        return to_open_unit_interval(total)

class EasyTask(BaseTask):
    def __init__(self):
        super().__init__({
            "max_time": 100,
            "arrival_rate": 2.0,
            "congestion_multiplier": 1.0,
            "emergency_prob": 0.0
        })

class MediumTask(BaseTask):
    def __init__(self):
        super().__init__({
            "max_time": 200,
            "arrival_rate": 2.2,
            "congestion_multiplier": 1.5, 
            "emergency_prob": 0.0
        })

class HardTask(BaseTask):
    def __init__(self):
        super().__init__({
            "max_time": 300,
            "arrival_rate": 2.0, 
            "congestion_multiplier": 1.75, 
            "emergency_prob": 0.08
        })
