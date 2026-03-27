from src.environment import TrafficEnv
from src.models import State, StepResult
from typing import Dict, Any, Optional

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
        expected_cleared = self.env.max_time * 2.5 * 2 
        clear_score = min(1.0, self.env.total_cleared / max(1, expected_cleared))
        
        avg_wait = self.env.total_waiting_time / max(1, self.env.total_cleared)
        wait_score = max(0.0, 1.0 - (avg_wait / 20.0))
        
        if self.config.get("emergency_prob", 0) > 0:
            handled = self.env.emergencies_handled
            em_score = min(1.0, handled / max(1, handled)) if handled > 0 else 0.5
            total = (clear_score * 0.4) + (wait_score * 0.4) + (em_score * 0.2)
        else:
            total = (clear_score * 0.5) + (wait_score * 0.5)
            
        return min(1.0, max(0.0, total))

class EasyTask(BaseTask):
    def __init__(self):
        super().__init__({
            "max_time": 100,
            "arrival_rate": 2.0,
            "congestion_multiplier": 0.0,
            "emergency_prob": 0.0
        })

class MediumTask(BaseTask):
    def __init__(self):
        super().__init__({
            "max_time": 200,
            "arrival_rate": 2.0,
            "congestion_multiplier": 1.5, 
            "emergency_prob": 0.0
        })

class HardTask(BaseTask):
    def __init__(self):
        super().__init__({
            "max_time": 300,
            "arrival_rate": 1.5, 
            "congestion_multiplier": 1.5, 
            "emergency_prob": 0.05
        })
