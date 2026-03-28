from dataclasses import dataclass
from typing import Dict, Any

@dataclass
class State:
    north_queue: int
    south_queue: int
    east_queue: int
    west_queue: int
    current_signal: str
    waiting_time_total: float
    emergency_vehicle_present: bool
    time_step: int
    ns_growth: float
    ew_growth: float
    emergency_direction: str
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "north_queue": self.north_queue,
            "south_queue": self.south_queue,
            "east_queue": self.east_queue,
            "west_queue": self.west_queue,
            "current_signal": self.current_signal,
            "waiting_time_total": self.waiting_time_total,
            "emergency_vehicle_present": self.emergency_vehicle_present,
            "time_step": self.time_step,
            "ns_growth": self.ns_growth,
            "ew_growth": self.ew_growth,
            "emergency_direction": self.emergency_direction
        }

@dataclass
class Action:
    action_type: int

@dataclass
class StepResult:
    state: State
    reward: float
    done: bool
    info: Dict[str, Any]
