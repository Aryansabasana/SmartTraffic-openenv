import random
import numpy as np
from typing import Dict, Any, Optional
from src.models import State, Action, StepResult

class TrafficEnv:
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.max_time = config.get("max_time", 100)
        self.arrival_rate_base = config.get("arrival_rate", 2)
        self.congestion_multiplier = config.get("congestion_multiplier", 1.0)
        self.emergency_prob = config.get("emergency_prob", 0.0)
        self.queue_cap = 100
        self.reset()
        
    def reset(self, seed: Optional[int] = None) -> State:
        if seed is not None:
            random.seed(seed)
            np.random.seed(seed)
            
        self.north = 0
        self.south = 0
        self.east = 0
        self.west = 0
        
        self.current_signal = "red"
        self.waiting_time_total = 0.0
        self.time_step = 0
        
        self.emergency_present = False
        self.emergency_direction_str = 'none'
        
        self.total_cleared = 0
        self.total_waiting_time = 0.0
        self.emergency_response_time = 0
        self.emergencies_handled = 0
        self.total_emergencies_generated = 0
        self.done = False
        
        self.prev_ns_total = 0
        self.prev_ew_total = 0
        self.ns_wait_time = 0.0
        self.ew_wait_time = 0.0
        self.reward_trends = []
        
        return self.state()
        
    def state(self) -> State:
        ns_total = self.north + self.south
        ew_total = self.east + self.west
        ns_growth = float(ns_total - self.prev_ns_total)
        ew_growth = float(ew_total - self.prev_ew_total)
        return State(
            north_queue=self.north,
            south_queue=self.south,
            east_queue=self.east,
            west_queue=self.west,
            current_signal=self.current_signal,
            waiting_time_total=self.waiting_time_total,
            emergency_vehicle_present=self.emergency_present,
            time_step=self.time_step,
            ns_growth=ns_growth,
            ew_growth=ew_growth,
            emergency_direction=self.emergency_direction_str,
            ns_wait_time=self.ns_wait_time,
            ew_wait_time=self.ew_wait_time
        )
        
    def step(self, action_idx: int) -> StepResult:
        if self.done:
            return StepResult(self.state(), 0, True, {"msg": "Done"})
            
        self.prev_ns_total = self.north + self.south
        self.prev_ew_total = self.east + self.west
            
        action = Action(action_idx)
        reward = 0.0
        
        prev_signal = self.current_signal
        
        if action.action_type == 0:
            self.current_signal = "red"
        elif action.action_type == 1:
            self.current_signal = "green_ns"
        elif action.action_type == 2:
            self.current_signal = "green_ew"
            
        if prev_signal != self.current_signal and prev_signal != "red":
            reward -= 1.0  # Stable switching penalty
            
        ns_total = self.north + self.south
        ew_total = self.east + self.west
        total_waiting = ns_total + ew_total
        
        # 2. Global Congestion Penalty (Secondary Gradient)
        # Drastically reduced multiplier so it does not overwhelm clearance in Hard mode
        reward -= (total_waiting * 0.01) 
        
        self.waiting_time_total += total_waiting
        self.total_waiting_time += total_waiting
        
        if self.emergency_present:
            self.emergency_response_time += 1
            reward -= 0.5            
        cleared_this_step = 0
        clearance_capacity = 8
        emergency_cleared = False
        
        if self.current_signal == "green_ns":
            c_n = min(self.north, clearance_capacity)
            c_s = min(self.south, clearance_capacity)
            self.north -= c_n
            self.south -= c_s
            cleared_this_step = c_n + c_s
            if self.emergency_present and self.emergency_direction_str == 'ns':
                emergency_cleared = True
                
        elif self.current_signal == "green_ew":
            c_e = min(self.east, clearance_capacity)
            c_w = min(self.west, clearance_capacity)
            self.east -= c_e
            self.west -= c_w
            cleared_this_step = c_e + c_w
            if self.emergency_present and self.emergency_direction_str == 'ew':
                emergency_cleared = True
                
        self.total_cleared += cleared_this_step
        
        # 1. Base Clearance Reward (Primary Goal)
        reward += cleared_this_step * 1.0 
        
        # 5. Empty Lane Inefficiency (Prevents wasting green lights)
        if self.current_signal == "green_ns" and ns_total == 0:
            reward -= 1.0
        elif self.current_signal == "green_ew" and ew_total == 0:
            reward -= 1.0
        
        if total_waiting > 0 and cleared_this_step == 0:
            reward -= 0.5
            
        if emergency_cleared:
            reward += 20.0 # Massive bonus for fast clearance
            self.emergency_present = False
            self.emergency_direction_str = 'none'
            self.emergencies_handled += 1
            
        # 3. Starvation Penalty (Targets poor timing, not environmental volume)
        if ns_total > 0 and self.current_signal != "green_ns":
            self.ns_wait_time += 1.0
            reward -= (self.ns_wait_time * 0.05)
        else:
            self.ns_wait_time = 0.0
            
        if ew_total > 0 and self.current_signal != "green_ew":
            self.ew_wait_time += 1.0
            reward -= (self.ew_wait_time * 0.05)
        else:
            self.ew_wait_time = 0.0
            
        reward -= 0.1  # Small step penalty to encourage speed

        

        current_multiplier = 1.0 + (self.congestion_multiplier * (self.time_step / self.max_time))
        total_expected_rate = (self.arrival_rate_base * 4) * current_multiplier
        

        noise_factor = random.uniform(0.85, 1.15)
        noisy_rate = total_expected_rate * noise_factor
        

        lane_split = np.random.dirichlet([5, 5, 5, 5])
        
        def arrive(r):
            base = int(r)
            return base + 1 if random.random() < (r - base) else base
            
        self.north = min(self.queue_cap, self.north + arrive(noisy_rate * lane_split[0]))
        self.south = min(self.queue_cap, self.south + arrive(noisy_rate * lane_split[1]))
        self.east = min(self.queue_cap, self.east + arrive(noisy_rate * lane_split[2]))
        self.west = min(self.queue_cap, self.west + arrive(noisy_rate * lane_split[3]))
        
        if not self.emergency_present and random.random() < self.emergency_prob:
            self.emergency_present = True
            self.emergency_direction_str = random.choice(['ns', 'ew'])
            self.total_emergencies_generated += 1
            

        reward += random.uniform(-0.1, 0.1)
        
        self.time_step += 1
        if self.time_step >= self.max_time:
            self.done = True
            
        self.reward_trends.append(reward)
            
        info = {
            "total_cleared": self.total_cleared,
            "avg_waiting_time": self.total_waiting_time / max(1, self.total_cleared),
            "emergencies_handled": self.emergencies_handled,
            "total_emergencies": self.total_emergencies_generated,
            "reward_trend_avg": sum(self.reward_trends[-10:]) / 10 if self.reward_trends else 0
        }
            
        return StepResult(self.state(), reward, self.done, info)
