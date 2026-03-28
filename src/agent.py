import random
from src.models import State

class DeterministicAgent:
    def __init__(self):
        self.last_switch_time = 0
        self.ns_active_time = 0
        self.ew_active_time = 0

    def get_action(self, state: State) -> int:
        ns_total = state.north_queue + state.south_queue
        ew_total = state.east_queue + state.west_queue
        
        current_idx = 1 if state.current_signal == "green_ns" else (2 if state.current_signal == "green_ew" else 0)
        time_since_switch = state.time_step - self.last_switch_time
        
        # B. EMERGENCY HANDLING
        emergency_bonus = 10000.0
        ns_em = 1 if (state.emergency_vehicle_present and state.emergency_direction == 'ns') else 0
        ew_em = 1 if (state.emergency_vehicle_present and state.emergency_direction == 'ew') else 0

        # Override and extend green logic
        if ns_em and current_idx != 1:
            self.last_switch_time = state.time_step
            return 1
        if ew_em and current_idx != 2:
            self.last_switch_time = state.time_step
            return 2
            
        # Weights for heuristic
        w1 = 12.0  # queue_length
        w2 = 3.0  # total_wait_time
        w3 = emergency_bonus # emergency_presence
        w4 = 6.0  # starvation_penalty
        w5 = 15.0 # recently_served_penalty
        
        # Starvation penalization - drastically spikes priority if ignored while waiting
        ns_starvation = (state.ns_wait_time ** 1.5) if current_idx != 1 else 0
        ew_starvation = (state.ew_wait_time ** 1.5) if current_idx != 2 else 0
        
        # Priority calculation per lane
        ns_priority = (
            w1 * ns_total +
            w2 * state.ns_wait_time +
            w3 * ns_em +
            w4 * ns_starvation -
            w5 * (time_since_switch if current_idx == 1 else 0)
        )
        
        ew_priority = (
            w1 * ew_total +
            w2 * state.ew_wait_time +
            w3 * ew_em +
            w4 * ew_starvation -
            w5 * (time_since_switch if current_idx == 2 else 0)
        )
        
        # Controlled variability to fix deterministic bounds
        epsilon = 0.5
        ns_priority += random.uniform(-epsilon, epsilon)
        ew_priority += random.uniform(-epsilon, epsilon)
            
        # Target determination
        target_idx = current_idx
        if current_idx == 1:
            if ew_priority > ns_priority: target_idx = 2
        elif current_idx == 2:
            if ns_priority > ew_priority: target_idx = 1
        else:
            target_idx = 1 if ns_priority >= ew_priority else 2
            
        # C. ADAPTIVE SIGNAL TIMING
        # Avoid fixed timing: longer green time for high congestion
        current_queue = ns_total if current_idx == 1 else (ew_total if current_idx == 2 else 0)
        dynamic_min_green = max(3, min(12, current_queue // 5))
        
        # Fast exit: Switch immediately if active lane is completely empty
        if current_queue == 0 and (ns_total > 0 or ew_total > 0):
            target_idx = 2 if current_idx == 1 else 1
        elif current_idx != 0 and time_since_switch < dynamic_min_green and current_queue > 0:
            target_idx = current_idx
            
        if target_idx != current_idx:
            self.last_switch_time = state.time_step
            
        return target_idx

