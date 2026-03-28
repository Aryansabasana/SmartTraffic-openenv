from src.models import State

class DeterministicAgent:
    def __init__(self):
        self.last_switch_time = 0

    def get_action(self, state: State) -> int:
        ns_total = state.north_queue + state.south_queue
        ew_total = state.east_queue + state.west_queue
        
        current_idx = 1 if state.current_signal == "green_ns" else (2 if state.current_signal == "green_ew" else 0)
        time_since_switch = state.time_step - self.last_switch_time
        
        # B. EMERGENCY HANDLING
        # If emergency detected: immediately prioritize that lane and reduce switching delay
        if state.emergency_vehicle_present and state.emergency_direction != 'none':
            em_idx = 1 if state.emergency_direction == 'ns' else 2
            if current_idx != em_idx:
                self.last_switch_time = state.time_step
                return em_idx
            return current_idx

        # A. PRIORITY SCORE PER LANE
        # Compute priority based on queue, wait time, and penalties
        weight_queue = 5.0
        weight_wait = 12.0
        
        # We calculate total priority per axis since signals control NS and EW together
        ns_priority = (ns_total * weight_queue) + (state.ns_wait_time * weight_wait)
        ew_priority = (ew_total * weight_queue) + (state.ew_wait_time * weight_wait)
        
        # D. SWITCHING PENALTY (Hysteresis & Cooldown)
        # Prevent rapid switching between lanes -> apply a hysteresis bonus to the currently active lane
        # And recently served penalty -> if you've been active for a while, decrease your priority to ensure fairness.
        hysteresis_bonus = 25.0
        recently_served_penalty = (time_since_switch * 2.5)
        
        if current_idx == 1:
            ns_priority += hysteresis_bonus
            ns_priority -= recently_served_penalty
        elif current_idx == 2:
            ew_priority += hysteresis_bonus
            ew_priority -= recently_served_penalty
            
        # Target determination
        target_idx = current_idx
        if current_idx == 1:
            if ew_priority > ns_priority: target_idx = 2
        elif current_idx == 2:
            if ns_priority > ew_priority: target_idx = 1
        else:
            target_idx = 1 if ns_priority >= ew_priority else 2
            
        # C. ADAPTIVE SIGNAL TIMING
        # Avoid fixed timing: longer green time for high congestion, shorter green time for low traffic
        current_queue = ns_total if current_idx == 1 else (ew_total if current_idx == 2 else 0)
        dynamic_min_green = max(2, min(10, current_queue // 6))
        
        # 4 & 5. REDUCE WAIT TIME & DYNAMIC RESPONSE
        # Switch immediately if active lane is completely empty to avoid idle greens
        if current_queue == 0 and (ns_total > 0 or ew_total > 0):
            target_idx = 2 if current_idx == 1 else 1
        elif current_idx != 0 and time_since_switch < dynamic_min_green and current_queue > 0:
            # Enforce minimum dynamic green time unless idle
            target_idx = current_idx
            
        if target_idx != current_idx:
            self.last_switch_time = state.time_step
            
        return target_idx
