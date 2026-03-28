from src.models import State

class DeterministicAgent:
    def __init__(self):
        self.last_switch_time = 0
        self.min_green_time = 3

    def get_action(self, state: State) -> int:
        ns_total = state.north_queue + state.south_queue
        ew_total = state.east_queue + state.west_queue
        
        current_idx = 1 if state.current_signal == "green_ns" else (2 if state.current_signal == "green_ew" else 0)
        time_since_switch = state.time_step - self.last_switch_time
        

        if state.emergency_vehicle_present and state.emergency_direction != 'none':
            em_idx = 1 if state.emergency_direction == 'ns' else 2
            if current_idx != em_idx:
                self.last_switch_time = state.time_step
            return em_idx
            

        if current_idx != 0 and time_since_switch < self.min_green_time:
            return current_idx
            

        ns_pressure = ns_total + (state.ns_growth * 1.5)
        ew_pressure = ew_total + (state.ew_growth * 1.5)
        

        if ns_total > 30: ns_pressure += 20
        if ew_total > 30: ew_pressure += 20
        
        threshold = 5.0
        
        if current_idx == 1:
            if ew_pressure > ns_pressure + threshold:
                target_idx = 2
            else:
                target_idx = 1
        elif current_idx == 2:
            if ns_pressure > ew_pressure + threshold:
                target_idx = 1
            else:
                target_idx = 2
        else:
            target_idx = 1 if ns_pressure >= ew_pressure else 2
            
        if target_idx != current_idx:
            self.last_switch_time = state.time_step
            
        return target_idx
