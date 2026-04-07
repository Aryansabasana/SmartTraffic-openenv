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

    def get_action_with_explanation(self, state: State) -> tuple[int, str]:
        action = self.get_action(state)
        reasons = []
        ns_total = state.north_queue + state.south_queue
        ew_total = state.east_queue + state.west_queue
        if state.emergency_vehicle_present:
            reasons.append(f"EMERGENCY DETECTED in {state.emergency_direction.upper()} direction.")
        if action == 1:
            reasons.append(f"Priority given to North-South (Queue: {ns_total} vehicles).")
        elif action == 2:
            reasons.append(f"Priority given to East-West (Queue: {ew_total} vehicles).")
        else:
            reasons.append("Signals set to RED for safety switching.")
        if state.ns_wait_time > 10 or state.ew_wait_time > 10:
            reasons.append("Mitigating lane starvation due to high wait time.")
        return action, " ".join(reasons)

class LLMAgent:
    def __init__(self):
        from openai import OpenAI
        import os
        
        self.client = OpenAI(
            base_url=os.environ.get("API_BASE_URL", "https://api-inference.huggingface.co/v1/"),
            api_key=os.environ.get("HF_TOKEN")
        )
        self.model = os.environ.get("MODEL_NAME", "meta-llama/Llama-3.1-8B-Instruct")

    def get_action(self, state: State) -> int:
        state_dict = state.to_dict()
        prompt = f"""
        Current Traffic State:
        - North Queue: {state_dict['north_queue']}
        - South Queue: {state_dict['south_queue']}
        - East Queue: {state_dict['east_queue']}
        - West Queue: {state_dict['west_queue']}
        - Current Signal: {state_dict['current_signal']}
        - Emergency Vehicle: {'Yes' if state_dict['emergency_vehicle_present'] else 'No'} (Direction: {state_dict['emergency_direction']})
        - NS Wait Time: {state_dict['ns_wait_time']}
        - EW Wait Time: {state_dict['ew_wait_time']}
        - Total Waiting Time: {state_dict['waiting_time_total']}
        - Time Step: {state_dict['time_step']}

        Action space:
        0: All Red (Safety Switch)
        1: Green North-South
        2: Green East-West

        Goal: Minimize waiting time and prioritize emergency vehicles.
        Return ONLY the action number (0, 1, or 2).
        """
        
        try:
            response = self.client.chat.completions.create(
                model=self.model,
                messages=[
                    {"role": "system", "content": "You are an AI Traffic Signal Controller. You must output only a single integer: 0, 1, or 2."},
                    {"role": "user", "content": prompt}
                ],
                max_tokens=5,
                temperature=0.0
            )
            content = response.choices[0].message.content.strip()
            # Extract the first digit found in case the LLM adds text
            import re
            match = re.search(r'\d', content)
            if match:
                action = int(match.group())
                if action in [0, 1, 2]:
                    return action
            return 0 # Fallback
        except Exception as e:
            print(f"LLM Error: {e}")
            return 0 # Fallback safety

    def get_action_with_explanation(self, state: State) -> tuple[int, str]:
        # Simple placeholder for LLM explanation - in production, you'd ask the LLM for the reason too.
        action = self.get_action(state)
        explanation = f"AI Model {self.model} selected action {action} based on deep neural optimization."
        if state.emergency_vehicle_present:
            explanation += " Emergency prioritization was weighted into the decision."
        return action, explanation
