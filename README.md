---
title: Smart Traffic Optimization
emoji: 🚦
colorFrom: green
colorTo: red
sdk: docker
app_file: app.py
pinned: false
---

# Smart Traffic Optimization Environment (OpenEnv)
> **A high-performance OpenEnv simulation designed to mathematically conquer urban gridlock and emergency response routing through dynamic AI signal orchestration.**

## 1. Overview
The **Smart Traffic Optimization Environment** is a production-ready, Hugging Face deployable simulation built strictly upon the OpenEnv specification. It simulates a busy 4-way intersection where a deterministic AI system orchestrates traffic lights to drastically reduce congestion, balance queue fairness, and prioritize emergency vehicles in real-time.

---

## 2. Problem Statement
Modern urban landscapes suffer constantly from static or poorly-timed traffic light schedules. These outdated systems ignore real-time vehicle influxes, resulting in:
* **Exponential congestion cascades** (queue explosions).
* **Starvation** (minority lanes waiting indefinitely).
* **Emergency Routing Failures** (ambulances stuck behind idle traffic).

Dynamic traffic optimization is critical to lowering global carbon emissions from idling cars and saving lives via immediate emergency clearance.

---

## 3. Solution Approach
This project solves the gridlock problem utilizing a rigorous **Environment-Based Modeling** approach under the OpenEnv API. A simulated junction actively feeds real-time pressure metrics to a **Heuristic AI Agent**. Rather than using naive signal timers, our optimization strategy evaluates queue sizes, traffic *growth rates*, and starvation limits to route traffic efficiently out of the intersection.

---

## 4. Architecture
The system employs a strict, typed modular setup:
* **Environment (`TrafficEnv`)**: The core OpenEnv API housing precise vehicle mechanics, bounding limits, and the complex reward calculation.
* **Agent (`DeterministicAgent`)**: A robust AI employing pressure-based heuristics and cooldown stabilization algorithms.
* **Tasks (`src/tasks.py`)**: Progressive difficulties (Easy, Medium, Hard) representing different curriculum learning distributions.
* **Evaluation System (`evaluate.py`)**: An automated script grading the mathematical efficiency of the agent on a `0.0` to `1.0` scale.

---

## 5. OpenEnv API Implementation
The system strictly implements the standard OpenEnv triad:
```python
# 1. Initializes the intersection and resets all queues
state = env.reset()

# 2. Applies the agent's signal decision
result = env.step(action_type)

# 3. Fetches the current observation space
current_state = env.state()
```

---

## 6. State Space
The environment returns a standard structured JSON tracking realistic intersection physics:
```json
{
  "north_queue": 15,
  "south_queue": 12,
  "east_queue": 2,
  "west_queue": 0,
  "current_signal": "green_ns",
  "waiting_time_total": 45.0,
  "emergency_vehicle_present": true,
  "ns_growth": 2.5,
  "ew_growth": 0.0,
  "emergency_direction": "ns",
  "time_step": 12
}
```

---

## 7. Action Space
The agent utilizes a discrete `[0, 1, 2]` action space, allowing for safety transitions and directional routing:
* `0` → **All Red** (Intersection clearing / safety pause)
* `1` → **Green North-South** 
* `2` → **Green East-West**

---

## 8. Reward Function
The heart of the simulation is a highly constrained, stabilized dense reward function (scaled to remain cleanly between `-5` and `+10` per step to prevent vanishing/exploding gradients in arbitrary networks):
* **- (Total Queue * 0.1)**: Continual small penalties tracking volumetric waiting time.
* **+ (Cleared Vehicles * 0.5)**: Rewarded proactively for pushing throughput.
* **- 1.0 Oscillation Penalty**: Penalizes the agent for flickering lights repeatedly.
* **- 0.5 Idle Penalty**: Penalizes leaving a light green while the lane is entirely empty but cross-traffic waits.
* **+ 10.0 Emergency Bonus**: Massive spike given when an active emergency vehicle is successfully routed through.

---

## 9. Tasks Curriculum
The grading framework evaluates the AI over three escalating complexities:
* **Easy**: Mild, fixed traffic flow. Goal: Basic queue reduction and API validation.
* **Medium**: Higher volumes scaling progressively over time. Goal: Enforce equal lane balancing against starvation contexts.
* **Hard**: Multi-objective routing featuring intense traffic surges (Rush Hour simulation) mixed dynamically with emergency vehicle spawns.

---

## 10. Agent Strategy
The optimized heuristic agent radically deviates from naive comparison models. Its logic checks multiple decision layers:
1. **Emergency Prioritization**: Immediate hard-override of signals toward active emergency routes.
2. **Cooldown Stability**: Signal execution locks for a minimum of 3 steps to prevent light-flickering.
3. **Pressure calculation**: Adds flat queue counts to the exact *growth rate* metric `(size + rate * 1.5)` to proactively switch before a lane overflows.
4. **Fairness Subroutine**: Any lane passing 30 vehicles forces an artificial pressure spike, guaranteeing traffic prevents permanent starvation.

---

## 11. Final Results
By optimizing the routing algorithm away from simple size-comparison to multi-layered, stability-controlled pressure metrics, **the AI achieved a total 0.94 / 1.00 Score.** 

| Difficulty | Baseline Agent | Optimized Agent | Improvement |
|------------|---------------|-----------------|-------------|
| **Easy**   | 0.94          | **0.95**        | Minimal     |
| **Medium** | 0.50          | **0.92**        | **+84%**    |
| **Hard**   | 0.60          | **0.96**        | **+60%**    |

The advanced logic drastically fixed the mathematical queue explosion that plagued the Medium/Hard tasks originally.

---

## 12. Installation & Setup
The project functions entirely upon Python standard libraries and is exceedingly lightweight.
```bash
# Clone the repository
git clone https://github.com/yourusername/SmartTrafficOpenEnv.git
cd SmartTrafficOpenEnv

# Create & activate the virtual environment (Windows/Bash)
python -m venv venv
source venv/Scripts/activate

# Install requirements (if modifying with external analytics)
pip install -r requirements.txt
```

---

## 13. Running Evaluation
To execute the task simulations and test the intelligence of the agent directly:
```bash
python evaluate.py
```
*Interpret the logs:* The shell prints `[Hard] Steps: X | Total Reward: X` alongside exact clearance metrics (Clearance quantity, Avg Wait per Car, and Emergencies handled) culminating in the `0 to 1` bounded metric.

---

## 14. Docker Setup
To test the environment natively with containerization limits:
```bash
# Build the highly optimized Python 3.11-slim container
docker build -t openenv-traffic .

# Run the execution evaluation container
docker run --rm openenv-traffic
```

---

## 15. Deployment (Hugging Face Spaces)
The configuration is **Hugging Face Spaces** ready using a standard Docker workflow. Simply push the provided `Dockerfile` and `requirements.txt` to a Docker-backed HF space, and the entrypoint will instantly validate the AI on the Hugging Face hardware endpoints.

---

## 16. Project Structure
```text
OpenEnv/
├── openenv.yaml         # Environment configuration metadata
├── Dockerfile           # Deployment container definition
├── requirements.txt     # Python dependencies mapping
├── README.md            # You are here
├── evaluate.py          # Unified execution script 
├── src/                 
│   ├── models.py        # Strongly typed API Dataclasses
│   ├── environment.py   # Core logic, dynamics, and dense reward generator
│   ├── tasks.py         # Tasks Configs & automated 0.0-1.0 Grader 
│   └── agent.py         # Advanced optimal heuristic logic
└── venv/                # Local virtual environment
```

---

## 17. Future Improvements
* **Multi-Intersection Network**: Connecting `TrafficEnv` schemas into a 3x3 grid where an AI must predict upstream congestion.
* **Deep Q-Network Integration**: Replacing the deterministic heuristic with a trainable RL module connecting to Pytorch arrays.
* **Live Camera Mapping**: Parsing real-world YOLOv8 intersections straight into the OpenEnv `State` object for live-world routing.

---

## 18. Conclusion
The **Smart Traffic Optimization Environment** successfully bridges the gap between simulated theory and real-world execution. By actively tracking queue trajectories and rigorously anchoring reward schemes through OpenEnv's standard dynamics, this architecture proves how lightweight, meticulously designed AI orchestrations can directly remedy catastrophic infrastructural problems cleanly and understandably.
