---
title: Smart Traffic Env V2
emoji: 🚦
colorFrom: blue
colorTo: green
sdk: docker
app_port: 7860
---

# Smart Traffic Optimization Environment (OpenEnv)

> **A production-ready OpenEnv simulation framework for dynamic traffic signal orchestration, emergency vehicle routing, and queue optimization.**

## 🚀 Live Demo
👉 https://huggingface.co/spaces/AryanSabasana/smart-traffic-openenv

## 🎯 Key Features
* **Adaptive Signal Timing:** Dynamically scales green lights based on immediate queue volume.
* **Emergency Override Protocol:** Instantly preempts signals to guarantee swift ambulance clearance.
* **Anti-Oscillation (Hysteresis):** Employs strict switching penalties to prevent erratic light flickering.
* **Deterministic Evaluation:** Ensures standardized performance testing across seeded environments.

## ⚡ Quick Start
This command runs all difficulty levels (Easy, Medium, Hard) and prints performance metrics and final scores.

```bash
# Clone the repository
git clone https://github.com/Aryansabasana/SmartTraffic-openenv.git
cd SmartTraffic-openenv

# Install requirements
pip install -r requirements.txt

# Run the system evaluation
python evaluate.py
```

## 📊 Simulation Preview

### 🖥️ Interface
![Dashboard](./assets/UI.png)

### 📈 Results After Execution
![Results](./assets/Results.png)

---

## 1. Overview
The **Smart Traffic Optimization Environment** is a rigorous, Hugging Face deployable simulation built upon the OpenEnv specification. It models a complex 4-way intersection where a deterministic AI agent coordinates traffic signals to minimize congestion, maintain lane fairness, and prioritize emergency vehicles through real-time heuristic modeling.

## 2. Problem Statement
Static or poorly timed traffic light schedules struggle to handle variable vehicle influxes. This results in:
* **Exponential congestion growth** during peak hours.
* **Minority lane starvation**, where sparse lanes experience high cumulative wait times.
* **Emergency routing delays**, trapping critical vehicles behind idle traffic.

Dynamic traffic modeling provides an algorithmic solution to minimize global idling times and expedite emergency response.

## 3. Solution Approach
This system implements rigorous **Environment-Based Modeling** utilizing the OpenEnv API. A simulated junction feeds real-time queue metrics to a **Multi-Factor Heuristic AI Agent**. Rather than traditional timing cycles, this system routes traffic optimally by evaluating queue length, waiting durations, and congestion density metrics.

## 4. Architecture
The framework relies on a strictly typed modular design:
* **Environment (`TrafficEnv`)**: The core OpenEnv API managing vehicle physics, throughput bounds, and dense reward tracking.
* **Agent (`DeterministicAgent`)**: A resilient heuristic model managing dynamic signal switching and hysteresis stabilization.
* **Tasks (`src/tasks.py`)**: Scaled complexity evaluations (Easy, Medium, Hard) representing varied traffic distributions.
* **Evaluator (`evaluate.py`)**: An automated script measuring agent efficiency on an absolute `0.0` to `1.0` scale.

## 5. OpenEnv API Implementation
The system interacts seamlessly using the standard OpenEnv state triad:
```python
# 1. Initialize intersection and define random seed
state = env.reset(seed=42)

# 2. Execute agent's signal action
result = env.step(action_type)

# 3. Retrieve updated observation state
current_state = env.state()
```

## 6. State & Action Spaces
### Observation State
The environment yields structured JSON data encapsulating intersection telemetry:
```json
{
  "north_queue": 15,
  "south_queue": 12,
  "east_queue": 2,
  "west_queue": 0,
  "current_signal": "green_ns",
  "waiting_time_total": 45.0,
  "emergency_vehicle_present": true,
  "ns_wait_time": 2.5,
  "ew_wait_time": 0.0,
  "emergency_direction": "ns",
  "time_step": 12
}
```

### Action Space
The agent navigates a discrete `[0, 1, 2]` action space:
* `0` → **All Red** (Safety clearance)
* `1` → **Green North-South** 
* `2` → **Green East-West**

## 7. Reward Function
The environment evaluates actions via a stable dense reward mechanism constrained to prevent exploding gradients:
* **Waiting Penalties**: Applies gradual penalties based on total cars waiting.
* **Throughput Rewards**: Positively scaled by the number of vehicles successfully cleared per step.
* **Hysteresis Penalty**: Discourages rapid signal switching.
* **Emergency Bonus**: Significantly rewards rapid clearance of active emergency situations.

## 8. 🔁 Reproducibility
The simulation execution strictly manages random number generator (RNG) states across environments. By passing a fixed seed (e.g., `env.reset(seed=42)`), the generated traffic patterns, emergency occurrences, and agent's subsequent calculations will remain completely reproducible. This deterministic evaluation allows for exact performance comparisons and precise metric audits.

## 9. 💡 Why This Is a Real-World Simulation
Unlike simple grid-worlds, this system enforces constraints common in actual physical infrastructure:
* **Max Capacity Bounds**: Throughput is physically limited by the duration of green lights.
* **Lane Fairness**: Continuous heavy traffic in one direction necessitates starvation limits to force clearing across cross-arteries over time.
* **System Oscillations**: Changing lights requires cooldowns, accurately preventing logical flickering.

## 10. Performance Results
By optimizing the routing algorithm away from simple size-comparisons toward multi-layered, stability-controlled metrics, the AI consistently achieves a **~0.93 – 0.96 overall score (varies by seed)** 

| Difficulty | Baseline Score | Optimized Agent |
|------------|----------------|-----------------|
| **Easy**   | 0.94           | **1.00**        |
| **Medium** | 0.50           | **0.95**        |
| **Hard**   | 0.60           | **~0.91**       |

## 11. Containerized Deployment
This platform includes a pre-configured Docker image designed for high-availability Hugging Face deployments.

```bash
# Build the Python 3.11-slim container
docker build -t openenv-traffic .

# Run locally
docker run -p 7860:7860 --rm openenv-traffic
```

## 12. Project Structure
```text
OpenEnv/
├── Dockerfile           # Deployment container definition
├── requirements.txt     # Python dependencies mapping
├── evaluate.py          # Unified execution logic
├── dashboard.py         # Streamlit interface
├── app.py               # Hugging Face deployment entrypoint
├── src/                 
│   ├── models.py        # Typed API Dataclasses
│   ├── environment.py   # State logic, dynamics, and rewards
│   ├── tasks.py         # Scored environments (Easy/Medium/Hard)
│   └── agent.py         # Priority-based heuristic logic
└── .dockerignore        # Build caching optimization
```

