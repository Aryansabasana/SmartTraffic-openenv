---
title: Smart Traffic AI Suite
emoji: 🚦
colorFrom: blue
colorTo: green
sdk: docker
app_port: 7860
---

# 🚦 Smart Traffic AI Optimization Suite
**A High-Fidelity, OpenEnv-Compliant AI Simulation Platform**

This project is a production-grade traffic simulation environment designed for evaluating reinforcement learning (RL) and Large Language Model (LLM) agents. Built on a **FastAPI/Gradio Hybrid Architecture**, it provides both a stunning interactive dashboard for demonstrations and standardized REST endpoints for automated evaluation.

## ✨ Key Features

- **🏰 Visual Intersection Dashboard**: An HTML/CSS-animated interface with real-time lane coloring, pulsing emergency alerts, and directional flow arrows.
- **🧠 Hybrid AI Engines**: 
  - **Heuristic Engine**: A robust deterministic state-machine for baseline performance.
  - **Neural AI Engine**: Support for OpenAI-compatible LLM agents (Llama 3.1, GPT-4, etc.) for high-level reasoning.
- **📊 Advanced Analytics**: 
  - **Infrastructure Load Plotting**: Real-time Matplotlib visualization of corridor density.
  - **Efficiency Index**: A composite, dynamic metric (30% Pressure, 30% Wait, 40% Throughput) for sub-second performance monitoring.
  - **Imbalance Meter**: A visual colored bar tracking traffic distribution quality.
- **🎯 Multi-Task Scenarios**: Pre-configured tasks ranging from "Easy" arrival rates to "Hard" emergency-heavy simulations.

## 🏗️ Technical Architecture

- **Backend**: FastAPI with asynchronous support for state/step management.
- **UI**: Gradio promoted to the root path (`/`) via `mount_gradio_app`.
- **Compliance**: Fully compliant with the **OpenEnv 1.0 Specification** for external validation.
- **Containerization**: Optimized Docker build targeting low-latency Hugging Face Spaces deployment.

## 📡 OpenEnv API Interface

The simulator exposes the following endpoints for automated agents:

| Method | Endpoint | Description |
| :--- | :--- | :--- |
| `POST` | `/api/reset` | Resets the interaction and returns the initial 13-field state. |
| `POST` | `/api/step` | Processes an action `{ "action": 0|1|2 }` and returns observation/reward. |
| `GET` | `/api/state` | Returns the current system observation without advancing time. |

### Observation Space (13 Fields)
Agents receive a state dictionary including:
- **Queue Sizes**: `north_queue`, `south_queue`, `east_queue`, `west_queue`.
- **Dynamics**: `ns_growth`, `ew_growth` (vehicles/step delta).
- **Wait Times**: `waiting_time_total`, `ns_wait_time`, `ew_wait_time`.
- **Emergencies**: `emergency_vehicle_present`, `emergency_direction`.

## 🚦 Action Space
The environment uses a **Discrete(3)** action space:
- `0`: **All Red** (Safety switching/Clearance).
- `1`: **Green North-South** (Corridor Priority).
- `2`: **Green East-West** (Corridor Priority).

## 🚀 Running Locally

1. **Clone the Repo**:
   ```bash
   git clone <repository_url>
   cd OpenEnv
   ```

2. **Install Dependencies**:
   ```bash
   pip install -r requirements.txt
   ```

3. **Configure LLM (Optional)**:
   Add these to your environment to enable the Neural AI Engine:
   - `HF_TOKEN`: Your Hugging Face API key.
   - `API_BASE_URL`: OpenAI-compatible endpoint.
   - `MODEL_NAME`: e.g., `meta-llama/Llama-3.1-8B-Instruct`.

4. **Launch**:
   ```bash
   python app.py
   ```
   *Visit `http://localhost:7860` to access the dashboard.*

## 📈 Optimization Logic (Scoring)
The system evaluates agents based on a weighted multi-objective formula:
- **50% Clearance Rate**: Maximizing vph (vehicles per hour) throughput.
- **30% Wait Quality**: Minimizing the cumulative wait-time penalty.
- **20% Safety Status**: Successful handling and clearance of priority emergency vehicles.

---
**Developed for the OpenEnv Smart Traffic Hackathon.**
