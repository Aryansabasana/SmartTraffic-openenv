import os
import random
import time
from typing import Dict, Any, Optional
from fastapi import FastAPI, HTTPException, Body, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
import gradio as gr
import uvicorn
import pandas as pd
import matplotlib.pyplot as plt

from src.tasks import BaseTask, EasyTask, MediumTask, HardTask
from src.models import State, StepResult
from src.agent import DeterministicAgent, LLMAgent

# --- FASTAPI BACKEND ---
app = FastAPI(title="Smart Traffic OpenEnv API")

# Global state to track the active task for the API
active_task: Optional[BaseTask] = None

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# API Helper
async def reset_logic(level: str = "easy", seed: Optional[int] = None):
    global active_task
    if level.lower() == "easy":
        active_task = EasyTask()
    elif level.lower() == "medium":
        active_task = MediumTask()
    elif level.lower() == "hard":
        active_task = HardTask()
    else:
        raise HTTPException(status_code=400, detail="Invalid level.")
    
    state = active_task.reset(seed=seed)
    return state.to_dict()

async def step_logic(action: int):
    global active_task
    if not active_task:
        raise HTTPException(status_code=400, detail="No active task.")
    result = active_task.step(action)
    return {
        "state": result.state.to_dict(),
        "reward": result.reward,
        "done": result.done,
        "info": result.info
    }

# Backward Compatible Routes (Root)
@app.post("/reset")
async def api_reset(level: str = "easy", seed: Optional[int] = None):
    return await reset_logic(level, seed)

@app.post("/step")
async def api_step(action: int = Body(..., embed=True)):
    return await step_logic(action)

@app.get("/state")
async def api_get_state():
    if not active_task: return JSONResponse({"error": "No task"}, status_code=400)
    return active_task.state().to_dict()

# /api Prefix Routes
@app.post("/api/reset")
async def api_v1_reset(level: str = "easy", seed: Optional[int] = None):
    return await reset_logic(level, seed)

@app.post("/api/step")
async def api_v1_step(action: int = Body(..., embed=True)):
    return await step_logic(action)

# --- GRADIO UI LOGIC ---

# Initialize Agents
heuristic_agent = DeterministicAgent()
llm_agent = None

def get_llm_agent():
    global llm_agent
    if llm_agent is None:
        try:
            if os.environ.get("HF_TOKEN") and os.environ.get("API_BASE_URL"):
                llm_agent = LLMAgent()
        except:
            pass
    return llm_agent

# Simulation Helper
ui_env = EasyTask()

def update_ui_step(n, s, e, w, emg, emg_dir, sig, use_llm):
    # Sync UI state to internal environment
    ui_env.env.north = n
    ui_env.env.south = s
    ui_env.env.east = e
    ui_env.env.west = w
    ui_env.env.emergency_present = emg
    ui_env.env.emergency_direction_str = emg_dir
    ui_env.env.current_signal = sig

    agent = get_llm_agent() if use_llm else heuristic_agent
    if use_llm and agent is None:
        agent = heuristic_agent # Fallback
        fallback_msg = " (Fallback: LLM not configured)"
    else:
        fallback_msg = ""

    current_state = ui_env.state()
    action, explanation = agent.get_action_with_explanation(current_state)
    
    result = ui_env.step(action)
    new_state = result.state.to_dict()
    
    # Dashboard visualization (ASCII representation)
    viz = f"""
    Queue Status:
       [N: {new_state['north_queue']}]
    [W: {new_state['west_queue']}]  +  [E: {new_state['east_queue']}]
       [S: {new_state['south_queue']}]
    
    Current Signal: {new_state['current_signal'].upper()}
    Emergency: {'⚠️ ' + new_state['emergency_direction'].upper() if new_state['emergency_vehicle_present'] else 'None'}
    """
    
    return (
        new_state['north_queue'], new_state['south_queue'], 
        new_state['east_queue'], new_state['west_queue'],
        new_state['current_signal'],
        round(result.reward, 2),
        explanation + fallback_msg,
        viz
    )

def run_simulation_batch(steps=50):
    ui_env.reset()
    history = []
    for i in range(steps):
        state = ui_env.state()
        action = heuristic_agent.get_action(state)
        res = ui_env.step(action)
        history.append({
            "Step": i,
            "NS_Queue": state.north_queue + state.south_queue,
            "EW_Queue": state.east_queue + state.west_queue,
            "Reward": res.reward
        })
    df = pd.DataFrame(history)
    
    plt.figure(figsize=(10, 4))
    plt.plot(df['Step'], df['NS_Queue'], label='NS Queue')
    plt.plot(df['Step'], df['EW_Queue'], label='EW Queue')
    plt.title("Traffic Density Over Time")
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    return df.head(10), plt, ui_env.evaluate()

# --- UI BUILDER ---
def create_ui():
    with gr.Blocks(title="Smart Traffic Control Simulator", theme=gr.themes.Soft()) as interface:
        gr.Markdown("# 🚦 Smart Traffic Control Simulator")
        gr.Markdown("An AI-driven optimization environment for urban intersection control.")
        
        with gr.Row():
            with gr.Column(scale=1):
                gr.Markdown("### 🛠️ Configuration")
                n_in = gr.Slider(0, 50, value=10, label="North Queue")
                s_in = gr.Slider(0, 50, value=10, label="South Queue")
                e_in = gr.Slider(0, 50, value=5, label="East Queue")
                w_in = gr.Slider(0, 50, value=5, label="West Queue")
                
                with gr.Row():
                    emg_in = gr.Checkbox(label="Emergency Present")
                    emg_dir_in = gr.Dropdown(["ns", "ew", "none"], value="none", label="Direction")
                
                sig_in = gr.Dropdown(["red", "green_ns", "green_ew"], value="red", label="Current Signal")
                use_llm = gr.Checkbox(label="Use AI Decision Engine (LLM)", value=False)
                
                with gr.Row():
                    step_btn = gr.Button("⏩ Next Step", variant="primary")
                    reset_btn = gr.Button("🔄 Reset Env")

            with gr.Column(scale=2):
                gr.Markdown("### 📊 Live Dashboard")
                viz_output = gr.Code(label="Intersection Map", language="markdown", lines=8)
                
                with gr.Row():
                    reward_out = gr.Number(label="Step Reward", precision=2)
                    status_out = gr.Textbox(label="Decision Logic / Explanation")

                gr.Markdown("### 📈 Batch Performance")
                batch_btn = gr.Button("🏃 Run 50-Step Simulation")
                with gr.Row():
                    table_out = gr.Dataframe(label="Recent Steps")
                    plot_out = gr.Plot(label="Queue Fluctuations")
                score_out = gr.Number(label="Efficiency Score (0-1)")

        # Event Handlers
        step_btn.click(
            fn=update_ui_step,
            inputs=[n_in, s_in, e_in, w_in, emg_in, emg_dir_in, sig_in, use_llm],
            outputs=[n_in, s_in, e_in, w_in, sig_in, reward_out, status_out, viz_output]
        )
        
        reset_btn.click(
            fn=lambda: (10, 10, 5, 5, False, "none", "red", 0, "Environment Reset", ""),
            outputs=[n_in, s_in, e_in, w_in, emg_in, emg_dir_in, sig_in, reward_out, status_out, viz_output]
        )
        
        batch_btn.click(
            fn=run_simulation_batch,
            outputs=[table_out, plot_out, score_out]
        )

    return interface

# --- FINAL ASSEMBLY ---
dashboard = create_ui()
# Mount Gradio at root '/' so it becomes the main web page.
# API endpoints remain accessible at /reset, /step, /api/reset, etc.
app = gr.mount_gradio_app(app, dashboard, path="/")

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 7860))
    uvicorn.run(app, host="0.0.0.0", port=port)
