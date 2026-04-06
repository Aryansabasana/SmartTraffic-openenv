import os
import random
import time
from typing import Dict, Any, Optional, List
from fastapi import FastAPI, HTTPException, Body, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
import gradio as gr
import uvicorn
import pandas as pd
import matplotlib.pyplot as plt
import io

from src.tasks import EasyTask
from src.models import State, StepResult
from src.agent import DeterministicAgent, LLMAgent

# --- FASTAPI BACKEND ---
app = FastAPI(title="Smart Traffic OpenEnv API")
active_task = EasyTask()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# API Helper
async def reset_logic(level: str = "easy", seed: Optional[int] = None):
    global active_task
    active_task.reset(seed=seed)
    return active_task.state().to_dict()

async def step_logic(action: int):
    result = active_task.step(action)
    return {
        "state": result.state.to_dict(),
        "reward": result.reward,
        "done": result.done,
        "info": result.info
    }

# API Routes
@app.post("/reset")
@app.post("/api/reset")
async def api_reset(level: str = "easy", seed: Optional[int] = None):
    return await reset_logic(level, seed)

@app.post("/step")
@app.post("/api/step")
async def api_step(action: int = Body(..., embed=True)):
    return await step_logic(action)

@app.get("/state")
@app.get("/api/state")
async def api_get_state():
    return active_task.state().to_dict()

# --- UI VISUALIZATION HELPERS ---

def generate_intersection_html(state_dict):
    """Creates a high-fidelity visual intersection with CSS."""
    sig = state_dict['current_signal']
    ns_color = "#2ecc71" if sig == "green_ns" else "#e74c3c"
    ew_color = "#2ecc71" if sig == "green_ew" else "#e74c3c"
    
    # Pulse animation for emergency
    emergency_style = ""
    if state_dict['emergency_vehicle_present']:
        dir_pulse = state_dict['emergency_direction']
        emergency_style = f"""
        .lane-{dir_pulse} {{
            box-shadow: 0 0 15px #f1c40f;
            animation: pulse 1s infinite;
        }}
        @keyframes pulse {{
            0% {{ opacity: 1; }}
            50% {{ opacity: 0.5; }}
            100% {{ opacity: 1; }}
        }}
        """

    html = f"""
    <style>
        .intersection-container {{
            background: #1a1a1a;
            padding: 20px;
            border-radius: 12px;
            font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
            text-align: center;
            color: white;
            min-height: 300px;
            display: flex;
            flex-direction: column;
            justify-content: center;
            align-items: center;
            {emergency_style}
        }}
        .road-vertical {{ width: 60px; height: 100px; background: #333; }}
        .road-horizontal {{ width: 100px; height: 60px; background: #333; }}
        .center-box {{
            width: 80px; height: 80px; 
            background: #222; 
            border: 2px solid #444;
            display: flex; justify-content: center; align-items: center;
            font-size: 24px;
        }}
        .lane-val {{ font-weight: bold; font-size: 1.2em; }}
        .signal-light {{
            width: 12px; height: 12px;
            border-radius: 50%;
            margin: 4px;
            display: inline-block;
        }}
    </style>
    <div class="intersection-container">
        <div class="lane-ns" style="color: {ns_color}">NORTH: {state_dict['north_queue']}</div>
        <div class="road-vertical" style="border-left: 2px dashed #555; border-right: 2px dashed #555;"></div>
        <div style="display: flex; align-items: center;">
            <div class="lane-ew" style="color: {ew_color}">WEST: {state_dict['west_queue']}</div>
            <div class="road-horizontal" style="border-top: 2px dashed #555; border-bottom: 2px dashed #555;"></div>
            <div class="center-box">
                <span style="color: {ns_color}">{'↕' if sig=='green_ns' else '·'}</span>
                <span style="color: {ew_color}">{'↔' if sig=='green_ew' else '·'}</span>
            </div>
            <div class="road-horizontal" style="border-top: 2px dashed #555; border-bottom: 2px dashed #555;"></div>
            <div class="lane-ew" style="color: {ew_color}">EAST: {state_dict['east_queue']}</div>
        </div>
        <div class="road-vertical" style="border-left: 2px dashed #555; border-right: 2px dashed #555;"></div>
        <div class="lane-ns" style="color: {ns_color}">SOUTH: {state_dict['south_queue']}</div>
        <div style="margin-top: 15px; font-size: 0.9em; opacity: 0.8;">
            {'⚠️ EMERGENCY: ' + state_dict['emergency_direction'].upper() if state_dict['emergency_vehicle_present'] else 'System Healthy'}
        </div>
    </div>
    """
    return html

def generate_signal_timeline(history):
    """Creates a horizontal timeline of the last 10 actions."""
    if not history: return "<div style='color: #666; font-style: italic;'>Waiting for first step...</div>"
    
    last_steps = history[-10:]
    timeline_html = "<div style='display: flex; gap: 8px; overflow-x: auto; padding-bottom: 10px;'>"
    
    for step in last_steps:
        sig = step['signal']
        color = "#2ecc71" if "green" in sig else "#e74c3c"
        label = "N-S" if sig == "green_ns" else ("E-W" if sig == "green_ew" else "RED")
        timeline_html += f"""
        <div style="background: #222; border: 1px solid #444; border-radius: 4px; padding: 4px 8px; text-align: center; min-width: 60px;">
            <div style="font-size: 0.7em; color: #888;">Step {step['step']}</div>
            <div style="color: {color}; font-weight: bold; font-size: 0.9em;">{label}</div>
        </div>
        """
    timeline_html += "</div>"
    return timeline_html

def update_plot(history):
    """Generates the queue fluctuation plot from history."""
    if not history: 
        fig, ax = plt.subplots(figsize=(10, 4))
        ax.set_title("Queue Distribution Over Time")
        ax.set_facecolor("#f9f9f9")
        return fig

    df = pd.DataFrame(history)
    fig, ax = plt.subplots(figsize=(10, 4), tight_layout=True)
    ax.plot(df['step'], df['ns_total'], label='North-South Queue', color='#3498db', linewidth=2)
    ax.plot(df['step'], df['ew_total'], label='East-West Queue', color='#e67e22', linewidth=2)
    
    ax.set_ylabel("Vehicle Count")
    ax.set_xlabel("Time Step")
    ax.legend(loc='upper right')
    ax.grid(True, alpha=0.2)
    ax.set_title("Real-Time Traffic Density")
    plt.close(fig)
    return fig

# --- AGENT LOGIC ---
heuristic_agent = DeterministicAgent()
llm_agent = None

def get_agent(use_llm):
    global llm_agent
    if use_llm:
        if llm_agent is None:
            try:
                llm_agent = LLMAgent()
            except: return heuristic_agent
        return llm_agent
    return heuristic_agent

# --- UI HANDLERS ---
ui_env = EasyTask()

def handle_step(history, n, s, e, w, emg, emg_dir, sig, use_llm):
    # Setup state
    ui_env.env.north, ui_env.env.south = n, s
    ui_env.env.east, ui_env.env.west = e, w
    ui_env.env.emergency_present, ui_env.env.emergency_direction_str = emg, emg_dir
    ui_env.env.current_signal = sig

    agent = get_agent(use_llm)
    current_state = ui_env.state()
    action, explanation = agent.get_action_with_explanation(current_state)
    
    result = ui_env.step(action)
    new_state = result.state.to_dict()
    
    # Update History
    history = history or []
    history.append({
        "step": len(history) + 1,
        "ns_total": new_state['north_queue'] + new_state['south_queue'],
        "ew_total": new_state['east_queue'] + new_state['west_queue'],
        "signal": new_state['current_signal'],
        "reward": result.reward
    })

    # Calc Efficiency (Clamped 0.0-1.0)
    avg_wait = result.info['avg_waiting_time']
    max_wait = 30.0
    efficiency = max(0.0, min(1.0, 1.0 - (avg_wait / max_wait)))

    return (
        history,
        new_state['north_queue'], new_state['south_queue'],
        new_state['east_queue'], new_state['west_queue'],
        new_state['current_signal'],
        generate_intersection_html(new_state),
        update_plot(history),
        round(efficiency, 3),
        explanation,
        generate_signal_timeline(history)
    )

def handle_batch(history):
    # Runs 50 steps from scratch
    ui_env.reset()
    batch_history = []
    for i in range(50):
        state = ui_env.state()
        action = heuristic_agent.get_action(state)
        res = ui_env.step(action)
        batch_history.append({
            "step": i + 1,
            "ns_total": state.north_queue + state.south_queue,
            "ew_total": state.east_queue + state.west_queue,
            "signal": state.current_signal,
            "reward": res.reward
        })
    score = ui_env.evaluate()
    return batch_history, update_plot(batch_history), round(score, 3), "Full Simulation Run Complete.", generate_signal_timeline(batch_history)

def handle_reset():
    return [], 10, 10, 5, 5, False, "none", "red", generate_intersection_html({'north_queue': 0, 'south_queue': 0, 'east_queue': 0, 'west_queue': 0, 'current_signal': 'red', 'emergency_vehicle_present': False}), update_plot([]), 0.0, "System Reset.", generate_signal_timeline([])

# --- UI BUILDER ---
def create_ui():
    with gr.Blocks(title="Smart Traffic AI Suite", theme=gr.themes.Soft()) as interface:
        history_state = gr.State([])
        
        with gr.Row():
            gr.Markdown("# 🚦 Smart Traffic Control Simulator")
            engine_status = gr.HTML("<div style='background: #333; padding: 5px 15px; border-radius: 20px; color: #2ecc71; border: 1px solid #2ecc71;'>⚙️ Heuristic Engine Active</div>")

        with gr.Row():
            with gr.Column(scale=1):
                gr.Markdown("### 🎛️ Control Panel")
                with gr.Group():
                    n_in = gr.Slider(0, 50, value=10, label="North Queue")
                    s_in = gr.Slider(0, 50, value=10, label="South Queue")
                    e_in = gr.Slider(0, 50, value=5, label="East Queue")
                    w_in = gr.Slider(0, 50, value=5, label="West Queue")
                
                with gr.Row():
                    emg_in = gr.Checkbox(label="🚨 Emergency", value=False)
                    emg_dir_in = gr.Dropdown(["ns", "ew"], value="ns", label="Direction")
                
                sig_in = gr.Dropdown(["red", "green_ns", "green_ew"], value="red", label="Active Signal")
                use_llm = gr.Checkbox(label="🧠 Use AI Decision Engine (LLM)", value=False)
                
                with gr.Row():
                    step_btn = gr.Button("⏩ Next Step", variant="primary")
                
                gr.Markdown("### 📚 Demo Presets")
                with gr.Row():
                    rush_btn = gr.Button("🏢 Rush Hour")
                    emg_btn = gr.Button("🚑 Emergency")
                
                reset_btn = gr.Button("🔄 Clear All History", variant="stop")

            with gr.Column(scale=2):
                with gr.Row():
                    with gr.Column(scale=1):
                        gr.Markdown("### 🏰 Visual Intersection")
                        viz_output = gr.HTML(generate_intersection_html({'north_queue': 10, 'south_queue': 10, 'east_queue': 5, 'west_queue': 5, 'current_signal': 'red', 'emergency_vehicle_present': False}))
                    with gr.Column(scale=1):
                        gr.Markdown("### 📊 Live Metrics")
                        efficiency_val = gr.Label(label="Efficiency Score", value=0.0)
                        explanation_out = gr.Textbox(label="Decision Rationale", lines=4)
                
                gr.Markdown("### 🕒 Signal History")
                timeline_out = gr.HTML(generate_signal_timeline([]))
                
                gr.Markdown("### 📈 Optimization Analytics")
                plot_out = gr.Plot(update_plot([]))
                batch_btn = gr.Button("🏃 Run Full Episode Simulation")

        # LLM Switch UI Update
        use_llm.change(
            fn=lambda x: f"<div style='background: #333; padding: 5px 15px; border-radius: 20px; color: {'#3498db' if x else '#2ecc71'}; border: 1px solid {'#3498db' if x else '#2ecc71'};'>{'🧠 Neural AI Active' if x else '⚙️ Heuristic Engine Active'}</div>",
            inputs=[use_llm],
            outputs=[engine_status]
        )

        # Event Mappings
        step_btn.click(
            fn=handle_step,
            inputs=[history_state, n_in, s_in, e_in, w_in, emg_in, emg_dir_in, sig_in, use_llm],
            outputs=[history_state, n_in, s_in, e_in, w_in, sig_in, viz_output, plot_out, efficiency_val, explanation_out, timeline_out]
        )

        batch_btn.click(
            fn=handle_batch,
            inputs=[history_state],
            outputs=[history_state, plot_out, efficiency_val, explanation_out, timeline_out]
        )

        reset_btn.click(
            fn=handle_reset,
            outputs=[history_state, n_in, s_in, e_in, w_in, emg_in, emg_dir_in, sig_in, viz_output, plot_out, efficiency_val, explanation_out, timeline_out]
        )

        # Demo Presets
        rush_btn.click(fn=lambda: (45, 42, 8, 12, False), outputs=[n_in, s_in, e_in, w_in, emg_in])
        emg_btn.click(fn=lambda: (5, 5, 20, 18, True, "ew"), outputs=[n_in, s_in, e_in, w_in, emg_in, emg_dir_in])

    return interface

dashboard = create_ui()
app = gr.mount_gradio_app(app, dashboard, path="/") # UI at root

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 7860))
    uvicorn.run(app, host="0.0.0.0", port=port)
