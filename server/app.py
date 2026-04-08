import os
import random
import time
from typing import Dict, Any, Optional, List
from fastapi import FastAPI, HTTPException, Body, Request
from fastapi.middleware.cors import CORSMiddleware
import uvicorn
import io

try:
    import gradio as gr
    HAS_GRADIO = True
except ImportError:
    HAS_GRADIO = False

try:
    import pandas as pd
    import matplotlib.pyplot as plt
    HAS_PLOTTING = True
except ImportError:
    HAS_PLOTTING = False
    pd = None
    plt = None


from src.tasks import EasyTask, MediumTask, HardTask, to_open_unit_interval
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

def sanitize_score_payload(obj):
    if isinstance(obj, dict):
        out = {}
        for k, v in obj.items():
            key = str(k).lower()
            if key.startswith("raw_"):
                out[k] = v
            elif any(token in key for token in ["score", "reward", "grade", "metric", "efficiency", "overall"]):
                if isinstance(v, (int, float)) or v is None:
                    out[k] = to_open_unit_interval(v)
                else:
                    out[k] = sanitize_score_payload(v)
            else:
                out[k] = sanitize_score_payload(v)
        return out
    elif isinstance(obj, list):
        return [sanitize_score_payload(x) for x in obj]
    else:
        return obj

# API Helper
async def reset_logic(level: str = "easy", seed: Optional[int] = None):
    global active_task
    lvl = (level or "easy").lower()

    if lvl == "easy":
        active_task = EasyTask()
    elif lvl == "medium":
        active_task = MediumTask()
    elif lvl == "hard":
        active_task = HardTask()
    else:
        raise HTTPException(status_code=400, detail=f"Unknown level: {level}")

    active_task.reset(seed=seed)
    return active_task.state().to_dict()

async def step_logic(action: int):
    result = active_task.step(action)
    safe_info = sanitize_score_payload(result.info)

    return {
        "state": result.state.to_dict(),
        "reward": to_open_unit_interval(result.reward),
        "raw_reward": result.reward,
        "done": result.done,
        "info": safe_info,
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

if not HAS_GRADIO:
    @app.get("/")
    async def root():
        return {
            "status": "active",
            "service": "Smart Traffic OpenEnv API",
            "ui_enabled": False,
            "message": "API endpoints are active at /api/reset and /api/step. UI is disabled."
        }



# --- UI VISUALIZATION HELPERS ---

def generate_intersection_html(state_dict):
    """Creates a professional visual intersection with flow indicators."""
    sig = state_dict['current_signal']
    ns_active = sig == "green_ns"
    ew_active = sig == "green_ew"
    
    ns_color = "#2ecc71" if ns_active else "#e74c3c"
    ew_color = "#2ecc71" if ew_active else "#e74c3c"
    
    # Opacity and highlighting
    ns_style = "opacity: 1;" if ns_active else "opacity: 0.4;"
    ew_style = "opacity: 1;" if ew_active else "opacity: 0.4;"
    
    # Directional Arrows
    ns_arrow = "↑" if ns_active else "·"
    ew_arrow = "↔" if ew_active else "·"

    # Pulse animation for emergency
    emergency_css = ""
    if state_dict['emergency_vehicle_present']:
        dir_pulse = state_dict['emergency_direction']
        emergency_css = f"""
        .lane-{dir_pulse} {{
            box-shadow: 0 0 20px #f1c40f;
            background: rgba(241, 196, 15, 0.1) !important;
            animation: pulse 1s infinite alternate;
        }}
        @keyframes pulse {{
            from {{ opacity: 1; }} to {{ opacity: 0.6; }}
        }}
        """

    html = f"""
    <style>
        .intersection-container {{
            background: #111;
            padding: 25px;
            border-radius: 16px;
            font-family: 'Inter', sans-serif;
            text-align: center;
            color: white;
            min-height: 350px;
            display: flex; flex-direction: column; justify-content: center; align-items: center;
            border: 1px solid #333;
            box-shadow: 0 10px 30px rgba(0,0,0,0.5);
            {emergency_css}
        }}
        .road-vertical {{ width: 70px; height: 110px; background: #222; position: relative; }}
        .road-horizontal {{ width: 110px; height: 70px; background: #222; position: relative; }}
        .center-box {{
            width: 90px; height: 90px; 
            background: #1a1a1a; 
            border: 1px solid #444;
            display: flex; flex-direction: column; justify-content: center; align-items: center;
            font-size: 32px;
            border-radius: 8px;
        }}
        .lane-val {{ font-weight: 800; letter-spacing: -0.5px; transition: all 0.3s; }}
        .flow-indicator {{ font-size: 24px; font-weight: bold; margin: -5px 0; }}
    </style>
    <div class="intersection-container">
        <div class="lane-ns" style="color: {ns_color}; {ns_style}">
            <div style="font-size: 0.7em;">NORTH</div>
            <div class="lane-val">{state_dict['north_queue']}</div>
        </div>
        <div class="road-vertical" style="border-left: 2px dashed #444; border-right: 2px dashed #444;">
             <div style="position: absolute; top: 0; left: 50%; transform: translateX(-50%); color: {ns_color}; font-size: 20px;">{ns_arrow}</div>
        </div>
        <div style="display: flex; align-items: center; gap: 5px;">
            <div class="lane-ew" style="color: {ew_color}; {ew_style}">
                <div style="font-size: 0.7em;">WEST</div>
                <div class="lane-val">{state_dict['west_queue']}</div>
            </div>
            <div class="road-horizontal" style="border-top: 2px dashed #444; border-bottom: 2px dashed #444;">
                 <div style="position: absolute; left: 0; top: 50%; transform: translateY(-50%); color: {ew_color}; font-size: 20px;">{ew_arrow}</div>
            </div>
            <div class="center-box">
                <div class="flow-indicator" style="color: {ns_color}; display: {'block' if ns_active else 'none'}">↑</div>
                <div class="flow-indicator" style="color: {ew_color}; display: {'block' if ew_active else 'none'}">↔</div>
                <div style="font-size: 10px; color: #666; margin-top: 5px;">FLOW</div>
            </div>
            <div class="road-horizontal" style="border-top: 2px dashed #444; border-bottom: 2px dashed #444;">
                 <div style="position: absolute; right: 0; top: 50%; transform: translateY(-50%); color: {ew_color}; font-size: 20px;">{ew_arrow}</div>
            </div>
            <div class="lane-ew" style="color: {ew_color}; {ew_style}">
                <div style="font-size: 0.7em;">EAST</div>
                <div class="lane-val">{state_dict['east_queue']}</div>
            </div>
        </div>
        <div class="road-vertical" style="border-left: 2px dashed #444; border-right: 2px dashed #444;">
             <div style="position: absolute; bottom: 0; left: 50%; transform: translateX(-50%); color: {ns_color}; font-size: 20px;">{ns_arrow}</div>
        </div>
        <div class="lane-ns" style="color: {ns_color}; {ns_style}">
            <div class="lane-val">{state_dict['south_queue']}</div>
            <div style="font-size: 0.7em;">SOUTH</div>
        </div>
        <div style="margin-top: 20px; font-size: 0.8em; padding: 5px 15px; border-radius: 20px; background: #222; border: 1px solid #333;">
            {'⚠️ PRIORITY: ' + state_dict['emergency_direction'].upper() if state_dict['emergency_vehicle_present'] else 'NORMAL OPERATION'}
        </div>
    </div>
    """
    return html

def generate_imbalance_meter(score):
    """Creates a visual balance meter (0-100%)."""
    color = "#2ecc71" if score < 30 else ("#f1c40f" if score < 60 else "#e74c3c")
    return f"""
    <div style="background: #222; border-radius: 10px; height: 8px; width: 100%; margin-top: 5px; overflow: hidden; border: 1px solid #333;">
        <div style="background: {color}; height: 100%; width: {score}%; transition: width 0.5s;"></div>
    </div>
    <div style="font-size: 10px; color: #888; text-align: right; margin-top: 2px;">{score}% Imbalance</div>
    """

def generate_signal_timeline(history):
    """Creates a clean signal timeline."""
    if not history: return "<div style='color: #666; font-size: 12px;'>Wait for steps...</div>"
    last_steps = history[-8:]
    html = "<div style='display: flex; gap: 6px; overflow-x: auto; padding-bottom: 5px;'>"
    for step in last_steps:
        color = "#2ecc71" if "green" in step['signal'] else "#e74c3c"
        label = "N-S" if step['signal']=="green_ns" else ("E-W" if step['signal']=="green_ew" else "RD")
        html += f"""
        <div style="background: #1a1a1a; border: 1px solid #333; border-radius: 6px; padding: 6px; min-width: 50px; text-align: center;">
            <div style="font-size: 9px; color: #555;">#{step['step']}</div>
            <div style="color: {color}; font-weight: 800; font-size: 11px;">{label}</div>
        </div>
        """
    html += "</div>"
    return html

def update_plot(history):
    """Professional stylized Matplotlib plot."""
    if not HAS_PLOTTING:
        # Return a blank figure or similar placeholder if possible, 
        # but in a headless environment this should just be bypassed.
        return None

    plt.close('all')
    fig, ax = plt.subplots(figsize=(10, 3.5), tight_layout=True)
    fig.patch.set_facecolor('#0f0f0f')
    ax.set_facecolor('#1a1a1a')
    
    if not history:
        ax.text(0.5, 0.5, "Awaiting Data Stream", ha='center', va='center', color='#444', fontsize=12)
        ax.set_title("Real-Time Infrastructure Load", color='#666', pad=15)
        ax.axis('off')
        return fig

    df = pd.DataFrame(history)
    ax.plot(df['step'], df['ns_total'], label='N-S Corridor', color='#3498db', linewidth=2.5, alpha=0.9)
    ax.plot(df['step'], df['ew_total'], label='E-W Corridor', color='#e67e22', linewidth=2.5, alpha=0.9)
    
    # Styling
    ax.set_title("Infrastructure Load Analysis", color='#fff', loc='left', pad=15, fontweight='bold')
    ax.set_xlabel("Operational Steps", color='#888', fontsize=9)
    ax.set_ylabel("Vehicle Load", color='#888', fontsize=9)
    ax.tick_params(colors='#666', labelsize=8)
    ax.set_ylim(0, 1.15)
    ax.grid(axis='y', linestyle='--', alpha=0.1)
    
    ax.legend(facecolor='#1a1a1a', edgecolor='#333', labelcolor='#ccc', fontsize=8, loc='upper right')
    plt.close(fig)
    return fig


# --- AGENT LOGIC ---
heuristic_agent = DeterministicAgent()
llm_agent = None

def get_agent(use_llm):
    global llm_agent
    if use_llm:
        if llm_agent is None:
            try: llm_agent = LLMAgent()
            except: return heuristic_agent
        return llm_agent
    return heuristic_agent

# --- UI HANDLERS ---
ui_env = EasyTask()

def handle_step(history, n, s, e, w, emg, emg_dir, sig, use_llm):
    # Sync current UI values to environment before step
    ui_env.env.north, ui_env.env.south = n, s
    ui_env.env.east, ui_env.env.west = e, w
    ui_env.env.emergency_present, ui_env.env.emergency_direction_str = emg, emg_dir
    ui_env.env.current_signal = sig

    agent = get_agent(use_llm)
    current_state = ui_env.state()
    action, raw_explanation = agent.get_action_with_explanation(current_state)
    
    result = ui_env.step(action)
    new_state = result.state.to_dict()
    
    # Update History
    history = history or []
    history.append({
        "step": len(history) + 1,
        "ns_total": new_state['north_queue'] + new_state['south_queue'],
        "ew_total": new_state['east_queue'] + new_state['west_queue'],
        "signal": new_state['current_signal']
    })

    # Calculations
    q_ns = new_state['north_queue'] + new_state['south_queue']
    q_ew = new_state['east_queue'] + new_state['west_queue']
    total_q = q_ns + q_ew
    max_cap = ui_env.env.queue_cap * 4
    
    # 1. Queue Pressure (30%) - Higher queue = lower efficiency
    pressure_factor = 1.0 - (total_q / max_cap)
    
    # 2. Wait Factor (30%) - Higher avg delay = lower efficiency
    avg_delay = result.info['avg_waiting_time']
    wait_factor = max(0.0, min(1.0, 1.0 - (avg_delay / 30.0)))
    
    # 3. Throughput Factor (40%) - Vehicles cleared this step vs capacity
    cleared_this_step = result.info.get('cleared_this_step', 0)
    max_clearance = 16.0 
    throughput_factor = min(1.0, cleared_this_step / max_clearance)
    
    # Composite Efficiency (Clamped 0.0 - 1.0)
    efficiency = (0.3 * pressure_factor) + (0.3 * wait_factor) + (0.4 * throughput_factor)
    efficiency = to_open_unit_interval(efficiency)
    
    imbalance = min(100, int((abs(q_ns - q_ew) / max(1, q_ns + q_ew)) * 100))
    congestion = min(100, int((total_q / max_cap) * 100))

    # Smart Rationale
    status = f"🟢 Flowing North-South" if action==1 else (f"🟢 Flowing East-West" if action==2 else "🔴 All Stop")
    if emg: status = f"🚑 EMERGENCY PRIORITY ({emg_dir.upper()})"
    
    smart_rationale = f"Intelligence selected {status}. "
    if action == 1 and q_ns > q_ew:
        smart_rationale += f"Prioritized N-S corridor to alleviate {q_ns}-vehicle congestion. "
    elif action == 2 and q_ew > q_ns:
        smart_rationale += f"Prioritized E-W corridor to alleviate {q_ew}-vehicle congestion. "
    
    if emg and (action == (1 if emg_dir=='ns' else 2)):
        smart_rationale += "Emergency vehicle path successfully cleared. "
    
    if len(history) > 1:
        prev_q = history[-2]['ns_total'] + history[-2]['ew_total']
        curr_q = q_ns + q_ew
        if curr_q < prev_q:
            smart_rationale += f"Optimized throughput results in a net clearance of {prev_q - curr_q} vehicles."

    return (
        history,
        new_state['north_queue'], new_state['south_queue'], new_state['east_queue'], new_state['west_queue'],
        new_state['current_signal'],
        generate_intersection_html(new_state),
        update_plot(history),
        to_open_unit_interval(efficiency),
        smart_rationale,
        generate_signal_timeline(history),
        result.info['total_cleared'],
        f"{avg_delay:.1f}s",
        status,
        f"{congestion}%",
        generate_imbalance_meter(imbalance)
    )

def handle_batch(history):
    ui_env.reset()
    batch_history = []
    for i in range(50):
        state = ui_env.state()
        action = heuristic_agent.get_action(state)
        res = ui_env.step(action)
        batch_history.append({
            "step": i + 1, "ns_total": state.north_queue + state.south_queue,
            "ew_total": state.east_queue + state.west_queue, "signal": state.current_signal
        })
    score = ui_env.evaluate()
    avg_d = res.info['avg_waiting_time']
    return batch_history, update_plot(batch_history), to_open_unit_interval(score), "Full Optimization Episode Completed Successfully. Performance verified.", generate_signal_timeline(batch_history), res.info['total_cleared'], f"{avg_d:.1f}s", "EPISODE COMPLETE", "VARIES", generate_imbalance_meter(50)

def handle_reset():
    return [], 10, 10, 5, 5, False, "ns", "red", generate_intersection_html({'north_queue': 0, 'south_queue': 0, 'east_queue': 0, 'west_queue': 0, 'current_signal': 'red', 'emergency_vehicle_present': False}), update_plot([]), to_open_unit_interval(1.0), "Ready.", generate_signal_timeline([]), 0, "0.0s", "IDLE", "0%", generate_imbalance_meter(0)

# --- UI BUILDER ---
def create_ui():
    with gr.Blocks(title="SmartTraffic AI Suite", theme=gr.themes.Soft()) as interface:
        history_state = gr.State([])
        
        with gr.Row(variant="compact"):
            gr.Markdown("# 🚦 Smart Traffic Optimizer")
            engine_status = gr.HTML("<div style='background: #222; padding: 5px 15px; border-radius: 20px; color: #2ecc71; border: 1px solid #2ecc71; font-weight: bold; font-size: 12px;'>HEURISTIC ENGINE</div>")

        with gr.Row():
            with gr.Column(scale=2):
                with gr.Group():
                    gr.Markdown("### 📡 Corridor Configuration", container=False)
                    with gr.Row():
                        n_in = gr.Slider(0, 50, value=10, label="North")
                        s_in = gr.Slider(0, 50, value=10, label="South")
                    with gr.Row():
                        e_in = gr.Slider(0, 50, value=5, label="East")
                        w_in = gr.Slider(0, 50, value=5, label="West")
                
                with gr.Row():
                    emg_in = gr.Checkbox(label="🚨 Emergency Active")
                    emg_dir_in = gr.Dropdown(["ns", "ew"], value="ns", label="Direction", scale=1)
                
                sig_in = gr.Dropdown(["red", "green_ns", "green_ew"], value="red", label="Manual Signal Fix", info="Overrides current flow")
                use_llm = gr.Checkbox(label="🧠 Deploy Neural AI Engine", value=False)
                
                with gr.Row():
                    step_btn = gr.Button("⏩ Process Next Frame", variant="primary")
                    batch_btn = gr.Button("🏃 Run Full Episode", variant="secondary")

                gr.Markdown("### 🏢 Preset Scenarios")
                with gr.Row():
                    rush_btn = gr.Button("Rush Hour", size="sm")
                    emg_preset_btn = gr.Button("Ambulance", size="sm")
                    night_btn = gr.Button("Late Night", size="sm")
                
                with gr.Row():
                    reset_btn = gr.Button("🔄 Reset Simulation & History", size="sm", variant="stop")

            with gr.Column(scale=3):
                with gr.Row():
                    with gr.Column(scale=4):
                        gr.Markdown("### 🏰 Infrastructure View")
                        viz_output = gr.HTML(generate_intersection_html({'north_queue': 0, 'south_queue': 0, 'east_queue': 0, 'west_queue': 0, 'current_signal': 'red', 'emergency_vehicle_present': False}))
                    with gr.Column(scale=3):
                        gr.Markdown("### 📊 Metrics Engine")
                        with gr.Group():
                            sys_status_out = gr.Textbox(label="System Status", value="IDLE", interactive=False)
                            efficiency_val = gr.Label(label="Efficiency Index", value=0.0)
                            
                        with gr.Row():
                            cleared_out = gr.Number(label="Vehicles Cleared", value=0)
                            wait_out = gr.Textbox(label="Avg Delay", value="0.0s")
                        
                        with gr.Row():
                            congestion_out = gr.Textbox(label="Congestion Index", value="0%")
                            imbalance_meter = gr.HTML(generate_imbalance_meter(0))

                gr.Markdown("### 💬 Decision Rationale")
                explanation_out = gr.Textbox(label="", value="System Ready.", lines=2, interactive=False)
                
                gr.Markdown("### 🕒 Flow History")
                timeline_out = gr.HTML(generate_signal_timeline([]))
                
                gr.Markdown("### 📉 Analysis & Forecasting")
                plot_out = gr.Plot(update_plot([]))

        # UI Styling Updates
        use_llm.change(
            fn=lambda x: f"<div style='background: #222; padding: 5px 15px; border-radius: 20px; color: {'#3498db' if x else '#2ecc71'}; border: 1px solid {'#3498db' if x else '#2ecc71'}; font-weight: bold; font-size: 12px;'>{'🧠 NEURAL AI ACTIVE' if x else '⚙️ HEURISTIC ENGINE'}</div>",
            inputs=[use_llm], outputs=[engine_status]
        )

        # Event Mappings
        step_btn.click(
            fn=handle_step,
            inputs=[history_state, n_in, s_in, e_in, w_in, emg_in, emg_dir_in, sig_in, use_llm],
            outputs=[history_state, n_in, s_in, e_in, w_in, sig_in, viz_output, plot_out, efficiency_val, explanation_out, timeline_out, cleared_out, wait_out, sys_status_out, congestion_out, imbalance_meter]
        )

        batch_btn.click(
            fn=handle_batch, inputs=[history_state],
            outputs=[history_state, plot_out, efficiency_val, explanation_out, timeline_out, cleared_out, wait_out, sys_status_out, congestion_out, imbalance_meter]
        )

        reset_btn.click(
            fn=handle_reset, 
            outputs=[history_state, n_in, s_in, e_in, w_in, emg_in, emg_dir_in, sig_in, viz_output, plot_out, efficiency_val, explanation_out, timeline_out, cleared_out, wait_out, sys_status_out, congestion_out, imbalance_meter]
        )

        # Presets
        rush_btn.click(fn=lambda: (42, 38, 8, 12, False, "ns"), outputs=[n_in, s_in, e_in, w_in, emg_in, emg_dir_in])
        emg_preset_btn.click(fn=lambda: (5, 8, 25, 22, True, "ew"), outputs=[n_in, s_in, e_in, w_in, emg_in, emg_dir_in])
        night_btn.click(fn=lambda: (2, 1, 1, 3, False, "ns"), outputs=[n_in, s_in, e_in, w_in, emg_in, emg_dir_in])

    return interface

if HAS_GRADIO:
    dashboard = create_ui()
    app = gr.mount_gradio_app(app, dashboard, path="/")
else:
    print("Warning: Gradio not found. Dashboard UI is disabled. API endpoints are still active.")


def main():
    port = int(os.environ.get("PORT", 7860))
    uvicorn.run(app, host="0.0.0.0", port=port)

if __name__ == "__main__":
    main()
