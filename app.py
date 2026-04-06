import os
import random
from typing import Dict, Any, Optional
from fastapi import FastAPI, HTTPException, Body
from fastapi.middleware.cors import CORSMiddleware
import gradio as gr
import uvicorn

from src.tasks import BaseTask, EasyTask, MediumTask, HardTask
from src.models import State, StepResult

# --- API CORE ---
app = FastAPI(title="Smart Traffic OpenEnv API")

# Global state to track the active task for the API
# In a real multi-user scenario, we'd use session IDs, but OpenEnv eval is single-streamed.
active_task: Optional[BaseTask] = None

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.get("/")
async def health_check():
    return {"status": "healthy", "service": "Smart Traffic OpenEnv"}

@app.post("/reset")
async def reset(level: str = "easy", seed: Optional[int] = None):
    global active_task
    if level.lower() == "easy":
        active_task = EasyTask()
    elif level.lower() == "medium":
        active_task = MediumTask()
    elif level.lower() == "hard":
        active_task = HardTask()
    else:
        raise HTTPException(status_code=400, detail="Invalid level. Choose 'easy', 'medium', or 'hard'.")
    
    state = active_task.reset(seed=seed)
    return state.to_dict()

@app.post("/step")
async def step(action: int = Body(..., embed=True)):
    global active_task
    if not active_task:
        raise HTTPException(status_code=400, detail="No active task. Call /reset first.")
    
    result = active_task.step(action)
    return {
        "state": result.state.to_dict(),
        "reward": result.reward,
        "done": result.done,
        "info": result.info
    }

@app.get("/state")
async def get_state():
    global active_task
    if not active_task:
        raise HTTPException(status_code=400, detail="No active task.")
    
    return active_task.state().to_dict()

@app.get("/evaluate")
async def evaluate():
    global active_task
    if not active_task:
        raise HTTPException(status_code=400, detail="No active task.")
    
    score = active_task.evaluate()
    return {"score": score}

# --- DASHBOARD LOGIC (Legacy Support & Visualization) ---
def run_ui_simulation(manual_seed=None):
    try:
        seed = int(manual_seed) if manual_seed and str(manual_seed).isdigit() else random.randint(1000, 99999)
        import io
        from contextlib import redirect_stdout
        from evaluate import run_evaluation
        from visualize import generate_graph
        
        f = io.StringIO()
        with redirect_stdout(f):
            scores = run_evaluation(base_seed=seed, silent=False)
        logs = f.getvalue()
        
        graph_path = "optimization_results.png"
        generate_graph(scores, seed, output_path=graph_path)
        
        return logs, graph_path
    except Exception as e:
        return f"CRITICAL RUNTIME ERROR: {str(e)}", None

def create_interface():
    with gr.Blocks() as interface:
        gr.Markdown("# 🚦 Smart Traffic Optimization Dashboard")
        
        with gr.Row():
            with gr.Column():
                seed_input = gr.Textbox(label="Seed", placeholder="42")
                run_btn = gr.Button("🚀 Run Full Simulation", variant="primary")
            with gr.Column():
                gr.Markdown("### Service Status")
                gr.JSON(value={"api": "active", "openenv": "compliant"})

        with gr.Row():
            output_text = gr.Textbox(label="Simulation Logs", lines=10)
            output_img = gr.Image(label="Results Graph")

        run_btn.click(fn=run_ui_simulation, inputs=[seed_input], outputs=[output_text, output_img])
        return interface

# Mount Gradio Dashboard
dashboard = create_interface()
app = gr.mount_gradio_app(app, dashboard, path="/dashboard", app_kwargs={"theme": gr.themes.Soft()})

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 7860))
    uvicorn.run(app, host="0.0.0.0", port=port)
