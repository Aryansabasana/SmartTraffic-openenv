import gradio as gr
import os
import random

# MOVED HEAVY IMPORTS INSIDE THE FUNCTION
# This guarantees app.py parses in milliseconds and the UI framework boots instantly.
def run_simulation(manual_seed=None):
    try:
        # 1. Determine Seed
        seed = int(manual_seed) if manual_seed and str(manual_seed).isdigit() else random.randint(1000, 99999)
        
        # 2. Localized Imports (Prevents blocking server UI startup!)
        # This stops matplotlib or other modules from hanging the global thread before launch
        import io
        from contextlib import redirect_stdout
        from evaluate import run_evaluation
        from visualize import generate_graph
        
        # 3. Execution
        f = io.StringIO()
        with redirect_stdout(f):
            scores = run_evaluation(base_seed=seed, silent=False)
        logs = f.getvalue()
        
        # 4. Visualization
        graph_path = "optimization_results.png"
        generate_graph(scores, seed, output_path=graph_path)
        
        return logs, graph_path
    except Exception as e:
        return f"CRITICAL ERROR during simulation: {str(e)}", None

# Build the UI Structure
with gr.Blocks() as interface:
    gr.Markdown("# 🚦 Smart Traffic Optimization Environment (OpenEnv)")
    gr.Markdown("Welcome to the Traffic Simulator. Watch how our AI improves traffic flow in real time.")
    
    with gr.Row():
        with gr.Column(scale=1): pass
        with gr.Column(scale=2, min_width=320):
            seed_input = gr.Textbox(label="Optional Seed (Empty for Random)", placeholder="e.g. 42")
            run_btn = gr.Button("🚀 Run Traffic Evaluator", variant="primary", size="lg")
            gr.HTML("<p style='text-align: center; color: gray; font-size: 0.9em; margin-bottom: 15px;'>Click to simulate AI-driven traffic optimization.</p>")
        with gr.Column(scale=1): pass

    gr.Markdown("""
    <div style='background-color: #ffeaea; border-left: 4px solid #ff4d4f; padding: 12px; margin: 15px 0px; border-radius: 4px;'>
        <span style='color: #a8071a; font-weight: 600;'>🚨 Active Protocol:</span> 
        <span style='color: #434343;'>System prioritizes emergency vehicles in real-time.</span>
    </div>
    """)
        
    with gr.Row():
        with gr.Column(scale=1):
            gr.Markdown("<h3 style='text-align: center;'>Simulation Logs</h3>")
            output_text = gr.Textbox(show_label=False, lines=22, interactive=False)
            
        with gr.Column(scale=1):
            gr.Markdown("<h3 style='text-align: center;'>Performance Improvement</h3>")
            output_img = gr.Image(show_label=False, type="filepath")

    # The lack of parentheses prevents execution at startup.
    run_btn.click(
        fn=run_simulation, 
        inputs=[seed_input], 
        outputs=[output_text, output_img]
    )

if __name__ == "__main__":
    # .queue() is mandatory for long-running functions on HF Spaces
    interface.queue().launch(server_name="0.0.0.0", server_port=7860, theme=gr.themes.Soft())
