import gradio as gr
import os
import random
from evaluate import run_evaluation
from visualize import generate_graph

def run_simulation(manual_seed=None):
    try:
        seed = int(manual_seed) if manual_seed and str(manual_seed).isdigit() else random.randint(1000, 99999)
        
        import io
        from contextlib import redirect_stdout
        
        f = io.StringIO()
        with redirect_stdout(f):
            scores = run_evaluation(base_seed=seed, silent=False)
        logs = f.getvalue()
        
        graph_path = "optimization_results.png"
        generate_graph(scores, seed, output_path=graph_path)
        
        return logs, graph_path
    except Exception as e:
        return f"Error running simulation: {str(e)}", None

with gr.Blocks(theme=gr.themes.Soft()) as interface:
    gr.Markdown("# 🚦 Smart Traffic Optimization Environment (OpenEnv)")
    gr.Markdown("Welcome to the Interactive Traffic Simulator. Watch as our heuristic AI resolves catastrophic urban gridlock.")
    
    with gr.Row():
        with gr.Column(scale=1):
            pass
        with gr.Column(scale=2, min_width=320):
            seed_input = gr.Textbox(label="Optional Seed (Empty for Random)", placeholder="e.g. 42")
            run_btn = gr.Button("🚀 Run Traffic Evaluator", variant="primary", size="lg")
            gr.HTML("""
                <p style='
                    text-align: center;
                    color: rgba(100, 100, 100, 0.75);
                    font-size: 0.88em;
                    margin: 4px 0 16px 0;
                    font-weight: 400;
                    letter-spacing: 0.01em;
                '>Click to simulate AI-driven traffic optimization across difficulty levels.</p>
            """)
        with gr.Column(scale=1):
            pass

    gr.Markdown("""
    <div style='background-color: #ffeaea; border-left: 4px solid #ff4d4f; padding: 12px; margin: 15px 0px; border-radius: 4px; box-shadow: 0 1px 3px rgba(0,0,0,0.1);'>
        <span style='color: #a8071a; font-weight: 600; font-size: 1.05em;'>🚨 Active Protocol:</span> 
        <span style='color: #434343;'>System prioritizes emergency vehicles in real-time.</span>
    </div>
    """)
        
    with gr.Row():
        with gr.Column(scale=1):
            gr.Markdown("<h3 style='text-align: center; color: #555; margin-bottom: 2px;'>Simulation Logs</h3>")
            output_text = gr.Textbox(show_label=False, lines=22, interactive=False)
            
        with gr.Column(scale=1):
            gr.Markdown("<h3 style='text-align: center; color: #222; margin-bottom: 10px; border-bottom: 1px solid #eaeaea; padding-bottom: 5px;'>Performance Improvement After Optimization</h3>")
            output_img = gr.Image(show_label=False, type="filepath")

    run_btn.click(fn=run_simulation, inputs=[seed_input], outputs=[output_text, output_img])

if __name__ == "__main__":
    interface.launch(server_name="0.0.0.0", server_port=7860)
