import gradio as gr
import os
import random

# 1. CORE SIMULATION LOGIC (Lazy loading)
def run_simulation(manual_seed=None):
    """
    Handles simulation and visualization in a single isolated environment.
    Prevents heavy imports at startup to avoid HF build/runtime jams.
    """
    try:
        # Determine Seed
        seed = int(manual_seed) if manual_seed and str(manual_seed).isdigit() else random.randint(1000, 99999)
        
        # Localized Imports (Absolute isolation)
        import io
        from contextlib import redirect_stdout
        from evaluate import run_evaluation
        from visualize import generate_graph
        
        # Execution with capture
        f = io.StringIO()
        with redirect_stdout(f):
            scores = run_evaluation(base_seed=seed, silent=False)
        logs = f.getvalue()
        
        # Persistence
        graph_path = "optimization_results.png"
        generate_graph(scores, seed, output_path=graph_path)
        
        return logs, graph_path
    except Exception as e:
        return f"CRITICAL RUNTIME ERROR: {str(e)}", None

# 2. UI CONSTRUCTOR (Strict encapsulation)
def create_interface():
    with gr.Blocks() as interface:
        gr.Markdown("# 🚦 Smart Traffic Optimization Environment (OpenEnv)")
        gr.Markdown("Production-grade AI Traffic Simulator built on OpenEnv principles.")
        
        with gr.Row():
            with gr.Column(scale=1): pass
            with gr.Column(scale=2, min_width=320):
                seed_input = gr.Textbox(label="Fix Randomness (Optional Seed)", placeholder="e.g. 42")
                run_btn = gr.Button("🚀 Start Evaluation", variant="primary", size="lg")
                gr.HTML("<p style='text-align: center; color: #666; font-size: 0.85em; margin-top: 10px;'>System handles real-time emergency prioritization automatically.</p>")
            with gr.Column(scale=1): pass

        with gr.Row():
            with gr.Column(scale=1):
                gr.Markdown("<h3 style='text-align: center;'>Processing Logs</h3>")
                output_text = gr.Textbox(show_label=False, lines=20, interactive=False)
            with gr.Column(scale=1):
                gr.Markdown("<h3 style='text-align: center;'>Optimization Delta</h3>")
                output_img = gr.Image(show_label=False, type="filepath")

        # Event Mapping
        run_btn.click(
            fn=run_simulation, 
            inputs=[seed_input], 
            outputs=[output_text, output_img]
        )
        
        return interface

# 3. PRODUCTION ENTRY POINT (Differentiates from local import)
if __name__ == "__main__":
    # Ensure no blocking calls reach global scope
    demo = create_interface()
    
    # Configure production binding for Hugging Face Spaces ingress router
    port = int(os.environ.get("PORT", 7860))
    
    # Theme belongs in launch() as of Gradio 6.0
    demo.queue().launch(
        server_name="0.0.0.0", 
        server_port=port, 
        share=False,
        theme="soft"
    )
