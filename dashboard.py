

import streamlit as st
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import numpy as np
import random
import sys
import os

sys.path.insert(0, os.path.dirname(__file__))
from src.tasks import EasyTask, MediumTask, HardTask, to_open_unit_interval
from src.models import State



class RandomAgent:
    def get_action(self, state: State) -> int:
        return random.choice([1, 2])


class FixedTimingAgent:
    def __init__(self, cycle_length: int = 10):
        self.cycle_length = cycle_length

    def get_action(self, state: State) -> int:
        phase = (state.time_step // self.cycle_length) % 2
        return 1 if phase == 0 else 2


class OptimizedAgent:
    def __init__(self):
        self.last_switch_time = 0
        self.min_green_time = 3

    def get_action(self, state: State) -> int:
        ns_total = state.north_queue + state.south_queue
        ew_total = state.east_queue + state.west_queue
        current_idx = 1 if state.current_signal == "green_ns" else (
            2 if state.current_signal == "green_ew" else 0
        )
        time_since_switch = state.time_step - self.last_switch_time

        if state.emergency_vehicle_present and state.emergency_direction != 'none':
            em_idx = 1 if state.emergency_direction == 'ns' else 2
            if current_idx != em_idx:
                self.last_switch_time = state.time_step
            return em_idx

        if current_idx != 0 and time_since_switch < self.min_green_time:
            return current_idx

        ns_pressure = ns_total + (state.ns_growth * 1.5)
        ew_pressure = ew_total + (state.ew_growth * 1.5)
        if ns_total > 30: ns_pressure += 20
        if ew_total > 30: ew_pressure += 20

        threshold = 5.0
        if current_idx == 1:
            target_idx = 2 if ew_pressure > ns_pressure + threshold else 1
        elif current_idx == 2:
            target_idx = 1 if ns_pressure > ew_pressure + threshold else 2
        else:
            target_idx = 1 if ns_pressure >= ew_pressure else 2

        if target_idx != current_idx:
            self.last_switch_time = state.time_step
        return target_idx


AGENTS = {
    "Random Agent":        RandomAgent,
    "Fixed-Timing Agent":  FixedTimingAgent,
    "Optimized Agent":     OptimizedAgent,
}

TASK_CONSTRUCTORS = {
    "Easy":   EasyTask,
    "Medium": MediumTask,
    "Hard":   HardTask,
}




def run_agent(agent_type: str, difficulty: str, seed: int) -> dict:
    random.seed(seed)
    np.random.seed(seed)

    task   = TASK_CONSTRUCTORS[difficulty]()
    agent  = AGENTS[agent_type]()
    state  = task.reset(seed=seed)

    done         = False
    total_reward = 0.0
    steps        = 0

    while not done:
        action = agent.get_action(state)
        result = task.step(action)
        state        = result.state
        total_reward += result.reward
        done          = result.done
        steps        += 1
        if steps > 500:
            break

    score = task.evaluate()
    info  = result.info

    from src.tasks import sanitize_score_payload
    raw_res = {
        "score":    to_open_unit_interval(score),
        "reward":   round(total_reward, 2),
        "cleared":  info["total_cleared"],
        "avg_wait": round(info["avg_waiting_time"], 2),
    }
    return sanitize_score_payload(raw_res)


def compute_metrics(seed: int) -> dict:
    metrics = {}
    for agent_name in AGENTS:
        metrics[agent_name] = {}
        for i, difficulty in enumerate(TASK_CONSTRUCTORS):
            task_seed = seed + i * 999
            metrics[agent_name][difficulty] = run_agent(agent_name, difficulty, task_seed)
        scores = [metrics[agent_name][d]["score"] for d in TASK_CONSTRUCTORS]
        metrics[agent_name]["Overall"] = {
            "score":    to_open_unit_interval(sum(scores) / len(scores)),
            "reward":   None,
            "cleared":  None,
            "avg_wait": None,
        }
    from src.tasks import sanitize_score_payload
    return sanitize_score_payload(metrics)




COLORS = {
    "Random Agent":        "#FF6B6B",
    "Fixed-Timing Agent":  "#FFD166",
    "Optimized Agent":     "#4ECDC4",
}


def plot_bar_chart(metrics: dict):
    difficulties = list(TASK_CONSTRUCTORS.keys()) + ["Overall"]
    agents       = list(AGENTS.keys())
    x            = np.arange(len(difficulties))
    width        = 0.22

    fig, ax = plt.subplots(figsize=(11, 5))
    for idx, agent in enumerate(agents):
        scores = [metrics[agent][d]["score"] for d in difficulties]
        offset = (idx - 1) * width
        bars = ax.bar(x + offset, scores, width, label=agent,
                      color=COLORS[agent], edgecolor="white", linewidth=0.6)
        for bar in bars:
            h = bar.get_height()
            ax.annotate(f"{h:.2f}",
                        xy=(bar.get_x() + bar.get_width() / 2, h),
                        xytext=(0, 3), textcoords="offset points",
                        ha="center", va="bottom", fontsize=8, fontweight="bold")

    ax.set_ylabel("Score (0 – 1)", fontsize=11, fontweight="bold")
    ax.set_title("Agent Performance Comparison by Difficulty", fontsize=13, fontweight="bold", pad=12)
    ax.set_xticks(x)
    ax.set_xticklabels(difficulties, fontsize=11)
    ax.set_ylim(0, 1.15)
    ax.legend(fontsize=10, loc="upper left")
    ax.grid(axis="y", linestyle="--", alpha=0.5)
    ax.spines[["top", "right"]].set_visible(False)
    fig.tight_layout()
    return fig


def plot_line_chart(metrics: dict):
    difficulties = ["Easy", "Medium", "Hard"]
    fig, ax = plt.subplots(figsize=(9, 4))

    for agent in AGENTS:
        scores = [metrics[agent][d]["score"] for d in difficulties]
        ax.plot(difficulties, scores, marker="o", linewidth=2.2,
                label=agent, color=COLORS[agent])
        for i, (d, s) in enumerate(zip(difficulties, scores)):
            ax.annotate(f"{s:.3f}", (d, s),
                        textcoords="offset points", xytext=(0, 8),
                        ha="center", fontsize=8, color=COLORS[agent], fontweight="bold")

    ax.set_ylabel("Score (0 – 1)", fontsize=11, fontweight="bold")
    ax.set_title("Performance Trend Across Difficulty Levels", fontsize=13, fontweight="bold", pad=12)
    ax.set_ylim(0, 1.10)
    ax.legend(fontsize=10)
    ax.grid(linestyle="--", alpha=0.4)
    ax.spines[["top", "right"]].set_visible(False)
    fig.tight_layout()
    return fig




def render():
    st.set_page_config(
        page_title="Traffic Optimization Benchmark",
        page_icon="🚦",
        layout="wide",
    )

    st.markdown("""
        <h1 style='text-align: center; margin-bottom: 0;'>🚦 Traffic Optimization Benchmark Dashboard</h1>
        <p style='text-align: center; color: gray; font-size: 1.05em; margin-top: 4px;'>
            Comparing Random, Fixed-Timing, and Optimized AI agents across Easy / Medium / Hard scenarios.
        </p>
        <hr style='margin: 10px 0 25px 0;'>
    """, unsafe_allow_html=True)

    col_left, col_mid, col_right = st.columns([2, 2, 1])
    with col_left:
        seed_input = st.text_input("🎲 Seed (leave blank for random)", placeholder="e.g. 42")
    with col_mid:
        run_btn = st.button("🚀 Run Benchmark", use_container_width=True, type="primary")

    if run_btn:
        seed = int(seed_input) if seed_input.strip().isdigit() else random.randint(1000, 99999)
        st.caption(f"📌 Seed used: **{seed}**  *(save this to reproduce results)*")

        with st.spinner("⏳ Running simulations across all agents and difficulty levels…"):
            metrics = compute_metrics(seed)

        st.success("✅ Benchmark complete!")

        st.markdown("### 📊 Score Summary")

        rows = []
        for agent in AGENTS:
            row = {"Agent": agent}
            for diff in list(TASK_CONSTRUCTORS.keys()) + ["Overall"]:
                row[diff] = f"{metrics[agent][diff]['score']:.4f}"
            rows.append(row)

        import pandas as pd
        df = pd.DataFrame(rows).set_index("Agent")
        st.dataframe(df.style.highlight_max(axis=0, color="#c6f6d5"), use_container_width=True)

        st.markdown("### 🔍 Detailed Metrics (per Difficulty)")
        tabs = st.tabs(list(TASK_CONSTRUCTORS.keys()))
        for tab, difficulty in zip(tabs, TASK_CONSTRUCTORS):
            with tab:
                detail_rows = []
                for agent in AGENTS:
                    m = metrics[agent][difficulty]
                    detail_rows.append({
                        "Agent":            agent,
                        "Score":            m["score"],
                        "Total Reward":     m["reward"],
                        "Vehicles Cleared": m["cleared"],
                        "Avg Wait (s)":     m["avg_wait"],
                    })
                st.dataframe(
                    pd.DataFrame(detail_rows).set_index("Agent"),
                    use_container_width=True
                )

        st.markdown("### 📈 Performance Charts")
        chart_col1, chart_col2 = st.columns([3, 2])
        with chart_col1:
            st.pyplot(plot_bar_chart(metrics))
        with chart_col2:
            st.pyplot(plot_line_chart(metrics))

        opt_overall = metrics["Optimized Agent"]["Overall"]["score"]
        rnd_overall = metrics["Random Agent"]["Overall"]["score"]
        gain = (opt_overall - rnd_overall) * 100

        st.markdown("---")
        st.markdown(f"""
        <div style='background:#e6f4ea; border-left:4px solid #34a853; padding:14px; border-radius:6px;'>
            <b>🧠 Key Insight:</b> The Optimized Agent achieved an overall score of 
            <b>{opt_overall:.4f}</b>, outperforming the Random Agent 
            (<b>{rnd_overall:.4f}</b>) by <b>{gain:.1f}%</b>.
        </div>
        """, unsafe_allow_html=True)

    else:
        st.info("👆 Configure your seed and click **Run Benchmark** to start the simulation.")


if __name__ == "__main__":
    render()
