

import argparse
import csv
import os
import random
import sys
import time

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import numpy as np
from tqdm import tqdm

sys.path.insert(0, os.path.dirname(__file__))
from src.tasks import EasyTask, MediumTask, HardTask, to_open_unit_interval
from src.agent import DeterministicAgent



TASK_MAP = {
    "Easy":   EasyTask,
    "Medium": MediumTask,
    "Hard":   HardTask,
}




def run_single(seed: int) -> dict:
    random.seed(seed)
    np.random.seed(seed)

    agent = DeterministicAgent()
    results = {"seed": seed}
    total_score = 0.0

    for i, (level, TaskClass) in enumerate(TASK_MAP.items()):
        task_seed = seed + i * 999
        task = TaskClass()
        state = task.reset(seed=task_seed)
        done = False
        steps = 0
        total_reward = 0.0

        while not done:
            action = agent.get_action(state)
            result = task.step(action)
            state = result.state
            total_reward += result.reward
            done = result.done
            steps += 1
            if steps > 500:
                break

        score = task.evaluate()
        info  = result.info
        total_score += score

        results[f"{level}_score"]   = score
        results[f"{level}_reward"]  = round(total_reward, 2)
        results[f"{level}_cleared"] = info["total_cleared"]
        results[f"{level}_wait"]    = round(info["avg_waiting_time"], 2)

    results["overall_score"] = to_open_unit_interval(total_score / len(TASK_MAP))
    return results




def compute_stats(values: np.ndarray, label: str) -> dict:
    return {
        "label":  label,
        "mean":   float(np.mean(values)),
        "std":    float(np.std(values)),
        "min":    float(np.min(values)),
        "max":    float(np.max(values)),
        "p5":     float(np.percentile(values, 5)),
        "p95":    float(np.percentile(values, 95)),
    }




COLORS = {
    "Easy":    "#4ECDC4",
    "Medium":  "#FFD166",
    "Hard":    "#EF476F",
    "Overall": "#6A0572",
}


def plot_histograms(all_records: list, output_path: str):
    metrics = ["Easy", "Medium", "Hard", "Overall"]
    fig, axes = plt.subplots(2, 2, figsize=(13, 9))
    axes = axes.flatten()

    for ax, m in zip(axes, metrics):
        key = f"{m}_score" if m != "Overall" else "overall_score"
        vals = np.array([r[key] for r in all_records])
        mean, std = vals.mean(), vals.std()

        ax.hist(vals, bins=40, color=COLORS[m], edgecolor="white",
                linewidth=0.5, alpha=0.85)
        ax.axvline(mean, color="black", linestyle="--", linewidth=1.4,
                   label=f"Mean={mean:.4f}")
        ax.axvline(mean - std, color="gray", linestyle=":", linewidth=1.1)
        ax.axvline(mean + std, color="gray", linestyle=":", linewidth=1.1,
                   label=f"±1σ={std:.4f}")

        ax.set_title(f"{m} Score Distribution", fontsize=12, fontweight="bold")
        ax.set_xlabel("Score (0–1)", fontsize=10)
        ax.set_ylabel("Frequency", fontsize=10)
        ax.legend(fontsize=9)
        ax.grid(axis="y", linestyle="--", alpha=0.4)
        ax.spines[["top", "right"]].set_visible(False)

    fig.suptitle(
        f"Stress Test Score Distributions  ({len(all_records)} runs)",
        fontsize=14, fontweight="bold", y=1.01
    )
    fig.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Histogram saved → {output_path}")


def plot_time_series(all_records: list, output_path: str):
    scores = [r["overall_score"] for r in all_records]
    runs   = list(range(1, len(scores) + 1))


    window = min(50, len(scores) // 10)
    rolling = np.convolve(scores, np.ones(window) / window, mode="valid")

    fig, ax = plt.subplots(figsize=(13, 4))
    ax.plot(runs, scores, color="#cccccc", linewidth=0.6, label="Run score")
    ax.plot(
        range(window, len(scores) + 1), rolling,
        color=COLORS["Overall"], linewidth=2.0, label=f"Rolling mean (w={window})"
    )
    ax.axhline(np.mean(scores), color="black", linestyle="--",
               linewidth=1.2, label=f"Global mean={np.mean(scores):.4f}")

    ax.set_title("Overall Score Over Simulation Runs", fontsize=13, fontweight="bold")
    ax.set_xlabel("Run number")
    ax.set_ylabel("Overall Score")
    ax.set_ylim(0, 1.05)
    ax.legend(fontsize=9)
    ax.grid(linestyle="--", alpha=0.35)
    ax.spines[["top", "right"]].set_visible(False)
    fig.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Time-series saved → {output_path}")




def save_csv(records: list, path: str):
    if not records:
        return
    fieldnames = list(records[0].keys())
    with open(path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(records)
    print(f"CSV saved      → {path}")




def validate(stats: dict, all_records: list) -> tuple[bool, list]:
    issues = []
    overall_scores = np.array([r["overall_score"] for r in all_records])


    if stats["std"] < 1e-6:
        issues.append("Std dev ≈ 0 → outputs are IDENTICAL across all runs (possible hardcoding)")


    if stats["min"] < 0 or stats["max"] > 1:
        issues.append(f"Scores out of range: min={stats['min']:.4f}, max={stats['max']:.4f}")


    z_scores = np.abs((overall_scores - stats["mean"]) / max(stats["std"], 1e-9))
    extreme_runs = int((z_scores > 3).sum())
    if extreme_runs > len(all_records) * 0.01:
        issues.append(f"{extreme_runs} extreme outliers detected (>3σ) — possible instability")


    if stats["mean"] < 0.5:
        issues.append(f"Mean score {stats['mean']:.4f} < 0.5 — agent performing worse than chance")

    return (len(issues) == 0), issues




def main():
    parser = argparse.ArgumentParser(description="Stress test the Smart Traffic environment")
    parser.add_argument("-n", "--runs", type=int, default=1000,
                        help="Number of simulation runs (default: 1000)")
    parser.add_argument("--base-seed", type=int, default=None,
                        help="Base seed for deterministic run sequence (optional)")
    parser.add_argument("--out-dir", type=str, default=".",
                        help="Output directory for CSV and plots")
    args = parser.parse_args()

    N        = args.runs
    base_rng = random.Random(args.base_seed)

    print(f"\n{'='*55}")
    print(f"  SMART TRAFFIC STRESS TEST")
    print(f"  Runs: {N}  |  Base seed: {args.base_seed or 'random'}")
    print(f"{'='*55}\n")


    all_records = []
    t0 = time.time()

    for _ in tqdm(range(N), desc="Simulating", unit="run",
                  ncols=72, colour="cyan"):
        seed = base_rng.randint(1, 999_999)
        rec  = run_single(seed)
        all_records.append(rec)

    elapsed = time.time() - t0
    print(f"\nCompleted {N} runs in {elapsed:.1f}s  ({elapsed/N*1000:.1f} ms/run)\n")


    overall_scores = np.array([r["overall_score"]  for r in all_records])
    easy_scores    = np.array([r["Easy_score"]      for r in all_records])
    medium_scores  = np.array([r["Medium_score"]    for r in all_records])
    hard_scores    = np.array([r["Hard_score"]      for r in all_records])

    overall_stats = compute_stats(overall_scores, "Overall")


    print(f"{'='*55}")
    print(f"  STRESS TEST REPORT")
    print(f"{'='*55}")
    print(f"  Runs           : {N}")
    print(f"  Elapsed        : {elapsed:.1f}s")
    print()
    print(f"  {'Metric':<18} {'Easy':>8} {'Medium':>8} {'Hard':>8} {'Overall':>9}")
    print(f"  {'-'*55}")
    for label, vals in [("Mean",  [easy_scores.mean(), medium_scores.mean(), hard_scores.mean(), overall_scores.mean()]),
                         ("Std Dev", [easy_scores.std(),  medium_scores.std(),  hard_scores.std(),  overall_scores.std()]),
                         ("Min",    [easy_scores.min(),  medium_scores.min(),  hard_scores.min(),  overall_scores.min()]),
                         ("Max",    [easy_scores.max(),  medium_scores.max(),  hard_scores.max(),  overall_scores.max()]),
                         ("P5",     [np.percentile(easy_scores, 5), np.percentile(medium_scores, 5),
                                     np.percentile(hard_scores, 5), np.percentile(overall_scores, 5)]),
                         ("P95",    [np.percentile(easy_scores,95), np.percentile(medium_scores,95),
                                     np.percentile(hard_scores,95), np.percentile(overall_scores,95)])]:
        print(f"  {label:<18} {vals[0]:>8.4f} {vals[1]:>8.4f} {vals[2]:>8.4f} {vals[3]:>9.4f}")


    print()
    stable, issues = validate(overall_stats, all_records)
    print(f"{'='*55}")
    print(f"  Stability Verdict:")
    if stable:
        print(f"  STABLE ✅  — All validation rules passed")
    else:
        print(f"  UNSTABLE ❌ — Issues detected:")
        for issue in issues:
            print(f"    • {issue}")
    print(f"{'='*55}\n")


    os.makedirs(args.out_dir, exist_ok=True)
    csv_path  = os.path.join(args.out_dir, "stress_test_results.csv")
    hist_path = os.path.join(args.out_dir, "stress_test_histogram.png")
    ts_path   = os.path.join(args.out_dir, "stress_test_timeseries.png")

    save_csv(all_records, csv_path)
    plot_histograms(all_records, hist_path)
    plot_time_series(all_records, ts_path)

    print("\nDone.")
    return 0 if stable else 1


if __name__ == "__main__":
    sys.exit(main())
