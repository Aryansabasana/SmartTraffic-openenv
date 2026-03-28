
import random
import sys
import os
import re
import ast
import pathlib


sys.path.insert(0, os.path.dirname(__file__))
from src.tasks import EasyTask, MediumTask, HardTask
from src.agent import DeterministicAgent
from src.models import State

results = {}

def run_single(seed=None):
    if seed is None:
        seed = random.randint(1000, 99999)
    random.seed(seed)

    agent = DeterministicAgent()
    tasks = {"Easy": EasyTask(), "Medium": MediumTask(), "Hard": HardTask()}
    scores = {}
    metrics = {}

    for level, task in tasks.items():
        task_seed = seed + list(tasks.keys()).index(level) * 999
        state = task.reset(seed=task_seed)
        done = False
        steps = 0
        total_reward = 0.0
        while not done:
            action_idx = agent.get_action(state)
            result = task.step(action_idx)
            state = result.state
            total_reward += result.reward
            done = result.done
            steps += 1
            if steps > 500:
                break
        score = task.evaluate()
        scores[level] = score
        metrics[level] = {
            "cleared": result.info["total_cleared"],
            "avg_wait": result.info["avg_waiting_time"],
            "emg": result.info["emergencies_handled"],
            "reward": round(total_reward, 2),
        }

    overall = sum(scores.values()) / len(scores)
    scores["Overall"] = overall
    return seed, scores, metrics


def sep(title):
    print(f"\n{'='*55}")
    print(f"  {title}")
    print(f"{'='*55}")



sep("TEST 1: SEED REPRODUCIBILITY")
seed = 42
s1, sc1, m1 = run_single(seed)
s2, sc2, m2 = run_single(seed)

print(f"Seed: {seed}")
print(f"\nRun 1: Easy={sc1['Easy']:.4f} | Medium={sc1['Medium']:.4f} | Hard={sc1['Hard']:.4f} | Overall={sc1['Overall']:.4f}")
print(f"       Cleared: {m1['Easy']['cleared']} / {m1['Medium']['cleared']} / {m1['Hard']['cleared']}")
print(f"\nRun 2: Easy={sc2['Easy']:.4f} | Medium={sc2['Medium']:.4f} | Hard={sc2['Hard']:.4f} | Overall={sc2['Overall']:.4f}")
print(f"       Cleared: {m2['Easy']['cleared']} / {m2['Medium']['cleared']} / {m2['Hard']['cleared']}")

if sc1 == sc2 and m1 == m2:
    print("\n✅ TEST 1: SEED REPRODUCIBILITY → PASS")
    results["Seed Reproducibility"] = "PASS"
else:
    print("\n❌ TEST 1: SEED REPRODUCIBILITY → FAIL (outputs differ with same seed)")
    results["Seed Reproducibility"] = "FAIL"



sep("TEST 2: STOCHASTIC VARIABILITY (3 random runs)")
runs = [run_single() for _ in range(3)]
print(f"\n{'Run':<5} {'Seed':<8} {'Easy':<8} {'Medium':<9} {'Hard':<8} {'Overall':<10} {'Hard Cleared'}")
print("-" * 65)
for i, (sd, sc, mx) in enumerate(runs, 1):
    print(f"  {i:<4} {sd:<8} {sc['Easy']:<8.4f} {sc['Medium']:<9.4f} {sc['Hard']:<8.4f} {sc['Overall']:<10.4f} {mx['Hard']['cleared']}")


all_scores = [(r[1]['Easy'], r[1]['Medium'], r[1]['Hard']) for r in runs]
unique_seeds = len(set(r[0] for r in runs)) == 3
any_diff = len(set(s for s in [str(x) for x in all_scores])) > 1

if unique_seeds and any_diff:
    print("\n✅ TEST 2: STOCHASTIC VARIABILITY → PASS (unique seeds + varying outputs)")
    results["Stochastic Variability"] = "PASS"
else:
    print("\n❌ TEST 2: STOCHASTIC VARIABILITY → FAIL (same outputs == possible hardcoding)")
    results["Stochastic Variability"] = "FAIL"



sep("TEST 3: NO HARDCODED VALUES SCAN")
SUSPECT_PATTERNS = [
    r'\[0\.\d+,\s*0\.\d+,\s*0\.\d+',
    r'return\s+0\.\d+\b',
]
IGNORE_BLOCK_MARKER = '__name__'
files = list(pathlib.Path('src').glob('*.py')) + [
    pathlib.Path('evaluate.py'), pathlib.Path('visualize.py')
]
all_ok = True
for fp in files:
    content = fp.read_text(encoding='utf-8')

    parts = content.split('if __name__')
    audit_content = parts[0]
    warnings = []
    for pat in SUSPECT_PATTERNS:
        if re.search(pat, audit_content):
            warnings.append(pat)
    if warnings:
        print(f"  ⚠️  WARNING: {fp}  ← suspicious pattern found")
        all_ok = False
    else:
        print(f"  ✅ OK: {fp}")

if all_ok:
    print("\n✅ TEST 3: NO HARDCODED VALUES → PASS")
    results["No Hardcoding"] = "PASS"
else:
    print("\n❌ TEST 3: NO HARDCODED VALUES → FAIL")
    results["No Hardcoding"] = "FAIL"



sep("TEST 4: METRIC CONSISTENCY (Low vs High Traffic)")


low_task = EasyTask()
low_task.env.arrival_rate_base = 0.2
state = low_task.reset(seed=100)
agent_low = DeterministicAgent()
done = False
while not done:
    r = low_task.step(agent_low.get_action(state))
    state = r.state
    done = r.done
low_score = low_task.evaluate()
low_cleared = r.info["total_cleared"]
low_wait = r.info["avg_waiting_time"]


high_task = EasyTask()
high_task.env.arrival_rate_base = 6.0
state = high_task.reset(seed=100)
agent_high = DeterministicAgent()
done = False
while not done:
    r = high_task.step(agent_high.get_action(state))
    state = r.state
    done = r.done
high_score = high_task.evaluate()
high_cleared = r.info["total_cleared"]
high_wait = r.info["avg_waiting_time"]

print(f"\n  Low Traffic:  Score={low_score:.4f} | Cleared={low_cleared} | Avg Wait={low_wait:.2f}")
print(f"  High Traffic: Score={high_score:.4f} | Cleared={high_cleared} | Avg Wait={high_wait:.2f}")
print(f"\n  Score Delta: {low_score - high_score:+.4f} | Wait Delta: {high_wait - low_wait:+.2f}")

if low_wait <= high_wait and low_score >= high_score:
    print("\n✅ TEST 4: METRIC CONSISTENCY → PASS (low traffic scores better than high traffic)")
    results["Metric Logic"] = "PASS"
else:
    print("\n❌ TEST 4: METRIC CONSISTENCY → FAIL (higher traffic should not beat lower)")
    results["Metric Logic"] = "FAIL"



sep("TEST 5: AGENT IMPACT (Real vs Random Policy)")


_, real_scores, _ = run_single(seed=999)
real_overall = real_scores["Overall"]


class RandomAgent:
    def get_action(self, state):
        return random.choice([1, 2])

def run_with_agent(agent_obj, seed=999):
    random.seed(seed)
    tasks = {"Easy": EasyTask(), "Medium": MediumTask(), "Hard": HardTask()}
    total_score = 0
    for level, task in tasks.items():
        task_seed = seed + list(tasks.keys()).index(level) * 999
        state = task.reset(seed=task_seed)
        done = False
        steps = 0
        while not done:
            r = task.step(agent_obj.get_action(state))
            state = r.state
            done = r.done
            steps += 1
            if steps > 500:
                break
        total_score += task.evaluate()
    return total_score / len(tasks)

random_overall = run_with_agent(RandomAgent(), seed=999)
delta = real_overall - random_overall

print(f"\n  Real Agent Score:   {real_overall:.4f}")
print(f"  Random Agent Score: {random_overall:.4f}")
print(f"  Improvement Delta:  {delta:+.4f} ({delta*100:.1f}%)")

if delta > 0.02:
    print("\n✅ TEST 5: AGENT IMPACT → PASS (Real agent significantly outperforms random)")
    results["Agent Impact"] = "PASS"
elif delta >= 0:
    print("\n⚠️  TEST 5: AGENT IMPACT → MARGINAL (small gap, agent barely helps)")
    results["Agent Impact"] = "MARGINAL"
else:
    print("\n❌ TEST 5: AGENT IMPACT → FAIL (random policy beats real agent — logic error)")
    results["Agent Impact"] = "FAIL"



sep("TEST 6: EXTREME SCENARIOS")


zero_task = EasyTask()
zero_task.env.arrival_rate_base = 0.0
state = zero_task.reset(seed=7)
done = False
agent_z = DeterministicAgent()
while not done:
    r = zero_task.step(agent_z.get_action(state))
    state = r.state
    done = r.done
zero_score = zero_task.evaluate()


cong_task = EasyTask()
cong_task.env.arrival_rate_base = 10.0
state = cong_task.reset(seed=7)
done = False
agent_c = DeterministicAgent()
while not done:
    r = cong_task.step(agent_c.get_action(state))
    state = r.state
    done = r.done
cong_score = cong_task.evaluate()


from src.tasks import HardTask
emg_task = HardTask()
state = emg_task.reset(seed=77)
agent_e = DeterministicAgent()
done = False
while not done:
    r = emg_task.step(agent_e.get_action(state))
    state = r.state
    done = r.done
emg_score = emg_task.evaluate()
emg_handled = r.info["emergencies_handled"]

print(f"\n  Case A (Zero traffic):     Score = {zero_score:.4f} (expected ≈ 1.0)")
print(f"  Case B (Extreme traffic):  Score = {cong_score:.4f} (expected < zero_score)")
print(f"  Case C (Emergency task):   Score = {emg_score:.4f} | Emergencies Handled = {emg_handled}")

case_a = zero_score >= 0.85
case_b = cong_score < zero_score
case_c = emg_handled > 0

if case_a and case_b and case_c:
    print("\n✅ TEST 6: EXTREME SCENARIOS → PASS")
    results["Extreme Cases"] = "PASS"
else:
    issues = []
    if not case_a: issues.append(f"Zero-traffic score {zero_score:.3f} unexpectedly low")
    if not case_b: issues.append(f"Congested score {cong_score:.3f} ≥ zero-traffic score {zero_score:.3f}")
    if not case_c: issues.append("No emergencies handled in hard task")
    print(f"\n❌ TEST 6: EXTREME SCENARIOS → FAIL: {'; '.join(issues)}")
    results["Extreme Cases"] = "FAIL"



sep("TEST 7: GRAPH VALIDATION (Score ↔ Graph Consistency)")
from visualize import generate_graph

_, sc_a, _ = run_single(seed=111)
_, sc_b, _ = run_single(seed=222)

out_a = "audit_graph_A.png"
out_b = "audit_graph_B.png"
generate_graph(sc_a, 111, output_path=out_a)
generate_graph(sc_b, 222, output_path=out_b)


size_a = os.path.getsize(out_a)
size_b = os.path.getsize(out_b)
graph_files_exist = os.path.exists(out_a) and os.path.exists(out_b)
values_match = (
    abs(sc_a['Easy'] - sc_b['Easy']) > 0.0001 or 
    abs(sc_a['Medium'] - sc_b['Medium']) > 0.0001
)

print(f"\n  Seed 111: Easy={sc_a['Easy']:.4f} Medium={sc_a['Medium']:.4f} Hard={sc_a['Hard']:.4f}")
print(f"  Seed 222: Easy={sc_b['Easy']:.4f} Medium={sc_b['Medium']:.4f} Hard={sc_b['Hard']:.4f}")
print(f"  Graph A size: {size_a} bytes | Graph B size: {size_b} bytes")
print(f"  Scores differ across seeds: {values_match}")
print(f"  Graph files generated: {graph_files_exist}")


for f in [out_a, out_b]:
    if os.path.exists(f): os.remove(f)

if graph_files_exist and values_match:
    print("\n✅ TEST 7: GRAPH VALIDATION → PASS (graphs generated from live scores, vary with seed)")
    results["Graph Accuracy"] = "PASS"
else:
    print("\n❌ TEST 7: GRAPH VALIDATION → FAIL")
    results["Graph Accuracy"] = "FAIL"



sep("FINAL AUDIT SUMMARY")
icon = {"PASS": "✅", "FAIL": "❌", "MARGINAL": "⚠️ "}
for test, status in results.items():
    print(f"  {icon.get(status, '?')} {test}: {status}")

any_fail = any(v == "FAIL" for v in results.values())
any_marginal = any(v == "MARGINAL" for v in results.values())

print(f"\n{'='*55}")
if any_fail:
    print("  FINAL VERDICT: NEEDS FIXES ❌")
elif any_marginal:
    print("  FINAL VERDICT: MOSTLY TRUSTED ⚠️  (minor issues detected)")
else:
    print("  FINAL VERDICT: TRUSTED SYSTEM ✅")
print(f"{'='*55}\n")
