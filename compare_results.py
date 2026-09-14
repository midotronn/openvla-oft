import json, numpy as np

base_dir = "eval_results/ablation/baseline"
full_dir = "eval_results/ablation/full_pipeline"

bs = json.load(open(f"{base_dir}/summary_libero_spatial.json"))

print("=" * 85)
print("BASELINE (no pruning, no biasing)")
bsr_all = bs["overall_success_rate"]
bvla = bs["timing_ms"]["vla_mean"]
bstep = bs["timing_ms"]["step_mean"]
bfps = bs["timing_ms"]["effective_fps"]
bavg_steps = bs["episode_steps"]["mean"]
print(f"  Overall SR: {bsr_all*100:.1f}%")
print(f"  VLA: {bvla:.0f}ms | Step: {bstep:.0f}ms | FPS: {bfps:.1f}")
print(f"  Avg steps: {bavg_steps:.0f}")
print()

print("=" * 85)
print("FULL PIPELINE (SigLIP-SAM + IG Bias)")
header = f"{'Task':>6} | {'SR':>7} {'Base':>7} {'Delta':>7} | {'VLA':>6} {'Step':>6} {'Prune':>6} | {'Steps':>6} {'Base':>6} {'Delta':>7}"
print(header)
print("-" * 85)
all_sr, all_vla, all_step, all_prune, all_steps_list = [], [], [], [], []
for i in range(9):
    try:
        d = json.load(open(f"{full_dir}/results_libero_spatial_task{i}.json"))
    except FileNotFoundError:
        continue
    t = d.get("timing_ms", {})
    p = d.get("pruning", {})
    sr = d["success_rate"]
    vla = t.get("vla_mean", 0)
    step = t.get("step_mean", 0)
    pr = p.get("avg_ratio", 0)
    steps = d.get("avg_steps", 0)
    all_sr.append(sr)
    all_vla.append(vla)
    all_step.append(step)
    all_prune.append(pr)
    all_steps_list.append(steps)
    task_bsr = bs["per_task_success_rates"].get(str(i), 0)
    task_bsteps = bs.get("per_task_avg_steps", {}).get(str(i), 0)
    dsr = (sr - task_bsr) * 100
    dsteps = steps - task_bsteps
    print(f"  T{i:>3} | {sr*100:6.0f}% {task_bsr*100:6.0f}% {dsr:+6.0f}pp | {vla:5.0f}ms {step:5.0f}ms {pr*100:5.0f}% | {steps:5.0f} {task_bsteps:5.0f} {dsteps:+6.0f}")

print("-" * 85)
avg_sr = np.mean(all_sr)
avg_vla = np.mean(all_vla)
avg_step = np.mean(all_step)
avg_prune = np.mean(all_prune)
avg_steps = np.mean(all_steps_list)
print(f"  AVG  | {avg_sr*100:6.1f}% {bsr_all*100:6.1f}% {(avg_sr-bsr_all)*100:+6.1f}pp | {avg_vla:5.0f}ms {avg_step:5.0f}ms {avg_prune*100:5.0f}% | {avg_steps:5.0f} {bavg_steps:5.0f} {avg_steps-bavg_steps:+6.0f}")
print()
print("KEY DELTAS (Full Pipeline vs Baseline):")
print(f"  Success Rate: {(avg_sr - bsr_all)*100:+.1f} pp")
print(f"  VLA Latency:  {avg_vla:.0f}ms vs {bvla:.0f}ms ({(avg_vla/bvla - 1)*100:+.0f}%)")
print(f"  Step Latency: {avg_step:.0f}ms vs {bstep:.0f}ms ({(avg_step/bstep - 1)*100:+.0f}%)")
print(f"  Avg Pruning:  {avg_prune*100:.0f}% of patches removed")
print(f"  Avg Steps:    {avg_steps:.0f} vs {bavg_steps:.0f} ({(avg_steps/bavg_steps - 1)*100:+.0f}%)")
