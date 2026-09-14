import json, numpy as np

base_dir = "eval_results/ablation/baseline"
full_dir = "eval_results/ablation/full_pipeline"
team_dir = "eval_results/ablation/teamvla"

bs = json.load(open(f"{base_dir}/summary_libero_spatial.json"))
ts = json.load(open(f"{team_dir}/summary_libero_spatial.json"))

# Build full pipeline aggregate from per-task JSONs
fp_tasks = {}
fp_vla, fp_step, fp_prune, fp_steps_list = [], [], [], []
for i in range(10):
    try:
        d = json.load(open(f"{full_dir}/results_libero_spatial_task{i}.json"))
    except FileNotFoundError:
        continue
    fp_tasks[i] = d
    t = d.get("timing_ms", {})
    p = d.get("pruning", {})
    fp_vla.append(t.get("vla_mean", 0))
    fp_step.append(t.get("step_mean", 0))
    fp_prune.append(p.get("avg_ratio", 0))
    fp_steps_list.append(d.get("avg_steps", 0))

fp_sr_all = np.mean([fp_tasks[i]["success_rate"] for i in fp_tasks])

print("=" * 90)
print("  ABLATION: 3 CONDITIONS x 10 TASKS x 10 EPISODES")
print("=" * 90)
print()

# ---- Overall Summary ----
print("OVERALL SUMMARY")
print("-" * 90)
print(f"{'Metric':<25} {'Baseline':>15} {'Full Pipeline':>15} {'TeamVLA':>15}")
print("-" * 90)
print(f"{'Success Rate':<25} {bs['overall_success_rate']*100:>14.1f}% {fp_sr_all*100:>14.1f}% {ts['overall_success_rate']*100:>14.1f}%")
print(f"{'VLA Latency (ms)':<25} {bs['timing_ms']['vla_mean']:>15.0f} {np.mean(fp_vla):>15.0f} {ts['vla_inference_ms']['mean']:>15.0f}")
print(f"{'Step Latency (ms)':<25} {bs['timing_ms']['step_mean']:>15.0f} {np.mean(fp_step):>15.0f} {ts['step_time_ms']['mean']:>15.0f}")
print(f"{'Effective FPS':<25} {bs['timing_ms']['effective_fps']:>15.1f} {1000.0/np.mean(fp_step):>15.1f} {ts['effective_fps']:>15.1f}")
print(f"{'Avg Steps/Episode':<25} {bs['episode_steps']['mean']:>15.0f} {np.mean(fp_steps_list):>15.0f} {ts['episode_steps']['mean']:>15.0f}")
print(f"{'Avg Pruning Ratio':<25} {'n/a':>15} {np.mean(fp_prune)*100:>14.0f}% {'n/a':>15}")
print("-" * 90)
print()

# ---- Deltas vs Baseline ----
print("DELTAS vs BASELINE")
print("-" * 90)
print(f"{'Metric':<25} {'Full Pipeline':>15} {'TeamVLA':>15}")
print("-" * 90)
fp_delta_sr = (fp_sr_all - bs["overall_success_rate"]) * 100
tv_delta_sr = (ts["overall_success_rate"] - bs["overall_success_rate"]) * 100
fp_delta_vla = np.mean(fp_vla) - bs["timing_ms"]["vla_mean"]
tv_delta_vla = ts["vla_inference_ms"]["mean"] - bs["timing_ms"]["vla_mean"]
fp_delta_step = np.mean(fp_step) - bs["timing_ms"]["step_mean"]
tv_delta_step = ts["step_time_ms"]["mean"] - bs["timing_ms"]["step_mean"]
fp_delta_steps = np.mean(fp_steps_list) - bs["episode_steps"]["mean"]
tv_delta_steps = ts["episode_steps"]["mean"] - bs["episode_steps"]["mean"]
print(f"{'Success Rate':<25} {fp_delta_sr:>+14.1f}pp {tv_delta_sr:>+14.1f}pp")
print(f"{'VLA Latency':<25} {fp_delta_vla:>+14.0f}ms ({fp_delta_vla/bs['timing_ms']['vla_mean']*100:>+.0f}%) {tv_delta_vla:>+14.0f}ms ({tv_delta_vla/bs['timing_ms']['vla_mean']*100:>+.0f}%)")
print(f"{'Step Latency':<25} {fp_delta_step:>+14.0f}ms ({fp_delta_step/bs['timing_ms']['step_mean']*100:>+.0f}%) {tv_delta_step:>+14.0f}ms ({tv_delta_step/bs['timing_ms']['step_mean']*100:>+.0f}%)")
print(f"{'Avg Steps':<25} {fp_delta_steps:>+14.0f} ({fp_delta_steps/bs['episode_steps']['mean']*100:>+.0f}%) {tv_delta_steps:>+14.0f} ({tv_delta_steps/bs['episode_steps']['mean']*100:>+.0f}%)")
print("-" * 90)
print()

# ---- Per-Task Comparison ----
print("PER-TASK SUCCESS RATES")
print("-" * 90)
print(f"{'Task':<8} {'Baseline':>10} {'Full Pipe':>10} {'TeamVLA':>10} {'FP Delta':>10} {'TV Delta':>10}")
print("-" * 90)
for i in range(10):
    b = bs["per_task_success_rates"].get(str(i), 0)
    f = fp_tasks[i]["success_rate"] if i in fp_tasks else 0
    t = ts["per_task_success_rates"].get(str(i), 0)
    fd = (f - b) * 100
    td = (t - b) * 100
    print(f"  T{i:<5} {b*100:>9.0f}% {f*100:>9.0f}% {t*100:>9.0f}% {fd:>+9.0f}pp {td:>+9.0f}pp")
print("-" * 90)
print(f"  AVG   {bs['overall_success_rate']*100:>9.1f}% {fp_sr_all*100:>9.1f}% {ts['overall_success_rate']*100:>9.1f}% {fp_delta_sr:>+9.1f}pp {tv_delta_sr:>+9.1f}pp")
print()

# ---- Per-Task Steps ----
print("PER-TASK AVG STEPS")
print("-" * 90)
print(f"{'Task':<8} {'Baseline':>10} {'Full Pipe':>10} {'TeamVLA':>10}")
print("-" * 90)
for i in range(10):
    b = bs.get("per_task_avg_steps", {}).get(str(i), 0)
    f = fp_tasks[i].get("avg_steps", 0) if i in fp_tasks else 0
    t = ts.get("episode_steps", {}).get("mean", 0)  # TeamVLA only has aggregate
    print(f"  T{i:<5} {b:>10.0f} {f:>10.0f}")
print("-" * 90)
print(f"  AVG   {bs['episode_steps']['mean']:>10.0f} {np.mean(fp_steps_list):>10.0f} {ts['episode_steps']['mean']:>10.0f}")
