"""
run_cogact_libero_eval.py — CogACT on LIBERO. Base + optional T3R (prune+bias).
Faithful to t3r: SigLIP-SAM vision-token pruning + IG attention bias on the LLaMA inside the VLM.
"""
import os, sys, json, time, argparse
sys.path.insert(0, "/workspace/CogACT")
sys.path.insert(0, "/workspace/T3R")
import numpy as np
import torch
from PIL import Image

from experiments.robot.cogact_loader import load_cogact_for_libero
from libero.libero import benchmark
from libero.libero import get_libero_path
from libero.libero.envs import OffScreenRenderEnv

def get_libero_env(task, resolution=256):
    bddl = os.path.join(get_libero_path("bddl_files"), task.problem_folder, task.bddl_file)
    env = OffScreenRenderEnv(bddl_file_name=bddl, camera_heights=resolution, camera_widths=resolution)
    env.seed(0)
    return env, task.language

def get_libero_image(obs):
    return obs["agentview_image"][::-1, ::-1]

def get_libero_dummy_action():
    return [0, 0, 0, 0, 0, 0, -1]

T3R = None  # lazy holder for t3r controller


def make_env(task_suite_name, task_id, resolution):
    bm = benchmark.get_benchmark_dict()[task_suite_name]()
    task = bm.get_task(task_id)
    env, desc = get_libero_env(task, resolution=resolution)
    init_states = bm.get_task_init_states(task_id)
    return env, desc, init_states, len(bm.tasks)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--checkpoint", default="/workspace/cogact_checkpoint/checkpoints/CogACT-Base.pt")
    ap.add_argument("--task_suite", default="libero_spatial")
    ap.add_argument("--num_tasks", type=int, default=10)
    ap.add_argument("--episodes", type=int, default=10)
    ap.add_argument("--max_steps", type=int, default=220)
    ap.add_argument("--replan", type=int, default=8)
    ap.add_argument("--unnorm_key", default="libero_spatial")
    ap.add_argument("--cfg_scale", type=float, default=1.5)
    ap.add_argument("--ddim", type=int, default=10)
    ap.add_argument("--use_prune", action="store_true")
    ap.add_argument("--use_bias", action="store_true")
    ap.add_argument("--out", default="/workspace/results/base.json")
    args = ap.parse_args()

    model, _, _ = load_cogact_for_libero(args.checkpoint, freeze_vlm=True, device="cuda")
    model.eval()

    if args.use_prune or args.use_bias:
        from t3r_cogact import T3RController
        global T3R
        T3R = T3RController(model, use_prune=args.use_prune, use_bias=args.use_bias)

    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    n_tasks = min(args.num_tasks, 10)
    total, succ, lat = 0, 0, []
    per_task = {}
    for tid in range(n_tasks):
        env, desc, init_states, _ = make_env(args.task_suite, tid, 256)
        t_s = 0
        for ep in range(args.episodes):
            env.reset()
            env.set_init_state(init_states[ep % len(init_states)])
            obs, done = None, False
            for _ in range(10):
                obs, _, _, _ = env.step(get_libero_dummy_action())
            if T3R: T3R.reset_episode(desc)
            chunk, ci, success = None, 0, False
            for st in range(args.max_steps):
                img = get_libero_image(obs)
                pil = Image.fromarray(img)
                if chunk is None or ci >= args.replan:
                    t0 = time.time()
                    if T3R:
                        actions, _ = T3R.predict(pil, desc, args.unnorm_key, args.cfg_scale, args.ddim)
                    else:
                        actions, _ = model.predict_action(pil, desc, unnorm_key=args.unnorm_key,
                                                           cfg_scale=args.cfg_scale, use_ddim=True,
                                                           num_ddim_steps=args.ddim)
                    lat.append(time.time() - t0); chunk, ci = actions, 0
                a = chunk[ci]; ci += 1
                act = a.tolist(); act[6] = 1.0 if act[6] > 0.5 else -1.0
                obs, _, done, _ = env.step(act)
                if done: success = True; break
            total += 1; t_s += int(success); succ += int(success)
        per_task[tid] = t_s / args.episodes
        env.close()
        print(f"task {tid}: {t_s}/{args.episodes}")
    res = {"success_rate": succ/total, "mean_latency_s": float(np.mean(lat)),
           "n": total, "per_task": per_task, "prune": args.use_prune, "bias": args.use_bias}
    if T3R: res.update(T3R.stats())
    json.dump(res, open(args.out, "w"), indent=2)
    print("RESULT", json.dumps(res))


if __name__ == "__main__":
    main()
