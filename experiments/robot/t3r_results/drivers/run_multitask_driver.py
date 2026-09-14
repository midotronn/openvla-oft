"""
run_multitask_driver.py — multi-task validation of the pruning-vs-SR comparison.

Loads CogACT ONCE, hot-swaps the token-reduction controller, and evaluates a focused set of
conditions on move_near (60 ep) and open_drawer (9 ep) — the two non-coke SimplerEnv tasks —
to test whether the coke Pareto pattern generalizes. Resumable: each (task, cond) result is
flushed to JSON and skipped on restart.

Conditions: base, adp(native), team(native,80), ours_k75(192), ours_k50(128), ours_k3125(80).
ours_k3125 is the direct head-to-head vs TeamVLA; ours_k75 vs ADP.
"""
import os, sys, json, time
sys.path.insert(0, "/workspace/CogACT")
sys.path.append("/workspace/T3R")
sys.path.append("/workspace/SimplerEnv")
sys.path.append("/workspace/EfficientTAM")
import numpy as np

os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
os.environ["DISPLAY"] = ""
os.environ["XLA_PYTHON_CLIENT_PREALLOCATE"] = "false"

from simpler_env.evaluation.argparse import get_args
from simpler_env.evaluation.maniskill2_evaluator import maniskill2_evaluator
from experiments.robot.cogact_simpler_policy import CogACTSimplerInference

CKPT = "/workspace/cogact_checkpoint/checkpoints/CogACT-Base.pt"
RESULTS = "/workspace/multitask_results.json"

COMMON = ["--policy-model", "cogact", "--policy-setup", "google_robot",
          "--ckpt-path", CKPT, "--robot", "google_robot_static",
          "--control-freq", "3", "--sim-freq", "513",
          "--robot-init-rot-quat-center", "0", "0", "0", "1",
          "--logging-dir", "/workspace/mt_videos"]

TASKS = {
    "move_near": COMMON + [
        "--max-episode-steps", "80",
        "--env-name", "MoveNearGoogleBakedTexInScene-v0", "--scene-name", "google_pick_coke_can_1_v4",
        "--rgb-overlay-path", "./ManiSkill2_real2sim/data/real_inpainting/google_move_near_real_eval_1.png",
        "--robot-init-x", "0.35", "0.35", "1", "--robot-init-y", "0.21", "0.21", "1",
        "--obj-variation-mode", "episode", "--obj-episode-range", "0", "60",
        "--robot-init-rot-rpy-range", "0", "0", "1", "0", "0", "1", "-0.09", "-0.09", "1",
        "--additional-env-build-kwargs", "urdf_version=None",
    ],
    "drawer": COMMON + [
        "--max-episode-steps", "113",
        "--env-name", "OpenTopDrawerCustomInScene-v0", "--scene-name", "dummy_drawer",
        "--robot-init-x", "0.65", "0.85", "3", "--robot-init-y", "-0.18", "0.22", "3",
        "--robot-init-rot-rpy-range", "0", "0", "1", "0", "0", "1", "0", "0", "1",
        "--obj-init-x-range", "0", "0", "1", "--obj-init-y-range", "0", "0", "1",
        "--rgb-overlay-path", "./ManiSkill2_real2sim/data/real_inpainting/open_drawer_b0.png",
        "--enable-raytracing",
        "--additional-env-build-kwargs", "station_name=mk_station_recolor", "light_mode=simple",
        "disable_bad_material=True", "urdf_version=None",
    ],
}

T3R_KEYS = ["T3R_METHOD", "T3R_PRUNE", "T3R_KEEP", "T3R_BIAS", "T3R_DITBIAS", "T3R_POOL",
            "T3R_ALPHA", "T3R_STRENGTH", "T3R_VSCALE", "T3R_POOLK", "T3R_SAM", "T3R_SAMFILL",
            "T3R_EMA", "T3R_RECOMPUTE", "ADP_KEEP", "ADP_DYNAMIC", "TEAM_TOPK"]


def clear_env():
    for k in T3R_KEYS:
        os.environ.pop(k, None)


def load_results():
    if os.path.exists(RESULTS):
        with open(RESULTS) as f:
            return json.load(f)
    return {}


def save_results(r):
    tmp = RESULTS + ".tmp"
    with open(tmp, "w") as f:
        json.dump(r, f, indent=2)
    os.replace(tmp, RESULTS)


def build_args(task):
    old = sys.argv
    sys.argv = ["main_inference.py"] + TASKS[task]
    try:
        return get_args()
    finally:
        sys.argv = old


def make_controller(cond, vla):
    clear_env()
    if cond == "base":
        return None
    if cond == "adp":
        os.environ["T3R_METHOD"] = "adp"
        from experiments.robot.baselines_cogact import BaselineController
        c = BaselineController(vla, method="adp"); c.attach(); return c
    if cond == "team":
        os.environ["T3R_METHOD"] = "team"; os.environ["TEAM_TOPK"] = "80"
        from experiments.robot.baselines_cogact import BaselineController
        c = BaselineController(vla, method="team"); c.attach(); return c
    if cond.startswith("ours_k"):
        keep = float("0." + cond[len("ours_k"):])
        os.environ.update({"T3R_PRUNE": "1", "T3R_KEEP": str(keep)})
        from experiments.robot.t3r_cogact import T3RController
        c = T3RController(vla, use_prune=True, use_bias=False, keep_ratio=keep); c.attach(); return c
    raise ValueError(cond)


def main():
    tasks = ["move_near", "drawer"]
    conditions = ["base", "team", "ours_k3125", "adp", "ours_k75", "ours_k50"]
    results = load_results()

    print("[mt] building CogACT (pristine) ...", flush=True)
    clear_env()
    model = CogACTSimplerInference(saved_model_path=CKPT, policy_setup="google_robot",
                                   cfg_scale=1.5, use_prune=False, use_bias=False)
    vla = model.vla
    PA0 = vla.predict_action
    VB0 = vla.vlm.vision_backbone.forward
    PJ0 = vla.vlm.projector.forward
    LM0 = vla.vlm.llm_backbone.llm.model.forward

    def restore():
        vla.predict_action = PA0
        vla.vlm.vision_backbone.forward = VB0
        vla.vlm.projector.forward = PJ0
        vla.vlm.llm_backbone.llm.model.forward = LM0
        model._t3r = None

    for task in tasks:
        for cond in conditions:
            key = f"{task}/{cond}"
            if key in results and results[key].get("sr") is not None:
                print(f"[mt] SKIP {key} (= {results[key]['sr']:.3f})", flush=True)
                continue
            restore()
            t0 = time.time()
            ctrl = make_controller(cond, vla)
            model._t3r = ctrl
            args = build_args(task)
            try:
                sr_arr = maniskill2_evaluator(model, args)
                sr = float(np.mean(sr_arr)); n = len(sr_arr)
                kept = None
                if ctrl is not None and getattr(ctrl, "kept_hist", None):
                    kept = float(np.mean(ctrl.kept_hist[-300:]))
                results[key] = {"sr": sr, "n": n, "kept_mean": kept,
                                "secs": round(time.time() - t0, 1)}
                save_results(results)
                print(f"[mt] DONE {key}: SR={sr:.3f} (n={n}) kept={kept} "
                      f"({results[key]['secs']}s)", flush=True)
            except Exception as e:
                import traceback; traceback.print_exc()
                results[key] = {"sr": None, "error": str(e)}
                save_results(results)
                print(f"[mt] FAIL {key}: {e}", flush=True)
            restore()

    print("[mt] ALL DONE", flush=True)
    print("\n==== MULTI-TASK VALIDATION ====", flush=True)
    for task in tasks:
        row = "  " + task.ljust(10)
        for cond in conditions:
            r = results.get(f"{task}/{cond}", {})
            s = r.get("sr")
            row += f" {cond}={s:.3f}" if s is not None else f" {cond}=NA"
        print(row, flush=True)


if __name__ == "__main__":
    main()
