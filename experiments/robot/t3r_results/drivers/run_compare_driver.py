"""
run_compare_driver.py — single-process, resumable head-to-head driver.

Loads CogACT ONCE, then evaluates every (condition x orientation) by hot-swapping the
token-reduction controller on the persistent model. Writes each result to a JSON file and
skips already-completed cells on restart, so it converges even if the process is killed.

Conditions (coke_can, 3 orient x 25 pos = 75 trials each):
  base       : no token reduction
  adp        : ADP native (QK importance @ layer0 + action-aware gate, qk_keep0.75)
  team       : TeamVLA native (expand + bipartite merge, topk=80)
  ours_bal   : T3R SigLIP-SAM keep0.8 + DiT-visual bias a0.15
  ours_aggr  : T3R SigLIP-SAM keep0.25 (75% prune)
Baselines run in NATIVE config (keep ratio not fixed to ours).
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
RESULTS = "/workspace/compare_selection_results.json"
ORIENTS = ["upright", "lr_switch", "laid_vertically"]

BASE_ARGV = [
    "--policy-model", "cogact", "--policy-setup", "google_robot",
    "--ckpt-path", CKPT, "--robot", "google_robot_static",
    "--control-freq", "3", "--sim-freq", "513", "--max-episode-steps", "80",
    "--env-name", "GraspSingleOpenedCokeCanInScene-v0", "--scene-name", "google_pick_coke_can_1_v4",
    "--rgb-overlay-path", "./ManiSkill2_real2sim/data/real_inpainting/google_coke_can_real_eval_1.png",
    "--robot-init-x", "0.35", "0.35", "1", "--robot-init-y", "0.20", "0.20", "1",
    "--obj-init-x", "-0.35", "-0.12", "5", "--obj-init-y", "-0.02", "0.42", "5",
    "--robot-init-rot-quat-center", "0", "0", "0", "1",
    "--robot-init-rot-rpy-range", "0", "0", "1", "0", "0", "1", "0", "0", "1",
    "--logging-dir", "/workspace/cmp_videos",
]

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


def build_args(orient):
    argv = list(BASE_ARGV) + ["--additional-env-build-kwargs", f"{orient}=True"]
    old = sys.argv
    sys.argv = ["main_inference.py"] + argv
    try:
        return get_args()
    finally:
        sys.argv = old


def make_controller(cond, vla):
    """Construct + attach the controller for a condition. Returns controller or None."""
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
    if "_grid" in cond and cond.startswith("ours_k"):
        # ours_kXX_grid[G] -> keep0.XX + SigLIP-guided spatial-coverage selection (faithful).
        # Optional trailing digit(s) set the supergrid size G (coverage anchors = G*G);
        # fewer anchors = more object detail. e.g. ours_k25_grid6 -> keep0.25, 6x6 coverage.
        keep = float("0." + cond[len("ours_k"):].split("_")[0])
        gpart = cond.split("_grid")[1]
        env = {"T3R_PRUNE": "1", "T3R_KEEP": str(keep), "T3R_GRID": "1"}
        if gpart.isdigit():
            env["T3R_GRIDG"] = gpart
        os.environ.update(env)
        from experiments.robot.t3r_cogact import T3RController
        c = T3RController(vla, use_prune=True, use_bias=False, keep_ratio=keep); c.attach(); return c
    if "_rc1" in cond and cond.startswith("ours_k"):
        # ours_kXX_rc1 -> keep0.XX + recompute the SigLIP mask EVERY step (track object).
        keep = float("0." + cond[len("ours_k"):].split("_")[0])
        os.environ.update({"T3R_PRUNE": "1", "T3R_KEEP": str(keep), "T3R_RECOMPUTE": "1"})
        from experiments.robot.t3r_cogact import T3RController
        c = T3RController(vla, use_prune=True, use_bias=False, keep_ratio=keep); c.attach(); return c
    if "_ab" in cond and cond.startswith("ours_k"):
        # ours_kXX_abYY -> keep=0.XX + FAITHFUL IG attention bias at strength Y.Y (no DiT bias).
        # e.g. ours_k25_ab10 -> keep0.25, attn-bias strength 1.0.
        base, ab = cond.split("_ab")
        keep = float("0." + base[len("ours_k"):])
        strength = float(ab[0] + "." + ab[1:])  # "10"->1.0, "08"->0.8, "15"->1.5
        os.environ.update({"T3R_PRUNE": "1", "T3R_KEEP": str(keep), "T3R_BIAS": "1",
                           "T3R_QWIN": "1", "T3R_STRENGTH": str(strength), "T3R_VSCALE": "0.5"})
        from experiments.robot.t3r_cogact import T3RController
        c = T3RController(vla, use_prune=True, use_bias=True, keep_ratio=keep,
                          bias_strength=strength); c.attach(); return c
    if cond.startswith("ours_k"):
        # ours_kXX -> raw faithful SigLIP-SAM pruning at keep=0.XX (no SAM-fill, no bias).
        # "75"->0.75, "3125"->0.3125, "50"->0.50, "25"->0.25, "15"->0.15, "90"->0.90
        keep = float("0." + cond[len("ours_k"):])
        os.environ.update({"T3R_PRUNE": "1", "T3R_KEEP": str(keep)})
        from experiments.robot.t3r_cogact import T3RController
        c = T3RController(vla, use_prune=True, use_bias=False, keep_ratio=keep); c.attach(); return c
    raise ValueError(cond)


def main():
    # Lighter grid coverage at keep0.25: fewer coverage anchors (g=5,6) + more SigLIP object fill,
    # to test if the hard-pose benefit survives with less easy-pose damage. prune-only=0.720,
    # full grid(g=8)=0.653 already measured this session-family.
    conditions = ["ours_k25_grid6", "ours_k25_grid5"]
    results = load_results()

    print("[driver] building CogACT (pristine, no controller) ...", flush=True)
    clear_env()
    model = CogACTSimplerInference(saved_model_path=CKPT, policy_setup="google_robot",
                                   cfg_scale=1.5, use_prune=False, use_bias=False)
    vla = model.vla

    # Save pristine method references for clean restore between conditions.
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

    for cond in conditions:
        for orient in ORIENTS:
            key = f"{cond}/{orient}"
            if key in results and results[key].get("sr") is not None:
                print(f"[driver] SKIP {key} (= {results[key]['sr']:.3f})", flush=True)
                continue
            restore()
            t0 = time.time()
            ctrl = make_controller(cond, vla)
            model._t3r = ctrl
            args = build_args(orient)
            try:
                sr_arr = maniskill2_evaluator(model, args)
                sr = float(np.mean(sr_arr))
                n = len(sr_arr)
                kept = None
                if ctrl is not None and getattr(ctrl, "kept_hist", None):
                    kept = float(np.mean(ctrl.kept_hist[-200:]))
                results[key] = {"sr": sr, "n": n, "kept_mean": kept,
                                "secs": round(time.time() - t0, 1)}
                save_results(results)
                print(f"[driver] DONE {key}: SR={sr:.3f} (n={n}) kept={kept} "
                      f"({results[key]['secs']}s)", flush=True)
            except Exception as e:
                import traceback; traceback.print_exc()
                results[key] = {"sr": None, "error": str(e)}
                save_results(results)
                print(f"[driver] FAIL {key}: {e}", flush=True)
            restore()

    print("[driver] ALL DONE", flush=True)
    # Print summary table
    print("\n==== COKE_CAN COMPARISON (75-trial, 3 orient x 25) ====", flush=True)
    for cond in conditions:
        srs = [results.get(f"{cond}/{o}", {}).get("sr") for o in ORIENTS]
        if all(s is not None for s in srs):
            kept = [results.get(f"{cond}/{o}", {}).get("kept_mean") for o in ORIENTS]
            kv = [k for k in kept if k]
            keptm = (sum(kv) / len(kv)) if kv else 256
            print(f"  {cond:10s}  up={srs[0]:.3f} lr={srs[1]:.3f} lv={srs[2]:.3f}  "
                  f"AVG={np.mean(srs):.3f}  kept~{keptm:.0f}/256 ({(1-keptm/256)*100:.0f}% red)",
                  flush=True)


if __name__ == "__main__":
    main()
