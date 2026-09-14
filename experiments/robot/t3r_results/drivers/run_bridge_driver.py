"""
run_bridge_driver.py — WidowX/Bridge suite, 4 methods (base, t3r_orig, adp64, team64).

CogACT-Base zero-shot on the 4 SimplerEnv Bridge tasks, policy_setup=widowx_bridge. All
PRUNING methods forced to 64 tokens (25% keep / 75% prune). t3r_dit EXCLUDED per request.
Resumable JSON keyed "unit/method".
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
RESULTS = "/workspace/bridge_results.json"
RGB = "./ManiSkill2_real2sim/data/real_inpainting"


def bridge_unit(env_name, robot, scene, overlay, rx, ry):
    return ["--policy-model", "cogact", "--policy-setup", "widowx_bridge",
            "--ckpt-path", CKPT, "--robot", robot,
            "--control-freq", "5", "--sim-freq", "500", "--max-episode-steps", "120",
            "--env-name", env_name, "--scene-name", scene,
            "--rgb-overlay-path", f"{RGB}/{overlay}",
            "--robot-init-x", rx, rx, "1", "--robot-init-y", ry, ry, "1",
            "--obj-variation-mode", "episode", "--obj-episode-range", "0", "24",
            "--robot-init-rot-quat-center", "0", "0", "0", "1",
            "--robot-init-rot-rpy-range", "0", "0", "1", "0", "0", "1", "0", "0", "1",
            "--logging-dir", "/workspace/bridge_videos"]


EVAL_UNITS = [
    ("stack_cube", bridge_unit("StackGreenCubeOnYellowCubeBakedTexInScene-v0", "widowx",
                               "bridge_table_1_v1", "bridge_real_eval_1.png", "0.147", "0.028")),
    ("carrot_on_plate", bridge_unit("PutCarrotOnPlateInScene-v0", "widowx",
                                    "bridge_table_1_v1", "bridge_real_eval_1.png", "0.147", "0.028")),
    ("spoon_on_towel", bridge_unit("PutSpoonOnTableClothInScene-v0", "widowx",
                                   "bridge_table_1_v1", "bridge_real_eval_1.png", "0.147", "0.028")),
    ("eggplant_in_basket", bridge_unit("PutEggplantInBasketScene-v0", "widowx_sink_camera_setup",
                                       "bridge_table_1_v2", "bridge_sink.png", "0.127", "0.06")),
]

METHODS = ["base", "t3r_orig", "adp64", "team64"]

T3R_KEYS = ["T3R_METHOD", "T3R_PRUNE", "T3R_KEEP", "T3R_BIAS", "T3R_DITBIAS", "T3R_POOL",
            "T3R_ALPHA", "T3R_STRENGTH", "T3R_VSCALE", "T3R_POOLK", "T3R_QWIN", "T3R_SAM",
            "T3R_SAMFILL", "T3R_EMA", "T3R_GRID", "T3R_GRIDG", "T3R_RECOMPUTE",
            "ADP_KEEP", "ADP_DYNAMIC", "TEAM_TOPK"]


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


def build_args(argv):
    old = sys.argv
    sys.argv = ["main_inference.py"] + argv
    try:
        return get_args()
    finally:
        sys.argv = old


def make_controller(method, vla):
    clear_env()
    if method == "base":
        return None
    if method == "t3r_orig":
        os.environ.update({"T3R_PRUNE": "1", "T3R_KEEP": "0.25", "T3R_BIAS": "1",
                           "T3R_QWIN": "1", "T3R_STRENGTH": "1.0", "T3R_VSCALE": "0.5"})
        from experiments.robot.t3r_cogact import T3RController
        c = T3RController(vla, use_prune=True, use_bias=True, keep_ratio=0.25,
                          bias_strength=1.0); c.attach(); return c
    if method == "adp64":
        os.environ.update({"T3R_METHOD": "adp", "ADP_KEEP": "0.25", "ADP_DYNAMIC": "0"})
        from experiments.robot.baselines_cogact import BaselineController
        c = BaselineController(vla, method="adp"); c.attach(); return c
    if method == "team64":
        os.environ.update({"T3R_METHOD": "team", "TEAM_TOPK": "64"})
        from experiments.robot.baselines_cogact import BaselineController
        c = BaselineController(vla, method="team"); c.attach(); return c
    raise ValueError(method)


def main():
    results = load_results()
    print("[bridge] building CogACT (widowx_bridge) ...", flush=True)
    clear_env()
    model = CogACTSimplerInference(saved_model_path=CKPT, policy_setup="widowx_bridge",
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

    for unit_id, argv in EVAL_UNITS:
        for method in METHODS:
            key = f"{unit_id}/{method}"
            if key in results and results[key].get("sr") is not None:
                print(f"[bridge] SKIP {key} (= {results[key]['sr']:.3f})", flush=True)
                continue
            restore()
            t0 = time.time()
            ctrl = make_controller(method, vla)
            model._t3r = ctrl
            args = build_args(argv)
            try:
                sr_arr = maniskill2_evaluator(model, args)
                sr = float(np.mean(sr_arr)); n = len(sr_arr)
                kept = 256.0
                if ctrl is not None and getattr(ctrl, "kept_hist", None):
                    kept = float(np.mean(ctrl.kept_hist[-400:]))
                results[key] = {"sr": sr, "n": n, "kept": round(kept, 1),
                                "prune_pct": round((1 - kept / 256.0) * 100, 1),
                                "secs": round(time.time() - t0, 1)}
                save_results(results)
                print(f"[bridge] DONE {key}: SR={sr:.3f} n={n} kept={kept:.0f} "
                      f"prune={results[key]['prune_pct']:.0f}% ({results[key]['secs']}s)", flush=True)
            except Exception as e:
                import traceback; traceback.print_exc()
                results[key] = {"sr": None, "error": str(e)[:200]}
                save_results(results)
                print(f"[bridge] FAIL {key}: {e}", flush=True)
            restore()

    print("[bridge] ALL DONE", flush=True)
    print("\n==== BRIDGE (SR %, pruning @64 tok / 75% prune) ====", flush=True)
    print("unit".ljust(20) + "".join(m.ljust(10) for m in METHODS), flush=True)
    for unit_id, _ in EVAL_UNITS:
        row = unit_id.ljust(20)
        for m in METHODS:
            r = results.get(f"{unit_id}/{m}", {})
            s = r.get("sr")
            row += (f"{s*100:.1f}".ljust(10)) if s is not None else "NA".ljust(10)
        print(row, flush=True)


if __name__ == "__main__":
    main()
