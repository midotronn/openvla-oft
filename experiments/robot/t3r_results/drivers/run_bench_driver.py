"""
run_bench_driver.py — comprehensive 5-method x all-Google-Robot-tasks benchmark on CogACT.

Loads CogACT ONCE, hot-swaps the token-reduction method, and evaluates every eval-unit.
All PRUNING methods are forced to the SAME budget (25% keep = 64 tokens) for a fair
matched-token comparison:
  base      : no reduction (256 tokens) — reference
  t3r_dit   : T3R full method = SigLIP-SAM prune(64) + DiT-conditioning bias (alpha 0.15)
  t3r_orig  : T3R full method = SigLIP-SAM prune(64) + original IG attention bias (str 1.0)
  adp64     : ADP forced to 64 (QK-importance, keep0.25, action-gate OFF)
  team64    : TeamVLA forced to 64 (expand + bipartite merge, topk=64)

Records SR, mean kept tokens, prune %, n, secs per (unit, method). Resumable JSON.
urdf_version=None (base appearance); the 4-URDF visual-matching aggregate is a 4x follow-up.
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
RESULTS = "/workspace/bench_results.json"
RGB = "./ManiSkill2_real2sim/data/real_inpainting"

COMMON = ["--policy-model", "cogact", "--policy-setup", "google_robot",
          "--ckpt-path", CKPT, "--robot", "google_robot_static",
          "--control-freq", "3", "--sim-freq", "513",
          "--robot-init-rot-quat-center", "0", "0", "0", "1",
          "--logging-dir", "/workspace/bench_videos"]

DRAWER_KW = ["station_name=mk_station_recolor", "light_mode=simple",
             "disable_bad_material=True", "urdf_version=None"]


def coke_unit(orient):
    return COMMON + [
        "--max-episode-steps", "80",
        "--env-name", "GraspSingleOpenedCokeCanInScene-v0", "--scene-name", "google_pick_coke_can_1_v4",
        "--rgb-overlay-path", f"{RGB}/google_coke_can_real_eval_1.png",
        "--robot-init-x", "0.35", "0.35", "1", "--robot-init-y", "0.20", "0.20", "1",
        "--obj-init-x", "-0.35", "-0.12", "5", "--obj-init-y", "-0.02", "0.42", "5",
        "--robot-init-rot-rpy-range", "0", "0", "1", "0", "0", "1", "0", "0", "1",
        "--additional-env-build-kwargs", f"{orient}=True", "urdf_version=None",
    ]


MOVE_NEAR = COMMON + [
    "--max-episode-steps", "80",
    "--env-name", "MoveNearGoogleBakedTexInScene-v0", "--scene-name", "google_pick_coke_can_1_v4",
    "--rgb-overlay-path", f"{RGB}/google_move_near_real_eval_1.png",
    "--robot-init-x", "0.35", "0.35", "1", "--robot-init-y", "0.21", "0.21", "1",
    "--obj-variation-mode", "episode", "--obj-episode-range", "0", "60",
    "--robot-init-rot-rpy-range", "0", "0", "1", "0", "0", "1", "-0.09", "-0.09", "1",
    "--additional-env-build-kwargs", "urdf_version=None",
]


def drawer_unit(env_name):
    return COMMON + [
        "--max-episode-steps", "113",
        "--env-name", env_name, "--scene-name", "dummy_drawer",
        "--robot-init-x", "0.65", "0.85", "3", "--robot-init-y", "-0.18", "0.22", "3",
        "--robot-init-rot-rpy-range", "0", "0", "1", "0", "0", "1", "0", "0", "1",
        "--obj-init-x-range", "0", "0", "1", "--obj-init-y-range", "0", "0", "1",
        "--rgb-overlay-path", f"{RGB}/open_drawer_b0.png", "--enable-raytracing",
        "--additional-env-build-kwargs", *DRAWER_KW,
    ]


PUT_IN_DRAWER = COMMON + [
    "--max-episode-steps", "200",
    "--env-name", "PlaceIntoClosedTopDrawerCustomInScene-v0", "--scene-name", "dummy_drawer",
    "--robot-init-x", "0.652", "0.652", "1", "--robot-init-y", "0.009", "0.009", "1",
    "--robot-init-rot-rpy-range", "0", "0", "1", "0", "0", "1", "0", "0", "1",
    "--obj-init-x-range", "-0.08", "-0.02", "3", "--obj-init-y-range", "-0.02", "0.08", "3",
    "--rgb-overlay-path", f"{RGB}/open_drawer_b0.png", "--enable-raytracing",
    "--additional-env-build-kwargs", *DRAWER_KW, "model_ids=baked_apple_v2",
]

# Ordered fast-first: coke(3) + move_near, then drawer(6), then put_in_drawer.
EVAL_UNITS = [
    ("coke_upright", coke_unit("upright")),
    ("coke_lr_switch", coke_unit("lr_switch")),
    ("coke_laid_vert", coke_unit("laid_vertically")),
    ("move_near", MOVE_NEAR),
    ("drawer_open_top", drawer_unit("OpenTopDrawerCustomInScene-v0")),
    ("drawer_open_mid", drawer_unit("OpenMiddleDrawerCustomInScene-v0")),
    ("drawer_open_bot", drawer_unit("OpenBottomDrawerCustomInScene-v0")),
    ("drawer_close_top", drawer_unit("CloseTopDrawerCustomInScene-v0")),
    ("drawer_close_mid", drawer_unit("CloseMiddleDrawerCustomInScene-v0")),
    ("drawer_close_bot", drawer_unit("CloseBottomDrawerCustomInScene-v0")),
    ("put_in_drawer", PUT_IN_DRAWER),
]

METHODS = ["base", "t3r_dit", "t3r_orig", "adp64", "team64"]

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
    if method == "t3r_dit":
        os.environ.update({"T3R_PRUNE": "1", "T3R_KEEP": "0.25", "T3R_BIAS": "1",
                           "T3R_DITBIAS": "1", "T3R_POOL": "visual", "T3R_ALPHA": "0.15",
                           "T3R_POOLK": "16", "T3R_VSCALE": "0.5"})
        from experiments.robot.t3r_cogact import T3RController
        c = T3RController(vla, use_prune=True, use_bias=True, keep_ratio=0.25); c.attach(); return c
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
    print("[bench] building CogACT (pristine) ...", flush=True)
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

    for unit_id, argv in EVAL_UNITS:
        for method in METHODS:
            key = f"{unit_id}/{method}"
            if key in results and results[key].get("sr") is not None:
                print(f"[bench] SKIP {key} (= {results[key]['sr']:.3f})", flush=True)
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
                print(f"[bench] DONE {key}: SR={sr:.3f} n={n} kept={kept:.0f} "
                      f"prune={results[key]['prune_pct']:.0f}% ({results[key]['secs']}s)", flush=True)
            except Exception as e:
                import traceback; traceback.print_exc()
                results[key] = {"sr": None, "error": str(e)[:200]}
                save_results(results)
                print(f"[bench] FAIL {key}: {e}", flush=True)
            restore()

    print("[bench] ALL DONE", flush=True)
    # Summary table
    print("\n==== BENCHMARK (SR %, all pruning @64 tok / 75% prune) ====", flush=True)
    header = "unit".ljust(18) + "".join(m.ljust(10) for m in METHODS)
    print(header, flush=True)
    for unit_id, _ in EVAL_UNITS:
        row = unit_id.ljust(18)
        for m in METHODS:
            r = results.get(f"{unit_id}/{m}", {})
            s = r.get("sr")
            row += (f"{s*100:.1f}".ljust(10)) if s is not None else "NA".ljust(10)
        print(row, flush=True)


if __name__ == "__main__":
    main()
