"""run_k60_driver.py — ABLATION: t3r_orig at keep=0.60 (~154 tok, ~40% prune) across the
full Google Robot suite. Matches the ~43% EFFECTIVE prune of the original 2-camera LIBERO
design (where T3R pruned the 3rd-person view ~85% but kept the full wrist camera).
Single method (t3r_orig, prune+IG bias), fp32. Compares vs the fp32 base already in `bench`.
Resumable JSON keyed 'unit'."""
import os, sys, json, time
sys.path.insert(0, "/workspace/CogACT")
sys.path.append("/workspace/T3R")
sys.path.append("/workspace/SimplerEnv")
sys.path.append("/workspace/EfficientTAM")
sys.path.append("/workspace/openvla-oft")
import numpy as np
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
os.environ["DISPLAY"] = ""
os.environ["XLA_PYTHON_CLIENT_PREALLOCATE"] = "false"

from simpler_env.evaluation.argparse import get_args
from simpler_env.evaluation.maniskill2_evaluator import maniskill2_evaluator
from experiments.robot.cogact_simpler_policy import CogACTSimplerInference

CKPT = "/workspace/cogact_checkpoint/checkpoints/CogACT-Base.pt"
RESULTS = "/workspace/k60_results.json"
RGB = "./ManiSkill2_real2sim/data/real_inpainting"
KEEP = "0.60"

COMMON = ["--policy-model", "cogact", "--policy-setup", "google_robot",
          "--ckpt-path", CKPT, "--robot", "google_robot_static",
          "--control-freq", "3", "--sim-freq", "513",
          "--robot-init-rot-quat-center", "0", "0", "0", "1",
          "--logging-dir", "/workspace/k60_videos"]
DRAWER_KW = ["station_name=mk_station_recolor", "light_mode=simple",
             "disable_bad_material=True", "urdf_version=None"]

def coke_unit(orient):
    return COMMON + ["--max-episode-steps", "80",
        "--env-name", "GraspSingleOpenedCokeCanInScene-v0", "--scene-name", "google_pick_coke_can_1_v4",
        "--rgb-overlay-path", f"{RGB}/google_coke_can_real_eval_1.png",
        "--robot-init-x", "0.35", "0.35", "1", "--robot-init-y", "0.20", "0.20", "1",
        "--obj-init-x", "-0.35", "-0.12", "5", "--obj-init-y", "-0.02", "0.42", "5",
        "--robot-init-rot-rpy-range", "0", "0", "1", "0", "0", "1", "0", "0", "1",
        "--additional-env-build-kwargs", f"{orient}=True", "urdf_version=None"]

MOVE_NEAR = COMMON + ["--max-episode-steps", "80",
    "--env-name", "MoveNearGoogleBakedTexInScene-v0", "--scene-name", "google_pick_coke_can_1_v4",
    "--rgb-overlay-path", f"{RGB}/google_move_near_real_eval_1.png",
    "--robot-init-x", "0.35", "0.35", "1", "--robot-init-y", "0.21", "0.21", "1",
    "--obj-variation-mode", "episode", "--obj-episode-range", "0", "60",
    "--robot-init-rot-rpy-range", "0", "0", "1", "0", "0", "1", "-0.09", "-0.09", "1",
    "--additional-env-build-kwargs", "urdf_version=None"]

def drawer_unit(env_name):
    return COMMON + ["--max-episode-steps", "113",
        "--env-name", env_name, "--scene-name", "dummy_drawer",
        "--robot-init-x", "0.65", "0.85", "3", "--robot-init-y", "-0.18", "0.22", "3",
        "--robot-init-rot-rpy-range", "0", "0", "1", "0", "0", "1", "0", "0", "1",
        "--obj-init-x-range", "0", "0", "1", "--obj-init-y-range", "0", "0", "1",
        "--rgb-overlay-path", f"{RGB}/open_drawer_b0.png", "--enable-raytracing",
        "--additional-env-build-kwargs", *DRAWER_KW]

PUT_IN_DRAWER = COMMON + ["--max-episode-steps", "200",
    "--env-name", "PlaceIntoClosedTopDrawerCustomInScene-v0", "--scene-name", "dummy_drawer",
    "--robot-init-x", "0.652", "0.652", "1", "--robot-init-y", "0.009", "0.009", "1",
    "--robot-init-rot-rpy-range", "0", "0", "1", "0", "0", "1", "0", "0", "1",
    "--obj-init-x-range", "-0.08", "-0.02", "3", "--obj-init-y-range", "-0.02", "0.08", "3",
    "--rgb-overlay-path", f"{RGB}/open_drawer_b0.png", "--enable-raytracing",
    "--additional-env-build-kwargs", *DRAWER_KW, "model_ids=baked_apple_v2"]

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

def load_r():
    return json.load(open(RESULTS)) if os.path.exists(RESULTS) else {}
def save_r(r):
    json.dump(r, open(RESULTS+".tmp","w"), indent=2); os.replace(RESULTS+".tmp", RESULTS)

def main():
    results = load_r()
    print(f"[k60] building CogACT fp32 + t3r_orig keep={KEEP} ...", flush=True)
    os.environ.update({"T3R_PRUNE":"1","T3R_KEEP":KEEP,"T3R_BIAS":"1","T3R_QWIN":"1",
                       "T3R_STRENGTH":"1.0","T3R_VSCALE":"0.5"})
    model = CogACTSimplerInference(saved_model_path=CKPT, policy_setup="google_robot",
                                   cfg_scale=1.5, use_prune=True, use_bias=True)
    ctrl = model._t3r
    for unit_id, argv in EVAL_UNITS:
        if unit_id in results and results[unit_id].get("sr") is not None:
            print(f"[k60] SKIP {unit_id} (={results[unit_id]['sr']:.3f})", flush=True); continue
        if ctrl is not None: ctrl.kept_hist = []
        sys.argv = ["main_inference.py"] + argv
        args = get_args()
        t0 = time.time()
        try:
            sr_arr = maniskill2_evaluator(model, args)
            sr = float(np.mean(sr_arr)); n = len(sr_arr)
            kept = float(np.mean(ctrl.kept_hist[-400:])) if getattr(ctrl,"kept_hist",None) else 256.0
            results[unit_id] = {"sr": sr, "n": n, "kept": round(kept,1),
                                "prune_pct": round((1-kept/256)*100,1), "secs": round(time.time()-t0,1)}
            save_r(results)
            print(f"[k60] DONE {unit_id}: SR={sr:.3f} n={n} kept={kept:.0f} "
                  f"prune={results[unit_id]['prune_pct']:.0f}% ({results[unit_id]['secs']}s)", flush=True)
        except Exception as e:
            import traceback; traceback.print_exc()
            results[unit_id] = {"sr": None, "error": str(e)[:150]}; save_r(results)
            print(f"[k60] FAIL {unit_id}: {e}", flush=True)
    print("[k60] ALL DONE", flush=True)

if __name__ == "__main__":
    main()
