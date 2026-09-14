"""run_k60_drawers.py — completes the keep=0.60 ablation: the 6 drawers + put_in_drawer,
PRUNE-ONLY (T3R_BIAS=0) to avoid the IG-backward GPU peak that OOMs/hangs raytraced envs.
Bias is neutral at these operating points, so prune-only ~= full method here. Appends to the
same /workspace/k60_results.json (skips the coke+move_near cells already done)."""
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
    ("drawer_open_top", drawer_unit("OpenTopDrawerCustomInScene-v0")),
    ("drawer_open_mid", drawer_unit("OpenMiddleDrawerCustomInScene-v0")),
    ("drawer_open_bot", drawer_unit("OpenBottomDrawerCustomInScene-v0")),
    ("drawer_close_top", drawer_unit("CloseTopDrawerCustomInScene-v0")),
    ("drawer_close_mid", drawer_unit("CloseMiddleDrawerCustomInScene-v0")),
    ("drawer_close_bot", drawer_unit("CloseBottomDrawerCustomInScene-v0")),
    ("put_in_drawer", PUT_IN_DRAWER),
]

def load_r(): return json.load(open(RESULTS)) if os.path.exists(RESULTS) else {}
def save_r(r):
    json.dump(r, open(RESULTS+".tmp","w"), indent=2); os.replace(RESULTS+".tmp", RESULTS)

def main():
    results = load_r()
    print(f"[k60d] building CogACT fp32 + t3r_orig keep={KEEP} PRUNE-ONLY (bias off) ...", flush=True)
    os.environ.update({"T3R_PRUNE":"1","T3R_KEEP":KEEP,"T3R_BIAS":"0"})
    os.environ.pop("T3R_QWIN", None); os.environ.pop("T3R_STRENGTH", None); os.environ.pop("T3R_VSCALE", None)
    model = CogACTSimplerInference(saved_model_path=CKPT, policy_setup="google_robot",
                                   cfg_scale=1.5, use_prune=True, use_bias=False)
    ctrl = model._t3r
    for unit_id, argv in EVAL_UNITS:
        key = unit_id + "_pruneonly"
        if key in results and results[key].get("sr") is not None:
            print(f"[k60d] SKIP {key} (={results[key]['sr']:.3f})", flush=True); continue
        if ctrl is not None: ctrl.kept_hist = []
        sys.argv = ["main_inference.py"] + argv
        args = get_args()
        t0 = time.time()
        try:
            sr_arr = maniskill2_evaluator(model, args)
            sr = float(np.mean(sr_arr)); n = len(sr_arr)
            kept = float(np.mean(ctrl.kept_hist[-400:])) if getattr(ctrl,"kept_hist",None) else 256.0
            results[key] = {"sr": sr, "n": n, "kept": round(kept,1),
                            "prune_pct": round((1-kept/256)*100,1), "secs": round(time.time()-t0,1)}
            save_r(results)
            print(f"[k60d] DONE {key}: SR={sr:.3f} n={n} kept={kept:.0f} "
                  f"prune={results[key]['prune_pct']:.0f}% ({results[key]['secs']}s)", flush=True)
        except Exception as e:
            import traceback; traceback.print_exc()
            results[key] = {"sr": None, "error": str(e)[:150]}; save_r(results)
            print(f"[k60d] FAIL {key}: {e}", flush=True)
    print("[k60d] ALL DONE", flush=True)

if __name__ == "__main__":
    main()
