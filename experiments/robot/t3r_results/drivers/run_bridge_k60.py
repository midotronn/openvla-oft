"""run_bridge_k60.py — t3r_orig at keep=0.60 (~154 tok, 40% prune) on the 4 Bridge tasks,
fp32. Completes the single-object claim across BOTH SimplerEnv suites. vs the fp32 base in
`bridge`. Resumable JSON keyed 'unit'."""
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
RESULTS = "/workspace/bridge_k60_results.json"
RGB = "./ManiSkill2_real2sim/data/real_inpainting"
KEEP = "0.60"

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
            "--logging-dir", "/workspace/bridge_k60_videos"]

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

def load_r(): return json.load(open(RESULTS)) if os.path.exists(RESULTS) else {}
def save_r(r):
    json.dump(r, open(RESULTS+".tmp","w"), indent=2); os.replace(RESULTS+".tmp", RESULTS)

def main():
    results = load_r()
    print(f"[bk60] building CogACT fp32 (widowx_bridge) + t3r_orig keep={KEEP} ...", flush=True)
    os.environ.update({"T3R_PRUNE":"1","T3R_KEEP":KEEP,"T3R_BIAS":"1","T3R_QWIN":"1",
                       "T3R_STRENGTH":"1.0","T3R_VSCALE":"0.5"})
    model = CogACTSimplerInference(saved_model_path=CKPT, policy_setup="widowx_bridge",
                                   cfg_scale=1.5, use_prune=True, use_bias=True)
    ctrl = model._t3r
    for unit_id, argv in EVAL_UNITS:
        if unit_id in results and results[unit_id].get("sr") is not None:
            print(f"[bk60] SKIP {unit_id} (={results[unit_id]['sr']:.3f})", flush=True); continue
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
            print(f"[bk60] DONE {unit_id}: SR={sr:.3f} n={n} kept={kept:.0f} "
                  f"prune={results[unit_id]['prune_pct']:.0f}% ({results[unit_id]['secs']}s)", flush=True)
        except Exception as e:
            import traceback; traceback.print_exc()
            results[unit_id] = {"sr": None, "error": str(e)[:150]}; save_r(results)
            print(f"[bk60] FAIL {unit_id}: {e}", flush=True)
    print("[bk60] ALL DONE", flush=True)

if __name__ == "__main__":
    main()
