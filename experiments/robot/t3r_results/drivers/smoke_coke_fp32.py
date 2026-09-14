"""smoke_coke_fp32.py — base CogACT coke_can, 3 orientations, fp32.
Compare vs recorded bf16: upright 0.80, lr_switch 0.88, laid_vert 0.60, avg 0.76."""
import os, sys, time
sys.path.insert(0, "/workspace/CogACT")
sys.path.append("/workspace/T3R")
sys.path.append("/workspace/SimplerEnv")
sys.path.append("/workspace/EfficientTAM")
sys.path.append("/workspace/openvla-oft")
import numpy as np
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
os.environ["DISPLAY"] = ""
os.environ["XLA_PYTHON_CLIENT_PREALLOCATE"] = "false"
for k in ("T3R_METHOD","T3R_PRUNE","T3R_KEEP","T3R_BIAS","ADP_KEEP","ADP_DYNAMIC","TEAM_TOPK","LOGACT"):
    os.environ.pop(k, None)

from simpler_env.evaluation.argparse import get_args
from simpler_env.evaluation.maniskill2_evaluator import maniskill2_evaluator
from experiments.robot.cogact_simpler_policy import CogACTSimplerInference

CKPT = "/workspace/cogact_checkpoint/checkpoints/CogACT-Base.pt"
RGB = "./ManiSkill2_real2sim/data/real_inpainting"
COMMON = ["--policy-model", "cogact", "--policy-setup", "google_robot",
          "--ckpt-path", CKPT, "--robot", "google_robot_static",
          "--control-freq", "3", "--sim-freq", "513",
          "--robot-init-rot-quat-center", "0", "0", "0", "1",
          "--logging-dir", "/workspace/coke_fp32_videos"]

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

print("[coke] building CogACT (fp32) ...", flush=True)
model = CogACTSimplerInference(saved_model_path=CKPT, policy_setup="google_robot",
                               cfg_scale=1.5, use_prune=False, use_bias=False)
import torch
print(f"[coke] DiT dtype={next(model.vla.action_model.net.parameters()).dtype}", flush=True)

allsr = []
for orient in ("upright", "lr_switch", "laid_vertically"):
    sys.argv = ["main_inference.py"] + coke_unit(orient)
    args = get_args()
    t = time.time()
    sr_arr = maniskill2_evaluator(model, args)
    sr = float(np.mean(sr_arr)); allsr.extend(sr_arr)
    print(f"[coke] {orient}: SR={sr:.3f} n={len(sr_arr)} ({time.time()-t:.0f}s)", flush=True)

print(f"[coke] RESULT_LINE coke base fp32 AVG SR={float(np.mean(allsr)):.4f} n={len(allsr)} "
      f"(bf16 was 0.76)", flush=True)
