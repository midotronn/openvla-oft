"""smoke_natural_t3r.py — NATURAL (SigLIP-SAM adaptive) T3R: variable token count.
T3R_SAM=1 -> bare object mask, scene-adaptive prune (NO forced keep ratio).
Reports adaptive kept-token distribution + SR on a single-object task (coke) and
a multi-object scene (move_near). fp32 base for faithful comparison."""
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
# NATURAL T3R: SAM-adaptive prune (variable count) + IG bias. NO forced keep ratio.
os.environ.update({"T3R_PRUNE": "1", "T3R_BIAS": "1", "T3R_SAM": "1",
                   "T3R_QWIN": "1", "T3R_STRENGTH": "1.0", "T3R_VSCALE": "0.5"})

from simpler_env.evaluation.argparse import get_args
from simpler_env.evaluation.maniskill2_evaluator import maniskill2_evaluator
from experiments.robot.cogact_simpler_policy import CogACTSimplerInference

CKPT = "/workspace/cogact_checkpoint/checkpoints/CogACT-Base.pt"
RGB = "./ManiSkill2_real2sim/data/real_inpainting"
COMMON = ["--policy-model", "cogact", "--policy-setup", "google_robot",
          "--ckpt-path", CKPT, "--robot", "google_robot_static",
          "--control-freq", "3", "--sim-freq", "513",
          "--robot-init-rot-quat-center", "0", "0", "0", "1",
          "--logging-dir", "/workspace/natural_t3r_videos"]

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
    "--obj-variation-mode", "episode", "--obj-episode-range", "0", "24",
    "--robot-init-rot-rpy-range", "0", "0", "1", "0", "0", "1", "-0.09", "-0.09", "1",
    "--additional-env-build-kwargs", "urdf_version=None"]

print("[nat] building CogACT fp32 + NATURAL T3R (T3R_SAM=1) ...", flush=True)
model = CogACTSimplerInference(saved_model_path=CKPT, policy_setup="google_robot",
                               cfg_scale=1.5, use_prune=True, use_bias=True)
import torch
print(f"[nat] DiT dtype={next(model.vla.action_model.net.parameters()).dtype}", flush=True)

for name, argv in (("coke_upright", coke_unit("upright")), ("move_near", MOVE_NEAR)):
    model._t3r.kept_hist = []
    sys.argv = ["main_inference.py"] + argv
    args = get_args()
    t = time.time()
    sr_arr = maniskill2_evaluator(model, args)
    sr = float(np.mean(sr_arr))
    kh = np.array(model._t3r.kept_hist) if model._t3r.kept_hist else np.array([256])
    prune = (1 - kh.mean() / 256) * 100
    print(f"[nat] RESULT {name}: SR={sr:.3f} n={len(sr_arr)} | "
          f"NAT_TOKENS mean={kh.mean():.1f} med={np.median(kh):.0f} "
          f"min={kh.min()} max={kh.max()} p10={np.percentile(kh,10):.0f} p90={np.percentile(kh,90):.0f} "
          f"| prune={prune:.0f}% ({time.time()-t:.0f}s)", flush=True)
