"""smoke_t3rorig_drawer.py — t3r_orig fp32 on RAYTRACED drawer_open_top: OOM check."""
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
os.environ.update({"T3R_PRUNE": "1", "T3R_KEEP": "0.25", "T3R_BIAS": "1",
                   "T3R_QWIN": "1", "T3R_STRENGTH": "1.0", "T3R_VSCALE": "0.5"})

from simpler_env.evaluation.argparse import get_args
from simpler_env.evaluation.maniskill2_evaluator import maniskill2_evaluator
from experiments.robot.cogact_simpler_policy import CogACTSimplerInference

CKPT = "/workspace/cogact_checkpoint/checkpoints/CogACT-Base.pt"
RGB = "./ManiSkill2_real2sim/data/real_inpainting"
DRAWER_KW = ["station_name=mk_station_recolor", "light_mode=simple",
             "disable_bad_material=True", "urdf_version=None"]
COMMON = ["--policy-model", "cogact", "--policy-setup", "google_robot",
          "--ckpt-path", CKPT, "--robot", "google_robot_static",
          "--control-freq", "3", "--sim-freq", "513",
          "--robot-init-rot-quat-center", "0", "0", "0", "1",
          "--logging-dir", "/workspace/t3r_drawer_videos"]
argv = COMMON + [
    "--max-episode-steps", "113",
    "--env-name", "OpenTopDrawerCustomInScene-v0", "--scene-name", "dummy_drawer",
    "--robot-init-x", "0.65", "0.85", "3", "--robot-init-y", "-0.18", "0.22", "3",
    "--robot-init-rot-rpy-range", "0", "0", "1", "0", "0", "1", "0", "0", "1",
    "--obj-init-x-range", "0", "0", "1", "--obj-init-y-range", "0", "0", "1",
    "--rgb-overlay-path", f"{RGB}/open_drawer_b0.png", "--enable-raytracing",
    "--additional-env-build-kwargs", *DRAWER_KW]

print("[oom] building CogACT fp32 + t3r_orig on RAYTRACED drawer ...", flush=True)
model = CogACTSimplerInference(saved_model_path=CKPT, policy_setup="google_robot",
                               cfg_scale=1.5, use_prune=True, use_bias=True)
import torch
sys.argv = ["main_inference.py"] + argv
args = get_args()
t = time.time()
try:
    sr_arr = maniskill2_evaluator(model, args)
    print(f"[oom] OK t3r_orig drawer_open_top fp32 SR={float(np.mean(sr_arr)):.3f} n={len(sr_arr)} "
          f"GPUpeak={torch.cuda.max_memory_allocated()/1e9:.1f}GB ({time.time()-t:.0f}s)", flush=True)
except RuntimeError as e:
    print(f"[oom] FAILED: {str(e)[:150]} GPUpeak={torch.cuda.max_memory_allocated()/1e9:.1f}GB", flush=True)
