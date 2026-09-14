"""smoke_t3rorig_fp32.py — t3r_orig (prune64 + attn IG bias) on coke upright, fp32 OOM check."""
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
COMMON = ["--policy-model", "cogact", "--policy-setup", "google_robot",
          "--ckpt-path", CKPT, "--robot", "google_robot_static",
          "--control-freq", "3", "--sim-freq", "513",
          "--robot-init-rot-quat-center", "0", "0", "0", "1",
          "--logging-dir", "/workspace/t3r_fp32_videos"]
argv = COMMON + [
    "--max-episode-steps", "80",
    "--env-name", "GraspSingleOpenedCokeCanInScene-v0", "--scene-name", "google_pick_coke_can_1_v4",
    "--rgb-overlay-path", f"{RGB}/google_coke_can_real_eval_1.png",
    "--robot-init-x", "0.35", "0.35", "1", "--robot-init-y", "0.20", "0.20", "1",
    "--obj-init-x", "-0.35", "-0.12", "5", "--obj-init-y", "-0.02", "0.42", "5",
    "--robot-init-rot-rpy-range", "0", "0", "1", "0", "0", "1", "0", "0", "1",
    "--additional-env-build-kwargs", "upright=True", "urdf_version=None"]

print("[t3r] building CogACT (fp32) + T3RController ...", flush=True)
model = CogACTSimplerInference(saved_model_path=CKPT, policy_setup="google_robot",
                               cfg_scale=1.5, use_prune=True, use_bias=True)
import torch
print(f"[t3r] DiT dtype={next(model.vla.action_model.net.parameters()).dtype} "
      f"GPU={torch.cuda.memory_allocated()/1e9:.1f}GB", flush=True)
sys.argv = ["main_inference.py"] + argv
args = get_args()
t = time.time()
sr_arr = maniskill2_evaluator(model, args)
kept = np.mean(model._t3r.kept_hist[-200:]) if getattr(model._t3r, "kept_hist", None) else -1
print(f"[t3r] RESULT_LINE t3r_orig coke_upright fp32 SR={float(np.mean(sr_arr)):.3f} "
      f"n={len(sr_arr)} kept={kept:.0f} GPUpeak={torch.cuda.max_memory_allocated()/1e9:.1f}GB "
      f"({time.time()-t:.0f}s)", flush=True)
