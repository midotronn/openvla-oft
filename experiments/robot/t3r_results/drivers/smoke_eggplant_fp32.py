"""smoke_eggplant_fp32.py — base CogACT on eggplant (published 100%), verify fp32 fix."""
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
for k in ("T3R_METHOD","T3R_PRUNE","T3R_KEEP","T3R_BIAS","ADP_KEEP","ADP_DYNAMIC","TEAM_TOPK"):
    os.environ.pop(k, None)

from simpler_env.evaluation.argparse import get_args
from simpler_env.evaluation.maniskill2_evaluator import maniskill2_evaluator
from experiments.robot.cogact_simpler_policy import CogACTSimplerInference

CKPT = "/workspace/cogact_checkpoint/checkpoints/CogACT-Base.pt"
RGB = "./ManiSkill2_real2sim/data/real_inpainting"

argv = ["--policy-model", "cogact", "--policy-setup", "widowx_bridge",
        "--ckpt-path", CKPT, "--robot", "widowx_sink_camera_setup",
        "--control-freq", "5", "--sim-freq", "500", "--max-episode-steps", "120",
        "--env-name", "PutEggplantInBasketScene-v0", "--scene-name", "bridge_table_1_v2",
        "--rgb-overlay-path", f"{RGB}/bridge_sink.png",
        "--robot-init-x", "0.127", "0.127", "1", "--robot-init-y", "0.06", "0.06", "1",
        "--obj-variation-mode", "episode", "--obj-episode-range", "0", "24",
        "--robot-init-rot-quat-center", "0", "0", "0", "1",
        "--robot-init-rot-rpy-range", "0", "0", "1", "0", "0", "1", "0", "0", "1",
        "--logging-dir", "/workspace/bridge_videos"]

print("[smoke] building CogACT (fp32) ...", flush=True)
t0 = time.time()
model = CogACTSimplerInference(saved_model_path=CKPT, policy_setup="widowx_bridge",
                               cfg_scale=1.5, use_prune=False, use_bias=False)
import torch
p = next(model.vla.action_model.net.parameters())
print(f"[smoke] DiT param dtype = {p.dtype}", flush=True)
vlmp = next(model.vla.vlm.llm_backbone.llm.parameters())
print(f"[smoke] LLM param dtype = {vlmp.dtype}", flush=True)
print(f"[smoke] load {time.time()-t0:.0f}s; GPU {torch.cuda.memory_allocated()/1e9:.1f}GB", flush=True)

sys.argv = ["main_inference.py"] + argv
args = get_args()
t1 = time.time()
sr_arr = maniskill2_evaluator(model, args)
sr = float(np.mean(sr_arr)); n = len(sr_arr)
print(f"[smoke] EGGPLANT base fp32: SR={sr:.3f} n={n} ({time.time()-t1:.0f}s)", flush=True)
print(f"[smoke] RESULT_LINE eggplant base SR={sr:.4f} n={n}", flush=True)
