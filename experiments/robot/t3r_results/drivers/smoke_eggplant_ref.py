"""smoke_eggplant_ref.py — TRUE reference sim_cogact policy (load_vla) on eggplant.
Decisive A/B vs my wrapper (0.167). If ref ~1.0 -> my loader/wrapper is buggy.
If ref ~0.17 -> env/harness issue (published 100% unreachable here)."""
import os, sys, time
sys.path.insert(0, "/workspace/CogACT")
sys.path.append("/workspace/SimplerEnv")
sys.path.append("/workspace/EfficientTAM")
import numpy as np
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
os.environ.setdefault("HF_HOME", "/workspace/.cache_huggingface")
os.environ["DISPLAY"] = ""
os.environ["XLA_PYTHON_CLIENT_PREALLOCATE"] = "false"

from simpler_env.evaluation.argparse import get_args
from simpler_env.evaluation.maniskill2_evaluator import maniskill2_evaluator
from simpler_env.policies.sim_cogact.cogact_policy import CogACTInference

SNAP = "/workspace/.cache_huggingface/hub/models--CogACT--CogACT-Base/snapshots/6550bf0992f162fc5d74f14ffee30771a9433363"
CKPT = f"{SNAP}/checkpoints/CogACT-Base.pt"
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
        "--logging-dir", "/workspace/bridge_videos_ref"]

print("[ref] building REFERENCE CogACTInference via load_vla ...", flush=True)
t0 = time.time()
try:
    model = CogACTInference(saved_model_path=CKPT, policy_setup="widowx_bridge",
                            action_model_type="DiT-B", cfg_scale=1.5)
except Exception as e:
    import traceback; traceback.print_exc()
    print(f"[ref] LOAD_FAILED: {e}", flush=True)
    sys.exit(3)
import torch
print(f"[ref] DiT dtype={next(model.vla.action_model.net.parameters()).dtype} "
      f"load={time.time()-t0:.0f}s GPU={torch.cuda.memory_allocated()/1e9:.1f}GB", flush=True)

sys.argv = ["main_inference.py"] + argv
args = get_args()
t1 = time.time()
sr_arr = maniskill2_evaluator(model, args)
sr = float(np.mean(sr_arr)); n = len(sr_arr)
print(f"[ref] RESULT_LINE eggplant REFERENCE SR={sr:.4f} n={n} ({time.time()-t1:.0f}s)", flush=True)
