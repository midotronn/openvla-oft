"""smoke_fastv.py — verify FastV (K=1, keep=154) on coke_upright (fp32). Writes JSON result."""
import os, sys, time, json
sys.path.insert(0, "/workspace/CogACT")
sys.path.append("/workspace/T3R"); sys.path.append("/workspace/SimplerEnv")
sys.path.append("/workspace/EfficientTAM"); sys.path.append("/workspace/openvla-oft")
import numpy as np
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
os.environ["DISPLAY"] = ""; os.environ["XLA_PYTHON_CLIENT_PREALLOCATE"] = "false"
os.environ["FASTV_KEEP"] = "154"; os.environ["FASTV_K"] = "1"
from simpler_env.evaluation.argparse import get_args
from simpler_env.evaluation.maniskill2_evaluator import maniskill2_evaluator
from experiments.robot.cogact_simpler_policy import CogACTSimplerInference
from experiments.robot.fastv_cogact import FastVController

CKPT = "/workspace/cogact_checkpoint/checkpoints/CogACT-Base.pt"
RGB = "./ManiSkill2_real2sim/data/real_inpainting"
COMMON = ["--policy-model","cogact","--policy-setup","google_robot","--ckpt-path",CKPT,
          "--robot","google_robot_static","--control-freq","3","--sim-freq","513",
          "--robot-init-rot-quat-center","0","0","0","1","--logging-dir","/workspace/smoke_fastv_vid"]
argv = COMMON+["--max-episode-steps","80","--env-name","GraspSingleOpenedCokeCanInScene-v0",
    "--scene-name","google_pick_coke_can_1_v4","--rgb-overlay-path",f"{RGB}/google_coke_can_real_eval_1.png",
    "--robot-init-x","0.35","0.35","1","--robot-init-y","0.20","0.20","1",
    "--obj-init-x","-0.35","-0.12","5","--obj-init-y","-0.02","0.42","5",
    "--robot-init-rot-rpy-range","0","0","1","0","0","1","0","0","1",
    "--additional-env-build-kwargs","upright=True","urdf_version=None"]
print("[fastv-smoke] building CogACT fp32 ...", flush=True)
model = CogACTSimplerInference(saved_model_path=CKPT, policy_setup="google_robot",
                               cfg_scale=1.5, use_prune=False, use_bias=False)
ctrl = FastVController(model.vla, keep=154, agg_layer=1); ctrl.attach()
sys.argv = ["main_inference.py"]+argv; args = get_args()
t = time.time(); res = {}
try:
    sr = maniskill2_evaluator(model, args); sr = float(np.mean(sr))
    kh = np.array(ctrl.kept_hist) if ctrl.kept_hist else np.array([256])
    res = {"ok": True, "sr": sr, "kept_img_mean": float(kh.mean()), "n_prune_steps": len(ctrl.kept_hist),
           "prune_pct": float((1-kh.mean()/256)*100), "secs": time.time()-t}
except Exception as e:
    import traceback; res = {"ok": False, "err": str(e)[:300], "tb": traceback.format_exc()[-1200:]}
json.dump(res, open("/workspace/smoke_fastv_result.json","w"), indent=2)
print("[fastv-smoke] RESULT " + json.dumps({k:v for k,v in res.items() if k!='tb'}), flush=True)
if not res.get("ok"): print(res.get("tb",""), flush=True)
