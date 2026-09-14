"""smoke_team.py — verify two-stage TeamVLA on coke_upright (fp32). Writes JSON result."""
import os, sys, time, json
sys.path.insert(0, "/workspace/CogACT")
sys.path.append("/workspace/T3R"); sys.path.append("/workspace/SimplerEnv")
sys.path.append("/workspace/EfficientTAM"); sys.path.append("/workspace/openvla-oft")
import numpy as np
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
os.environ["DISPLAY"] = ""; os.environ["XLA_PYTHON_CLIENT_PREALLOCATE"] = "false"
for k in ("T3R_METHOD","T3R_PRUNE","T3R_BIAS","ADP_KEEP","ADP_DYNAMIC","ADP_INIT","TEAM_TOPK","TEAM_TWOSTAGE","TEAM_U","TEAM_K"):
    os.environ.pop(k, None)
os.environ.update({"T3R_METHOD":"team","TEAM_TOPK":"80","TEAM_TWOSTAGE":"1","TEAM_U":"0.35","TEAM_K":"3"})
from simpler_env.evaluation.argparse import get_args
from simpler_env.evaluation.maniskill2_evaluator import maniskill2_evaluator
from experiments.robot.cogact_simpler_policy import CogACTSimplerInference
from experiments.robot.baselines_cogact import BaselineController

CKPT = "/workspace/cogact_checkpoint/checkpoints/CogACT-Base.pt"
RGB = "./ManiSkill2_real2sim/data/real_inpainting"
COMMON = ["--policy-model","cogact","--policy-setup","google_robot","--ckpt-path",CKPT,
          "--robot","google_robot_static","--control-freq","3","--sim-freq","513",
          "--robot-init-rot-quat-center","0","0","0","1","--logging-dir","/workspace/smoke_team_vid"]
argv = COMMON+["--max-episode-steps","80","--env-name","GraspSingleOpenedCokeCanInScene-v0",
    "--scene-name","google_pick_coke_can_1_v4","--rgb-overlay-path",f"{RGB}/google_coke_can_real_eval_1.png",
    "--robot-init-x","0.35","0.35","1","--robot-init-y","0.20","0.20","1",
    "--obj-init-x","-0.35","-0.12","5","--obj-init-y","-0.02","0.42","5",
    "--robot-init-rot-rpy-range","0","0","1","0","0","1","0","0","1",
    "--additional-env-build-kwargs","upright=True","urdf_version=None"]
print("[team] building CogACT fp32 ...", flush=True)
model = CogACTSimplerInference(saved_model_path=CKPT, policy_setup="google_robot",
                               cfg_scale=1.5, use_prune=False, use_bias=False)
ctrl = BaselineController(model.vla, method="team"); ctrl.attach(); model._t3r = ctrl
sys.argv = ["main_inference.py"]+argv; args = get_args()
t = time.time()
res = {}
try:
    sr = maniskill2_evaluator(model, args); sr = float(np.mean(sr))
    kh = np.array(ctrl.kept_hist) if ctrl.kept_hist else np.array([256])
    res = {"ok": True, "sr": sr, "kept_mean": float(kh.mean()), "kept_min": int(kh.min()),
           "kept_max": int(kh.max()), "prune_pct": float((1-kh.mean()/256)*100), "secs": time.time()-t}
except Exception as e:
    import traceback; tb = traceback.format_exc()
    res = {"ok": False, "err": str(e)[:300], "tb": tb[-800:]}
json.dump(res, open("/workspace/smoke_team_result.json","w"), indent=2)
print("[team] RESULT " + json.dumps(res), flush=True)
