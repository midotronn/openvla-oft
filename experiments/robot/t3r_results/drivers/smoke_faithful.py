"""smoke_faithful.py — verify faithful ADP gate + two-stage TeamVLA on coke_upright (fp32)."""
import os, sys, time
sys.path.insert(0, "/workspace/CogACT")
sys.path.append("/workspace/T3R"); sys.path.append("/workspace/SimplerEnv")
sys.path.append("/workspace/EfficientTAM"); sys.path.append("/workspace/openvla-oft")
import numpy as np
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
os.environ["DISPLAY"] = ""; os.environ["XLA_PYTHON_CLIENT_PREALLOCATE"] = "false"
from simpler_env.evaluation.argparse import get_args
from simpler_env.evaluation.maniskill2_evaluator import maniskill2_evaluator
from experiments.robot.cogact_simpler_policy import CogACTSimplerInference

CKPT = "/workspace/cogact_checkpoint/checkpoints/CogACT-Base.pt"
RGB = "./ManiSkill2_real2sim/data/real_inpainting"
COMMON = ["--policy-model","cogact","--policy-setup","google_robot","--ckpt-path",CKPT,
          "--robot","google_robot_static","--control-freq","3","--sim-freq","513",
          "--robot-init-rot-quat-center","0","0","0","1","--logging-dir","/workspace/smoke_faith_vid"]
def coke():
    return COMMON+["--max-episode-steps","80","--env-name","GraspSingleOpenedCokeCanInScene-v0",
        "--scene-name","google_pick_coke_can_1_v4","--rgb-overlay-path",f"{RGB}/google_coke_can_real_eval_1.png",
        "--robot-init-x","0.35","0.35","1","--robot-init-y","0.20","0.20","1",
        "--obj-init-x","-0.35","-0.12","5","--obj-init-y","-0.02","0.42","5",
        "--robot-init-rot-rpy-range","0","0","1","0","0","1","0","0","1",
        "--additional-env-build-kwargs","upright=True","urdf_version=None"]

KEYS=["T3R_METHOD","T3R_PRUNE","T3R_BIAS","ADP_KEEP","ADP_DYNAMIC","ADP_INIT","TEAM_TOPK","TEAM_TWOSTAGE","TEAM_U","TEAM_K"]
def clear():
    for k in KEYS: os.environ.pop(k,None)

print("[smoke] building CogACT fp32 ...", flush=True)
clear()
model = CogACTSimplerInference(saved_model_path=CKPT, policy_setup="google_robot",
                               cfg_scale=1.5, use_prune=False, use_bias=False)
vla = model.vla
PA0=vla.predict_action; VB0=vla.vlm.vision_backbone.forward
PJ0=vla.vlm.projector.forward; LM0=vla.vlm.llm_backbone.llm.model.forward
def restore():
    vla.predict_action=PA0; vla.vlm.vision_backbone.forward=VB0
    vla.vlm.projector.forward=PJ0; vla.vlm.llm_backbone.llm.model.forward=LM0
    model._t3r=None

from experiments.robot.baselines_cogact import BaselineController
for name, env in (
    ("adp_native_faithful", {"T3R_METHOD":"adp","ADP_KEEP":"0.75","ADP_DYNAMIC":"1","ADP_INIT":"0"}),
    ("team_twostage",       {"T3R_METHOD":"team","TEAM_TOPK":"80","TEAM_TWOSTAGE":"1","TEAM_U":"0.35","TEAM_K":"3"}),
):
    restore(); clear(); os.environ.update(env)
    ctrl = BaselineController(vla, method=env["T3R_METHOD"]); ctrl.attach(); model._t3r=ctrl
    sys.argv=["main_inference.py"]+coke(); args=get_args()
    t=time.time()
    try:
        sr=maniskill2_evaluator(model,args); sr=float(np.mean(sr))
        kh=np.array(ctrl.kept_hist) if ctrl.kept_hist else np.array([256])
        print(f"[smoke] {name}: SR={sr:.3f} kept mean={kh.mean():.0f} min={kh.min()} max={kh.max()} "
              f"prune={(1-kh.mean()/256)*100:.0f}% ({time.time()-t:.0f}s)", flush=True)
    except Exception as e:
        import traceback; traceback.print_exc(); print(f"[smoke] {name} FAIL: {str(e)[:150]}", flush=True)
    restore()
print("[smoke] DONE", flush=True)
