"""run_bridge_faithful.py — FAITHFUL baselines on the 4 Bridge (WidowX) tasks, fp32:
  adp_faithful  = action-aware DYNAMIC gate, ADP_KEEP=0.75  (~native, barely prunes)
  adp40         = ADP QK-importance, gate OFF, keep 0.60 -> ~154 tok (matched to t3r_orig@40%)
  team_faithful = two-stage TeamVLA, TEAM_TOPK=80 -> <=80 anchors
Completes the 4-method x both-suite comparison the user asked for. vs fp32 base in `bridge` and
t3r_orig@40% in `bridge_k60`. Resumable JSON keyed 'unit/method'."""
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
RESULTS = "/workspace/bridge_faithful_results.json"
RGB = "./ManiSkill2_real2sim/data/real_inpainting"

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
            "--logging-dir", "/workspace/bridge_faithful_videos"]

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
METHODS = ["adp_faithful", "adp40", "team_faithful"]
KEYS = ["T3R_METHOD","T3R_PRUNE","T3R_KEEP","T3R_BIAS","ADP_KEEP","ADP_DYNAMIC","ADP_INIT",
        "TEAM_TOPK","TEAM_TWOSTAGE","TEAM_U","TEAM_K"]

def clear_env():
    for k in KEYS: os.environ.pop(k, None)

def make_controller(method, vla):
    clear_env()
    from experiments.robot.baselines_cogact import BaselineController
    if method == "adp_faithful":
        os.environ.update({"T3R_METHOD":"adp","ADP_KEEP":"0.75","ADP_DYNAMIC":"1","ADP_INIT":"0"})
        c = BaselineController(vla, method="adp"); c.attach(); return c
    if method == "adp40":
        os.environ.update({"T3R_METHOD":"adp","ADP_KEEP":"0.60","ADP_DYNAMIC":"0"})
        c = BaselineController(vla, method="adp"); c.attach(); return c
    if method == "team_faithful":
        os.environ.update({"T3R_METHOD":"team","TEAM_TOPK":"80","TEAM_TWOSTAGE":"1",
                           "TEAM_U":"0.35","TEAM_K":"3"})
        c = BaselineController(vla, method="team"); c.attach(); return c
    raise ValueError(method)

def load_r(): return json.load(open(RESULTS)) if os.path.exists(RESULTS) else {}
def save_r(r):
    json.dump(r, open(RESULTS+".tmp","w"), indent=2); os.replace(RESULTS+".tmp", RESULTS)
def build_args(argv):
    old=sys.argv; sys.argv=["main_inference.py"]+argv
    try: return get_args()
    finally: sys.argv=old

def main():
    results = load_r()
    print("[bfaith] building CogACT fp32 (widowx_bridge) ...", flush=True)
    clear_env()
    model = CogACTSimplerInference(saved_model_path=CKPT, policy_setup="widowx_bridge",
                                   cfg_scale=1.5, use_prune=False, use_bias=False)
    vla = model.vla
    PA0=vla.predict_action; VB0=vla.vlm.vision_backbone.forward
    PJ0=vla.vlm.projector.forward; LM0=vla.vlm.llm_backbone.llm.model.forward
    def restore():
        vla.predict_action=PA0; vla.vlm.vision_backbone.forward=VB0
        vla.vlm.projector.forward=PJ0; vla.vlm.llm_backbone.llm.model.forward=LM0
        model._t3r=None
    for unit_id, argv in EVAL_UNITS:
        for method in METHODS:
            key = f"{unit_id}/{method}"
            if key in results and results[key].get("sr") is not None:
                print(f"[bfaith] SKIP {key} (={results[key]['sr']:.3f})", flush=True); continue
            restore()
            ctrl = make_controller(method, vla); model._t3r = ctrl
            args = build_args(argv)
            t0=time.time()
            try:
                sr_arr = maniskill2_evaluator(model, args)
                sr=float(np.mean(sr_arr)); n=len(sr_arr)
                kh = getattr(ctrl,"kept_hist",None) or []
                kept=float(np.mean(kh[-400:])) if kh else 256.0
                results[key]={"sr":sr,"n":n,"kept":round(kept,1),
                              "prune_pct":round((1-kept/256)*100,1),
                              "kh_len":len(kh),"secs":round(time.time()-t0,1)}
                save_r(results)
                print(f"[bfaith] DONE {key}: SR={sr:.3f} n={n} kept={kept:.0f} "
                      f"prune={results[key]['prune_pct']:.0f}% ({results[key]['secs']}s)", flush=True)
            except Exception as e:
                import traceback; traceback.print_exc()
                results[key]={"sr":None,"error":str(e)[:150]}; save_r(results)
                print(f"[bfaith] FAIL {key}: {e}", flush=True)
            restore()
    print("[bfaith] ALL DONE", flush=True)
    open("/workspace/bridge_faithful_DONE", "w").write("done")

if __name__ == "__main__":
    main()
