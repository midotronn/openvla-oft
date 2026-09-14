"""run_faithful_driver.py — FAITHFUL ADP + FAITHFUL two-stage TeamVLA, full Google Robot suite, fp32.
  adp_faithful  = action-aware DYNAMIC gate (net-displacement rule, initial=0, max_consec_prune=3),
                  ADP_KEEP=0.75 -> ~192 tok in coarse phases, 256 in fine phases (variable avg).
  team_faithful = TWO-STAGE (paper 2512.09927): stage-1 sim-seed + KxK spatial expand + u-context
                  hard PRUNE, stage-2 soft-bipartite MERGE survivors -> <=80 anchors.
Fresh results (faithful_results.json) so the older `native` data is preserved for old-vs-new.
No SAM / no IG backward -> no raytraced-drawer OOM. Resumable JSON keyed 'unit/method'."""
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
RESULTS = "/workspace/faithful_results.json"
RGB = "./ManiSkill2_real2sim/data/real_inpainting"
COMMON = ["--policy-model", "cogact", "--policy-setup", "google_robot",
          "--ckpt-path", CKPT, "--robot", "google_robot_static",
          "--control-freq", "3", "--sim-freq", "513",
          "--robot-init-rot-quat-center", "0", "0", "0", "1",
          "--logging-dir", "/workspace/faithful_videos"]
DRAWER_KW = ["station_name=mk_station_recolor", "light_mode=simple",
             "disable_bad_material=True", "urdf_version=None"]

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
    "--obj-variation-mode", "episode", "--obj-episode-range", "0", "60",
    "--robot-init-rot-rpy-range", "0", "0", "1", "0", "0", "1", "-0.09", "-0.09", "1",
    "--additional-env-build-kwargs", "urdf_version=None"]

def drawer_unit(env_name):
    return COMMON + ["--max-episode-steps", "113",
        "--env-name", env_name, "--scene-name", "dummy_drawer",
        "--robot-init-x", "0.65", "0.85", "3", "--robot-init-y", "-0.18", "0.22", "3",
        "--robot-init-rot-rpy-range", "0", "0", "1", "0", "0", "1", "0", "0", "1",
        "--obj-init-x-range", "0", "0", "1", "--obj-init-y-range", "0", "0", "1",
        "--rgb-overlay-path", f"{RGB}/open_drawer_b0.png", "--enable-raytracing",
        "--additional-env-build-kwargs", *DRAWER_KW]

PUT_IN_DRAWER = COMMON + ["--max-episode-steps", "200",
    "--env-name", "PlaceIntoClosedTopDrawerCustomInScene-v0", "--scene-name", "dummy_drawer",
    "--robot-init-x", "0.652", "0.652", "1", "--robot-init-y", "0.009", "0.009", "1",
    "--robot-init-rot-rpy-range", "0", "0", "1", "0", "0", "1", "0", "0", "1",
    "--obj-init-x-range", "-0.08", "-0.02", "3", "--obj-init-y-range", "-0.02", "0.08", "3",
    "--rgb-overlay-path", f"{RGB}/open_drawer_b0.png", "--enable-raytracing",
    "--additional-env-build-kwargs", *DRAWER_KW, "model_ids=baked_apple_v2"]

EVAL_UNITS = [
    ("coke_upright", coke_unit("upright")),
    ("coke_lr_switch", coke_unit("lr_switch")),
    ("coke_laid_vert", coke_unit("laid_vertically")),
    ("move_near", MOVE_NEAR),
    ("drawer_open_top", drawer_unit("OpenTopDrawerCustomInScene-v0")),
    ("drawer_open_mid", drawer_unit("OpenMiddleDrawerCustomInScene-v0")),
    ("drawer_open_bot", drawer_unit("OpenBottomDrawerCustomInScene-v0")),
    ("drawer_close_top", drawer_unit("CloseTopDrawerCustomInScene-v0")),
    ("drawer_close_mid", drawer_unit("CloseMiddleDrawerCustomInScene-v0")),
    ("drawer_close_bot", drawer_unit("CloseBottomDrawerCustomInScene-v0")),
    ("put_in_drawer", PUT_IN_DRAWER),
]
METHODS = ["adp_faithful", "team_faithful", "adp40"]
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
        # ADP's faithful QK-importance selection, gate OFF, forced to ~40% prune (keep 60% = 154 tok)
        # -> matched-budget rival to t3r_orig@40% (also 154 tok).
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
    print("[faith] building CogACT fp32 ...", flush=True)
    clear_env()
    model = CogACTSimplerInference(saved_model_path=CKPT, policy_setup="google_robot",
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
                print(f"[faith] SKIP {key} (={results[key]['sr']:.3f})", flush=True); continue
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
                print(f"[faith] DONE {key}: SR={sr:.3f} n={n} kept={kept:.0f} "
                      f"prune={results[key]['prune_pct']:.0f}% kh_len={len(kh)} "
                      f"({results[key]['secs']}s)", flush=True)
            except Exception as e:
                import traceback; traceback.print_exc()
                results[key]={"sr":None,"error":str(e)[:150]}; save_r(results)
                print(f"[faith] FAIL {key}: {e}", flush=True)
            restore()
    print("[faith] ALL DONE", flush=True)
    open("/workspace/faithful_DONE", "w").write("done")

if __name__ == "__main__":
    main()
