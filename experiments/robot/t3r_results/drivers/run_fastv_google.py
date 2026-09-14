"""run_fastv_google.py — FastV (K=1, keep=154, 40% prune) on the 10 informative Google Robot units
(put_in_drawer excluded: 0 for all methods incl base). fp32. Matched-budget vs t3r_orig@40% & adp40.
Resumable JSON keyed 'unit'."""
import os, sys, json, time
sys.path.insert(0, "/workspace/CogACT")
sys.path.append("/workspace/T3R"); sys.path.append("/workspace/SimplerEnv")
sys.path.append("/workspace/EfficientTAM"); sys.path.append("/workspace/openvla-oft")
import numpy as np
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
os.environ["DISPLAY"] = ""; os.environ["XLA_PYTHON_CLIENT_PREALLOCATE"] = "false"
os.environ["FASTV_KEEP"] = os.environ.get("FASTV_KEEP", "154")
os.environ["FASTV_K"] = os.environ.get("FASTV_K", "1")
from simpler_env.evaluation.argparse import get_args
from simpler_env.evaluation.maniskill2_evaluator import maniskill2_evaluator
from experiments.robot.cogact_simpler_policy import CogACTSimplerInference
from experiments.robot.fastv_cogact import FastVController

CKPT = "/workspace/cogact_checkpoint/checkpoints/CogACT-Base.pt"
RESULTS = "/workspace/fastv_google_results.json"
RGB = "./ManiSkill2_real2sim/data/real_inpainting"
COMMON = ["--policy-model","cogact","--policy-setup","google_robot","--ckpt-path",CKPT,
          "--robot","google_robot_static","--control-freq","3","--sim-freq","513",
          "--robot-init-rot-quat-center","0","0","0","1","--logging-dir","/workspace/fastv_google_videos"]
DRAWER_KW = ["station_name=mk_station_recolor","light_mode=simple","disable_bad_material=True","urdf_version=None"]

def coke_unit(orient):
    return COMMON+["--max-episode-steps","80","--env-name","GraspSingleOpenedCokeCanInScene-v0",
        "--scene-name","google_pick_coke_can_1_v4","--rgb-overlay-path",f"{RGB}/google_coke_can_real_eval_1.png",
        "--robot-init-x","0.35","0.35","1","--robot-init-y","0.20","0.20","1",
        "--obj-init-x","-0.35","-0.12","5","--obj-init-y","-0.02","0.42","5",
        "--robot-init-rot-rpy-range","0","0","1","0","0","1","0","0","1",
        "--additional-env-build-kwargs",f"{orient}=True","urdf_version=None"]

MOVE_NEAR = COMMON+["--max-episode-steps","80","--env-name","MoveNearGoogleBakedTexInScene-v0",
    "--scene-name","google_pick_coke_can_1_v4","--rgb-overlay-path",f"{RGB}/google_move_near_real_eval_1.png",
    "--robot-init-x","0.35","0.35","1","--robot-init-y","0.21","0.21","1",
    "--obj-variation-mode","episode","--obj-episode-range","0","60",
    "--robot-init-rot-rpy-range","0","0","1","0","0","1","-0.09","-0.09","1",
    "--additional-env-build-kwargs","urdf_version=None"]

def drawer_unit(env_name):
    return COMMON+["--max-episode-steps","113","--env-name",env_name,"--scene-name","dummy_drawer",
        "--robot-init-x","0.65","0.85","3","--robot-init-y","-0.18","0.22","3",
        "--robot-init-rot-rpy-range","0","0","1","0","0","1","0","0","1",
        "--obj-init-x-range","0","0","1","--obj-init-y-range","0","0","1",
        "--rgb-overlay-path",f"{RGB}/open_drawer_b0.png","--enable-raytracing",
        "--additional-env-build-kwargs",*DRAWER_KW]

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
]

def load_r(): return json.load(open(RESULTS)) if os.path.exists(RESULTS) else {}
def save_r(r):
    json.dump(r, open(RESULTS+".tmp","w"), indent=2); os.replace(RESULTS+".tmp", RESULTS)
def build_args(argv):
    old=sys.argv; sys.argv=["main_inference.py"]+argv
    try: return get_args()
    finally: sys.argv=old

def main():
    results = load_r()
    print("[fvg] building CogACT fp32 + FastV K=1 keep=154 ...", flush=True)
    model = CogACTSimplerInference(saved_model_path=CKPT, policy_setup="google_robot",
                                   cfg_scale=1.5, use_prune=False, use_bias=False)
    ctrl = FastVController(model.vla, keep=int(os.environ["FASTV_KEEP"]),
                           agg_layer=int(os.environ["FASTV_K"])); ctrl.attach()
    for unit_id, argv in EVAL_UNITS:
        if unit_id in results and results[unit_id].get("sr") is not None:
            print(f"[fvg] SKIP {unit_id} (={results[unit_id]['sr']:.3f})", flush=True); continue
        ctrl.kept_hist = []
        args = build_args(argv); t0=time.time()
        try:
            sr_arr = maniskill2_evaluator(model, args)
            sr=float(np.mean(sr_arr)); n=len(sr_arr)
            kh = ctrl.kept_hist or [256]
            kept=float(np.mean(kh[-400:]))
            results[unit_id]={"sr":sr,"n":n,"kept_img":round(kept,1),
                              "prune_pct":round((1-kept/256)*100,1),"secs":round(time.time()-t0,1)}
            save_r(results)
            print(f"[fvg] DONE {unit_id}: SR={sr:.3f} n={n} kept_img={kept:.0f} "
                  f"prune={results[unit_id]['prune_pct']:.0f}% ({results[unit_id]['secs']}s)", flush=True)
        except Exception as e:
            import traceback; traceback.print_exc()
            results[unit_id]={"sr":None,"error":str(e)[:150]}; save_r(results)
            print(f"[fvg] FAIL {unit_id}: {e}", flush=True)
    print("[fvg] ALL DONE", flush=True)
    open("/workspace/fastv_google_DONE","w").write("done")

if __name__ == "__main__":
    main()
