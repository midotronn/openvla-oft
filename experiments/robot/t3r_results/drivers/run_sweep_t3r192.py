"""run_sweep_t3r192.py — LIGHTER-PRUNE SWEEP point: t3r_orig at keep=0.75 (~192 tok, 25% prune).
Tests whether T3R closes the gap vs adp/fastv when pruning is gentler than the 40% (154-tok) point.
SWEEP_SUITE env selects 'google' (10 informative units) or 'bridge' (4 tasks). fp32, prune+IG bias
(same config as the k60 @keep=0.60 run). Resumable JSON keyed 'unit'."""
import os, sys, json, time
sys.path.insert(0, "/workspace/CogACT")
sys.path.append("/workspace/T3R"); sys.path.append("/workspace/SimplerEnv")
sys.path.append("/workspace/EfficientTAM"); sys.path.append("/workspace/openvla-oft")
import numpy as np
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
os.environ["DISPLAY"] = ""; os.environ["XLA_PYTHON_CLIENT_PREALLOCATE"] = "false"
from simpler_env.evaluation.argparse import get_args
from simpler_env.evaluation.maniskill2_evaluator import maniskill2_evaluator
from experiments.robot.cogact_simpler_policy import CogACTSimplerInference

SUITE = os.environ.get("SWEEP_SUITE", "google")
KEEP = "0.75"
CKPT = "/workspace/cogact_checkpoint/checkpoints/CogACT-Base.pt"
RESULTS = f"/workspace/sweep_t3r192_{SUITE}_results.json"
RGB = "./ManiSkill2_real2sim/data/real_inpainting"
GCOMMON = ["--policy-model","cogact","--policy-setup","google_robot","--ckpt-path",CKPT,
           "--robot","google_robot_static","--control-freq","3","--sim-freq","513",
           "--robot-init-rot-quat-center","0","0","0","1","--logging-dir","/workspace/sweep_t3r192_vid"]
DKW = ["station_name=mk_station_recolor","light_mode=simple","disable_bad_material=True","urdf_version=None"]

def coke_unit(o):
    return GCOMMON+["--max-episode-steps","80","--env-name","GraspSingleOpenedCokeCanInScene-v0",
        "--scene-name","google_pick_coke_can_1_v4","--rgb-overlay-path",f"{RGB}/google_coke_can_real_eval_1.png",
        "--robot-init-x","0.35","0.35","1","--robot-init-y","0.20","0.20","1",
        "--obj-init-x","-0.35","-0.12","5","--obj-init-y","-0.02","0.42","5",
        "--robot-init-rot-rpy-range","0","0","1","0","0","1","0","0","1",
        "--additional-env-build-kwargs",f"{o}=True","urdf_version=None"]
MOVE_NEAR = GCOMMON+["--max-episode-steps","80","--env-name","MoveNearGoogleBakedTexInScene-v0",
    "--scene-name","google_pick_coke_can_1_v4","--rgb-overlay-path",f"{RGB}/google_move_near_real_eval_1.png",
    "--robot-init-x","0.35","0.35","1","--robot-init-y","0.21","0.21","1",
    "--obj-variation-mode","episode","--obj-episode-range","0","60",
    "--robot-init-rot-rpy-range","0","0","1","0","0","1","-0.09","-0.09","1",
    "--additional-env-build-kwargs","urdf_version=None"]
def drawer_unit(e):
    return GCOMMON+["--max-episode-steps","113","--env-name",e,"--scene-name","dummy_drawer",
        "--robot-init-x","0.65","0.85","3","--robot-init-y","-0.18","0.22","3",
        "--robot-init-rot-rpy-range","0","0","1","0","0","1","0","0","1",
        "--obj-init-x-range","0","0","1","--obj-init-y-range","0","0","1",
        "--rgb-overlay-path",f"{RGB}/open_drawer_b0.png","--enable-raytracing","--additional-env-build-kwargs",*DKW]
def bridge_unit(env_name, robot, scene, overlay, rx, ry):
    return ["--policy-model","cogact","--policy-setup","widowx_bridge","--ckpt-path",CKPT,"--robot",robot,
            "--control-freq","5","--sim-freq","500","--max-episode-steps","120",
            "--env-name",env_name,"--scene-name",scene,"--rgb-overlay-path",f"{RGB}/{overlay}",
            "--robot-init-x",rx,rx,"1","--robot-init-y",ry,ry,"1",
            "--obj-variation-mode","episode","--obj-episode-range","0","24",
            "--robot-init-rot-quat-center","0","0","0","1",
            "--robot-init-rot-rpy-range","0","0","1","0","0","1","0","0","1",
            "--logging-dir","/workspace/sweep_t3r192_vid"]

GOOGLE = [("coke_upright",coke_unit("upright")),("coke_lr_switch",coke_unit("lr_switch")),
    ("coke_laid_vert",coke_unit("laid_vertically")),("move_near",MOVE_NEAR),
    ("drawer_open_top",drawer_unit("OpenTopDrawerCustomInScene-v0")),
    ("drawer_open_mid",drawer_unit("OpenMiddleDrawerCustomInScene-v0")),
    ("drawer_open_bot",drawer_unit("OpenBottomDrawerCustomInScene-v0")),
    ("drawer_close_top",drawer_unit("CloseTopDrawerCustomInScene-v0")),
    ("drawer_close_mid",drawer_unit("CloseMiddleDrawerCustomInScene-v0")),
    ("drawer_close_bot",drawer_unit("CloseBottomDrawerCustomInScene-v0"))]
BRIDGE = [("stack_cube",bridge_unit("StackGreenCubeOnYellowCubeBakedTexInScene-v0","widowx","bridge_table_1_v1","bridge_real_eval_1.png","0.147","0.028")),
    ("carrot_on_plate",bridge_unit("PutCarrotOnPlateInScene-v0","widowx","bridge_table_1_v1","bridge_real_eval_1.png","0.147","0.028")),
    ("spoon_on_towel",bridge_unit("PutSpoonOnTableClothInScene-v0","widowx","bridge_table_1_v1","bridge_real_eval_1.png","0.147","0.028")),
    ("eggplant_in_basket",bridge_unit("PutEggplantInBasketScene-v0","widowx_sink_camera_setup","bridge_table_1_v2","bridge_sink.png","0.127","0.06"))]
UNITS = GOOGLE if SUITE == "google" else BRIDGE
POLICY = "google_robot" if SUITE == "google" else "widowx_bridge"

def load_r(): return json.load(open(RESULTS)) if os.path.exists(RESULTS) else {}
def save_r(r): json.dump(r, open(RESULTS+".tmp","w"), indent=2); os.replace(RESULTS+".tmp", RESULTS)

def main():
    results = load_r()
    print(f"[t3r192-{SUITE}] building CogACT fp32 + t3r keep={KEEP} ...", flush=True)
    os.environ.update({"T3R_PRUNE":"1","T3R_KEEP":KEEP,"T3R_BIAS":"1","T3R_QWIN":"1",
                       "T3R_STRENGTH":"1.0","T3R_VSCALE":"0.5"})
    model = CogACTSimplerInference(saved_model_path=CKPT, policy_setup=POLICY,
                                   cfg_scale=1.5, use_prune=True, use_bias=True)
    ctrl = model._t3r
    for unit_id, argv in UNITS:
        if unit_id in results and results[unit_id].get("sr") is not None:
            print(f"[t3r192-{SUITE}] SKIP {unit_id} (={results[unit_id]['sr']:.3f})", flush=True); continue
        if ctrl is not None: ctrl.kept_hist = []
        sys.argv = ["main_inference.py"]+argv; args = get_args(); t0=time.time()
        try:
            sr_arr = maniskill2_evaluator(model, args); sr=float(np.mean(sr_arr)); n=len(sr_arr)
            kept = float(np.mean(ctrl.kept_hist[-400:])) if getattr(ctrl,"kept_hist",None) else 256.0
            results[unit_id]={"sr":sr,"n":n,"kept":round(kept,1),
                              "prune_pct":round((1-kept/256)*100,1),"secs":round(time.time()-t0,1)}
            save_r(results)
            print(f"[t3r192-{SUITE}] DONE {unit_id}: SR={sr:.3f} n={n} kept={kept:.0f} "
                  f"prune={results[unit_id]['prune_pct']:.0f}% ({results[unit_id]['secs']}s)", flush=True)
        except Exception as e:
            import traceback; traceback.print_exc()
            results[unit_id]={"sr":None,"error":str(e)[:150]}; save_r(results)
            print(f"[t3r192-{SUITE}] FAIL {unit_id}: {e}", flush=True)
    print(f"[t3r192-{SUITE}] ALL DONE", flush=True)
    open(f"/workspace/sweep_t3r192_{SUITE}_DONE","w").write("done")

if __name__ == "__main__":
    main()
