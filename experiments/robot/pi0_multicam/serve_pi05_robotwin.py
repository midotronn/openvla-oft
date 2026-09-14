"""serve_pi05_robotwin.py — openpi pi0.5 (pytorch) policy server for RoboTwin, over websocket.
Builds the validated pi05_robotwin policy (motus ckpt, 3-cam aloha, delta+adapt_to_pi) and serves it.
T3R / baseline hooks will be attached to the underlying model here in Phase 2.
Run in the openpi uv env:  uv run python serve_pi05_robotwin.py --port 8000
"""
import argparse, logging
logging.basicConfig(level=logging.INFO)

from openpi.training import config as _config
from openpi.policies import policy_config as _policy_config
from openpi.models import pi0_config
from openpi.serving import websocket_policy_server

CKPT = "/workspace/pi05_robotwin2_ckpt"

def build_policy():
    cfg = _config.TrainConfig(
        name="pi05_robotwin",
        model=pi0_config.Pi0Config(pi05=True, action_horizon=32, pytorch_compile_mode=None),
        data=_config.LeRobotAlohaDataConfig(
            assets=_config.AssetsConfig(asset_id="pi0.5_clean_randomize_joint_training"),
        ),
    )
    return _policy_config.create_trained_policy(cfg, CKPT, pytorch_device="cuda")

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--port", type=int, default=8000)
    args = ap.parse_args()
    print("[serve] building pi0.5 RoboTwin policy ...", flush=True)
    policy = build_policy()
    try:
        import t3r_pi0
        t3r_pi0.install(policy._model)
        policy._sample_actions = policy._model.sample_actions  # re-point (patched after capture)
        # stash the raw instruction on the model each infer (standalone SigLIP-SAM pruning needs the text)
        _orig_infer = policy.infer
        def _infer_with_prompt(obs, *a, **k):
            try:
                if isinstance(obs, dict):
                    p = obs.get("prompt") or obs.get("instruction") or obs.get("task")
                    if p is not None:
                        policy._model._t3r_prompt = p
            except Exception:
                pass
            return _orig_infer(obs, *a, **k)
        policy.infer = _infer_with_prompt
        print("[serve] t3r_pi0 installed + _sample_actions re-pointed + prompt hook", flush=True)
    except Exception as e:
        import traceback; traceback.print_exc()
        print(f"[serve] t3r_pi0 install skipped: {e}", flush=True)
    print(f"[serve] policy ready. serving on 0.0.0.0:{args.port}", flush=True)
    server = websocket_policy_server.WebsocketPolicyServer(policy, host="0.0.0.0", port=args.port)
    server.serve_forever()

if __name__ == "__main__":
    main()
