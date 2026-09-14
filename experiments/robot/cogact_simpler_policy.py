"""
cogact_simpler_policy.py — CogACT policy for SimplerEnv that loads via cogact_loader
(local LLaMA build, no gated meta-llama download), with optional T3R prune+bias.
Native step/ensemble/gripper logic mirrors sim_cogact/cogact_policy.py.
"""
import os, sys
from collections import deque
from typing import Optional
import numpy as np
import cv2 as cv
from PIL import Image
from transforms3d.euler import euler2axangle

sys.path.insert(0, "/workspace/CogACT")
sys.path.insert(0, "/workspace/T3R")
from sim_cogact.adaptive_ensemble import AdaptiveEnsembler
from experiments.robot.cogact_loader import load_cogact_for_libero


class CogACTSimplerInference:
    def __init__(self, saved_model_path, policy_setup="google_robot", action_scale=1.0,
                 action_model_type="DiT-B", cfg_scale=1.5, use_ddim=True, num_ddim_steps=10,
                 image_size=(224, 224), action_ensemble=True, adaptive_ensemble_alpha=0.1,
                 use_prune=False, use_bias=False):
        os.environ["TOKENIZERS_PARALLELISM"] = "false"
        if policy_setup == "widowx_bridge":
            self.unnorm_key = "bridge_orig"; aeh = 7; self.sticky_gripper_num_repeat = 1
        else:
            self.unnorm_key = "fractal20220817_data"; aeh = 2; self.sticky_gripper_num_repeat = 10
        self.policy_setup = policy_setup
        self.vla, _, _ = load_cogact_for_libero(saved_model_path, freeze_vlm=True, device="cuda")
        self.vla.eval()
        self.cfg_scale = cfg_scale; self.use_ddim = use_ddim; self.num_ddim_steps = num_ddim_steps
        self.image_size = list(image_size); self.action_scale = action_scale
        self.action_ensemble = action_ensemble; self.action_ensemble_horizon = aeh
        self.adaptive_ensemble_alpha = adaptive_ensemble_alpha
        self.sticky_action_is_on = False; self.gripper_action_repeat = 0
        self.sticky_gripper_action = 0.0; self.previous_gripper_action = None
        self.task_description = None
        self.action_ensembler = AdaptiveEnsembler(aeh, adaptive_ensemble_alpha) if action_ensemble else None
        self._t3r = None
        import os as _o
        method = _o.environ.get("T3R_METHOD", "siglip").lower()
        if method in ("adp", "team"):
            # Baseline comparison condition (ADP or TeamVLA) — native token-reduction,
            # NOT a fixed keep ratio (that is applicable to our SigLIP-SAM method only).
            from experiments.robot.baselines_cogact import BaselineController
            self._t3r = BaselineController(self.vla, method=method)
            self._t3r.attach()
            print(f"[BASELINE] attached method={method} "
                  f"(adp_keep={self._t3r.adp_keep_ratio} dynamic={self._t3r.adp_dynamic} "
                  f"team_topk={self._t3r.team_topk})")
        elif use_prune or use_bias:
            kr = float(_o.environ.get("T3R_KEEP", "0.15"))
            st = float(_o.environ.get("T3R_STRENGTH", "2.0"))
            from experiments.robot.t3r_cogact import T3RController
            self._t3r = T3RController(self.vla, use_prune=use_prune, use_bias=use_bias,
                                      keep_ratio=kr, bias_strength=st)
            self._t3r.attach()
            print(f"[T3R] attached prune={use_prune} bias={use_bias} keep={kr} strength={st}")

    def reset(self, task_description):
        self.task_description = task_description
        if self.action_ensemble:
            self.action_ensembler.reset()
        self.sticky_action_is_on = False; self.gripper_action_repeat = 0
        self.sticky_gripper_action = 0.0; self.previous_gripper_action = None
        if self._t3r:
            self._t3r.reset_episode(task_description)
        self._log_n = 0; self._log_grip = []

    def step(self, image, task_description=None, *args, **kwargs):
        if task_description is not None and task_description != self.task_description:
            self.reset(task_description)
        assert image.dtype == np.uint8
        pil = Image.fromarray(image)
        raw_actions, _ = self.vla.predict_action(image=pil, instruction=self.task_description,
                                                 unnorm_key=self.unnorm_key, cfg_scale=self.cfg_scale,
                                                 use_ddim=self.use_ddim, num_ddim_steps=self.num_ddim_steps)
        if self.action_ensemble:
            raw_actions = self.action_ensembler.ensemble_action(raw_actions)[None]
        raw_action = {"world_vector": np.array(raw_actions[0, :3]),
                      "rotation_delta": np.array(raw_actions[0, 3:6]),
                      "open_gripper": np.array(raw_actions[0, 6:7])}
        action = {}
        action["world_vector"] = raw_action["world_vector"] * self.action_scale
        roll, pitch, yaw = np.asarray(raw_action["rotation_delta"], dtype=np.float64)
        axes, angles = euler2axangle(roll, pitch, yaw)
        action["rot_axangle"] = axes * angles * self.action_scale
        if self.policy_setup == "google_robot":
            current = raw_action["open_gripper"]
            if self.previous_gripper_action is None:
                rel = np.array([0]); self.previous_gripper_action = current
            else:
                rel = self.previous_gripper_action - current
            if np.abs(rel) > 0.5 and (not self.sticky_action_is_on):
                self.sticky_action_is_on = True; self.sticky_gripper_action = rel
                self.previous_gripper_action = current
            if self.sticky_action_is_on:
                self.gripper_action_repeat += 1; rel = self.sticky_gripper_action
            if self.gripper_action_repeat == self.sticky_gripper_num_repeat:
                self.sticky_action_is_on = False; self.gripper_action_repeat = 0; self.sticky_gripper_action = 0.0
            action["gripper"] = rel
        else:
            action["gripper"] = 2.0 * (raw_action["open_gripper"] > 0.5) - 1.0
        action["terminate_episode"] = np.array([0.0])
        if os.environ.get("LOGACT") == "1":
            self._log_n = getattr(self, "_log_n", 0) + 1
            g = float(np.asarray(action["gripper"]).reshape(-1)[0])
            og = float(np.asarray(raw_action["open_gripper"]).reshape(-1)[0])
            wz = float(action["world_vector"][2])
            self._log_grip = getattr(self, "_log_grip", [])
            self._log_grip.append(g)
            if self._log_n % 15 == 0:
                arr = np.array(self._log_grip)
                print(f"[LOGACT] step={self._log_n} wz={wz:+.3f} og={og:+.2f} grip={g:+.2f} "
                      f"gmin={arr.min():+.2f} gmax={arr.max():+.2f} nclose={(arr<0).sum()} "
                      f"nopen={(arr>0).sum()}", flush=True)
        return raw_action, action

    def visualize_epoch(self, *a, **k):
        pass
