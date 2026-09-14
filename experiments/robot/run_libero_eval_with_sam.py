"""
Example script for running LIBERO evaluation with SigLIP similarity-guided SAM/EfficientTAM.

This script demonstrates how to integrate the SigLIPGuidedSAM module into OpenVLA
for mask-based token filtering without requiring VLM-predicted bounding boxes.

Supports two backends:
- EfficientTAM (recommended): Faster, lightweight
- SAM: Original Segment Anything Model

Usage:
    # With EfficientTAM
    python run_libero_eval_with_sam.py \
        --model_path openvla/openvla-7b \
        --task_suite_name libero_spatial \
        --center_crop True \
        --use_sam True \
        --sam_backend efficienttam \
        --sam_checkpoint EfficientTAM/checkpoints \
        --sam_model_type efficienttam_s
    
    # With original SAM
    python run_libero_eval_with_sam.py \
        --model_path openvla/openvla-7b \
        --task_suite_name libero_spatial \
        --use_sam True \
        --sam_backend sam \
        --sam_checkpoint /path/to/sam_vit_h_4b8939.pth \
        --sam_model_type vit_h

SigLIP Similarity-Guided Approach:
    The new approach implements the architecture shown in the diagram:
    1. Text instruction -> SigLIP Text Encoder -> Text embeddings
    2. Image -> SigLIP Image Encoder -> Image patch embeddings  
    3. Compute cosine similarity between text tokens and image patches
    4. Select Top-K most similar patches as positive point prompts
    5. Use EfficientTAM/SAM to generate segmentation mask
    6. Convert mask to patch selection for token filtering in VLA
    
    This approach provides better semantic alignment between the instruction
    and the visual regions, potentially improving task performance.
"""

import argparse
import collections
import json
import os
import sys
import time
from pathlib import Path
from types import SimpleNamespace
from typing import Dict, List, Optional, Tuple

import imageio
import numpy as np
import torch
import torch.nn.functional as F
import yaml
from PIL import Image

# Monkey-patch torch.load for PyTorch 2.6+ compatibility with LIBERO state files
_original_torch_load = torch.load
def _safe_torch_load(*args, **kwargs):
    kwargs.setdefault("weights_only", False)
    return _original_torch_load(*args, **kwargs)
torch.load = _safe_torch_load

from libero.libero import benchmark

# Append current directory so that interpreter can find experiments.robot
sys.path.append("../..")
from experiments.robot.libero.libero_utils import (
    get_libero_env,
    get_libero_image,
    get_libero_wrist_image,
    quat2axisangle,
    save_rollout_video,
)
from experiments.robot.openvla_utils import (
    get_vla_action,
    get_vla,
    get_processor,
    get_action_head,
    get_proprio_projector,
    get_noisy_action_projector,
    resize_image_for_policy,
    center_crop_image,
)
from experiments.robot.robot_utils import (
    get_image_resize_size,
    set_seed_everywhere,
)
from prismatic.models.siglip_guided_sam import (
    create_siglip_guided_sam,
    visualize_similarity_heatmap,
    visualize_topk_patches,
    visualize_mask_with_points,
)
from prismatic.models.attention_bias import (
    SaliencyInfo,
    install_layer_hooks,
    remove_layer_hooks,
    install_biased_sdpa,
    uninstall_biased_sdpa,
    bias_context,
    compute_saliency_for_episode,
)

TASK_MAX_STEPS = {
    "libero_spatial": 220,  # longest training demo has 193 steps
    "libero_object": 280,  # longest training demo has 254 steps
    "libero_goal": 300,  # longest training demo has 270 steps
    "libero_10": 520,  # longest training demo has 505 steps
    "libero_90": 400,  # longest training demo has 373 steps
}


# ============================================================
# OOD Perturbation: Object Position Displacement
# ============================================================
# Adds small Gaussian noise to object positions in the MuJoCo
# initial state to test robustness to out-of-distribution scenarios.
# The MuJoCo state is a flattened array: [qpos, qvel, ...]
# Free bodies have 7 DOFs in qpos: [x, y, z, qw, qx, qy, qz]

def _get_object_qpos_indices(env) -> dict:
    """Identify which qpos indices correspond to free-body object positions."""
    object_indices = {}
    try:
        sim = env.sim
        model = sim.model
        for i in range(model.njnt):
            joint_name = model.joint_id2name(i)
            joint_type = model.jnt_type[i]
            if joint_type == 0 and joint_name:
                if "robot" in joint_name.lower() or "gripper" in joint_name.lower():
                    continue
                qpos_addr = model.jnt_qposadr[i]
                object_indices[joint_name] = (qpos_addr, qpos_addr + 7)
    except Exception as e:
        print(f"  Warning: Could not analyze joint structure: {e}")
    return object_indices


def _quaternion_multiply(q1, q2):
    """Multiply two quaternions (w, x, y, z format)."""
    w1, x1, y1, z1 = q1
    w2, x2, y2, z2 = q2
    return np.array([
        w1*w2 - x1*x2 - y1*y2 - z1*z2,
        w1*x2 + x1*w2 + y1*z2 - z1*y2,
        w1*y2 - x1*z2 + y1*w2 + z1*x2,
        w1*z2 + x1*y2 - y1*x2 + z1*w2,
    ])


def perturb_init_state(
    init_state: np.ndarray,
    env,
    position_noise_std: float = 0.04,
    rotation_noise_std: float = 0.0,
    seed: int = None,
) -> np.ndarray:
    """Add small perturbations to object positions in the init state.

    Args:
        init_state: Flattened MuJoCo state array.
        env: LIBERO environment (needed to identify object indices).
        position_noise_std: Std dev for position noise in meters.
        rotation_noise_std: Std dev for rotation noise in radians.
        seed: Random seed for reproducibility.

    Returns:
        Perturbed init state array (copy of original).
    """
    if init_state is None:
        return None

    rng = np.random.RandomState(seed) if seed is not None else np.random
    perturbed_state = init_state.copy()
    object_indices = _get_object_qpos_indices(env)

    if not object_indices:
        print("  Warning: No objects found to perturb")
        return perturbed_state

    for obj_name, (start_idx, end_idx) in object_indices.items():
        if end_idx > len(perturbed_state):
            continue

        if position_noise_std > 0:
            position_noise = rng.normal(0, position_noise_std, 3)
            perturbed_state[start_idx:start_idx + 3] += position_noise

        if rotation_noise_std > 0:
            axis = rng.randn(3)
            axis = axis / (np.linalg.norm(axis) + 1e-8)
            angle = rng.normal(0, rotation_noise_std)
            half_angle = angle / 2
            dq = np.array([
                np.cos(half_angle),
                axis[0] * np.sin(half_angle),
                axis[1] * np.sin(half_angle),
                axis[2] * np.sin(half_angle),
            ])
            quat_idx = start_idx + 3
            current_quat = perturbed_state[quat_idx:quat_idx + 4]
            perturbed_state[quat_idx:quat_idx + 4] = _quaternion_multiply(dq, current_quat)

    return perturbed_state


def perturb_init_states_batch(
    init_states: list,
    env,
    position_noise_std: float = 0.04,
    rotation_noise_std: float = 0.0,
    base_seed: int = 42,
) -> list:
    """Perturb a batch of init states with deterministic per-state seeds."""
    perturbed = []
    for i, state in enumerate(init_states):
        seed = (base_seed + i) if base_seed is not None else None
        perturbed.append(perturb_init_state(
            state, env,
            position_noise_std=position_noise_std,
            rotation_noise_std=rotation_noise_std,
            seed=seed,
        ))
    return perturbed


def visualize_prune_mask(image, patch_mask, size=(256, 256)):
    """
    Visualize the prune mask by dimming pruned regions.
    
    Args:
        image: Original image (H, W, 3)
        patch_mask: (256,) or (16, 16) tensor/array
        size: Target visualization size
        
    Returns:
        (H, W, 3) uint8 numpy array
    """
    if torch.is_tensor(patch_mask):
        mask = patch_mask.detach().cpu().to(torch.float32).numpy()
    else:
        mask = patch_mask
        
    # Handle multiple images (e.g., 512 tokens for 2 images)
    if mask.size >= 256:
        # Take the first 256 tokens for the primary image visualization
        mask = mask[:256].reshape(16, 16)
        
    # Resize mask to image size using nearest neighbor
    mask_img = Image.fromarray((mask * 255).astype(np.uint8)).resize(size, resample=Image.NEAREST)
    mask_np = np.array(mask_img) > 128
    
    # Ensure image is numpy array
    if not isinstance(image, np.ndarray):
        image = np.array(image)
        
    # Create visualization: dimmed image where mask is 0
    vis = image.copy().astype(np.float32)
    vis[~mask_np] *= 0.3  # Dim the pruned regions
    
    return vis.astype(np.uint8)


def parse_task_ids(task_id_str):
    """Parse task_id from config. Supports single integer or range string like '0-9'.
    
    Args:
        task_id_str: Task ID as integer, string integer, or range string (e.g., "0-9")
        
    Returns:
        List of task IDs to evaluate
    """
    if isinstance(task_id_str, int):
        return [task_id_str]
    
    task_id_str = str(task_id_str).strip()
    
    # Check if it's a range (e.g., "0-9")
    if '-' in task_id_str:
        parts = task_id_str.split('-')
        if len(parts) == 2:
            try:
                start = int(parts[0])
                end = int(parts[1])
                return list(range(start, end + 1))
            except ValueError:
                raise ValueError(f"Invalid task_id range format: {task_id_str}")
    
    # Single task ID
    try:
        return [int(task_id_str)]
    except ValueError:
        raise ValueError(f"Invalid task_id format: {task_id_str}")


def load_config(config_path: str = "experiments/robot/config.yaml"):
    """Load configuration from YAML file.
    
    Args:
        config_path: Path to YAML config file
        
    Returns:
        SimpleNamespace object with configuration parameters
    """
    # Check if config file exists
    if not os.path.exists(config_path):
        raise FileNotFoundError(f"Config file not found: {config_path}")
    
    # Load YAML config
    with open(config_path, 'r') as f:
        config_dict = yaml.safe_load(f)
    
    # Validate required fields
    required_fields = ['model_path', 'task_suite_name']
    missing_fields = [field for field in required_fields if field not in config_dict or config_dict[field] is None]
    if missing_fields:
        raise ValueError(f"Missing required fields in config file: {', '.join(missing_fields)}")
    
    # Parse task_ids from task_id field
    if 'task_id' in config_dict:
        config_dict['task_ids'] = parse_task_ids(config_dict['task_id'])
    else:
        config_dict['task_ids'] = [0]  # Default to task 0
    
    # Convert dict to SimpleNamespace for dot notation access
    config = SimpleNamespace(**config_dict)
    
    return config


def _compute_sam_mask(
    obs, cfg, vla, processor, sam_module, resize_size,
    task_label, use_weighted_similarity, target_extraction,
    cached_text_features, cached_token_weights,
    top_k_points, num_neg_points, mask_to_patch_threshold,
    use_spatial_clustering, k_initial, dbscan_eps, min_cluster_size,
):
    """Compute SAM-guided patch mask from the current observation.

    Returns (patch_mask, patch_scores, similarity_map, mask, point_coords, point_labels)
    where patch_mask is a (256,) boolean tensor or None if generation failed.
    """
    img = get_libero_image(obs)
    img_resized = resize_image_for_policy(img, resize_size)
    if isinstance(img_resized, np.ndarray):
        img_np = img_resized.astype(np.uint8)
    else:
        img_np = np.array(img_resized).astype(np.uint8)

    vision_backbone = vla.vision_backbone if hasattr(vla, 'vision_backbone') else vla.model.vision_backbone

    img_for_siglip = Image.fromarray(img_np)
    processed = processor("", img_for_siglip)
    pixel_values_siglip = processed["pixel_values"].to('cuda' if torch.cuda.is_available() else 'cpu')

    patch_features = sam_module.extract_siglip_image_features(
        pixel_values=pixel_values_siglip, vision_backbone=vision_backbone,
    )

    if use_weighted_similarity:
        if cached_text_features is not None and cached_token_weights is not None:
            text_features = cached_text_features
            token_weights = cached_token_weights
        else:
            text_features, token_weights, _ = sam_module.extract_siglip_text_features_with_weights(
                text=task_label, vision_backbone=vision_backbone,
            )
        patch_scores = sam_module.compute_weighted_similarity(
            text_features=text_features, patch_features=patch_features,
            token_weights=token_weights, normalize=True,
        )
        similarity_map = patch_scores.unsqueeze(0)
    else:
        if cached_text_features is not None:
            text_features = cached_text_features
        else:
            text_features = sam_module.extract_siglip_text_features(
                text=task_label, vision_backbone=vision_backbone,
                target_extraction=target_extraction,
            )
        similarity_map = sam_module.compute_text_image_similarity(
            text_features=text_features, patch_features=patch_features, normalize=True,
        )
        patch_scores = similarity_map.mean(dim=0)

    mask, point_coords, point_labels = sam_module.generate_mask_from_precomputed(
        image=img_np, patch_scores=patch_scores,
        image_size=(224, 224), patch_size=14,
        num_points=top_k_points, num_neg_points=num_neg_points,
        return_vis_data=True, use_spatial_clustering=use_spatial_clustering,
        k_initial=k_initial, dbscan_eps=dbscan_eps, min_cluster_size=min_cluster_size,
    )

    if mask is None:
        return None, patch_scores, similarity_map, None, point_coords, point_labels

    mask_224 = F.interpolate(
        mask.unsqueeze(0).unsqueeze(0).float(),
        size=(224, 224), mode='bilinear', align_corners=False
    ).squeeze(0).squeeze(0) > 0.5
    patch_mask = sam_module.mask_to_patch_selection(
        mask=mask_224, patch_size=14, threshold=mask_to_patch_threshold,
    )
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    if patch_mask.device.type != device:
        patch_mask = patch_mask.to(device)

    num_selected = int(patch_mask.sum().item())
    if num_selected == 0:
        best_idx = patch_scores.argmax().item()
        patch_mask[best_idx] = True
        num_selected = 1
        print(f"  [SAM fallback] 0 patches survived threshold, keeping top-1 (idx={best_idx})")

    return patch_mask, patch_scores, similarity_map, mask, point_coords, point_labels


SAM_RECOMPUTE_INTERVAL = 5  # recompute SAM mask every N VLA calls


def run_episode(
    env,
    vla,
    processor,
    cfg,
    task_label: str,
    resize_size,
    initial_state=None,
    sam_module=None,
    use_mask_filtering: bool = False,
    episode_idx: int = 0,
    max_steps: int = 300,
    num_steps_wait: int = 10,
    top_k_points: int = 10,
    num_neg_points: int = 5,
    mask_to_patch_threshold: float = 0.1,
    action_head=None,
    proprio_projector=None,
    noisy_action_projector=None,
    target_extraction: str = "none",
    use_weighted_similarity: bool = False,
    use_spatial_clustering: bool = False,
    k_initial: int = 50,
    dbscan_eps: float = 2.0,
    min_cluster_size: int = 3,
    # Attention biasing parameters
    use_attention_bias: bool = False,
    bias_strength: float = 2.0,
    bias_active_layers: Optional[set] = None,
    saliency_top_k_ratio: float = 0.3,
    saliency_method: str = "ig",
    # Visual highlighting parameters (Vea-inspired)
    use_visual_highlight: bool = False,
    highlight_alpha: float = 0.4,
    # Cached SigLIP text features (computed once, passed in)
    cached_text_features=None,
    cached_token_weights=None,
) -> Tuple[Dict, List[np.ndarray]]:
    """
    Run a single episode with the VLA policy.

    SAM mask is computed ONCE at episode start (after warm-up) and reused for
    all subsequent VLA calls. SigLIP text features are passed in pre-cached.
    """
    from experiments.robot.libero.libero_utils import get_libero_dummy_action
    from experiments.robot.robot_utils import normalize_gripper_action, invert_gripper_action
    
    # Reset environment
    env.reset()
    
    # Set initial state if provided
    if initial_state is not None:
        obs = env.set_init_state(initial_state)
    else:
        obs = env.get_observation()
    
    t = 0
    success = False
    total_masks_generated = 0
    replay_images = []
    patch_mask = None
    similarity_map = None
    saliency_info = None
    action_queue = collections.deque()
    mask = None
    point_coords = None
    point_labels = None
    
    # Timing statistics
    timing_stats = {
        "sam_times": [],
        "vla_times": [],
        "step_times": [],
        "preprocess_times": [],
        "env_times": [],
    }
    
    # ---- Warm-up phase ----
    while t < num_steps_wait:
        obs, reward, done, info = env.step(get_libero_dummy_action(cfg.model_family))
        img = get_libero_image(obs)
        img_resized = np.array(Image.fromarray(img).resize((256, 256)))
        if sam_module is not None and use_mask_filtering:
            black = np.zeros((256, 256, 3), dtype=np.uint8)
            top_row = np.concatenate([img_resized, black], axis=1)
            bottom_row = np.concatenate([black, black], axis=1)
        else:
            top_row = np.concatenate([img_resized, img_resized], axis=1)
            bottom_row = np.concatenate([img_resized, img_resized], axis=1)
        replay_images.append(np.concatenate([top_row, bottom_row], axis=0))
        t += 1

    # ---- Initial SAM mask computation at episode start ----
    patch_scores = None
    sam_common_args = dict(
        cfg=cfg, vla=vla, processor=processor, sam_module=sam_module,
        resize_size=resize_size, task_label=task_label,
        use_weighted_similarity=use_weighted_similarity,
        target_extraction=target_extraction,
        cached_text_features=cached_text_features,
        cached_token_weights=cached_token_weights,
        top_k_points=top_k_points, num_neg_points=num_neg_points,
        mask_to_patch_threshold=mask_to_patch_threshold,
        use_spatial_clustering=use_spatial_clustering,
        k_initial=k_initial, dbscan_eps=dbscan_eps, min_cluster_size=min_cluster_size,
    )

    use_pruning = getattr(cfg, "use_pruning", True)

    if sam_module is not None and use_mask_filtering:
        sam_start_time = time.time()

        patch_mask, patch_scores, similarity_map, mask, point_coords, point_labels = \
            _compute_sam_mask(obs=obs, **sam_common_args)

        total_masks_generated += 1

        if patch_mask is not None:
            num_selected = int(patch_mask.sum().item())
            pruning_ratio = 1.0 - (num_selected / 256)
            timing_stats.setdefault("pruning_ratios", []).append(pruning_ratio)
            timing_stats.setdefault("selected_patches", []).append(num_selected)
            print(f"  SAM mask computed: {num_selected}/256 patches kept ({pruning_ratio*100:.0f}% pruned)")
        else:
            print("  Warning: SAM mask generation failed, running without pruning")

        if not use_pruning:
            print("  Pruning DISABLED — keeping all patches (scores used for highlighting only)")
            patch_mask = None

        if torch.cuda.is_available():
            torch.cuda.synchronize()
        sam_time = time.time() - sam_start_time
        timing_stats["sam_times"].append(sam_time)
        print(f"  SAM initial cost: {sam_time*1000:.0f}ms")

    # ---- One-time saliency (IG) computation ----
    if use_attention_bias:
        img = get_libero_image(obs)
        wrist_img = get_libero_wrist_image(obs)
        img_resized_sal = resize_image_for_policy(img, resize_size)
        wrist_img_resized_sal = resize_image_for_policy(wrist_img, resize_size)
        observation_sal = {
            "full_image": img_resized_sal,
            "wrist_image": wrist_img_resized_sal,
        }
        sal_start = time.time()
        saliency_info = compute_saliency_for_episode(
            model=vla, processor=processor,
            observation=observation_sal, task_label=task_label,
            top_k_ratio=saliency_top_k_ratio,
            n_steps=5,
            method=saliency_method,
        )
        sal_time = time.time() - sal_start
        timing_stats["saliency_time"] = sal_time
        method_label = saliency_method.upper()
        if saliency_info is not None:
            print(f"  {method_label} saliency computed in {sal_time:.2f}s "
                  f"(top5={saliency_info.top_positions[-5:][::-1].tolist()})")
        else:
            print(f"  {method_label} saliency computation failed, running without bias")

    # Pre-compute num_patches_for_bias (constant for the episode since mask doesn't change)
    num_patches_for_bias = 0
    visual_patch_indices = None
    visual_patch_weights = None
    if use_attention_bias and saliency_info is not None:
        num_patches_per_image = 256
        if patch_mask is not None:
            num_selected = int(patch_mask.sum().item())
            num_patches_for_bias = num_selected + num_patches_per_image
            # Multimodal layout: [BOS(0)] [selected_3rd_person(1..N)] [wrist(N+1..N+256)] ...
            # The kept third-person patches sit at positions 1..N
            visual_patch_indices = np.arange(1, num_selected + 1)
            # Weight each kept patch by its SigLIP similarity score (normalized to 0-1)
            if patch_scores is not None:
                kept_scores = patch_scores[patch_mask].detach().cpu().float().numpy()
                score_max = kept_scores.max()
                if score_max > 0:
                    visual_patch_weights = kept_scores / score_max  # normalize to [0, 1]
                else:
                    visual_patch_weights = np.ones(num_selected, dtype=np.float32)
        else:
            num_patches_for_bias = num_patches_per_image * 2
        if proprio_projector is not None:
            num_patches_for_bias += 1

    # ---- Main action loop ----
    while t < max_steps + num_steps_wait:
        step_start_time = time.time()

        if len(action_queue) == 0:
            preprocess_start = time.time()
            img = get_libero_image(obs)
            wrist_img = get_libero_wrist_image(obs)

            img_resized = resize_image_for_policy(img, resize_size)
            wrist_img_resized = resize_image_for_policy(wrist_img, resize_size)

            # Visual Evidence Highlighting (Vea-inspired): dim irrelevant regions
            # using continuous SigLIP similarity scores so the vision encoder
            # naturally produces stronger features for task-relevant patches.
            if use_visual_highlight and patch_scores is not None:
                scores_np = patch_scores.detach().cpu().float().numpy()
                scores_2d = scores_np.reshape(16, 16)
                highlight_map = F.interpolate(
                    torch.tensor(scores_2d, dtype=torch.float32).unsqueeze(0).unsqueeze(0),
                    size=(224, 224), mode='bilinear', align_corners=False,
                ).squeeze().numpy()
                h_min, h_max = highlight_map.min(), highlight_map.max()
                if h_max > h_min:
                    highlight_map = (highlight_map - h_min) / (h_max - h_min)
                else:
                    highlight_map = np.ones_like(highlight_map)
                pixel_weights = highlight_alpha + (1.0 - highlight_alpha) * highlight_map
                if isinstance(img_resized, np.ndarray):
                    img_resized = (img_resized.astype(np.float32) * pixel_weights[..., None]).clip(0, 255).astype(np.uint8)
                else:
                    img_np = np.array(img_resized).astype(np.float32)
                    img_resized = Image.fromarray(
                        (img_np * pixel_weights[..., None]).clip(0, 255).astype(np.uint8)
                    )

            preprocess_time = time.time() - preprocess_start
            timing_stats["preprocess_times"].append(preprocess_time)

            observation = {
                "full_image": img_resized,
                "wrist_image": wrist_img_resized,
                "state": np.concatenate(
                    (obs["robot0_eef_pos"], quat2axisangle(obs["robot0_eef_quat"]), obs["robot0_gripper_qpos"])
                ),
            }

            # VLA inference — produces a chunk of actions
            if torch.cuda.is_available():
                torch.cuda.synchronize()
            vla_start_time = time.time()

            with bias_context(
                saliency_info=saliency_info if use_attention_bias else None,
                num_patches=num_patches_for_bias,
                strength=bias_strength,
                active_layers=bias_active_layers,
                top_k_ratio=saliency_top_k_ratio,
                visual_patch_indices=visual_patch_indices,
                visual_patch_weights=visual_patch_weights,
            ):
                result = get_vla_action(
                    cfg=cfg, vla=vla, processor=processor,
                    obs=observation, task_label=task_label,
                    action_head=action_head,
                    proprio_projector=proprio_projector,
                    noisy_action_projector=noisy_action_projector,
                    use_film=cfg.use_film, patch_mask=patch_mask,
                )
            if torch.cuda.is_available():
                torch.cuda.synchronize()
            vla_time = time.time() - vla_start_time
            timing_stats["vla_times"].append(vla_time)
            n_vla = len(timing_stats["vla_times"])
            print("    VLA call %d: %.0fms" % (n_vla, vla_time * 1000), end="")
            if timing_stats.get("selected_patches"):
                print(" | patches: %d/256" % timing_stats["selected_patches"][-1], end="")
            print()

            if isinstance(result, tuple):
                new_actions, debug_info = result
            else:
                new_actions = result

            action_queue.extend(new_actions)

            # Periodic SAM mask recomputation to track scene changes
            if (sam_module is not None and use_mask_filtering
                    and n_vla > 1 and n_vla % SAM_RECOMPUTE_INTERVAL == 0):
                sam_start_time = time.time()
                new_mask, new_scores, similarity_map, mask, point_coords, point_labels = \
                    _compute_sam_mask(obs=obs, **sam_common_args)
                if new_mask is not None:
                    patch_scores = new_scores
                    total_masks_generated += 1
                    if use_pruning:
                        patch_mask = new_mask
                        num_selected = int(patch_mask.sum().item())
                        pruning_ratio = 1.0 - (num_selected / 256)
                        timing_stats.setdefault("pruning_ratios", []).append(pruning_ratio)
                        timing_stats.setdefault("selected_patches", []).append(num_selected)
                        # Update visual patch bias weights
                        if use_attention_bias and saliency_info is not None:
                            num_patches_for_bias = num_selected + 256
                            if proprio_projector is not None:
                                num_patches_for_bias += 1
                            visual_patch_indices = np.arange(1, num_selected + 1)
                            kept_scores = patch_scores[patch_mask].detach().cpu().float().numpy()
                            if kept_scores.size > 0:
                                score_max = kept_scores.max()
                                visual_patch_weights = (kept_scores / score_max) if score_max > 0 \
                                    else np.ones(num_selected, dtype=np.float32)
                            else:
                                visual_patch_weights = np.ones(max(num_selected, 1), dtype=np.float32)
                        print(f"    SAM recomputed (VLA call {n_vla}): {num_selected}/256 kept")
                    else:
                        print(f"    SAM scores recomputed (VLA call {n_vla}), pruning disabled")
                if torch.cuda.is_available():
                    torch.cuda.synchronize()
                timing_stats["sam_times"].append(time.time() - sam_start_time)

        # Pop next action from the chunk queue
        action = action_queue.popleft()
        action_step = t - num_steps_wait
        if action_step % 50 == 0 and action_step > 0:
            elapsed_vla = sum(timing_stats["vla_times"])
            avg_vla = elapsed_vla / len(timing_stats["vla_times"]) * 1000
            print("    [step %d] avg VLA: %.0fms | VLA calls: %d | elapsed: %.0fs"
                  % (action_step, avg_vla, len(timing_stats["vla_times"]), sum(timing_stats["step_times"])))

        # Visualization for rollout video
        cur_img = get_libero_image(obs)
        if sam_module is not None and use_mask_filtering:
            vis_img = np.array(Image.fromarray(cur_img).resize((256, 256)))
            if cfg.center_crop:
                vis_img = np.array(center_crop_image(Image.fromarray(vis_img)).resize((256, 256)))

            if similarity_map is not None:
                heatmap_vis = visualize_similarity_heatmap(
                    image=vis_img, similarity_map=similarity_map,
                    size=(256, 256), alpha=0.7, aggregation="mean",
                )
            else:
                heatmap_vis = vis_img.copy()

            if mask is not None and point_coords is not None:
                mask_for_vis = F.interpolate(
                    mask.unsqueeze(0).unsqueeze(0).float(),
                    size=(256, 256), mode='nearest',
                ).squeeze(0).squeeze(0)
                sam_vis = visualize_mask_with_points(
                    image=vis_img, mask=mask_for_vis,
                    point_coords=point_coords, point_labels=point_labels,
                    size=(256, 256), mask_alpha=0.5,
                )
            else:
                sam_vis = vis_img.copy()

            if patch_mask is not None:
                mask_vis = visualize_prune_mask(vis_img, patch_mask, size=(256, 256))
            else:
                mask_vis = vis_img.copy()

            top_row = np.concatenate([vis_img, heatmap_vis], axis=1)
            bottom_row = np.concatenate([sam_vis, mask_vis], axis=1)
            combined_img = np.concatenate([top_row, bottom_row], axis=0)
            replay_images.append(combined_img)
        else:
            vis_img = np.array(Image.fromarray(cur_img).resize((256, 256)))
            top_row = np.concatenate([vis_img, vis_img], axis=1)
            bottom_row = np.concatenate([vis_img, vis_img], axis=1)
            combined_img = np.concatenate([top_row, bottom_row], axis=0)
            replay_images.append(combined_img)

        # Process action: normalize gripper and invert for OpenVLA
        action = normalize_gripper_action(action, binarize=True)
        if cfg.model_family == "openvla":
            action = invert_gripper_action(action)

        # Execute action in environment
        env_start_time = time.time()
        obs, reward, done, info = env.step(action.tolist())
        env_time = time.time() - env_start_time
        timing_stats["env_times"].append(env_time)

        # Record total step time
        step_time = time.time() - step_start_time
        timing_stats["step_times"].append(step_time)

        t += 1

        # Check for success
        if done:
            success = True
            break
    
    # Compute statistics
    step_count = t - num_steps_wait
    stats = {
        "success": success,
        "steps": step_count,
        "masks_generated": total_masks_generated,
        "timing": timing_stats,  # Include timing stats
    }
    
    return stats, replay_images


def main():
    """Main evaluation loop."""
    parser = argparse.ArgumentParser(description="LIBERO evaluation with optional SAM pruning and attention biasing")
    parser.add_argument("--config", type=str, default="experiments/robot/config.yaml",
                        help="Path to YAML config file")
    cli_args = parser.parse_args()

    args = load_config(cli_args.config)
    
    # Set random seed
    set_seed_everywhere(args.seed if hasattr(args, 'seed') else 7)
    
    # Create output directory
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Create config object from args
    cfg = SimpleNamespace(
        model_family=args.model_family,
        pretrained_checkpoint=args.pretrained_checkpoint or args.model_path,
        use_l1_regression=not args.use_diffusion,
        use_diffusion=args.use_diffusion,
        use_proprio=args.use_proprio,
        use_film=args.use_film,
        center_crop=args.center_crop,
        load_in_8bit=False,
        load_in_4bit=False,
        num_images_in_input=2,  # head + wrist camera
        lora_rank=32,
        num_diffusion_steps_train=50,
        num_diffusion_steps_inference=50,
        unnorm_key=args.task_suite_name,  # Will be updated after model loading
        use_pruning=getattr(args, "use_pruning", True),
    )
    
    # Load VLA model
    print(f"Loading VLA model from {cfg.pretrained_checkpoint}...")
    vla = get_vla(cfg)
    processor = get_processor(cfg)
    print("VLA model loaded successfully!")
    
    # Check and fix unnorm_key based on model's available keys
    unnorm_key = cfg.unnorm_key
    if unnorm_key not in vla.norm_stats and f"{unnorm_key}_no_noops" in vla.norm_stats:
        unnorm_key = f"{unnorm_key}_no_noops"
        print(f"Updated unnorm_key to: {unnorm_key}")
    assert unnorm_key in vla.norm_stats, (
        f"Action un-norm key '{unnorm_key}' not found in VLA norm_stats! "
        f"Available keys: {list(vla.norm_stats.keys())}"
    )
    cfg.unnorm_key = unnorm_key
    
    # Load proprio projector if needed
    proprio_projector = None
    if cfg.use_proprio:
        print("Loading proprio projector...")
        proprio_projector = get_proprio_projector(
            cfg,
            vla.llm_dim,
            proprio_dim=8,  # 8-dimensional proprio for LIBERO
        )
        print("Proprio projector loaded successfully!")
    
    # Load action head if needed
    action_head = None
    if cfg.use_l1_regression or cfg.use_diffusion:
        print("Loading action head...")
        action_head = get_action_head(cfg, vla.llm_dim)
        print("Action head loaded successfully!")
    
    # Load noisy action projector if using diffusion
    noisy_action_projector = None
    if cfg.use_diffusion:
        print("Loading noisy action projector...")
        noisy_action_projector = get_noisy_action_projector(cfg, vla.llm_dim)
        print("Noisy action projector loaded successfully!")
    
    # Set up attention biasing if enabled
    use_bias = getattr(args, "use_attention_bias", False)
    bias_active_layers = None
    if use_bias:
        install_layer_hooks(vla)
        install_biased_sdpa()

        # Parse bias_active_layers from config (e.g., "8-23" or "0-31")
        layer_spec = getattr(args, "bias_active_layers", None)
        if layer_spec is not None:
            layer_spec = str(layer_spec).strip()
            if "-" in layer_spec:
                lo, hi = layer_spec.split("-")
                bias_active_layers = set(range(int(lo), int(hi) + 1))
            else:
                bias_active_layers = {int(layer_spec)}
            print(f"Attention biasing: layers {sorted(bias_active_layers)}, "
                  f"strength={getattr(args, 'bias_strength', 2.0)}")
        else:
            print("Attention biasing: ALL layers")

    # Print visual highlighting config
    if getattr(args, "use_visual_highlight", False):
        print(f"Visual highlighting: ENABLED (alpha={getattr(args, 'highlight_alpha', 0.4)})")
    else:
        print("Visual highlighting: DISABLED")

    # Print OOD perturbation config
    if getattr(args, "ood_enabled", False):
        print(f"OOD perturbation: ENABLED (+/-{getattr(args, 'ood_position_noise_cm', 4.0):.1f}cm, "
              f"seed={getattr(args, 'ood_seed', 42)})")
    else:
        print("OOD perturbation: DISABLED (nominal initial states)")

    # Get expected image dimensions
    resize_size = get_image_resize_size(cfg)
    
    # Initialize SAM/EfficientTAM module if requested
    sam_module = None
    if args.use_sam:
        if args.sam_checkpoint is None:
            raise ValueError(
                "--use_sam is True but --sam_checkpoint is not provided. "
                "Please specify checkpoint path with --sam_checkpoint."
            )
        
        print(f"Initializing similarity-guided {args.sam_backend} with model {args.sam_model_type}...")
        try:
            sam_module = create_siglip_guided_sam(
                sam_checkpoint=args.sam_checkpoint,
                sam_model_type=args.sam_model_type,
                device="cuda" if torch.cuda.is_available() else "cpu",
                backend=args.sam_backend,
                efficienttam_base_dir=args.efficienttam_base_dir,
            )
            print(f"{args.sam_backend.upper()} module initialized successfully!")
        except Exception as e:
            print(f"Error initializing SAM module: {e}")
            print("Continuing without SAM...")
            args.use_sam = False
            sam_module = None
    
    # Load LIBERO task suite
    print(f"Loading LIBERO task suite: {args.task_suite_name}")
    benchmark_dict = benchmark.get_benchmark_dict()
    task_suite = benchmark_dict[args.task_suite_name]()
    
    # Get task IDs to evaluate
    task_ids = args.task_ids
    print(f"Tasks to evaluate: {task_ids}")
    
    # Store results for all tasks
    all_tasks_stats = {}
    all_tasks_success_rates = {}
    max_steps = TASK_MAX_STEPS[args.task_suite_name]
    
    # Loop over all tasks
    for task_id in task_ids:
        print(f"\n{'='*70}")
        print(f"Evaluating Task {task_id}")
        print(f"{'='*70}")
        
        task = task_suite.get_task(task_id)
        
        # Get environment and task description
        env, task_label = get_libero_env(task, cfg.model_family, resolution=256)
        print(f"Task description: {task_label}")
        
        # Get initial states
        initial_states = task_suite.get_task_init_states(task_id)

        # Apply OOD perturbation to test generalizability
        ood_enabled = getattr(args, "ood_enabled", False)
        if ood_enabled and len(initial_states) > 0 and initial_states[0] is not None:
            ood_cm = getattr(args, "ood_position_noise_cm", 4.0)
            ood_seed = getattr(args, "ood_seed", 42)
            print(f"  Applying OOD perturbation: +/-{ood_cm:.1f}cm position noise (seed={ood_seed})")
            initial_states = perturb_init_states_batch(
                initial_states, env,
                position_noise_std=ood_cm / 100.0,
                base_seed=ood_seed,
            )

        # Cache SigLIP text features once per task (constant across episodes)
        cached_text_features = None
        cached_token_weights = None
        if sam_module is not None and args.use_sam:
            vision_backbone = vla.vision_backbone if hasattr(vla, 'vision_backbone') else vla.model.vision_backbone
            use_weighted = getattr(args, "use_weighted_similarity", False)
            if use_weighted:
                cached_text_features, cached_token_weights, _ = sam_module.extract_siglip_text_features_with_weights(
                    text=task_label, vision_backbone=vision_backbone,
                )
                print(f"  Cached weighted text features for task: {task_label[:50]}...")
            else:
                cached_text_features = sam_module.extract_siglip_text_features(
                    text=task_label, vision_backbone=vision_backbone,
                    target_extraction=getattr(args, "target_extraction", "none"),
                )
                print(f"  Cached text features for task: {task_label[:50]}...")

        # Run evaluation episodes for this task
        all_stats = []
        success_count = 0
        
        print(f"\nRunning {args.num_episodes} episodes for task {task_id}...")
        for ep_idx in range(args.num_episodes):
            print(f"\n=== Episode {ep_idx + 1}/{args.num_episodes} ===")
            
            initial_state = initial_states[ep_idx % len(initial_states)]
            
            stats, replay_images = run_episode(
                env=env,
                vla=vla,
                processor=processor,
                cfg=cfg,
                task_label=task_label,
                resize_size=resize_size,
                initial_state=initial_state,
                sam_module=sam_module,
                use_mask_filtering=args.use_sam,
                episode_idx=ep_idx,
                top_k_points=args.top_k_points,
                num_neg_points=getattr(args, "num_neg_points", 5),
                mask_to_patch_threshold=getattr(args, "mask_to_patch_threshold", 0.1),
                max_steps=max_steps,
                action_head=action_head,
                proprio_projector=proprio_projector,
                noisy_action_projector=noisy_action_projector,
                target_extraction=getattr(args, "target_extraction", "none"),
                use_weighted_similarity=getattr(args, "use_weighted_similarity", False),
                use_spatial_clustering=getattr(args, "use_spatial_clustering", False),
                k_initial=getattr(args, "k_initial", 50),
                dbscan_eps=getattr(args, "dbscan_eps", 2.0),
                min_cluster_size=getattr(args, "min_cluster_size", 3),
                use_attention_bias=use_bias,
                bias_strength=getattr(args, "bias_strength", 2.0),
                bias_active_layers=bias_active_layers,
                saliency_top_k_ratio=getattr(args, "saliency_top_k_ratio", 0.3),
                saliency_method=getattr(args, "saliency_method", "ig"),
                use_visual_highlight=getattr(args, "use_visual_highlight", False),
                highlight_alpha=getattr(args, "highlight_alpha", 0.4),
                cached_text_features=cached_text_features,
                cached_token_weights=cached_token_weights,
            )
            
            all_stats.append(stats)
            if stats["success"]:
                success_count += 1
            
            # Save rollout video
            if args.save_visualizations and len(replay_images) > 0:
                save_rollout_video(
                    replay_images,
                    idx=ep_idx,
                    success=stats["success"],
                    task_description=task_label,
                    rollout_dir=getattr(args, "rollout_dir", None),
                )
            
            # Print episode results
            print(f"Success: {stats['success']}, Steps: {stats['steps']}")
            if args.use_sam:
                print(f"Masks generated: {stats.get('masks_generated', 0)}")
        
            # Print timing for this episode
            timing = stats.get("timing", {})
            if timing:
                vla_times = timing.get("vla_times", [])
                sam_times = timing.get("sam_times", [])
                step_times = timing.get("step_times", [])
                preprocess_times = timing.get("preprocess_times", [])
                env_times = timing.get("env_times", [])
                pruning_ratios = timing.get("pruning_ratios", [])
                selected_patches = timing.get("selected_patches", [])
                
                print(f"Timing (this episode):")
                if step_times:
                    print(f"  Total step time: {np.mean(step_times)*1000:.1f}ms avg, {np.sum(step_times):.2f}s total")
                if vla_times:
                    print(f"  VLA inference:   {np.mean(vla_times)*1000:.1f}ms avg, {np.sum(vla_times):.2f}s total")
                if sam_times:
                    print(f"  SAM inference:   {np.mean(sam_times)*1000:.1f}ms avg, {np.sum(sam_times):.2f}s total")
                if preprocess_times:
                    print(f"  Preprocessing:   {np.mean(preprocess_times)*1000:.1f}ms avg")
                if env_times:
                    print(f"  Env step:        {np.mean(env_times)*1000:.1f}ms avg")
                
                # Print token pruning statistics
                if pruning_ratios:
                    print(f"Token Pruning (this episode):")
                    print(f"  Avg pruning ratio: {np.mean(pruning_ratios)*100:.1f}%")
                    print(f"  Avg patches kept:  {np.mean(selected_patches):.1f}/256")
                    print(f"  Min/Max patches:   {min(selected_patches)}/{max(selected_patches)}")
    
        # Aggregate all timing stats for this task
        all_vla_times = []
        all_sam_times = []
        all_step_times = []
        all_preprocess_times = []
        all_env_times = []
        all_pruning_ratios = []
        all_selected_patches = []
        for s in all_stats:
            timing = s.get("timing", {})
            all_vla_times.extend(timing.get("vla_times", []))
            all_sam_times.extend(timing.get("sam_times", []))
            all_step_times.extend(timing.get("step_times", []))
            all_preprocess_times.extend(timing.get("preprocess_times", []))
            all_env_times.extend(timing.get("env_times", []))
            all_pruning_ratios.extend(timing.get("pruning_ratios", []))
            all_selected_patches.extend(timing.get("selected_patches", []))
        
        # Compute and print task statistics
        print("\n" + "="*50)
        print(f"TASK {task_id} RESULTS")
        print("="*50)
        print(f"Task: {task_label}")
        task_success_rate = success_count/args.num_episodes if args.num_episodes > 0 else 0
        print(f"Success Rate: {success_count}/{args.num_episodes} ({task_success_rate*100:.1f}%)")
        if all_stats:
            print(f"Average Steps: {np.mean([s['steps'] for s in all_stats]):.1f}")
        else:
            print("No episodes completed.")
        
        # Print aggregated timing statistics for this task
        if all_step_times or all_vla_times or all_sam_times:
            print("\n" + "-"*50)
            print("TIMING STATISTICS (across all episodes)")
            print("-"*50)
            if all_step_times:
                print(f"Total step time:   {np.mean(all_step_times)*1000:.1f}ms avg, {np.std(all_step_times)*1000:.1f}ms std")
                print(f"                   {np.min(all_step_times)*1000:.1f}ms min, {np.max(all_step_times)*1000:.1f}ms max")
            if all_vla_times:
                print(f"VLA inference:     {np.mean(all_vla_times)*1000:.1f}ms avg, {np.std(all_vla_times)*1000:.1f}ms std")
                print(f"                   {np.min(all_vla_times)*1000:.1f}ms min, {np.max(all_vla_times)*1000:.1f}ms max")
                print(f"                   Total: {np.sum(all_vla_times):.2f}s")
            if all_sam_times:
                print(f"SAM inference:     {np.mean(all_sam_times)*1000:.1f}ms avg, {np.std(all_sam_times)*1000:.1f}ms std")
                print(f"                   {np.min(all_sam_times)*1000:.1f}ms min, {np.max(all_sam_times)*1000:.1f}ms max")
                print(f"                   Total: {np.sum(all_sam_times):.2f}s")
            if all_preprocess_times:
                print(f"Preprocessing:     {np.mean(all_preprocess_times)*1000:.1f}ms avg, {np.std(all_preprocess_times)*1000:.1f}ms std")
            if all_env_times:
                print(f"Env step:          {np.mean(all_env_times)*1000:.1f}ms avg, {np.std(all_env_times)*1000:.1f}ms std")
            
            # Calculate FPS
            if all_step_times:
                avg_fps = 1.0 / np.mean(all_step_times)
                print(f"\nEffective FPS:     {avg_fps:.2f}")
        
        if args.use_sam and sam_module is not None and len(all_stats) > 0:
            total_masks = sum([s.get('masks_generated', 0) for s in all_stats])
            print(f"\nSAM Statistics:")
            print(f"  Backend: {args.sam_backend}")
            print(f"  Model: {args.sam_model_type}")
            print(f"  Total masks generated: {total_masks}")
            print(f"  Avg masks per episode: {total_masks/len(all_stats):.1f}")
            
            # Print token pruning statistics
            if all_pruning_ratios:
                print(f"\nToken Pruning Statistics (across all episodes):")
                print(f"  Avg pruning ratio: {np.mean(all_pruning_ratios)*100:.1f}%")
                print(f"  Std pruning ratio: {np.std(all_pruning_ratios)*100:.1f}%")
                print(f"  Avg patches kept:  {np.mean(all_selected_patches):.1f}/256 ({np.mean(all_selected_patches)/256*100:.1f}%)")
                print(f"  Min/Max patches:   {min(all_selected_patches)}/{max(all_selected_patches)}")
        
        # Save results to file for this task
        results_file = output_dir / f"results_{args.task_suite_name}_task{task_id}.txt"
        with open(results_file, "w") as f:
            f.write(f"Task ID: {task_id}\n")
            f.write(f"Task: {task_label}\n")
            f.write(f"Model: {args.model_path}\n")
            f.write(f"Seed: {args.seed}\n")
            f.write(f"Use SAM: {args.use_sam}\n")
            if args.use_sam and sam_module is not None:
                f.write(f"SAM Backend: {args.sam_backend}\n")
                f.write(f"SAM Model: {args.sam_model_type}\n")
            f.write(f"Success Rate: {success_count}/{args.num_episodes} ({task_success_rate*100:.1f}%)\n")
            if all_stats:
                f.write(f"Average Steps: {np.mean([s['steps'] for s in all_stats]):.1f}\n")
            if all_step_times or all_vla_times or all_sam_times:
                f.write(f"\n--- Timing Statistics ---\n")
                if all_step_times:
                    f.write(f"Total step time: {np.mean(all_step_times)*1000:.1f}ms avg, {np.std(all_step_times)*1000:.1f}ms std\n")
                    f.write(f"                 {np.min(all_step_times)*1000:.1f}ms min, {np.max(all_step_times)*1000:.1f}ms max\n")
                if all_vla_times:
                    f.write(f"VLA inference: {np.mean(all_vla_times)*1000:.1f}ms avg, {np.std(all_vla_times)*1000:.1f}ms std, {np.sum(all_vla_times):.2f}s total\n")
                if all_sam_times:
                    f.write(f"SAM inference: {np.mean(all_sam_times)*1000:.1f}ms avg, {np.std(all_sam_times)*1000:.1f}ms std, {np.sum(all_sam_times):.2f}s total\n")
                if all_preprocess_times:
                    f.write(f"Preprocessing: {np.mean(all_preprocess_times)*1000:.1f}ms avg\n")
                if all_env_times:
                    f.write(f"Env step: {np.mean(all_env_times)*1000:.1f}ms avg\n")
                if all_step_times:
                    f.write(f"Effective FPS: {1.0/np.mean(all_step_times):.2f}\n")
            
            # Save SAM and token pruning statistics
            if args.use_sam and sam_module is not None:
                total_masks = sum([s.get('masks_generated', 0) for s in all_stats])
                f.write(f"\n--- SAM Statistics ---\n")
                f.write(f"Total masks generated: {total_masks}\n")
                f.write(f"Avg masks per episode: {total_masks/len(all_stats) if all_stats else 0:.1f}\n")
                
                if all_pruning_ratios:
                    f.write(f"\n--- Token Pruning Statistics ---\n")
                    f.write(f"Avg pruning ratio: {np.mean(all_pruning_ratios)*100:.1f}%\n")
                    f.write(f"Std pruning ratio: {np.std(all_pruning_ratios)*100:.1f}%\n")
                    f.write(f"Avg patches kept: {np.mean(all_selected_patches):.1f}/256 ({np.mean(all_selected_patches)/256*100:.1f}%)\n")
                    f.write(f"Min/Max patches: {min(all_selected_patches)}/{max(all_selected_patches)}\n")
        
        print(f"Results saved to {results_file}")

        # Save structured JSON results for this task
        all_saliency_times = [
            s.get("timing", {}).get("saliency_time", 0.0) for s in all_stats
            if s.get("timing", {}).get("saliency_time") is not None
        ]
        task_json = {
            "config": {
                "use_sam": args.use_sam,
                "use_attention_bias": getattr(args, "use_attention_bias", False),
                "bias_strength": getattr(args, "bias_strength", 0.0),
                "bias_active_layers": getattr(args, "bias_active_layers", None),
                "saliency_top_k_ratio": getattr(args, "saliency_top_k_ratio", 0.0),
                "saliency_method": getattr(args, "saliency_method", "ig"),
                "use_visual_highlight": getattr(args, "use_visual_highlight", False),
                "highlight_alpha": getattr(args, "highlight_alpha", 0.4),
                "sam_mode": "once_per_episode",
                "model_path": args.model_path,
                "seed": getattr(args, "seed", 7),
                "num_episodes": args.num_episodes,
                "ood_enabled": getattr(args, "ood_enabled", False),
                "ood_position_noise_cm": getattr(args, "ood_position_noise_cm", 0.0),
                "ood_seed": getattr(args, "ood_seed", None),
            },
            "task_id": task_id,
            "task_label": task_label,
            "success_rate": task_success_rate,
            "success_count": success_count,
            "num_episodes": args.num_episodes,
            "avg_steps": float(np.mean([s["steps"] for s in all_stats])) if all_stats else 0.0,
            "timing_ms": {
                "step_mean": float(np.mean(all_step_times) * 1000) if all_step_times else 0.0,
                "step_std": float(np.std(all_step_times) * 1000) if all_step_times else 0.0,
                "vla_mean": float(np.mean(all_vla_times) * 1000) if all_vla_times else 0.0,
                "vla_std": float(np.std(all_vla_times) * 1000) if all_vla_times else 0.0,
                "sam_mean": float(np.mean(all_sam_times) * 1000) if all_sam_times else 0.0,
                "sam_std": float(np.std(all_sam_times) * 1000) if all_sam_times else 0.0,
                "preprocess_mean": float(np.mean(all_preprocess_times) * 1000) if all_preprocess_times else 0.0,
                "env_mean": float(np.mean(all_env_times) * 1000) if all_env_times else 0.0,
                "saliency_mean": float(np.mean(all_saliency_times) * 1000) if all_saliency_times else 0.0,
                "effective_fps": float(1.0 / np.mean(all_step_times)) if all_step_times else 0.0,
            },
            "pruning": {
                "avg_ratio": float(np.mean(all_pruning_ratios)) if all_pruning_ratios else 0.0,
                "avg_patches_kept": float(np.mean(all_selected_patches)) if all_selected_patches else 256.0,
            },
            "per_episode": [
                {
                    "success": s["success"],
                    "steps": s["steps"],
                    "masks_generated": s.get("masks_generated", 0),
                }
                for s in all_stats
            ],
        }
        json_file = output_dir / f"results_{args.task_suite_name}_task{task_id}.json"
        with open(json_file, "w") as f:
            json.dump(task_json, f, indent=2)
        print(f"JSON results saved to {json_file}")

        # Store task results for summary
        all_tasks_stats[task_id] = all_stats
        all_tasks_success_rates[task_id] = task_success_rate
        
        # Clean up environment
        env.close()
    
    # Print overall summary across all tasks
    print("\n" + "="*70)
    print("OVERALL SUMMARY - ALL TASKS")
    print("="*70)
    total_success_rate = np.mean(list(all_tasks_success_rates.values())) if all_tasks_success_rates else 0
    print(f"Tasks evaluated: {len(task_ids)}")
    print(f"Overall success rate: {total_success_rate*100:.1f}%")
    print("\nPer-task success rates:")
    for tid in task_ids:
        if tid in all_tasks_success_rates:
            print(f"  Task {tid}: {all_tasks_success_rates[tid]*100:.1f}%")
    
    # Save overall summary
    summary_file = output_dir / f"summary_{args.task_suite_name}_tasks_{min(task_ids)}-{max(task_ids)}.txt"
    with open(summary_file, "w") as f:
        f.write(f"Task Suite: {args.task_suite_name}\n")
        f.write(f"Model: {args.model_path}\n")
        f.write(f"Tasks evaluated: {task_ids}\n")
        f.write(f"Episodes per task: {args.num_episodes}\n")
        f.write(f"Overall success rate: {total_success_rate*100:.1f}%\n\n")
        f.write("Per-task success rates:\n")
        for tid in task_ids:
            if tid in all_tasks_success_rates:
                f.write(f"  Task {tid}: {all_tasks_success_rates[tid]*100:.1f}%\n")
    
    print(f"\nOverall summary saved to {summary_file}")

    # Save overall JSON summary (aggregates all per-task JSON files)
    summary_json = {
        "config": {
            "use_sam": args.use_sam,
            "use_attention_bias": getattr(args, "use_attention_bias", False),
            "bias_strength": getattr(args, "bias_strength", 0.0),
            "bias_active_layers": getattr(args, "bias_active_layers", None),
            "saliency_method": getattr(args, "saliency_method", "ig"),
            "use_visual_highlight": getattr(args, "use_visual_highlight", False),
            "highlight_alpha": getattr(args, "highlight_alpha", 0.4),
            "model_path": args.model_path,
            "seed": getattr(args, "seed", 7),
            "num_episodes": args.num_episodes,
            "ood_enabled": getattr(args, "ood_enabled", False),
            "ood_position_noise_cm": getattr(args, "ood_position_noise_cm", 0.0),
            "ood_seed": getattr(args, "ood_seed", None),
        },
        "task_suite": args.task_suite_name,
        "overall_success_rate": float(total_success_rate),
        "per_task_success_rates": {
            str(tid): float(all_tasks_success_rates[tid])
            for tid in task_ids if tid in all_tasks_success_rates
        },
    }
    # Aggregate step counts and timing across all tasks
    global_vla, global_sam, global_step = [], [], []
    global_episode_steps = []
    for tid, stats_list in all_tasks_stats.items():
        for s in stats_list:
            global_episode_steps.append(s["steps"])
            t = s.get("timing", {})
            global_vla.extend(t.get("vla_times", []))
            global_sam.extend(t.get("sam_times", []))
            global_step.extend(t.get("step_times", []))
    summary_json["timing_ms"] = {
        "vla_mean": float(np.mean(global_vla) * 1000) if global_vla else 0.0,
        "vla_std": float(np.std(global_vla) * 1000) if global_vla else 0.0,
        "sam_mean": float(np.mean(global_sam) * 1000) if global_sam else 0.0,
        "sam_std": float(np.std(global_sam) * 1000) if global_sam else 0.0,
        "step_mean": float(np.mean(global_step) * 1000) if global_step else 0.0,
        "step_std": float(np.std(global_step) * 1000) if global_step else 0.0,
        "effective_fps": float(1.0 / np.mean(global_step)) if global_step else 0.0,
    }
    summary_json["episode_steps"] = {
        "mean": float(np.mean(global_episode_steps)) if global_episode_steps else 0.0,
        "std": float(np.std(global_episode_steps)) if global_episode_steps else 0.0,
        "min": int(min(global_episode_steps)) if global_episode_steps else 0,
        "max": int(max(global_episode_steps)) if global_episode_steps else 0,
    }
    summary_json["per_task_avg_steps"] = {}
    for tid in task_ids:
        if tid in all_tasks_stats:
            task_steps = [s["steps"] for s in all_tasks_stats[tid]]
            summary_json["per_task_avg_steps"][str(tid)] = float(np.mean(task_steps))

    summary_json_file = output_dir / f"summary_{args.task_suite_name}.json"
    with open(summary_json_file, "w") as f:
        json.dump(summary_json, f, indent=2)
    print(f"JSON summary saved to {summary_json_file}")

    if args.save_visualizations:
        rollout_dir = getattr(args, "rollout_dir", "./rollouts/")
        print(f"Rollout videos saved to {rollout_dir}")


if __name__ == "__main__":
    main()
