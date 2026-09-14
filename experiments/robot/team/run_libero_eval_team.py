"""
run_libero_eval.py

Evaluates a trained policy in a LIBERO simulation benchmark task suite.

Usage:
    # From YAML config file:
    python run_libero_eval_team.py --config configs/libero_eval_config.yaml

    # Override specific parameters:
    python run_libero_eval_team.py --config configs/libero_eval_config.yaml --use_token_pruning true
"""

import json
import logging
import os
import sys
import time
from collections import deque
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import Optional, Union

import draccus
import torch
import numpy as np
import tqdm
import yaml
from libero.libero import benchmark

import wandb

# PyTorch 2.6+ compatibility with LIBERO state files
_original_torch_load = torch.load
def _safe_torch_load(*args, **kwargs):
    kwargs.setdefault("weights_only", False)
    return _original_torch_load(*args, **kwargs)
torch.load = _safe_torch_load

# Append current directory so that interpreter can find experiments.robot
sys.path.append("../..")
from prismatic.models.token_pruning import TokenPruningConfig
from experiments.robot.libero.libero_utils import (
    get_libero_dummy_action,
    get_libero_env,
    get_libero_image,
    get_libero_wrist_image,
    quat2axisangle,
    save_rollout_video,
)
from experiments.robot.openvla_utils import (
    get_action_head,
    get_noisy_action_projector,
    get_processor,
    get_proprio_projector,
    resize_image_for_policy,
)
from experiments.robot.robot_utils import (
    DATE_TIME,
    get_action,
    get_image_resize_size,
    get_model,
    invert_gripper_action,
    normalize_gripper_action,
    set_seed_everywhere,
)
from prismatic.vla.constants import NUM_ACTIONS_CHUNK


# Define task suite constants
class TaskSuite(str, Enum):
    LIBERO_SPATIAL = "libero_spatial"
    LIBERO_OBJECT = "libero_object"
    LIBERO_GOAL = "libero_goal"
    LIBERO_10 = "libero_10"
    LIBERO_90 = "libero_90"


# Define max steps for each task suite
TASK_MAX_STEPS = {
    TaskSuite.LIBERO_SPATIAL: 220,  # longest training demo has 193 steps
    TaskSuite.LIBERO_OBJECT: 280,  # longest training demo has 254 steps
    TaskSuite.LIBERO_GOAL: 300,  # longest training demo has 270 steps
    TaskSuite.LIBERO_10: 520,  # longest training demo has 505 steps
    TaskSuite.LIBERO_90: 400,  # longest training demo has 373 steps
}


# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[logging.StreamHandler()],
)
logger = logging.getLogger(__name__)


@dataclass
class GenerateConfig:
    # fmt: off

    #################################################################################################################
    # Config file path (for loading from YAML)
    #################################################################################################################
    config: Optional[str] = None                     # Path to YAML config file

    #################################################################################################################
    # Model-specific parameters
    #################################################################################################################
    model_family: str = "openvla"                    # Model family
    pretrained_checkpoint: Union[str, Path] = ""     # Pretrained checkpoint path

    use_l1_regression: bool = True                   # If True, uses continuous action head with L1 regression objective
    use_diffusion: bool = False                      # If True, uses continuous action head with diffusion modeling objective (DDIM)
    num_diffusion_steps_train: int = 50              # (When `diffusion==True`) Number of diffusion steps used for training
    num_diffusion_steps_inference: int = 50          # (When `diffusion==True`) Number of diffusion steps used for inference
    use_film: bool = False                           # If True, uses FiLM to infuse language inputs into visual features
    num_images_in_input: int = 2                     # Number of images in the VLA input (default: 1)
    use_proprio: bool = True                         # Whether to include proprio state in input

    center_crop: bool = True                         # Center crop? (if trained w/ random crop image aug)
    num_open_loop_steps: int = 8                     # Number of actions to execute open-loop before requerying policy

    lora_rank: int = 32                              # Rank of LoRA weight matrix (MAKE SURE THIS MATCHES TRAINING!)

    unnorm_key: Union[str, Path] = ""                # Action un-normalization key

    load_in_8bit: bool = False                       # (For OpenVLA only) Load with 8-bit quantization
    load_in_4bit: bool = False                       # (For OpenVLA only) Load with 4-bit quantization

    #################################################################################################################
    # LIBERO environment-specific parameters
    #################################################################################################################
    task_suite_name: str = TaskSuite.LIBERO_SPATIAL  # Task suite
    num_steps_wait: int = 10                         # Number of steps to wait for objects to stabilize in sim
    num_trials_per_task: int = 50                    # Number of rollouts per task
    initial_states_path: str = "DEFAULT"             # "DEFAULT", or path to initial states JSON file
    env_img_res: int = 256                           # Resolution for environment images (not policy input resolution)

    #################################################################################################################
    # Utils
    #################################################################################################################
    run_id_note: Optional[str] = None                # Extra note to add to end of run ID for logging
    local_log_dir: str = "./experiments/logs"        # Local directory for eval logs

    use_wandb: bool = False                          # Whether to also log results in Weights & Biases
    wandb_entity: str = "your-wandb-entity"          # Name of WandB entity
    wandb_project: str = "your-wandb-project"        # Name of WandB project

    seed: int = 7                                    # Random Seed (for reproducibility)

    #################################################################################################################
    # Token Pruning (TEAM) parameters
    #################################################################################################################
    use_token_pruning: bool = False                  # Whether to enable token pruning
    token_density_threshold: float = 1.0             # Threshold τ for dense/sparse area classification
    token_context_sample_ratio: float = 0.1          # Ratio u for context sampling (background tokens)
    token_expansion_kernel_size: int = 3             # Kernel size K for mask expansion
    use_token_merging: bool = True                   # Whether to use token merging
    token_merge_topk: int = 80                       # Top-M tokens for source set in merging
    token_merge_layer: int = 16                      # Layer index for token merging (middle layer)

    # fmt: on


def load_config_from_yaml(yaml_path: str) -> dict:
    """Load configuration from a YAML file."""
    with open(yaml_path, 'r') as f:
        config = yaml.safe_load(f)
    return config if config else {}


def merge_configs(cfg: GenerateConfig, yaml_config: dict) -> GenerateConfig:
    """Merge YAML config into dataclass config (YAML values are used as defaults, CLI args override)."""
    for key, value in yaml_config.items():
        if hasattr(cfg, key):
            # Only set from YAML if the CLI didn't explicitly set it (i.e., it's still default)
            current_value = getattr(cfg, key)
            default_value = GenerateConfig.__dataclass_fields__[key].default
            
            # Check if field has default_factory
            if default_value is dataclass_field_missing:
                default_factory = GenerateConfig.__dataclass_fields__[key].default_factory
                if default_factory is not dataclass_field_missing:
                    default_value = default_factory()
            
            # If current value equals default, use YAML value
            if current_value == default_value:
                setattr(cfg, key, value)
    return cfg


# Import for checking default field values
from dataclasses import MISSING as dataclass_field_missing


def validate_config(cfg: GenerateConfig) -> None:
    """Validate configuration parameters."""
    assert cfg.pretrained_checkpoint is not None, "pretrained_checkpoint must not be None!"

    if "image_aug" in str(cfg.pretrained_checkpoint):
        assert cfg.center_crop, "Expecting `center_crop==True` because model was trained with image augmentations!"

    assert not (cfg.load_in_8bit and cfg.load_in_4bit), "Cannot use both 8-bit and 4-bit quantization!"

    # Validate task suite
    assert cfg.task_suite_name in [suite.value for suite in TaskSuite], f"Invalid task suite: {cfg.task_suite_name}"


def initialize_model(cfg: GenerateConfig):
    """Initialize model and associated components."""
    # Load model
    model = get_model(cfg)

    # Load proprio projector if needed
    proprio_projector = None
    if cfg.use_proprio:
        proprio_projector = get_proprio_projector(
            cfg,
            model.llm_dim,
            proprio_dim=8,  # 8-dimensional proprio for LIBERO
        )

    # Load action head if needed
    action_head = None
    if cfg.use_l1_regression or cfg.use_diffusion:
        action_head = get_action_head(cfg, model.llm_dim)

    # Load noisy action projector if using diffusion
    noisy_action_projector = None
    if cfg.use_diffusion:
        noisy_action_projector = get_noisy_action_projector(cfg, model.llm_dim)

    # Get OpenVLA processor if needed
    processor = None
    if cfg.model_family == "openvla":
        processor = get_processor(cfg)
        check_unnorm_key(cfg, model)

    # Initialize token pruning config if enabled
    token_pruning_config = None
    if cfg.use_token_pruning:
        token_pruning_config = TokenPruningConfig(
            enabled=cfg.use_token_pruning,
            density_threshold=cfg.token_density_threshold,
            context_sample_ratio=cfg.token_context_sample_ratio,
            expansion_kernel_size=cfg.token_expansion_kernel_size,
            use_token_merging=cfg.use_token_merging,
            merge_topk=cfg.token_merge_topk,
            merge_layer=cfg.token_merge_layer,
        )
        logger.info(f"Token pruning enabled with config: {token_pruning_config.__dict__}")

    return model, action_head, proprio_projector, noisy_action_projector, processor, token_pruning_config


def check_unnorm_key(cfg: GenerateConfig, model) -> None:
    """Check that the model contains the action un-normalization key."""
    # Initialize unnorm_key
    unnorm_key = cfg.task_suite_name

    # In some cases, the key must be manually modified (e.g. after training on a modified version of the dataset
    # with the suffix "_no_noops" in the dataset name)
    if unnorm_key not in model.norm_stats and f"{unnorm_key}_no_noops" in model.norm_stats:
        unnorm_key = f"{unnorm_key}_no_noops"

    assert unnorm_key in model.norm_stats, f"Action un-norm key {unnorm_key} not found in VLA `norm_stats`!"

    # Set the unnorm_key in cfg
    cfg.unnorm_key = unnorm_key


def setup_logging(cfg: GenerateConfig):
    """Set up logging to file and optionally to wandb."""
    # Create run ID
    run_id = f"EVAL-{cfg.task_suite_name}-{cfg.model_family}-{DATE_TIME}"
    if cfg.run_id_note is not None:
        run_id += f"--{cfg.run_id_note}"

    # Set up local logging
    os.makedirs(cfg.local_log_dir, exist_ok=True)
    local_log_filepath = os.path.join(cfg.local_log_dir, run_id + ".txt")
    log_file = open(local_log_filepath, "w")
    logger.info(f"Logging to local log file: {local_log_filepath}")

    # Initialize Weights & Biases logging if enabled
    if cfg.use_wandb:
        wandb.init(
            entity=cfg.wandb_entity,
            project=cfg.wandb_project,
            name=run_id,
        )

    return log_file, local_log_filepath, run_id


def log_message(message: str, log_file=None):
    """Log a message to console and optionally to a log file."""
    logger.info(message)
    if log_file:
        log_file.write(message + "\n")
        log_file.flush()


def load_initial_states(cfg: GenerateConfig, task_suite, task_id: int, log_file=None):
    """Load initial states for the given task."""
    # Get default initial states
    initial_states = task_suite.get_task_init_states(task_id)

    # If using custom initial states, load them from file
    if cfg.initial_states_path != "DEFAULT":
        with open(cfg.initial_states_path, "r") as f:
            all_initial_states = json.load(f)
        log_message(f"Using initial states from {cfg.initial_states_path}", log_file)
        return initial_states, all_initial_states
    else:
        log_message("Using default initial states", log_file)
        return initial_states, None


def prepare_observation(obs, resize_size):
    """Prepare observation for policy input."""
    # Get preprocessed images
    img = get_libero_image(obs)
    wrist_img = get_libero_wrist_image(obs)

    # Resize images to size expected by model
    img_resized = resize_image_for_policy(img, resize_size)
    wrist_img_resized = resize_image_for_policy(wrist_img, resize_size)

    # Prepare observations dict
    observation = {
        "full_image": img_resized,
        "wrist_image": wrist_img_resized,
        "state": np.concatenate(
            (obs["robot0_eef_pos"], quat2axisangle(obs["robot0_eef_quat"]), obs["robot0_gripper_qpos"])
        ),
    }

    return observation, img  # Return both processed observation and original image for replay


def process_action(action, model_family):
    """Process action before sending to environment."""
    # Normalize gripper action [0,1] -> [-1,+1] because the environment expects the latter
    action = normalize_gripper_action(action, binarize=True)

    # [OpenVLA] The dataloader flips the sign of the gripper action to align with other datasets
    # (0 = close, 1 = open), so flip it back (-1 = open, +1 = close) before executing the action
    if model_family == "openvla":
        action = invert_gripper_action(action)

    return action


def run_episode(
    cfg: GenerateConfig,
    env,
    task_description: str,
    model,
    resize_size,
    processor=None,
    action_head=None,
    proprio_projector=None,
    noisy_action_projector=None,
    initial_state=None,
    log_file=None,
    token_pruning_config=None,
):
    """Run a single episode in the environment."""
    # Reset environment
    env.reset()

    # Set initial state if provided
    if initial_state is not None:
        obs = env.set_init_state(initial_state)
    else:
        obs = env.get_observation()

    # Initialize action queue
    if cfg.num_open_loop_steps != NUM_ACTIONS_CHUNK:
        print(f"WARNING: cfg.num_open_loop_steps ({cfg.num_open_loop_steps}) does not match the NUM_ACTIONS_CHUNK "
              f"({NUM_ACTIONS_CHUNK}) constant defined in prismatic.vla.constants! For best performance (in terms of "
               "both speed and success rate), we recommend executing the full action chunk.")
    action_queue = deque(maxlen=cfg.num_open_loop_steps)

    # Setup
    t = 0
    replay_images = []
    max_steps = TASK_MAX_STEPS[cfg.task_suite_name]

    # Run episode
    success = False
    vla_times = []
    step_times = []
    episode_start = time.time()
    try:
        while t < max_steps + cfg.num_steps_wait:
            if t < cfg.num_steps_wait:
                obs, reward, done, info = env.step(get_libero_dummy_action(cfg.model_family))
                t += 1
                continue

            step_start = time.time()
            observation, img = prepare_observation(obs, resize_size)
            replay_images.append(img)

            if len(action_queue) == 0:
                vla_start = time.time()
                actions = get_action(
                    cfg,
                    model,
                    observation,
                    task_description,
                    processor=processor,
                    action_head=action_head,
                    proprio_projector=proprio_projector,
                    noisy_action_projector=noisy_action_projector,
                    use_film=cfg.use_film,
                    token_pruning_config=token_pruning_config,
                )
                vla_time = time.time() - vla_start
                vla_times.append(vla_time)
                n_vla = len(vla_times)
                print(f"    VLA call {n_vla}: {vla_time*1000:.0f}ms")
                action_queue.extend(actions)

            action = action_queue.popleft()
            action = process_action(action, cfg.model_family)

            obs, reward, done, info = env.step(action.tolist())
            step_time = time.time() - step_start
            step_times.append(step_time)
            action_step = t - cfg.num_steps_wait
            if action_step > 0 and action_step % 50 == 0:
                avg_vla = sum(vla_times) / len(vla_times) * 1000
                print(f"    [step {action_step}] avg VLA: {avg_vla:.0f}ms | VLA calls: {len(vla_times)} | elapsed: {sum(step_times):.0f}s")
            if done:
                success = True
                break
            t += 1

    except Exception as e:
        log_message(f"Episode error: {e}", log_file)

    episode_time = time.time() - episode_start
    step_count = t - cfg.num_steps_wait

    import numpy as _np
    print(f"Timing (this episode):")
    if step_times:
        print(f"  Total step time: {_np.mean(step_times)*1000:.1f}ms avg, {sum(step_times):.2f}s total")
    if vla_times:
        print(f"  VLA inference:   {_np.mean(vla_times)*1000:.1f}ms avg, {sum(vla_times):.2f}s total")
        print(f"  Effective FPS:   {len(step_times)/sum(step_times):.1f}" if step_times else "")

    episode_stats = {
        "success": success,
        "steps": step_count,
        "episode_time": episode_time,
        "vla_times": vla_times,
        "step_times": step_times,
    }
    return success, replay_images, episode_stats


def run_task(
    cfg: GenerateConfig,
    task_suite,
    task_id: int,
    model,
    resize_size,
    processor=None,
    action_head=None,
    proprio_projector=None,
    noisy_action_projector=None,
    total_episodes=0,
    total_successes=0,
    log_file=None,
    token_pruning_config=None,
):
    """Run evaluation for a single task."""
    # Get task
    task = task_suite.get_task(task_id)

    # Get initial states
    initial_states, all_initial_states = load_initial_states(cfg, task_suite, task_id, log_file)

    # Initialize environment and get task description
    env, task_description = get_libero_env(task, cfg.model_family, resolution=cfg.env_img_res)

    # Start episodes
    task_episodes, task_successes = 0, 0
    task_stats = []
    for episode_idx in tqdm.tqdm(range(cfg.num_trials_per_task)):
        log_message(f"\nTask: {task_description}", log_file)

        # Handle initial state
        if cfg.initial_states_path == "DEFAULT":
            # Use default initial state
            initial_state = initial_states[episode_idx]
        else:
            # Get keys for fetching initial episode state from JSON
            initial_states_task_key = task_description.replace(" ", "_")
            episode_key = f"demo_{episode_idx}"

            # Skip episode if expert demonstration failed to complete the task
            if not all_initial_states[initial_states_task_key][episode_key]["success"]:
                log_message(f"Skipping task {task_id} episode {episode_idx} due to failed expert demo!", log_file)
                continue

            # Get initial state
            initial_state = np.array(all_initial_states[initial_states_task_key][episode_key]["initial_state"])

        log_message(f"Starting episode {task_episodes + 1}...", log_file)

        # Run episode
        success, replay_images, episode_stats = run_episode(
            cfg,
            env,
            task_description,
            model,
            resize_size,
            processor,
            action_head,
            proprio_projector,
            noisy_action_projector,
            initial_state,
            log_file,
            token_pruning_config,
        )

        # Update counters
        task_episodes += 1
        total_episodes += 1
        task_stats.append(episode_stats)
        if success:
            task_successes += 1
            total_successes += 1

        # Save replay video
        save_rollout_video(
            replay_images, total_episodes, success=success, task_description=task_description, log_file=log_file
        )

        # Log results for this rollout
        log_message(f"Success: {success}", log_file)
        log_message(f"Episode {task_episodes} completed - Task: {task_description}", log_file)
        log_message(f"Episode total count: {total_episodes}, Total successes: {total_successes} ({total_successes / total_episodes * 100:.1f}%)", log_file)
        log_message(f"Task progress: {task_successes}/{task_episodes} ({task_successes / task_episodes * 100:.1f}% success rate)", log_file)
        log_message("-" * 80, log_file)

    # Log task results
    task_success_rate = float(task_successes) / float(task_episodes) if task_episodes > 0 else 0
    total_success_rate = float(total_successes) / float(total_episodes) if total_episodes > 0 else 0

    log_message("=" * 80, log_file)
    log_message(f"TASK {task_id} COMPLETED: {task_description}", log_file)
    log_message(f"Task episodes: {task_episodes}", log_file)
    log_message(f"Task successes: {task_successes}", log_file)
    log_message(f"Task success rate: {task_success_rate:.4f} ({task_success_rate * 100:.1f}%)", log_file)
    log_message(f"Cumulative total episodes: {total_episodes}", log_file)
    log_message(f"Cumulative total successes: {total_successes}", log_file)
    log_message(f"Cumulative success rate: {total_success_rate:.4f} ({total_success_rate * 100:.1f}%)", log_file)

    import numpy as _np
    task_vla = [t for s in task_stats for t in s["vla_times"]]
    task_step = [t for s in task_stats for t in s["step_times"]]
    task_steps_counts = [s["steps"] for s in task_stats]
    if task_vla:
        log_message(f"Task {task_id} Timing:", log_file)
        log_message(f"  VLA latency:  {_np.mean(task_vla)*1000:.0f}ms avg (std {_np.std(task_vla)*1000:.0f}ms)", log_file)
        log_message(f"  Step latency: {_np.mean(task_step)*1000:.0f}ms avg", log_file)
        log_message(f"  Effective FPS: {len(task_step)/sum(task_step):.1f}", log_file)
        log_message(f"  Avg steps:    {_np.mean(task_steps_counts):.0f}", log_file)
    log_message("=" * 80, log_file)

    # Log to wandb if enabled
    if cfg.use_wandb:
        wandb.log(
            {
                f"success_rate/{task_description}": task_success_rate,
                f"num_episodes/{task_description}": task_episodes,
            }
        )

    return total_episodes, total_successes, task_stats



def _write_json_summary(cfg, all_task_stats, total_episodes, total_successes, total_eval_time, num_tasks):
    import numpy as _np

    all_vla_times, all_step_times, all_steps, all_episode_times = [], [], [], []
    per_task = {}
    for tid, stats_list in all_task_stats.items():
        successes = sum(1 for s in stats_list if s["success"])
        per_task[str(tid)] = successes / len(stats_list) if stats_list else 0.0
        for s in stats_list:
            all_steps.append(s["steps"])
            all_episode_times.append(s["episode_time"])
            all_vla_times.extend(s["vla_times"])
            all_step_times.extend(s["step_times"])

    summary = {
        "condition": "teamvla",
        "config": {
            "use_token_pruning": cfg.use_token_pruning,
            "token_density_threshold": cfg.token_density_threshold,
            "token_context_sample_ratio": cfg.token_context_sample_ratio,
            "token_merge_topk": cfg.token_merge_topk,
            "model_path": str(cfg.pretrained_checkpoint),
            "seed": cfg.seed,
            "num_episodes": cfg.num_trials_per_task,
        },
        "task_suite": cfg.task_suite_name,
        "overall_success_rate": total_successes / total_episodes if total_episodes else 0.0,
        "per_task_success_rates": per_task,
        "episode_steps": {
            "mean": float(_np.mean(all_steps)) if all_steps else 0,
            "std": float(_np.std(all_steps)) if all_steps else 0,
            "min": int(min(all_steps)) if all_steps else 0,
            "max": int(max(all_steps)) if all_steps else 0,
        },
        "vla_inference_ms": {
            "mean": float(_np.mean(all_vla_times)) * 1000 if all_vla_times else 0,
            "std": float(_np.std(all_vla_times)) * 1000 if all_vla_times else 0,
        },
        "step_time_ms": {
            "mean": float(_np.mean(all_step_times)) * 1000 if all_step_times else 0,
            "std": float(_np.std(all_step_times)) * 1000 if all_step_times else 0,
        },
        "effective_fps": len(all_step_times) / sum(all_step_times) if all_step_times else 0,
        "total_eval_time_s": total_eval_time,
    }

    out_dir = getattr(cfg, "output_dir", "./eval_results/ablation/teamvla")
    os.makedirs(out_dir, exist_ok=True)
    path = os.path.join(out_dir, f"summary_{cfg.task_suite_name}.json")
    with open(path, "w") as f:
        json.dump(summary, f, indent=2)
    logger.info(f"JSON summary saved to {path}")


@draccus.wrap()
def eval_libero(cfg: GenerateConfig) -> float:
    """Main function to evaluate a trained policy on LIBERO benchmark tasks."""
    # Load config from YAML if provided
    if cfg.config is not None:
        yaml_config = load_config_from_yaml(cfg.config)
        cfg = merge_configs(cfg, yaml_config)
        logger.info(f"Loaded config from: {cfg.config}")

    # Validate configuration
    validate_config(cfg)

    # Set random seed
    set_seed_everywhere(cfg.seed)

    # Initialize model and components
    model, action_head, proprio_projector, noisy_action_projector, processor, token_pruning_config = initialize_model(cfg)

    # Get expected image dimensions
    resize_size = get_image_resize_size(cfg)

    # Setup logging
    log_file, local_log_filepath, run_id = setup_logging(cfg)

    # Initialize LIBERO task suite
    benchmark_dict = benchmark.get_benchmark_dict()
    task_suite = benchmark_dict[cfg.task_suite_name]()
    num_tasks = task_suite.n_tasks

    log_message(f"Task suite: {cfg.task_suite_name}", log_file)
    if cfg.use_token_pruning:
        log_message(f"Token pruning enabled: density_threshold={cfg.token_density_threshold}, "
                    f"context_sample_ratio={cfg.token_context_sample_ratio}, merge_topk={cfg.token_merge_topk}", log_file)

    # Start evaluation
    total_episodes, total_successes = 0, 0
    all_task_stats = {}
    eval_start_time = time.time()
    for task_id in tqdm.tqdm(range(num_tasks)):
        total_episodes, total_successes, task_stats = run_task(
            cfg,
            task_suite,
            task_id,
            model,
            resize_size,
            processor,
            action_head,
            proprio_projector,
            noisy_action_projector,
            total_episodes,
            total_successes,
            log_file,
            token_pruning_config,
        )
        all_task_stats[task_id] = task_stats

    total_eval_time = time.time() - eval_start_time

    # Write JSON summary
    _write_json_summary(cfg, all_task_stats, total_episodes, total_successes, total_eval_time, num_tasks)

    # Calculate final success rate
    final_success_rate = float(total_successes) / float(total_episodes) if total_episodes > 0 else 0

    # Log final results
    log_message("\n" + "=" * 80, log_file)
    log_message("FINAL EVALUATION RESULTS", log_file)
    log_message("=" * 80, log_file)
    log_message(f"Task suite: {cfg.task_suite_name}", log_file)
    log_message(f"Total tasks evaluated: {num_tasks}", log_file)
    log_message(f"Total episodes: {total_episodes}", log_file)
    log_message(f"Total successes: {total_successes}", log_file)
    log_message(f"Overall success rate: {final_success_rate:.4f} ({final_success_rate * 100:.1f}%)", log_file)
    if cfg.use_token_pruning:
        log_message(f"Token pruning config: density_threshold={cfg.token_density_threshold}, "
                    f"context_sample_ratio={cfg.token_context_sample_ratio}, merge_topk={cfg.token_merge_topk}", log_file)
    log_message("=" * 80, log_file)

    # Log to wandb if enabled
    if cfg.use_wandb:
        wandb.log(
            {
                "success_rate/total": final_success_rate,
                "num_episodes/total": total_episodes,
            }
        )
        wandb.save(local_log_filepath)

    # Close log file
    if log_file:
        log_file.close()

    return final_success_rate


if __name__ == "__main__":
    eval_libero()
