"""
Fine-tune CogACT-Base on LIBERO-Spatial for cross-architecture T3R validation.

Usage:
    cd /workspace/CogACT && python -u /workspace/T3R/experiments/robot/train_cogact_libero.py \
        --cogact_checkpoint /workspace/cogact_checkpoint/checkpoints/CogACT-Base.pt \
        --libero_data_dir /workspace/libero_demos/libero_spatial \
        --output_dir /workspace/cogact_libero_spatial \
        --freeze_vlm
"""
import argparse
import glob
import json
import os
import sys
import time
from pathlib import Path

import h5py
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from PIL import Image
from tqdm import tqdm

sys.path.insert(0, "/workspace/CogACT")
sys.path.insert(0, "/workspace/T3R")
from experiments.robot.cogact_loader import load_cogact_for_libero


class LIBEROCogACTDataset(Dataset):
    """LIBERO dataset for CogACT training. Each sample: (image, instruction, action_chunk_16x7)"""
    
    def __init__(self, data_dir, image_transform, future_action_window=15, action_norm_stats=None):
        self.data_dir = data_dir
        self.image_transform = image_transform
        self.chunk_size = future_action_window + 1  # 16 steps
        
        self.samples = []
        self._load_demos()
        
        if action_norm_stats is not None:
            self.action_norm_stats = action_norm_stats
        else:
            self.action_norm_stats = self._compute_norm_stats()
        
        print(f"Dataset: {len(self.samples)} samples from {data_dir}")
    
    def _load_demos(self):
        hdf5_files = sorted(glob.glob(os.path.join(self.data_dir, "*.hdf5")))
        for fpath in hdf5_files:
            task_name = os.path.basename(fpath).replace("_demo.hdf5", "")
            instruction = task_name.replace("_", " ")
            with h5py.File(fpath, "r") as f:
                demo_keys = sorted(f["data"].keys(), key=lambda x: int(x.split("_")[1]))
                for demo_key in demo_keys:
                    n_steps = f[f"data/{demo_key}/actions"].shape[0]
                    for step_idx in range(n_steps):
                        self.samples.append((fpath, demo_key, step_idx, instruction))
    
    def _compute_norm_stats(self):
        all_actions = []
        hdf5_files = sorted(glob.glob(os.path.join(self.data_dir, "*.hdf5")))
        for fpath in hdf5_files:
            with h5py.File(fpath, "r") as f:
                for demo_key in f["data"].keys():
                    all_actions.append(f[f"data/{demo_key}/actions"][:])
        all_actions = np.concatenate(all_actions, axis=0)
        return {
            "q01": np.percentile(all_actions, 1, axis=0).tolist(),
            "q99": np.percentile(all_actions, 99, axis=0).tolist(),
            "mean": np.mean(all_actions, axis=0).tolist(),
            "std": np.std(all_actions, axis=0).tolist(),
        }

    def normalize_action(self, action):
        q01 = np.array(self.action_norm_stats["q01"])
        q99 = np.array(self.action_norm_stats["q99"])
        denom = q99 - q01
        denom = np.where(np.abs(denom) < 1e-6, 1.0, denom)
        normalized = 2.0 * (action - q01) / denom - 1.0
        return np.clip(normalized, -1.0, 1.0)

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        fpath, demo_key, step_idx, instruction = self.samples[idx]

        with h5py.File(fpath, "r") as f:
            demo = f[f"data/{demo_key}"]
            img_array = demo["obs/agentview_rgb"][step_idx]  # (128, 128, 3)
            image = Image.fromarray(img_array)

            all_actions = demo["actions"][:]
            n_steps = all_actions.shape[0]

            chunk_indices = list(range(step_idx, min(step_idx + self.chunk_size, n_steps)))
            action_chunk = all_actions[chunk_indices]

            if len(chunk_indices) < self.chunk_size:
                pad_size = self.chunk_size - len(chunk_indices)
                action_chunk = np.concatenate(
                    [action_chunk, np.repeat(action_chunk[-1:], pad_size, axis=0)], axis=0
                )

            action_chunk = np.array([self.normalize_action(a) for a in action_chunk])

        pixel_values = self.image_transform(image)
        # pixel_values is a dict {"dino": tensor, "siglip": tensor} for fused backbone

        return {
            "pixel_values": pixel_values,
            "instruction": instruction,
            "actions": torch.tensor(action_chunk, dtype=torch.float32),
        }


def collate_fn(batch):
    """Custom collate that handles dict pixel_values."""
    instructions = [b["instruction"] for b in batch]
    actions = torch.stack([b["actions"] for b in batch])
    
    pv0 = batch[0]["pixel_values"]
    if isinstance(pv0, dict):
        pixel_values = {k: torch.stack([b["pixel_values"][k] for b in batch]) for k in pv0.keys()}
    else:
        pixel_values = torch.stack([b["pixel_values"] for b in batch])
    
    return {"pixel_values": pixel_values, "instruction": instructions, "actions": actions}


def train(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    # Load model using our custom loader
    model, image_transform, tokenizer = load_cogact_for_libero(
        args.cogact_checkpoint, freeze_vlm=args.freeze_vlm, device=str(device)
    )

    dataset = LIBEROCogACTDataset(
        data_dir=args.libero_data_dir,
        image_transform=image_transform,
        future_action_window=model.future_action_window_size,
    )

    os.makedirs(args.output_dir, exist_ok=True)
    norm_stats = {"libero_spatial": {"action": dataset.action_norm_stats}}
    stats_path = os.path.join(args.output_dir, "dataset_statistics.json")
    with open(stats_path, "w") as f:
        json.dump(norm_stats, f, indent=2)
    print(f"Saved norm stats to {stats_path}")

    dataloader = DataLoader(
        dataset, batch_size=args.batch_size, shuffle=True,
        num_workers=4, pin_memory=True, drop_last=True, collate_fn=collate_fn,
    )

    trainable_params = [p for p in model.parameters() if p.requires_grad]
    print(f"Trainable params: {sum(p.numel() for p in trainable_params)/1e6:.1f}M")

    optimizer = torch.optim.AdamW(trainable_params, lr=args.lr, weight_decay=0.01)

    global_step = 0
    best_loss = float("inf")

    optimizer.zero_grad()
    for epoch in range(args.num_epochs):
        epoch_loss = 0.0
        num_batches = 0
        model.action_model.train()
        model.vlm.projector.train()

        pbar = tqdm(dataloader, desc=f"Epoch {epoch+1}/{args.num_epochs}")
        for batch in pbar:
            # Move pixel_values to device
            if isinstance(batch["pixel_values"], dict):
                pixel_values = {k: v.to(device) for k, v in batch["pixel_values"].items()}
            else:
                pixel_values = batch["pixel_values"].to(device)
            actions = batch["actions"].to(device)
            instructions = batch["instruction"]

            # Tokenize instructions
            input_ids_list = []
            for instr in instructions:
                pb = model.vlm.get_prompt_builder()
                pb.add_turn(role="human", message=f"What action should the robot take to {instr}?")
                prompt_text = pb.get_prompt()
                ids = tokenizer(prompt_text, truncation=True, return_tensors="pt").input_ids.squeeze(0)
                ids = torch.cat([ids, torch.tensor([29871, 2])])  # EOS
                input_ids_list.append(ids)

            max_len = max(ids.shape[0] for ids in input_ids_list)
            padded_ids = torch.full((len(input_ids_list), max_len), 0, dtype=torch.long)
            attention_mask = torch.zeros((len(input_ids_list), max_len), dtype=torch.long)
            for i, ids in enumerate(input_ids_list):
                padded_ids[i, :ids.shape[0]] = ids
                attention_mask[i, :ids.shape[0]] = 1
            padded_ids = padded_ids.to(device)
            attention_mask = attention_mask.to(device)

            with torch.cuda.amp.autocast(dtype=torch.bfloat16):
                loss, _ = model(
                    input_ids=padded_ids,
                    attention_mask=attention_mask,
                    pixel_values=pixel_values,
                    labels=padded_ids,
                    actions=actions,
                    output_hidden_states=True,
                    repeated_diffusion_steps=args.repeated_diffusion_steps,
                )

            (loss / args.grad_accum_steps).backward()

            if (num_batches) % args.grad_accum_steps == 0:
                torch.nn.utils.clip_grad_norm_(trainable_params, max_norm=1.0)
                optimizer.step()
                optimizer.zero_grad()

            epoch_loss += loss.item()
            num_batches += 1
            global_step += 1
            pbar.set_postfix({"loss": f"{loss.item():.4f}", "avg": f"{epoch_loss/num_batches:.4f}"})

            if global_step % args.save_every == 0:
                save_checkpoint(model, optimizer, epoch, global_step, args.output_dir)

        avg_loss = epoch_loss / max(num_batches, 1)
        print(f"Epoch {epoch+1}: avg_loss = {avg_loss:.4f}")

        save_checkpoint(model, optimizer, epoch, global_step, args.output_dir, is_epoch_end=True)
        if avg_loss < best_loss:
            best_loss = avg_loss
            save_checkpoint(model, optimizer, epoch, global_step, args.output_dir, best=True)

    print(f"Training complete! Best loss: {best_loss:.4f}")


def save_checkpoint(model, optimizer, epoch, step, output_dir, best=False, is_epoch_end=False):
    os.makedirs(os.path.join(output_dir, "checkpoints"), exist_ok=True)
    state = {
        "model": {
            "projector": model.vlm.projector.state_dict(),
            "llm_backbone": model.vlm.llm_backbone.state_dict(),
            "vision_backbone": model.vlm.vision_backbone.state_dict(),
            "action_model": model.action_model.state_dict(),
        },
        "epoch": epoch,
        "step": step,
    }
    if best:
        path = os.path.join(output_dir, "checkpoints", "best.pt")
    elif is_epoch_end:
        path = os.path.join(output_dir, "checkpoints", f"epoch_{epoch+1}.pt")
    else:
        path = os.path.join(output_dir, "checkpoints", f"step_{step}.pt")
    torch.save(state, path)
    print(f"  Saved: {path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--cogact_checkpoint", type=str, required=True)
    parser.add_argument("--libero_data_dir", type=str, required=True)
    parser.add_argument("--output_dir", type=str, default="/workspace/cogact_libero_spatial")
    parser.add_argument("--freeze_vlm", action="store_true")
    parser.add_argument("--batch_size", type=int, default=4)
    parser.add_argument("--lr", type=float, default=2e-5)
    parser.add_argument("--num_epochs", type=int, default=50)
    parser.add_argument("--repeated_diffusion_steps", type=int, default=8)
    parser.add_argument("--save_every", type=int, default=2000)
    parser.add_argument("--grad_accum_steps", type=int, default=4)
    args = parser.parse_args()

    train(args)
