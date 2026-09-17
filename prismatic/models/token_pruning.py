"""
token_pruning.py

Token Pruning and Merging module for accelerating VLA inference.
Based on the TEAM algorithm from TeamVLA (arXiv:2512.09927).

Stages:
1. Similarity Sampling: Top-K most similar visual tokens per language token
2. Token Expanding: Density-based convolution expansion
3. Context Sampling: Importance-weighted background token sampling
4. Token Merging: Soft bipartite matching at middle decoder layer
"""

import math
from typing import Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F


class TokenPruningConfig:
    """Configuration for token pruning and merging."""

    def __init__(
        self,
        enabled: bool = False,
        density_threshold: float = 1.0,
        context_sample_ratio: float = 0.1,
        expansion_kernel_size: int = 3,
        use_token_merging: bool = True,
        merge_topk: int = 80,
        merge_layer: int = 16,
    ):
        self.enabled = enabled
        self.density_threshold = density_threshold
        self.context_sample_ratio = context_sample_ratio
        self.expansion_kernel_size = expansion_kernel_size
        self.use_token_merging = use_token_merging
        self.merge_topk = merge_topk
        self.merge_layer = merge_layer


class TokenPruningModule(nn.Module):
    """TEAM-VLA Token Pruning and Merging Module."""

    _log_counter: int = 0

    def __init__(self, config: TokenPruningConfig):
        super().__init__()
        self.config = config
        kernel_size = config.expansion_kernel_size
        self.register_buffer(
            "expansion_kernel",
            torch.ones(1, 1, kernel_size, kernel_size),
        )

    # ------------------------------------------------------------------
    # Stage 1 - Similarity Sampling  (Top-K per language token)
    # ------------------------------------------------------------------
    def compute_similarity_mask(
        self,
        visual_tokens: torch.Tensor,
        language_tokens: torch.Tensor,
        num_patches_per_side: int,
        top_k: int = 3,
    ) -> torch.Tensor:
        B, num_patches, D = visual_tokens.shape

        visual_norm = F.normalize(visual_tokens, p=2, dim=-1)
        language_norm = F.normalize(language_tokens, p=2, dim=-1)

        similarity_matrix = torch.bmm(
            language_norm, visual_norm.transpose(1, 2)
        )  # [B, num_lang, num_patches]

        top_k = min(top_k, num_patches)
        _, top_k_indices = similarity_matrix.topk(k=top_k, dim=-1)

        similarity_mask = torch.zeros(
            B, num_patches, dtype=torch.bool, device=visual_tokens.device
        )
        for b in range(B):
            unique_indices = top_k_indices[b].flatten().unique()
            similarity_mask[b, unique_indices] = True

        return similarity_mask

    # ------------------------------------------------------------------
    # Stage 2 - Token Expanding
    # ------------------------------------------------------------------
    def expand_mask(
        self,
        similarity_mask: torch.Tensor,
        num_patches_per_side: int,
        num_images: int = 1,
    ) -> torch.Tensor:
        B = similarity_mask.shape[0]
        H = W = num_patches_per_side
        patches_per_image = H * W
        kernel_size = self.config.expansion_kernel_size
        padding = kernel_size // 2
        tau = self.config.density_threshold

        all_expanded_masks = []

        for img_idx in range(num_images):
            start_idx = img_idx * patches_per_image
            end_idx = (img_idx + 1) * patches_per_image

            img_mask = similarity_mask[:, start_idx:end_idx]
            mask_2d = img_mask.float().view(B, 1, H, W)
            kernel = self.expansion_kernel.to(device=mask_2d.device, dtype=mask_2d.dtype)

            density_map = F.conv2d(mask_2d, kernel, padding=padding)

            expanded_mask = mask_2d.clone()

            # Dense expansion: F > τ → dilate full neighbourhood
            dense_positions = (density_map > tau)
            dense_expanded = F.conv2d(dense_positions.float(), kernel, padding=padding)
            dense_expanded = dense_expanded > 0

            # Sparse expansion: 0 < F ≤ τ → random neighbourhood activation
            sparse_positions = (density_map > 0) & (density_map <= tau)
            if sparse_positions.any():
                random_expansion = torch.rand_like(mask_2d)
                sparse_neighborhood = F.conv2d(
                    sparse_positions.float(), kernel, padding=padding
                )
                sparse_expanded = (sparse_neighborhood > 0) & (random_expansion > 0.5)
            else:
                sparse_expanded = torch.zeros_like(dense_expanded)

            final_mask = expanded_mask.bool() | dense_expanded | sparse_expanded
            final_mask = final_mask.view(B, -1).bool()
            all_expanded_masks.append(final_mask)

        return torch.cat(all_expanded_masks, dim=1)

    # ------------------------------------------------------------------
    # Stage 3 - Context Sampling (importance-weighted)
    # ------------------------------------------------------------------
    def context_sampling(
        self,
        expanded_mask: torch.Tensor,
        num_patches: int,
        visual_tokens: Optional[torch.Tensor] = None,
        language_tokens: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        B = expanded_mask.shape[0]
        u = self.config.context_sample_ratio

        if u <= 0:
            return expanded_mask

        final_mask = expanded_mask.clone()
        use_importance = visual_tokens is not None and language_tokens is not None

        if use_importance:
            visual_norm = F.normalize(visual_tokens, p=2, dim=-1)
            language_norm = F.normalize(language_tokens, p=2, dim=-1)
            similarity = torch.bmm(
                visual_norm, language_norm.transpose(1, 2)
            ).max(dim=-1)[0]

        for b in range(B):
            bg = torch.where(~expanded_mask[b])[0]
            if len(bg) == 0:
                continue
            num_context = max(1, int(len(bg) * u))

            if use_importance:
                bg_scores = similarity[b, bg]
                probs = F.softmax(bg_scores * 10, dim=0)
                if num_context < len(bg):
                    idx = torch.multinomial(probs, num_samples=num_context, replacement=False)
                    sampled = bg[idx]
                else:
                    sampled = bg
            else:
                if num_context < len(bg):
                    interval = len(bg) // num_context
                    sampled = bg[::interval][:num_context]
                else:
                    sampled = bg

            final_mask[b, sampled] = True

        return final_mask

    # ------------------------------------------------------------------
    # Stage 4 - Token Merging  (soft bipartite matching)
    # ------------------------------------------------------------------
    def merge_tokens_bipartite(
        self,
        image_tokens: torch.Tensor,
        task_tokens: torch.Tensor,
        action_tokens: Optional[torch.Tensor] = None,
        topk: int = 80,
    ) -> torch.Tensor:
        """Soft bipartite merging.

        For each target token, compute distribution over source tokens
        (which source should absorb this target). Then aggregate target
        information into sources via W^T @ T.
        """
        B, num_img, D = image_tokens.shape

        if num_img == 0:
            return image_tokens
        if task_tokens is None or task_tokens.shape[1] == 0:
            topk = min(topk, num_img)
            return image_tokens[:, :topk, :]
        if num_img <= topk:
            return image_tokens

        if action_tokens is not None:
            guide_tokens = torch.cat([task_tokens, action_tokens], dim=1)
        else:
            guide_tokens = task_tokens

        img_norm = F.normalize(image_tokens, p=2, dim=-1)
        guide_norm = F.normalize(guide_tokens, p=2, dim=-1)

        sim_guide = torch.bmm(
            img_norm, guide_norm.transpose(1, 2)
        )
        similarity, _ = sim_guide.max(dim=-1)

        topk = min(topk, num_img)
        _, top_indices = similarity.topk(topk, dim=1)

        source_mask = torch.zeros(
            B, num_img, dtype=torch.bool, device=image_tokens.device
        )
        source_mask.scatter_(1, top_indices, True)

        merged_list = []
        for b in range(B):
            source_idx = torch.where(source_mask[b])[0]
            target_idx = torch.where(~source_mask[b])[0]

            S = image_tokens[b, source_idx]  # [N_S, D]
            T = image_tokens[b, target_idx]  # [N_T, D]

            if T.shape[0] == 0:
                merged_list.append(S)
                continue

            def rms_norm(x, eps=1e-6):
                return x / (x.pow(2).mean(dim=-1, keepdim=True).sqrt() + eps)

            S_n = rms_norm(S)
            T_n = rms_norm(T)

            # Sim: [N_S, N_T]
            sim_matrix = torch.mm(S_n, T_n.transpose(0, 1)) / math.sqrt(D)

            # W: [N_T, N_S] - each target gets distribution over sources
            W = F.softmax(sim_matrix.transpose(0, 1), dim=-1)

            # A = W^T @ T → [N_S, D]
            A = torch.mm(W.transpose(0, 1), T)

            # s = W^T @ 1 → [N_S, 1]
            ones = torch.ones(T.shape[0], 1, device=T.device, dtype=T.dtype)
            s = torch.mm(W.transpose(0, 1), ones)

            # S' = (S + A) / (1 + s)
            S_merged = (S + A) / (1 + s)
            merged_list.append(S_merged)

        return torch.stack(merged_list, dim=0)

    # ------------------------------------------------------------------
    # Prune helper
    # ------------------------------------------------------------------
    def prune_visual_tokens(
        self,
        visual_tokens: torch.Tensor,
        context_mask: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        B, num_patches, D = visual_tokens.shape
        num_kept_per_sample = context_mask.sum(dim=1)
        max_kept = num_kept_per_sample.max().item()

        pruned_list = []
        idx_list = []

        for b in range(B):
            kept_idx = torch.where(context_mask[b])[0]
            if len(kept_idx) < max_kept:
                pad = kept_idx[-1].repeat(max_kept - len(kept_idx)) if len(kept_idx) > 0 \
                    else torch.zeros(max_kept, dtype=torch.long, device=kept_idx.device)
                kept_idx = torch.cat([kept_idx, pad])
            pruned_list.append(visual_tokens[b, kept_idx])
            idx_list.append(kept_idx)

        return torch.stack(pruned_list), torch.stack(idx_list)

    # ------------------------------------------------------------------
    # Forward
    # ------------------------------------------------------------------
    def forward(
        self,
        visual_tokens: torch.Tensor,
        language_tokens: torch.Tensor,
        num_patches_per_side: int = 16,
        num_images: int = 1,
        return_mask: bool = False,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        if not self.config.enabled:
            return visual_tokens, None

        num_total = visual_tokens.shape[1]

        similarity_mask = self.compute_similarity_mask(
            visual_tokens, language_tokens, num_patches_per_side, top_k=3,
        )
        n_anchor = similarity_mask.sum(dim=1).float().mean().item()

        expanded_mask = self.expand_mask(similarity_mask, num_patches_per_side, num_images)
        n_expanded = expanded_mask.sum(dim=1).float().mean().item()

        context_mask = self.context_sampling(
            expanded_mask, visual_tokens.shape[1],
            visual_tokens=visual_tokens,
            language_tokens=language_tokens,
        )
        n_final = context_mask.sum(dim=1).float().mean().item()

        TokenPruningModule._log_counter += 1
        if TokenPruningModule._log_counter <= 5 or TokenPruningModule._log_counter % 50 == 0:
            print(
                f"[TokenPruning] {num_total} patches -> "
                f"anchor={n_anchor:.0f}, expanded={n_expanded:.0f}, "
                f"final={n_final:.0f} ({n_final / num_total * 100:.1f}%)"
            )

        pruned_tokens, kept_indices = self.prune_visual_tokens(visual_tokens, context_mask)

        if return_mask:
            return pruned_tokens, context_mask
        return pruned_tokens, kept_indices


# ======================================================================
# Convenience functions (with module caching)
# ======================================================================

_cached_pruning_module: Optional[TokenPruningModule] = None
_cached_pruning_device: Optional[torch.device] = None


def _get_pruning_module(config: TokenPruningConfig, device: torch.device) -> TokenPruningModule:
    global _cached_pruning_module, _cached_pruning_device
    if _cached_pruning_module is None or _cached_pruning_device != device:
        _cached_pruning_module = TokenPruningModule(config)
        _cached_pruning_module = _cached_pruning_module.to(device)
        _cached_pruning_device = device
    return _cached_pruning_module


def apply_token_pruning(
    visual_tokens: torch.Tensor,
    language_tokens: torch.Tensor,
    config: TokenPruningConfig,
    num_patches_per_side: int = 16,
    num_images: int = 1,
) -> Tuple[torch.Tensor, torch.Tensor]:
    if not config.enabled:
        return visual_tokens, None
    module = _get_pruning_module(config, visual_tokens.device)
    return module(visual_tokens, language_tokens, num_patches_per_side, num_images)


def apply_token_merging(
    image_tokens: torch.Tensor,
    task_tokens: torch.Tensor,
    config: TokenPruningConfig,
    action_tokens: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    if not config.enabled or not config.use_token_merging:
        return image_tokens
    module = _get_pruning_module(config, image_tokens.device)
    return module.merge_tokens_bipartite(
        image_tokens, task_tokens, action_tokens,
        topk=config.merge_topk,
    )
