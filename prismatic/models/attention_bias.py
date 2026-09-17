"""
attention_bias.py

Layer-selective attention biasing via Captum Integrated Gradients for VLA models.

This module implements the attention biasing pipeline:
1. Compute text-token saliency using Captum's LayerIntegratedGradients
2. Monkey-patch F.scaled_dot_product_attention to inject additive bias at those positions
3. Support layer-selective biasing (e.g., only layers 8-23)

The bias is applied in the KEY dimension of SDPA, steering attention toward high-saliency
text token positions in the multimodal sequence.

IMPORTANT — Position mapping:
  The multimodal sequence layout in predict_action is:
    [BOS (1)] [visual_patches (NUM_PATCHES)] [text_tokens (input_ids[1:])] [action_tokens]
  Saliency is computed over input_ids (text tokens). To map a text token at position `i`
  in input_ids to its position in the multimodal sequence:
    multimodal_pos = 1 + NUM_PATCHES + (i - 1)   [since input_ids[0] is BOS]
  This offset changes when patch_mask prunes visual tokens.
"""

from contextlib import contextmanager
from dataclasses import dataclass
from typing import Dict, List, Optional, Set, Tuple

import numpy as np
import torch
import torch.nn.functional as F


# ============ Saliency Data ============

@dataclass
class SaliencyInfo:
    """Stores saliency information for attention biasing."""
    token_saliency: np.ndarray   # Per-token saliency scores (over input_ids)
    top_positions: np.ndarray    # Indices of high-saliency tokens (in input_ids space)
    prompt_len: int              # Length of the prompt in input_ids


# ============ Global Bias State ============

class BiasState:
    """Global mutable state that the monkey-patched SDPA reads at each call."""
    active: bool = False
    # Positions to bias in the MULTIMODAL sequence (already offset-adjusted)
    saliency_positions: Optional[np.ndarray] = None
    saliency_weights: Optional[np.ndarray] = None
    strength: float = 2.0

    # Layer/head selection
    current_layer: int = -1
    active_layers: Optional[Set[int]] = None
    active_heads: Optional[Set[int]] = None
    num_layers: int = 32
    num_heads: int = 32

    # Row mask: only bias the last q_window query positions (action tokens)
    q_window: int = 64

    # Pre-built tensors (computed once per forward pass, reused across layers)
    _cached_bias_1d: Optional[torch.Tensor] = None  # [K] bias vector
    _cached_combined_mask: Optional[torch.Tensor] = None  # [1, 1, Q, K] causal+bias
    _cached_mask_key: Optional[tuple] = None  # (Q, K, B, H, is_causal, has_attn_mask) cache key
    _cached_seq_len: int = -1  # K that the bias_1d cache was built for
    _cached_device: Optional[torch.device] = None
    _cached_dtype: Optional[torch.dtype] = None


_BIAS_STATE = BiasState()

# Keep a reference to the real SDPA before any patching
_ORIGINAL_SDPA = F.scaled_dot_product_attention

# Track installed hooks for cleanup
_LAYER_HOOKS: List[torch.utils.hooks.RemovableHandle] = []


# ============ Layer Hook Management ============

def install_layer_hooks(model: torch.nn.Module) -> None:
    """Register forward_pre_hooks on each self_attn module to track current layer.

    This sets _BIAS_STATE.current_layer before each layer's attention,
    allowing the biased SDPA to know which layer is calling it.
    """
    global _LAYER_HOOKS

    remove_layer_hooks()

    lm = getattr(model, "language_model", model)
    inner = getattr(lm, "model", lm)
    layers = getattr(inner, "layers", None)

    if layers is None:
        attn_modules = []
        for name, mod in model.named_modules():
            if "self_attn" in name and not any(
                sub in name for sub in ["q_proj", "k_proj", "v_proj", "o_proj"]
            ):
                attn_modules.append((name, mod))
        if not attn_modules:
            print("[attention_bias] ERROR: No self_attn modules found.")
            return
        layers_list = attn_modules
    else:
        layers_list = [
            (f"layer_{i}", getattr(layer, "self_attn", layer))
            for i, layer in enumerate(layers)
        ]

    _BIAS_STATE.num_layers = len(layers_list)

    for layer_idx, (_name, attn_module) in enumerate(layers_list):
        def _make_hook(idx: int):
            def hook(module, args):
                _BIAS_STATE.current_layer = idx
            return hook

        handle = attn_module.register_forward_pre_hook(_make_hook(layer_idx))
        _LAYER_HOOKS.append(handle)

    config = getattr(model, "config", None)
    lm_config = getattr(getattr(model, "language_model", None), "config", None)
    for cfg in [lm_config, config]:
        if cfg is not None and hasattr(cfg, "num_attention_heads"):
            _BIAS_STATE.num_heads = cfg.num_attention_heads
            break

    print(
        f"[attention_bias] Installed {len(_LAYER_HOOKS)} layer hooks "
        f"({_BIAS_STATE.num_layers} layers, {_BIAS_STATE.num_heads} heads/layer)"
    )


def remove_layer_hooks() -> None:
    """Remove all registered layer hooks."""
    global _LAYER_HOOKS
    for h in _LAYER_HOOKS:
        try:
            h.remove()
        except Exception:
            pass
    _LAYER_HOOKS = []


# ============ Biased SDPA ============

def _build_bias_1d(K: int, device: torch.device, dtype: torch.dtype) -> Optional[torch.Tensor]:
    """Build the [K] bias vector once and cache it on _BIAS_STATE.

    Returns the cached tensor if K/device/dtype match, otherwise rebuilds.
    """
    if (
        _BIAS_STATE._cached_bias_1d is not None
        and _BIAS_STATE._cached_seq_len == K
        and _BIAS_STATE._cached_device == device
        and _BIAS_STATE._cached_dtype == dtype
    ):
        return _BIAS_STATE._cached_bias_1d

    all_positions = torch.as_tensor(
        _BIAS_STATE.saliency_positions, device=device, dtype=torch.long
    )
    valid_mask = (all_positions >= 0) & (all_positions < K)
    idx = all_positions[valid_mask]

    if idx.numel() == 0:
        _BIAS_STATE._cached_bias_1d = None
        _BIAS_STATE._cached_seq_len = K
        _BIAS_STATE._cached_device = device
        _BIAS_STATE._cached_dtype = dtype
        return None

    weights = _BIAS_STATE.saliency_weights
    strength = _BIAS_STATE.strength

    if weights is not None:
        all_weights = torch.as_tensor(weights, device=device, dtype=dtype)
        filtered_weights = all_weights[valid_mask]
    else:
        filtered_weights = torch.ones(idx.numel(), device=device, dtype=dtype)

    bias_k = torch.zeros(K, device=device, dtype=dtype)
    bias_k.scatter_(0, idx, strength * filtered_weights)

    _BIAS_STATE._cached_bias_1d = bias_k
    _BIAS_STATE._cached_seq_len = K
    _BIAS_STATE._cached_device = device
    _BIAS_STATE._cached_dtype = dtype
    return bias_k


def _biased_sdpa(
    query, key, value, attn_mask=None, dropout_p=0.0, is_causal=False, scale=None, **kwargs
):
    """SDPA wrapper that injects additive attention bias at salient key positions.

    Respects layer-level and head-level selection via _BIAS_STATE.
    Falls through to the original SDPA when bias is inactive or the
    current layer is not in the active set.

    Both the [K] bias vector and the full [1,1/H,Q,K] combined mask
    (causal + bias) are built once per forward pass and reused across
    all layers, avoiding repeated allocation, triu, and addition.
    """
    if not _BIAS_STATE.active or _BIAS_STATE.saliency_positions is None:
        return _ORIGINAL_SDPA(
            query, key, value, attn_mask=attn_mask,
            dropout_p=dropout_p, is_causal=is_causal, scale=scale, **kwargs
        )

    current_layer = _BIAS_STATE.current_layer
    if (
        _BIAS_STATE.active_layers is not None
        and current_layer not in _BIAS_STATE.active_layers
    ):
        return _ORIGINAL_SDPA(
            query, key, value, attn_mask=attn_mask,
            dropout_p=dropout_p, is_causal=is_causal, scale=scale, **kwargs
        )

    B, H, Q, D = query.shape
    K = key.shape[-2]

    # Check if we have a cached combined mask that matches this call's shape
    mask_key = (Q, K, B, H, is_causal, attn_mask is not None)
    if _BIAS_STATE._cached_combined_mask is not None and _BIAS_STATE._cached_mask_key == mask_key:
        return _ORIGINAL_SDPA(
            query, key, value, attn_mask=_BIAS_STATE._cached_combined_mask,
            dropout_p=dropout_p, is_causal=False, scale=scale, **kwargs
        )

    # Build the [K] bias vector (cached across layers)
    bias_k = _build_bias_1d(K, query.device, query.dtype)
    if bias_k is None:
        return _ORIGINAL_SDPA(
            query, key, value, attn_mask=attn_mask,
            dropout_p=dropout_p, is_causal=is_causal, scale=scale, **kwargs
        )

    # [1, 1, 1, K] or [1, H, 1, K] with head masking
    bias_4d = bias_k.view(1, 1, 1, K)
    if _BIAS_STATE.active_heads is not None and H > 0:
        head_mask = torch.zeros(H, device=query.device, dtype=query.dtype)
        for h_idx in _BIAS_STATE.active_heads:
            if 0 <= h_idx < H:
                head_mask[h_idx] = 1.0
        bias_4d = bias_k.view(1, 1, 1, K) * head_mask.view(1, H, 1, 1)

    # Row mask: only bias the last q_window query positions (action tokens).
    # This preserves learned attention patterns for visual and text queries;
    # only the action-prediction queries get steered toward salient keys.
    q_window = getattr(_BIAS_STATE, 'q_window', 64)
    if Q > 1 and q_window < Q:
        row_mask = torch.zeros(Q, device=query.device, dtype=query.dtype)
        row_mask[max(0, Q - q_window):] = 1.0
        bias_4d = bias_4d * row_mask.view(1, 1, Q, 1)
    
    # Expand to [B, 1/H, Q, K]
    bias_4d = bias_4d.expand(B, -1, Q, -1)

    # Build causal mask if needed
    if attn_mask is None and is_causal:
        causal = torch.ones(Q, K, device=query.device, dtype=query.dtype)
        causal = torch.triu(causal, diagonal=K - Q + 1) * torch.finfo(query.dtype).min
        attn_mask = causal.view(1, 1, Q, K)

    if attn_mask is None:
        combined_mask = bias_4d
    else:
        combined_mask = attn_mask.to(dtype=query.dtype) + bias_4d

    # Cache for subsequent layers in this forward pass
    _BIAS_STATE._cached_combined_mask = combined_mask
    _BIAS_STATE._cached_mask_key = mask_key

    return _ORIGINAL_SDPA(
        query, key, value, attn_mask=combined_mask,
        dropout_p=dropout_p, is_causal=False, scale=scale, **kwargs
    )


def install_biased_sdpa() -> None:
    """Replace F.scaled_dot_product_attention with the biased wrapper."""
    F.scaled_dot_product_attention = _biased_sdpa
    print("[attention_bias] Installed biased SDPA wrapper")


def uninstall_biased_sdpa() -> None:
    """Restore the original F.scaled_dot_product_attention."""
    F.scaled_dot_product_attention = _ORIGINAL_SDPA


# ============ Bias Context Manager ============

@contextmanager
def bias_context(
    saliency_info: Optional[SaliencyInfo],
    num_patches: int,
    strength: float = 2.0,
    active_layers: Optional[Set[int]] = None,
    active_heads: Optional[Set[int]] = None,
    top_k_ratio: float = 0.3,
    visual_patch_indices: Optional[np.ndarray] = None,
    visual_patch_weights: Optional[np.ndarray] = None,
    visual_patch_weight_scale: float = 0.5,
    weight_norm: str = "max",
):
    """Context manager that activates attention biasing for a forward pass.

    Converts saliency positions from input_ids space to multimodal sequence space
    by applying the visual-token offset, then sets up _BIAS_STATE.

    For base OpenVLA (where input_ids already contains image placeholder tokens),
    pass num_patches=0 so positions map directly and weight_norm="none" to
    preserve the sum-normalized saliency weights from Captum IG.

    Args:
        saliency_info: SaliencyInfo with token saliency in input_ids space.
        num_patches: Number of visual tokens in the multimodal sequence
                     (already accounts for pruning, proprio, etc.).
                     Set to 0 for base OpenVLA where input_ids already
                     contains image placeholder tokens.
        strength: Bias strength multiplier.
        active_layers: Set of layer indices to bias (None = all).
        active_heads: Set of head indices to bias (None = all).
        top_k_ratio: Fraction of tokens to bias (if saliency_info.top_positions
                     was not pre-filtered).
        visual_patch_indices: Multimodal-space indices of kept visual patches
                              to also receive attention bias.
        visual_patch_weights: Per-patch weights (normalized SigLIP similarity).
                              If None, all patches get uniform weight.
        visual_patch_weight_scale: Global scale factor for visual patch bias.
                                   Final weight = scale * per_patch_weight.
        weight_norm: Weight normalization strategy for text saliency weights.
                     "max" = divide by max so peak weight = 1.0 (OFT default).
                     "none" = use weights as-is (sum-normalized from IG).
    """
    if saliency_info is None or len(saliency_info.top_positions) == 0:
        _BIAS_STATE.active = False
        try:
            yield
        finally:
            pass
        return

    # Map positions from input_ids space → multimodal sequence space.
    # OFT layout: [BOS(1)] [patches(num_patches)] [text(input_ids[1:])]
    #   mm_pos = num_patches + pos  (for pos > 0)
    # Base OpenVLA: input_ids already has image placeholders, so num_patches=0
    #   mm_pos = pos  (positions map directly)
    multimodal_positions = []
    multimodal_weights = []

    for pos in saliency_info.top_positions:
        if pos == 0:
            mm_pos = 0  # BOS stays at 0
        else:
            mm_pos = num_patches + pos  # text token offset (0 for base model)
        multimodal_positions.append(mm_pos)
        w = saliency_info.token_saliency[pos] if pos < len(saliency_info.token_saliency) else 0.0
        multimodal_weights.append(w)

    multimodal_positions = np.array(multimodal_positions, dtype=np.int64)
    multimodal_weights = np.array(multimodal_weights, dtype=np.float32)

    if weight_norm == "max":
        w_max = multimodal_weights.max()
        if w_max > 0:
            multimodal_weights = multimodal_weights / w_max

    # Also bias toward kept visual patches (SAM-selected task-relevant regions)
    if visual_patch_indices is not None and len(visual_patch_indices) > 0:
        vp_indices = np.asarray(visual_patch_indices, dtype=np.int64)
        if visual_patch_weights is not None:
            vp_weights = np.asarray(visual_patch_weights, dtype=np.float32) * visual_patch_weight_scale
        else:
            vp_weights = np.full(len(vp_indices), visual_patch_weight_scale, dtype=np.float32)
        multimodal_positions = np.concatenate([multimodal_positions, vp_indices])
        multimodal_weights = np.concatenate([multimodal_weights, vp_weights])

    _BIAS_STATE.active = True
    _BIAS_STATE.saliency_positions = multimodal_positions
    _BIAS_STATE.saliency_weights = multimodal_weights
    _BIAS_STATE.strength = strength
    _BIAS_STATE.active_layers = active_layers
    _BIAS_STATE.active_heads = active_heads
    # Invalidate caches so they're rebuilt on first layer call
    _BIAS_STATE._cached_bias_1d = None
    _BIAS_STATE._cached_seq_len = -1
    _BIAS_STATE._cached_combined_mask = None
    _BIAS_STATE._cached_mask_key = None

    try:
        yield
    finally:
        _BIAS_STATE.active = False
        _BIAS_STATE.saliency_positions = None
        _BIAS_STATE.saliency_weights = None
        _BIAS_STATE._cached_bias_1d = None
        _BIAS_STATE._cached_seq_len = -1
        _BIAS_STATE._cached_combined_mask = None
        _BIAS_STATE._cached_mask_key = None


# ============ Captum IG Saliency Computation ============

def compute_ig_saliency(
    model: torch.nn.Module,
    input_ids: torch.Tensor,
    pixel_values: torch.Tensor,
    processor,
    target_id: Optional[int] = None,
    window: int = 256,
    n_steps: int = 3,
) -> Tuple[np.ndarray, int]:
    """Compute text-token saliency using Captum Integrated Gradients.

    Uses LayerIntegratedGradients on the embedding layer to attribute the
    model's prediction back to individual input tokens. This is more
    accurate than GradInput (single-pass approximation) because it
    integrates gradients along a path from a baseline (pad tokens) to
    the actual input.

    Args:
        model: The VLA model (OpenVLAForActionPrediction).
        input_ids: (1, seq_len) token IDs for the prompt.
        pixel_values: (1, C, H, W) or (1, 2C, H, W) preprocessed images.
        processor: The tokenizer/processor (needed for pad_token_id).
        target_id: Token ID to attribute to (None = use argmax prediction).
        window: Number of tokens from the end to include.
        n_steps: Number of IG integration steps (higher = more accurate, slower).

    Returns:
        saliency: (window_len,) array of per-token saliency scores.
        start_idx: The starting index of the window in input_ids.
    """
    from captum.attr import LayerIntegratedGradients
    from prismatic.vla.constants import IGNORE_INDEX

    device = next(model.parameters()).device
    model_dtype = next(model.parameters()).dtype
    seq_len = input_ids.shape[1]
    start_idx = max(0, seq_len - window)
    input_ids_win = input_ids[:, start_idx:].to(device)
    pv = pixel_values.to(device, dtype=model_dtype)

    lm = getattr(model, "language_model", model)
    inner = getattr(lm, "model", lm)
    emb_layer = inner.embed_tokens

    # Determine target token via a quick forward pass if not specified
    if target_id is None:
        with torch.no_grad():
            labels = torch.full_like(input_ids_win, IGNORE_INDEX)
            with torch.amp.autocast(device_type="cuda", dtype=model_dtype, enabled=True):
                out = model(
                    input_ids=input_ids_win, pixel_values=pv,
                    attention_mask=torch.ones_like(input_ids_win),
                    labels=labels, use_cache=False, return_dict=True,
                )
            target_id = out.logits[:, -1, :].float().argmax(dim=-1).item()

    # Forward wrapper for Captum — must accept (input_ids, pixel_values)
    # and return logits at last position.
    def _forward_for_ig(ids, pvs):
        labels = torch.full_like(ids, IGNORE_INDEX)
        with torch.amp.autocast(device_type="cuda", dtype=model_dtype, enabled=True):
            outputs = model(
                input_ids=ids, pixel_values=pvs,
                attention_mask=torch.ones_like(ids),
                labels=labels, use_cache=False, return_dict=True,
            )
        return outputs.logits[:, -1, :]

    lig = LayerIntegratedGradients(_forward_for_ig, emb_layer)

    pad_id = getattr(processor, "pad_token_id", None)
    if pad_id is None:
        tokenizer = getattr(processor, "tokenizer", processor)
        pad_id = getattr(tokenizer, "pad_token_id", 0)
    baselines = torch.full_like(input_ids_win, pad_id)

    model.zero_grad()

    with torch.enable_grad():
        attributions = lig.attribute(
            inputs=input_ids_win,
            baselines=baselines,
            target=target_id,
            additional_forward_args=(pv,),
            n_steps=n_steps,
            internal_batch_size=1,
        )

    saliency = attributions.sum(dim=-1).squeeze(0).abs().float().detach().cpu().numpy()

    model.zero_grad()
    if emb_layer.weight.grad is not None:
        emb_layer.weight.grad = None

    return saliency, start_idx


def compute_gradinput_saliency(
    model: torch.nn.Module,
    input_ids: torch.Tensor,
    pixel_values: torch.Tensor,
    processor,
    target_id: Optional[int] = None,
    window: int = 256,
) -> Tuple[np.ndarray, int]:
    """Compute text-token saliency via Gradient x Input (single pass).

    Uses manual gradient capture at the embedding layer with gradient
    checkpointing to minimize GPU memory during backward.

    Returns the same (saliency, start_idx) format as compute_ig_saliency.
    """
    import gc
    from prismatic.vla.constants import IGNORE_INDEX

    device = next(model.parameters()).device
    model_dtype = next(model.parameters()).dtype
    seq_len = input_ids.shape[1]
    start_idx = max(0, seq_len - window)
    input_ids_win = input_ids[:, start_idx:].to(device)
    pv = pixel_values.to(device, dtype=model_dtype)

    lm = getattr(model, "language_model", model)
    inner = getattr(lm, "model", lm)
    emb_layer = inner.embed_tokens

    gc_was_enabled = getattr(inner, "gradient_checkpointing", False)
    if hasattr(model, "gradient_checkpointing_enable"):
        model.gradient_checkpointing_enable()
    elif hasattr(inner, "gradient_checkpointing_enable"):
        inner.gradient_checkpointing_enable()

    if target_id is None:
        with torch.no_grad():
            labels = torch.full_like(input_ids_win, IGNORE_INDEX)
            with torch.amp.autocast(device_type="cuda", dtype=model_dtype, enabled=True):
                out = model(
                    input_ids=input_ids_win, pixel_values=pv,
                    attention_mask=torch.ones_like(input_ids_win),
                    labels=labels, use_cache=False, return_dict=True,
                )
            target_id = out.logits[:, -1, :].float().argmax(dim=-1).item()
            del out
        gc.collect()
        torch.cuda.empty_cache()

    model.zero_grad()
    embedding_output = None
    def _hook_fn(module, inp, out):
        nonlocal embedding_output
        embedding_output = out
        out.retain_grad()
        return out
    handle = emb_layer.register_forward_hook(_hook_fn)

    try:
        with torch.enable_grad():
            labels = torch.full_like(input_ids_win, IGNORE_INDEX)
            with torch.amp.autocast(device_type="cuda", dtype=model_dtype, enabled=True):
                outputs = model(
                    input_ids=input_ids_win, pixel_values=pv,
                    attention_mask=torch.ones_like(input_ids_win),
                    labels=labels, use_cache=False, return_dict=True,
                )
            logits = outputs.logits[:, -1, :].float()
            target_logit = logits[0, target_id]
            target_logit.backward()

        grad = embedding_output.grad
        act = embedding_output.detach()
        gi = (grad * act).sum(dim=-1).squeeze(0).abs().float().detach().cpu().numpy()
    finally:
        handle.remove()
        if not gc_was_enabled:
            if hasattr(model, "gradient_checkpointing_disable"):
                model.gradient_checkpointing_disable()
            elif hasattr(inner, "gradient_checkpointing_disable"):
                inner.gradient_checkpointing_disable()

    model.zero_grad()
    if emb_layer.weight.grad is not None:
        emb_layer.weight.grad = None
    del embedding_output, outputs, logits, target_logit
    gc.collect()
    torch.cuda.empty_cache()

    return gi, start_idx


def compute_saliency_for_episode(
    model: torch.nn.Module,
    processor,
    observation: dict,
    task_label: str,
    top_k_ratio: float = 0.3,
    n_steps: int = 3,
    method: str = "ig",
) -> Optional[SaliencyInfo]:
    """Compute token saliency once at episode start.

    Args:
        model: The VLA model.
        processor: The VLA processor (tokenizer + image processor).
        observation: Dict with "full_image" key (numpy array or PIL Image).
        task_label: The text instruction.
        top_k_ratio: Fraction of prompt tokens to mark as salient.
        n_steps: Number of IG integration steps (only used when method="ig").
        method: "ig" for Integrated Gradients, "gradinput" for Gradient x Input.

    Returns:
        SaliencyInfo or None if computation fails.
    """
    from PIL import Image as PILImage

    device = next(model.parameters()).device
    model_dtype = next(model.parameters()).dtype

    img = observation["full_image"]
    if isinstance(img, np.ndarray):
        img = PILImage.fromarray(img)

    prompt = f"In: What action should the robot take to {task_label.lower()}?\nOut:"
    inputs = processor(prompt, img)
    input_ids = inputs["input_ids"].to(device)
    pixel_values = inputs["pixel_values"].to(device, dtype=model_dtype)

    if "wrist_image" in observation:
        wrist_img = observation["wrist_image"]
        if isinstance(wrist_img, np.ndarray):
            wrist_img = PILImage.fromarray(wrist_img)
        wrist_inputs = processor(prompt, wrist_img)
        wrist_pv = wrist_inputs["pixel_values"].to(device, dtype=model_dtype)
        pixel_values = torch.cat([pixel_values, wrist_pv], dim=1)

    prompt_len = input_ids.shape[1]

    try:
        if method == "gradinput":
            saliency, start_idx = compute_gradinput_saliency(
                model, input_ids, pixel_values, processor,
                target_id=None,
                window=min(256, prompt_len),
            )
        else:
            saliency, start_idx = compute_ig_saliency(
                model, input_ids, pixel_values, processor,
                target_id=None,
                window=min(256, prompt_len),
                n_steps=n_steps,
            )

        full_saliency = np.zeros(prompt_len)
        end_idx = start_idx + len(saliency)
        full_saliency[start_idx:end_idx] = saliency

        sal_sum = full_saliency.sum()
        if sal_sum > 0:
            full_saliency = full_saliency / sal_sum

        k = max(1, int(prompt_len * top_k_ratio))
        top_pos = np.argsort(full_saliency)[-k:]

        return SaliencyInfo(
            token_saliency=full_saliency,
            top_positions=top_pos,
            prompt_len=prompt_len,
        )

    except Exception as e:
        print(f"[attention_bias] {method} saliency computation failed: {e}")
        import traceback
        traceback.print_exc()
        return None

    finally:
        model.zero_grad()
        torch.cuda.empty_cache()
