"""
siglip_guided_sam.py

SigLIP similarity-guided SAM for vision-language-action models.
Uses SigLIP text-image similarity to guide SAM/EfficientTAM for precise object segmentation without bounding boxes.

Supports both:
- Original SAM (segment_anything)
- EfficientTAM (efficient_track_anything) - faster and more lightweight

Key Features:
- Target object text token extraction: Focus on object-related tokens instead of action verbs
- Weighted similarity aggregation: Give higher weights to nouns/adjectives
- Spatial clustering: Use DBSCAN to select spatially coherent point prompts
- Temporal smoothing: EMA smoothing of similarity maps across frames
"""

import os
import re
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Tuple, List, Dict, Any, Union
from PIL import Image
from transformers import AutoTokenizer, AutoModel

# EfficientTAM model variants and their configs
EFFICIENTTAM_MODELS = {
    # 1024x1024 resolution models
    "efficienttam_s": ("configs/efficienttam/efficienttam_s.yaml", "efficienttam_s.pt"),
    "efficienttam_ti": ("configs/efficienttam/efficienttam_ti.yaml", "efficienttam_ti.pt"),
    # 512x512 resolution models (faster)
    "efficienttam_s_512x512": ("configs/efficienttam/efficienttam_s_512x512.yaml", "efficienttam_s_512x512.pt"),
    "efficienttam_ti_512x512": ("configs/efficienttam/efficienttam_ti_512x512.yaml", "efficienttam_ti_512x512.pt"),
}


class SigLIPGuidedSAM(nn.Module):
    """
    SigLIP similarity-guided SAM/EfficientTAM module that uses text-image similarity to generate masks.
    
    This eliminates the need for bounding boxes by leveraging the model's own understanding
    of which visual regions are semantically related to the instruction.
    
    Supports two backends:
    - 'sam': Original Segment Anything Model
    - 'efficienttam': EfficientTAM (faster and more lightweight)
    """
    
    def __init__(
        self,
        sam_checkpoint: str = "sam_vit_h_4b8939.pth",
        sam_model_type: str = "vit_h",
        device: str = "cuda",
        backend: str = "efficienttam",  # 'sam' or 'efficienttam'
        efficienttam_base_dir: Optional[str] = None,  # Base dir containing EfficientTAM
    ):
        super().__init__()
        
        self.backend = backend
        self.device = device
        self.has_sam = False
        self.predictor = None
        
        if backend == "efficienttam":
            self._load_efficienttam(sam_checkpoint, sam_model_type, efficienttam_base_dir)
        else:
            self._load_sam(sam_checkpoint, sam_model_type)
    
    def _load_efficienttam(
        self, 
        checkpoint: str, 
        model_type: str,
        base_dir: Optional[str] = None
    ):
        """Load EfficientTAM image predictor."""
        try:
            # Add EfficientTAM to path if needed
            if base_dir is not None:
                import sys
                if base_dir not in sys.path:
                    sys.path.insert(0, base_dir)
            
            from efficient_track_anything.build_efficienttam import build_efficienttam
            from efficient_track_anything.efficienttam_image_predictor import EfficientTAMImagePredictor
            
            # Determine config and checkpoint paths
            if model_type in EFFICIENTTAM_MODELS:
                config_file, default_ckpt = EFFICIENTTAM_MODELS[model_type]
                # If checkpoint is a directory, look for the checkpoint file inside
                if os.path.isdir(checkpoint):
                    checkpoint = os.path.join(checkpoint, default_ckpt)
            else:
                # Assume model_type is the config file path
                config_file = model_type
            
            print(f"Loading EfficientTAM with config: {config_file}, checkpoint: {checkpoint}")
            
            # Build image predictor (always needed)
            efficienttam_model = build_efficienttam(
                config_file=config_file,
                ckpt_path=checkpoint,
                device=self.device,
                mode="eval",
            )
            self.predictor = EfficientTAMImagePredictor(efficienttam_model)
            self.has_sam = True
            print(f"✓ Loaded EfficientTAM image predictor: {model_type}")
            
        except Exception as e:
            print(f"⚠ Warning: Could not load EfficientTAM: {e}")
            print("  Attempting to fall back to original SAM...")
            self._load_sam(checkpoint, "vit_h")
    
    def _load_sam(self, checkpoint: str, model_type: str):
        """Load original SAM model as fallback."""
        try:
            from segment_anything import sam_model_registry, SamPredictor
            sam = sam_model_registry[model_type](checkpoint=checkpoint)
            self.predictor = SamPredictor(sam)
            self.predictor.model.to(self.device)
            self.has_sam = True
            self.backend = "sam"
            print(f"✓ Loaded SAM model: {model_type}")
        except Exception as e:
            print(f"⚠ Warning: Could not load SAM: {e}")
            print("  Mask generation will be disabled.")
            self.has_sam = False
    
    # ==================== SigLIP Text-Image Similarity Methods ====================
    
    # -------------------- Target Object Text Extraction --------------------
    
    def _extract_target_object_simple(self, text: str) -> str:
        """
        Extract target object phrase from task instruction using simple rules.
        
        Assumes instruction format: "action + target object + location description"
        Example: "pick up the black bowl between the plate and the rack"
                 -> returns "black bowl"
        
        Args:
            text: Full task instruction
            
        Returns:
            Extracted target object phrase
        """
        text_lower = text.lower().strip()
        
        # Common action words to skip
        action_patterns = [
            r"^pick\s+up\s+",
            r"^put\s+",
            r"^place\s+",
            r"^move\s+",
            r"^grab\s+",
            r"^take\s+",
            r"^push\s+",
            r"^pull\s+",
            r"^open\s+",
            r"^close\s+",
            r"^lift\s+",
            r"^drop\s+",
            r"^insert\s+",
            r"^remove\s+",
        ]
        
        # Remove action prefix
        after_action = text_lower
        for pattern in action_patterns:
            match = re.match(pattern, after_action)
            if match:
                after_action = after_action[match.end():].strip()
                break
        
        # Position/location words to truncate at
        position_patterns = [
            r"\s+between\s+",
            r"\s+next\s+to\s+",
            r"\s+on\s+the\s+",
            r"\s+near\s+",
            r"\s+beside\s+",
            r"\s+in\s+front\s+of\s+",
            r"\s+behind\s+",
            r"\s+on\s+top\s+of\s+",
            r"\s+under\s+",
            r"\s+above\s+",
            r"\s+into\s+",
            r"\s+onto\s+",
            r"\s+from\s+",
            r"\s+to\s+the\s+",
        ]
        
        # Truncate at position words
        for pattern in position_patterns:
            match = re.search(pattern, after_action)
            if match:
                after_action = after_action[:match.start()].strip()
                break
        
        # Remove articles
        after_action = re.sub(r"^(the|a|an)\s+", "", after_action)
        
        result = after_action.strip()
        
        # Fallback: if result is empty or too short, return original text
        if len(result) < 2:
            return text
        
        return result
    
    def _compute_token_weights(
        self,
        tokens: List[str],
        method: str = "pos_based",
    ) -> torch.Tensor:
        """
        Compute importance weights for each token.
        
        Weights are higher for nouns/adjectives (object descriptors) and lower for
        verbs, prepositions, articles, and special tokens.
        
        Args:
            tokens: List of token strings from tokenizer
            method: Weighting method ("pos_based", "uniform")
            
        Returns:
            weights: (num_tokens,) tensor of weights
        """
        if method == "uniform":
            return torch.ones(len(tokens))
        
        # Token categories and their weights
        # Low weight: action verbs, prepositions, articles, special tokens
        low_weight_tokens = {
            # Special tokens
            "[cls]", "[sep]", "[pad]", "<s>", "</s>", "<pad>", "<unk>",
            # Articles
            "the", "a", "an",
            # Prepositions
            "to", "and", "or", "in", "on", "at", "of", "for", "with", "from",
            # Action verbs (we want to focus on objects, not actions)
            "pick", "up", "put", "place", "move", "grab", "take", "push", "pull",
            "open", "close", "lift", "drop", "insert", "remove",
        }
        
        # Medium weight: location/relation words
        medium_weight_tokens = {
            "between", "next", "near", "beside", "front", "behind",
            "left", "right", "top", "bottom", "above", "below", "under",
            "into", "onto", "inside", "outside",
        }
        
        weights = []
        for token in tokens:
            # Clean token (SigLIP uses ▁ as word separator)
            token_clean = token.lower().replace("▁", "").replace("##", "").strip()
            
            if not token_clean or token.startswith("[") or token.startswith("<"):
                weights.append(0.05)  # Very low for special tokens
            elif token_clean in low_weight_tokens:
                weights.append(0.1)
            elif token_clean in medium_weight_tokens:
                weights.append(0.3)
            else:
                # Nouns, adjectives, colors, etc. get high weight
                weights.append(1.0)
        
        return torch.tensor(weights)

    def extract_siglip_text_features(
        self,
        text: str,
        vision_backbone: torch.nn.Module,
        target_extraction: str = "none",
    ) -> torch.Tensor:
        """
        Extract SigLIP text features from instruction text.
        
        Args:
            text: Task instruction text
            vision_backbone: VLA's vision backbone
            target_extraction: Target extraction method
                - "none": Use full text (original behavior)
                - "simple": Extract target object phrase only
                
        Returns:
            text_features: (seq_len, embed_dim) text token embeddings
        """
        model_name = "google/siglip-so400m-patch14-384"        
        if not hasattr(self, '_siglip_text_model'):
            self._siglip_tokenizer = AutoTokenizer.from_pretrained(model_name)
            # 自动匹配 VLA 主干的数据类型 (bfloat16/float32)
            v_backbone = vision_backbone.vision_backbone if hasattr(vision_backbone, "vision_backbone") else vision_backbone
            # 寻找 embed_dim 为 1152 的 SigLIP 模块以获取 dtype
            dtype = torch.bfloat16 # 默认为 bfloat16
            for attr in ["siglip_featurizer", "fused_featurizer", "featurizer"]:
                feat = getattr(v_backbone, attr, None)
                if feat is not None and getattr(feat, "embed_dim", 0) == 1152:
                    dtype = feat.patch_embed.proj.weight.dtype
                    break
            self._siglip_text_model = AutoModel.from_pretrained(model_name, torch_dtype=dtype).text_model
            self._siglip_text_model = self._siglip_text_model.to(self.device).eval()

        # Apply target extraction if requested
        if target_extraction == "simple":
            text = self._extract_target_object_simple(text)
        
        inputs = self._siglip_tokenizer(text, return_tensors="pt", padding=True).to(self.device)
        with torch.no_grad():
            outputs = self._siglip_text_model(**inputs)
        return outputs.last_hidden_state[0]  # (seq_len, 1152)
    
    def extract_siglip_text_features_with_weights(
        self,
        text: str,
        vision_backbone: torch.nn.Module,
    ) -> Tuple[torch.Tensor, torch.Tensor, List[str]]:
        """
        Extract SigLIP text features with per-token importance weights.
        
        Args:
            text: Task instruction text
            vision_backbone: VLA's vision backbone
            
        Returns:
            text_features: (seq_len, embed_dim) text token embeddings
            token_weights: (seq_len,) importance weights
            tokens: List of token strings
        """
        model_name = "google/siglip-so400m-patch14-384"        
        if not hasattr(self, '_siglip_text_model'):
            self._siglip_tokenizer = AutoTokenizer.from_pretrained(model_name)
            v_backbone = vision_backbone.vision_backbone if hasattr(vision_backbone, "vision_backbone") else vision_backbone
            dtype = torch.bfloat16
            for attr in ["siglip_featurizer", "fused_featurizer", "featurizer"]:
                feat = getattr(v_backbone, attr, None)
                if feat is not None and getattr(feat, "embed_dim", 0) == 1152:
                    dtype = feat.patch_embed.proj.weight.dtype
                    break
            self._siglip_text_model = AutoModel.from_pretrained(model_name, torch_dtype=dtype).text_model
            self._siglip_text_model = self._siglip_text_model.to(self.device).eval()

        inputs = self._siglip_tokenizer(text, return_tensors="pt", padding=True).to(self.device)
        tokens = self._siglip_tokenizer.convert_ids_to_tokens(inputs["input_ids"][0])
        
        with torch.no_grad():
            outputs = self._siglip_text_model(**inputs)
        
        text_features = outputs.last_hidden_state[0]  # (seq_len, 1152)
        token_weights = self._compute_token_weights(tokens).to(self.device)
        
        return text_features, token_weights, tokens
            

    def extract_siglip_image_features(
        self,
        pixel_values: Union[torch.Tensor, Dict[str, torch.Tensor]],
        vision_backbone: torch.nn.Module,
    ) -> torch.Tensor:
        v_backbone = vision_backbone.vision_backbone if hasattr(vision_backbone, "vision_backbone") else vision_backbone
        
        siglip_feat, is_secondary = None, False
        if getattr(v_backbone, "siglip_featurizer", None):
            siglip_feat = v_backbone.siglip_featurizer
            is_secondary = (siglip_feat == getattr(v_backbone, "fused_featurizer", None))
        elif getattr(v_backbone.featurizer, "embed_dim", 0) == 1152:
            siglip_feat, is_secondary = v_backbone.featurizer, False
        else:
            siglip_feat, is_secondary = v_backbone.fused_featurizer, True

        if isinstance(pixel_values, dict):
            pv = pixel_values.get("siglip", next(iter(pixel_values.values())))
        elif pixel_values.shape[1] == 6:
            pv = pixel_values[:, 3:, :, :] if is_secondary else pixel_values[:, :3, :, :]
        else:
            pv = pixel_values

        with torch.no_grad():
            dtype = siglip_feat.patch_embed.proj.weight.dtype
            x = siglip_feat.patch_embed(pv.to(dtype))
            x = siglip_feat._pos_embed(x)
            x = siglip_feat.norm_pre(x)
            for blk in siglip_feat.blocks: x = blk(x)
            x = siglip_feat.norm(x)
            
        return x.squeeze(0) # (num_patches, 1152)
    
    def compute_text_image_similarity(
        self,
        text_features: torch.Tensor,
        patch_features: torch.Tensor,
        normalize: bool = True,
    ) -> torch.Tensor:
        """
        Compute similarity between text tokens and image patches.
        
        Args:
            text_features: (N_text, embed_dim) text token embeddings
            patch_features: (N_patches, embed_dim) patch embeddings  
            normalize: Whether to L2-normalize features before computing similarity
            
        Returns:
            similarity: (N_text, N_patches) similarity matrix
        """
        if normalize:
            text_features = F.normalize(text_features, p=2, dim=-1)
            patch_features = F.normalize(patch_features, p=2, dim=-1)
        
        # Ensure same dtype for matrix multiplication
        if text_features.dtype != patch_features.dtype:
            text_features = text_features.to(dtype=patch_features.dtype)
        
        # Compute cosine similarity: (N_text, embed_dim) @ (embed_dim, N_patches)
        similarity = text_features @ patch_features.T
        
        return similarity
    
    def compute_weighted_similarity(
        self,
        text_features: torch.Tensor,
        patch_features: torch.Tensor,
        token_weights: torch.Tensor,
        normalize: bool = True,
    ) -> torch.Tensor:
        """
        Compute weighted similarity between text tokens and image patches.
        
        Instead of max/mean aggregation, uses token-specific weights to give
        higher importance to object-related tokens (nouns, adjectives) and
        lower importance to action verbs and prepositions.
        
        Args:
            text_features: (N_text, embed_dim) text token embeddings
            patch_features: (N_patches, embed_dim) patch embeddings
            token_weights: (N_text,) importance weights for each token
            normalize: Whether to L2-normalize features
            
        Returns:
            patch_scores: (N_patches,) weighted similarity scores for each patch
        """
        if normalize:
            text_features = F.normalize(text_features, p=2, dim=-1)
            patch_features = F.normalize(patch_features, p=2, dim=-1)
        
        if text_features.dtype != patch_features.dtype:
            text_features = text_features.to(dtype=patch_features.dtype)
        
        # Compute similarity matrix: (N_text, N_patches)
        similarity = text_features @ patch_features.T
        
        # Ensure weights are same dtype and device
        token_weights = token_weights.to(device=similarity.device, dtype=similarity.dtype)
        
        # Weighted aggregation: sum(weight[i] * sim[i, patch]) / sum(weights)
        # Shape: (N_patches,)
        weighted_sum = (similarity * token_weights.unsqueeze(1)).sum(dim=0)
        weight_total = token_weights.sum() + 1e-8
        patch_scores = weighted_sum / weight_total
        
        return patch_scores
    
    def select_topk_patches_from_similarity(
        self,
        similarity: torch.Tensor,
        k: int = 10,
        aggregation: str = "max",
    ) -> torch.Tensor:
        """
        Select top-K most similar patches based on text-image similarity.
        
        Args:
            similarity: (N_text, N_patches) similarity matrix
            k: Number of top patches to select
            aggregation: How to aggregate across text tokens ("max", "mean", "sum")
            
        Returns:
            topk_indices: (k,) indices of top-k patches
        """
        # Aggregate similarity across text tokens
        if aggregation == "max":
            # For each patch, take max similarity across all text tokens
            patch_scores = similarity.max(dim=0)[0]  # (N_patches,)
        elif aggregation == "mean":
            patch_scores = similarity.mean(dim=0)
        elif aggregation == "sum":
            patch_scores = similarity.sum(dim=0)
        else:
            raise ValueError(f"Unknown aggregation method: {aggregation}")
        
        # Select top-k patches
        topk_values, topk_indices = torch.topk(patch_scores, k=min(k, len(patch_scores)))
        
        return topk_indices
    
    # ==================== Mask Generation Methods ====================
    
    def mask_to_patch_selection(
        self,
        mask: torch.Tensor,
        patch_size: int = 14,
        threshold: float = 0.5,
    ) -> torch.Tensor:
        """
        Convert pixel-level mask to patch-level selection.
        
        Args:
            mask: (H, W) binary mask tensor (torch.Tensor or np.ndarray)
            patch_size: Size of each patch
            threshold: Ratio of masked pixels needed to keep a patch
            
        Returns:
            patch_mask: (num_patches,) boolean mask for patch selection
        """
        # Convert to torch tensor if needed
 
        # Ensure float for mean computation
        if mask.dtype == torch.bool:
            mask = mask.float()
        elif mask.dtype in [torch.uint8, torch.int32, torch.int64]:
            mask = mask.float()
        
        H, W = mask.shape
        patches_h = H // patch_size
        patches_w = W // patch_size
        
        # Resize mask to ensure divisibility using torch
        if H % patch_size != 0 or W % patch_size != 0:
            new_h = patches_h * patch_size
            new_w = patches_w * patch_size
            # Use torch F.interpolate for resizing (needs 4D input: NCHW)
            mask = mask.unsqueeze(0).unsqueeze(0)  # Add batch and channel dims
            mask = F.interpolate(mask, size=(new_h, new_w), mode='nearest')
            mask = mask.squeeze(0).squeeze(0)  # Remove batch and channel dims
        
        # Reshape mask into patches: (patches_h, patch_size, patches_w, patch_size)
        mask_patches = mask.reshape(patches_h, patch_size, patches_w, patch_size)
        
        # Permute to (patches_h, patches_w, patch_size, patch_size)
        mask_patches = mask_patches.permute(0, 2, 1, 3)
        
        # Compute ratio for each patch (vectorized)
        patch_ratios = mask_patches.mean(dim=(2, 3))
        
        # Apply threshold (vectorized)
        patch_mask = patch_ratios > threshold
        
        # Flatten to 1D and return
        return patch_mask.flatten()
    
    def similarity_to_point_prompts(
        self,
        similarity: torch.Tensor,
        image_size: Tuple[int, int] = (224, 224),
        patch_size: int = 14,
        num_points: int = 10,
        num_neg_points: int = 5,
        aggregation: str = "max",
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Convert text-image similarity matrix to point prompts for SAM.
        
        Args:
            similarity: (N_text, N_patches) similarity matrix
            image_size: (H, W) original image size
            patch_size: Size of each patch
            num_points: Number of positive points (top-K most similar patches)
            num_neg_points: Number of negative points (bottom-K least similar patches)
            aggregation: How to aggregate similarity across text tokens ("max", "mean", "sum")
            
        Returns:
            point_coords: (N, 2) array of [x, y] coordinates
            point_labels: (N,) array of labels (1 for positive, 0 for negative)
        """
        # Aggregate similarity scores across text tokens to get per-patch scores
        if aggregation == "max":
            patch_scores = similarity.max(dim=0)[0]  # (N_patches,)
        elif aggregation == "mean":
            patch_scores = similarity.mean(dim=0)
        elif aggregation == "sum":
            patch_scores = similarity.sum(dim=0)
        else:
            raise ValueError(f"Unknown aggregation: {aggregation}")
        
        # Convert patch scores to point prompts
        return self.importance_map_to_point_prompts(
            importance_map=patch_scores,
            image_size=image_size,
            patch_size=patch_size,
            num_points=num_points,
            num_neg_points=num_neg_points,
        )
    
    def importance_map_to_point_prompts(
        self,
        importance_map: torch.Tensor,
        image_size: Tuple[int, int] = (224, 224),
        patch_size: int = 14,
        num_points: int = 10,
        num_neg_points: int = 5,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Convert importance map to point prompts for SAM.
        
        Args:
            importance_map: (num_patches,) importance weights
            image_size: (H, W) original image size
            patch_size: Size of each patch
            num_points: Number of positive points to sample
            num_neg_points: Number of negative points to sample
            
        Returns:
            point_coords: (N, 2) array of [x, y] coordinates
            point_labels: (N,) array of labels (1 for positive, 0 for negative)
        """
        H, W = image_size
        patches_per_side = H // patch_size
        
        # 1. Positive points (Top-K)
        top_k = min(num_points, importance_map.numel())
        top_values, top_indices = torch.topk(importance_map, k=top_k)
        
        # Filter out very low importance points (noise)
        pos_mask = top_values > (top_values.max() * 0.1)
        top_indices = top_indices[pos_mask]
        
        # 2. Negative points (Bottom-K)
        neg_k = min(num_neg_points, importance_map.numel())
        _, bottom_indices = torch.topk(importance_map, k=neg_k, largest=False)
        
        # Combine positive and negative points
        all_indices = torch.cat([top_indices, bottom_indices])
        point_labels = torch.cat([
            torch.ones(len(top_indices), dtype=torch.int32, device=importance_map.device),
            torch.zeros(len(bottom_indices), dtype=torch.int32, device=importance_map.device)
        ])
        
        # Vectorized computation of patch coordinates
        patch_y = all_indices // patches_per_side
        patch_x = all_indices % patches_per_side
        
        # Get patch centers in image coordinates (vectorized)
        center_y = ((patch_y + 0.5) * patch_size).to(torch.int32)
        center_x = ((patch_x + 0.5) * patch_size).to(torch.int32)
        
        # Clamp to image bounds (vectorized)
        center_y = torch.clip(center_y, 0, H - 1)
        center_x = torch.clip(center_x, 0, W - 1)
        
        # Stack into coordinate array
        point_coords = torch.stack([center_x, center_y], dim=1)
        
        return point_coords, point_labels
    
    def select_topk_with_spatial_clustering(
        self,
        importance_map: torch.Tensor,
        image_size: Tuple[int, int] = (224, 224),
        patch_size: int = 14,
        k_initial: int = 50,
        k_final: int = 10,
        num_neg_points: int = 5,
        dbscan_eps: float = 2.0,
        min_cluster_size: int = 3,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Select top-K points with spatial clustering for robustness.
        
        This method addresses the "jumping points" problem when similarity
        scores are uniformly distributed. By first selecting more candidates
        and then clustering them spatially, we ensure selected points are
        spatially coherent rather than scattered randomly.
        
        Algorithm:
        1. Select Top-K_initial patches (e.g., 50)
        2. Convert to 2D grid coordinates
        3. Apply DBSCAN clustering to find spatially coherent groups
        4. From the largest cluster, select the K_final highest-scoring points
        
        Args:
            importance_map: (num_patches,) importance scores
            image_size: (H, W) image size
            patch_size: Patch size
            k_initial: Number of initial candidates (larger for robustness)
            k_final: Final number of positive points to return
            num_neg_points: Number of negative points
            dbscan_eps: DBSCAN epsilon (neighborhood radius in patch grid units)
            min_cluster_size: Minimum points to form a cluster
            
        Returns:
            point_coords: (N, 2) array of [x, y] coordinates
            point_labels: (N,) array of labels (1 for positive, 0 for negative)
        """
        H, W = image_size
        patches_per_side = H // patch_size
        device = importance_map.device
        
        # Step 1: Select Top-K_initial candidates
        k_initial = min(k_initial, importance_map.numel())
        top_values, top_indices = torch.topk(importance_map, k=k_initial)
        
        # Filter out very low importance points
        valid_mask = top_values > (top_values.max() * 0.05)
        top_indices = top_indices[valid_mask]
        top_values = top_values[valid_mask]
        
        if len(top_indices) == 0:
            # Fallback: return top k_final without clustering
            return self.importance_map_to_point_prompts(
                importance_map, image_size, patch_size, k_final, num_neg_points
            )
        
        # Step 2: Convert to 2D grid coordinates
        patch_y = (top_indices // patches_per_side).float()
        patch_x = (top_indices % patches_per_side).float()
        coords_2d = torch.stack([patch_x, patch_y], dim=1).cpu().numpy()
        
        # Step 3: Apply DBSCAN clustering
        try:
            from sklearn.cluster import DBSCAN
            clustering = DBSCAN(eps=dbscan_eps, min_samples=min_cluster_size).fit(coords_2d)
            labels = clustering.labels_
            
            # Find unique clusters (excluding noise labeled as -1)
            unique_labels = set(labels) - {-1}
            
            if len(unique_labels) > 0:
                # Find the largest cluster
                cluster_sizes = {l: (labels == l).sum() for l in unique_labels}
                largest_cluster_label = max(cluster_sizes, key=cluster_sizes.get)
                cluster_mask = labels == largest_cluster_label
                
                # Get indices belonging to the largest cluster
                cluster_indices = top_indices[torch.tensor(cluster_mask, device=device)]
                cluster_scores = top_values[torch.tensor(cluster_mask, device=device)]
                
                # From this cluster, select top k_final
                k_select = min(k_final, len(cluster_indices))
                _, best_in_cluster = torch.topk(cluster_scores, k=k_select)
                final_positive_indices = cluster_indices[best_in_cluster]
            else:
                # No valid clusters found, fallback to top k_final
                final_positive_indices = top_indices[:min(k_final, len(top_indices))]
                
        except ImportError:
            # sklearn not available, fallback to simple top-k
            print("Warning: sklearn not available for DBSCAN clustering. Using simple top-k.")
            final_positive_indices = top_indices[:min(k_final, len(top_indices))]
        
        # Step 4: Select negative points (bottom-K, spatially away from positives)
        neg_k = min(num_neg_points, importance_map.numel())
        _, bottom_indices = torch.topk(importance_map, k=neg_k, largest=False)
        
        # Combine positive and negative points
        all_indices = torch.cat([final_positive_indices, bottom_indices])
        point_labels = torch.cat([
            torch.ones(len(final_positive_indices), dtype=torch.int32, device=device),
            torch.zeros(len(bottom_indices), dtype=torch.int32, device=device)
        ])
        
        # Convert to image coordinates
        patch_y = all_indices // patches_per_side
        patch_x = all_indices % patches_per_side
        center_y = ((patch_y + 0.5) * patch_size).to(torch.int32)
        center_x = ((patch_x + 0.5) * patch_size).to(torch.int32)
        center_y = torch.clip(center_y, 0, H - 1)
        center_x = torch.clip(center_x, 0, W - 1)
        point_coords = torch.stack([center_x, center_y], dim=1)
        
        return point_coords, point_labels
    
    @torch.no_grad()
    def generate_mask_with_siglip_similarity(
        self,
        image: np.ndarray,
        text: str,
        pixel_values: torch.Tensor,
        vision_backbone: torch.nn.Module,
        image_size: Tuple[int, int] = (224, 224),
        patch_size: int = 14,
        num_points: int = 10,
        num_neg_points: int = 5,
        aggregation: str = "max",
        return_vis_data: bool = False,
        # New parameters for improved stability
        target_extraction: str = "none",  # "none", "simple"
        use_weighted_similarity: bool = False,
        use_spatial_clustering: bool = False,
        k_initial: int = 50,  # For spatial clustering
        dbscan_eps: float = 2.0,
        min_cluster_size: int = 3,
    ) -> Union[torch.Tensor, Tuple[torch.Tensor, torch.Tensor, torch.Tensor]]:
        """
        Generate mask using SigLIP text-image similarity to guide SAM/EfficientTAM.
        
        This implements the pipeline from the diagram:
        1. Extract SigLIP text features from instruction
        2. Extract SigLIP image patch features
        3. Compute similarity matrix
        4. Select top-K most similar patches as point prompts
        5. Use EfficientTAM to generate mask
        
        Args:
            image: (H, W, 3) RGB image as numpy array (uint8) for SAM
            text: Task instruction text (e.g., "pick the bowl next to the plate")
            pixel_values: Preprocessed image tensor for SigLIP feature extraction
            vision_backbone: VLA's vision backbone containing SigLIP
            image_size: Size used for patch computation
            patch_size: Patch size
            num_points: Number of positive point prompts (top-K)
            num_neg_points: Number of negative point prompts (bottom-K)
            aggregation: How to aggregate similarity across text tokens
            return_vis_data: If True, return mask with point coords and labels for visualization
            
            # New parameters for improved stability:
            target_extraction: Method to extract target object from text
                - "none": Use full instruction text (original behavior)
                - "simple": Extract target object phrase only (e.g., "black bowl")
            use_weighted_similarity: If True, use token-weighted similarity
                (higher weights for nouns/adjectives, lower for verbs/prepositions)
            use_spatial_clustering: If True, use DBSCAN clustering to select
                spatially coherent point prompts (reduces jumping)
            k_initial: Number of initial candidates for spatial clustering
            dbscan_eps: DBSCAN epsilon parameter
            min_cluster_size: Minimum cluster size for DBSCAN
            
        Returns:
            If return_vis_data=False:
                mask: (H, W) binary mask
            If return_vis_data=True:
                (mask, point_coords, point_labels): mask and visualization data
        """
        if not self.has_sam:
            # Return full mask if SAM is not available
            default_mask = torch.ones(image.shape[:2], dtype=torch.bool)
            if return_vis_data:
                return default_mask, None, None
            return default_mask
        
        # Step 1: Extract SigLIP image patch features
        patch_features = self.extract_siglip_image_features(
            pixel_values=pixel_values,
            vision_backbone=vision_backbone,
        )  # (num_patches, embed_dim)
        
        # Step 2: Extract text features with optional target extraction and weighting
        if use_weighted_similarity:
            # Get features with per-token weights
            text_features, token_weights, _ = self.extract_siglip_text_features_with_weights(
                text=text,
                vision_backbone=vision_backbone,
            )
            
            # Step 3: Compute weighted similarity (directly returns patch scores)
            patch_scores = self.compute_weighted_similarity(
                text_features=text_features,
                patch_features=patch_features,
                token_weights=token_weights,
                normalize=True,
            )  # (num_patches,)
            
            # For visualization, also compute full similarity matrix
            similarity = self.compute_text_image_similarity(
                text_features=text_features,
                patch_features=patch_features,
                normalize=True,
            )
        else:
            # Original approach: extract features with optional target extraction
            text_features = self.extract_siglip_text_features(
                text=text,
                vision_backbone=vision_backbone,
                target_extraction=target_extraction,
            )  # (N_text, embed_dim)
            
            # Step 3: Compute text-image similarity
            similarity = self.compute_text_image_similarity(
                text_features=text_features,
                patch_features=patch_features,
                normalize=True,
            )  # (N_text, num_patches)
            
            # Aggregate to get patch scores
            if aggregation == "max":
                patch_scores = similarity.max(dim=0)[0]
            elif aggregation == "mean":
                patch_scores = similarity.mean(dim=0)
            elif aggregation == "sum":
                patch_scores = similarity.sum(dim=0)
            else:
                raise ValueError(f"Unknown aggregation: {aggregation}")
        
        # Step 4: Convert similarity to point prompts
        if use_spatial_clustering:
            # Use DBSCAN clustering for spatially coherent points
            point_coords, point_labels = self.select_topk_with_spatial_clustering(
                importance_map=patch_scores,
                image_size=image_size,
                patch_size=patch_size,
                k_initial=k_initial,
                k_final=num_points,
                num_neg_points=num_neg_points,
                dbscan_eps=dbscan_eps,
                min_cluster_size=min_cluster_size,
            )
        else:
            # Original top-k selection
            point_coords, point_labels = self.importance_map_to_point_prompts(
                importance_map=patch_scores,
                image_size=image_size,
                patch_size=patch_size,
                num_points=num_points,
                num_neg_points=num_neg_points,
            )

        # Step 5: Use SAM/EfficientTAM to generate mask with the point prompts
        mask, point_coords, point_labels = self._run_sam_with_points(
            image, point_coords, point_labels, return_vis_data
        )
        return (mask, point_coords, point_labels) if return_vis_data else mask

    def generate_mask_from_precomputed(
        self,
        image: np.ndarray,
        patch_scores: torch.Tensor,
        image_size: Tuple[int, int] = (224, 224),
        patch_size: int = 14,
        num_points: int = 10,
        num_neg_points: int = 5,
        return_vis_data: bool = False,
        use_spatial_clustering: bool = False,
        k_initial: int = 50,
        dbscan_eps: float = 2.0,
        min_cluster_size: int = 3,
    ) -> Union[torch.Tensor, Tuple[torch.Tensor, torch.Tensor, torch.Tensor]]:
        """Generate mask from pre-computed patch scores (skips redundant feature extraction)."""
        if not self.has_sam:
            default_mask = torch.ones(image.shape[:2], dtype=torch.bool)
            return (default_mask, None, None) if return_vis_data else default_mask

        if use_spatial_clustering:
            point_coords, point_labels = self.select_topk_with_spatial_clustering(
                importance_map=patch_scores, image_size=image_size, patch_size=patch_size,
                k_initial=k_initial, k_final=num_points, num_neg_points=num_neg_points,
                dbscan_eps=dbscan_eps, min_cluster_size=min_cluster_size,
            )
        else:
            point_coords, point_labels = self.importance_map_to_point_prompts(
                importance_map=patch_scores, image_size=image_size, patch_size=patch_size,
                num_points=num_points, num_neg_points=num_neg_points,
            )

        return self._run_sam_with_points(image, point_coords, point_labels, return_vis_data)

    def _run_sam_with_points(self, image, point_coords, point_labels, return_vis_data=False):
        """Run SAM/EfficientTAM prediction with given point prompts."""
        self.predictor.set_image(image)
        
        point_coords_np = point_coords.cpu().numpy()
        point_labels_np = point_labels.cpu().numpy()
        
        masks, scores, _ = self.predictor.predict(
            point_coords=point_coords_np,
            point_labels=point_labels_np,
            multimask_output=True,
        )
        
        best_idx = np.argmax(scores)
        best_mask = torch.from_numpy(masks[best_idx]).to(self.device)
        
        if return_vis_data:
            return best_mask, point_coords, point_labels
        return best_mask


# ==================== Visualization Functions ====================

def visualize_similarity_heatmap(
    image: Union[np.ndarray, Image.Image],
    similarity_map: Union[torch.Tensor, np.ndarray],
    size: Tuple[int, int] = (256, 256),
    alpha: float = 0.9,
    aggregation: str = "max",
) -> np.ndarray:
    """
    Convert SigLIP similarity map to a heatmap visualization overlaid on the image.
    
    This function creates a JET-like colormap heatmap showing which image patches
    are most semantically similar to the text instruction.
    
    Args:
        image: Original image (H, W, 3) as numpy array or PIL Image
        similarity_map: (N_text, N_patches) or (N_patches,) similarity matrix/vector
            - If 2D: will be aggregated across text tokens using the specified method
            - If 1D: used directly as patch importance scores
        size: Target visualization size (width, height)
        alpha: Heatmap opacity (0.0 = only image, 1.0 = only heatmap)
        aggregation: How to aggregate similarity across text tokens ("max", "mean", "sum")
            Only used when similarity_map is 2D
        
    Returns:
        (H, W, 3) uint8 numpy array with heatmap overlaid on image
        
    Example:
        >>> similarity = sam_module.compute_text_image_similarity(text_features, patch_features)
        >>> heatmap_vis = visualize_similarity_heatmap(image, similarity, aggregation="max")
        >>> Image.fromarray(heatmap_vis).save("similarity_heatmap.png")
    """
    # Convert torch tensor to numpy
    if torch.is_tensor(similarity_map):
        sim = similarity_map.detach().cpu().to(torch.float32).numpy()
    else:
        sim = similarity_map.astype(np.float32)
    
    # Aggregate if 2D (N_text, N_patches)
    if len(sim.shape) == 2:
        if aggregation == "max":
            sim = sim.max(axis=0)  # (N_patches,)
        elif aggregation == "mean":
            sim = sim.mean(axis=0)
        elif aggregation == "sum":
            sim = sim.sum(axis=0)
        else:
            raise ValueError(f"Unknown aggregation: {aggregation}")
    
    # Handle multiple images (e.g., 512 tokens for 2 images)
    # Take the first 256 tokens for primary image visualization
    if sim.size > 256:
        sim = sim[:256]
    
    # Reshape to 16x16 patch grid (for 224x224 image with patch_size=14)
    num_patches = sim.size
    grid_size = int(np.sqrt(num_patches))
    sim = sim.reshape(grid_size, grid_size)
    
    # Apply gamma correction (sqrt) to make low values more visible
    sim = np.sqrt(np.clip(sim, 0, None) + 1e-8)
    
    # Normalize to [0, 1]
    sim_min, sim_max = sim.min(), sim.max()
    if sim_max - sim_min > 1e-8:
        sim = (sim - sim_min) / (sim_max - sim_min)
    else:
        sim = np.zeros_like(sim)
    
    # Resize to target size using PIL (bilinear interpolation)
    sim_img = Image.fromarray((sim * 255).astype(np.uint8))
    sim_img = sim_img.resize(size, resample=Image.BILINEAR)
    sim_np = np.array(sim_img).astype(np.float32) / 255.0
    
    # Apply JET-like colormap manually
    # JET: Blue -> Cyan -> Green -> Yellow -> Red
    heatmap = np.zeros((size[1], size[0], 3), dtype=np.float32)
    heatmap[:, :, 0] = np.clip(4 * sim_np - 2, 0, 1)                    # Red
    heatmap[:, :, 1] = np.clip(1.5 - np.abs(4 * sim_np - 2), 0, 1)      # Green
    heatmap[:, :, 2] = np.clip(2 - 4 * sim_np, 0, 1)                    # Blue
    
    # Prepare image
    if isinstance(image, Image.Image):
        image = np.array(image)
    
    # Resize image to target size if needed
    if image.shape[:2] != (size[1], size[0]):
        img_pil = Image.fromarray(image)
        img_pil = img_pil.resize(size, resample=Image.BILINEAR)
        image = np.array(img_pil)
    
    # Normalize image to [0, 1]
    img_vis = image.astype(np.float32) / 255.0
    
    # Blend image and heatmap
    # Use alpha-weighted blending where high similarity regions show more heatmap
    combined = img_vis * (1 - alpha * sim_np[..., None]) + heatmap * (alpha * sim_np[..., None])
    
    return (np.clip(combined, 0, 1) * 255).astype(np.uint8)


def visualize_topk_patches(
    image: Union[np.ndarray, Image.Image],
    similarity_map: Union[torch.Tensor, np.ndarray],
    k: int = 10,
    patch_size: int = 14,
    image_size: Tuple[int, int] = (224, 224),
    aggregation: str = "max",
    positive_color: Tuple[int, int, int] = (0, 255, 0),      # Green for positive
    negative_color: Tuple[int, int, int] = (255, 0, 0),      # Red for negative
    num_neg_points: int = 5,
    line_width: int = 2,
) -> np.ndarray:
    """
    Visualize top-K most similar and bottom-K least similar patches with colored rectangles.
    
    This is useful for debugging point prompts sent to SAM/EfficientTAM.
    
    Args:
        image: Original image (H, W, 3) as numpy array or PIL Image
        similarity_map: (N_text, N_patches) or (N_patches,) similarity matrix/vector
        k: Number of top (positive) patches to highlight
        patch_size: Size of each patch in the original image
        image_size: Size of the image used for patch computation (H, W)
        aggregation: How to aggregate similarity across text tokens
        positive_color: RGB color for positive (high similarity) patches
        negative_color: RGB color for negative (low similarity) patches
        num_neg_points: Number of negative patches to highlight
        line_width: Width of rectangle border
        
    Returns:
        (H, W, 3) uint8 numpy array with patch rectangles drawn
        
    Example:
        >>> similarity = sam_module.compute_text_image_similarity(text_features, patch_features)
        >>> topk_vis = visualize_topk_patches(image, similarity, k=10, num_neg_points=5)
        >>> Image.fromarray(topk_vis).save("topk_patches.png")
    """
    # Convert torch tensor to numpy
    if torch.is_tensor(similarity_map):
        sim = similarity_map.detach().cpu().to(torch.float32).numpy()
    else:
        sim = similarity_map.astype(np.float32)
    
    # Aggregate if 2D
    if len(sim.shape) == 2:
        if aggregation == "max":
            sim = sim.max(axis=0)
        elif aggregation == "mean":
            sim = sim.mean(axis=0)
        elif aggregation == "sum":
            sim = sim.sum(axis=0)
    
    # Take first 256 patches for primary image
    if sim.size > 256:
        sim = sim[:256]
    
    # Prepare image
    if isinstance(image, Image.Image):
        image = np.array(image)
    img_vis = image.copy()
    
    # Calculate scale factor if image size differs from patch computation size
    H, W = image_size
    img_h, img_w = img_vis.shape[:2]
    scale_y = img_h / H
    scale_x = img_w / W
    
    patches_per_side = H // patch_size
    
    # Get top-k and bottom-k indices
    top_k = min(k, len(sim))
    neg_k = min(num_neg_points, len(sim))
    
    top_indices = np.argsort(sim)[-top_k:][::-1]  # Highest first
    bottom_indices = np.argsort(sim)[:neg_k]       # Lowest first
    
    def draw_rect(img, patch_idx, color, thickness):
        """Draw rectangle around a patch."""
        patch_y = patch_idx // patches_per_side
        patch_x = patch_idx % patches_per_side
        
        # Calculate pixel coordinates
        x1 = int(patch_x * patch_size * scale_x)
        y1 = int(patch_y * patch_size * scale_y)
        x2 = int((patch_x + 1) * patch_size * scale_x)
        y2 = int((patch_y + 1) * patch_size * scale_y)
        
        # Draw rectangle border
        img[y1:y1+thickness, x1:x2] = color          # Top
        img[y2-thickness:y2, x1:x2] = color          # Bottom
        img[y1:y2, x1:x1+thickness] = color          # Left
        img[y1:y2, x2-thickness:x2] = color          # Right
        
        return img
    
    # Draw positive patches (green)
    for idx in top_indices:
        img_vis = draw_rect(img_vis, idx, positive_color, line_width)
    
    # Draw negative patches (red)
    for idx in bottom_indices:
        img_vis = draw_rect(img_vis, idx, negative_color, line_width)
    
    return img_vis


def visualize_mask_with_points(
    image: Union[np.ndarray, Image.Image],
    mask: Union[torch.Tensor, np.ndarray],
    point_coords: Optional[Union[torch.Tensor, np.ndarray]] = None,
    point_labels: Optional[Union[torch.Tensor, np.ndarray]] = None,
    size: Tuple[int, int] = (256, 256),
    mask_alpha: float = 0.5,
    positive_color: Tuple[int, int, int] = (0, 0, 255),  # Blue for positive points
    negative_color: Tuple[int, int, int] = (255, 0, 0),  # Red for negative points
    point_size: int = 3,
) -> np.ndarray:
    """
    Visualize SAM segmentation mask with point prompts overlaid on the image.
    
    This creates the visualization shown in the center panel of your diagram:
    - Green mask overlay showing the segmented region
    - Blue dots for positive point prompts
    - Red dots for negative point prompts
    
    Args:
        image: Original image (H, W, 3) as numpy array or PIL Image
        mask: (H, W) binary mask from SAM
        point_coords: (N, 2) array of [x, y] coordinates for point prompts
        point_labels: (N,) array of labels (1 for positive, 0 for negative)
        size: Target visualization size (width, height)
        mask_alpha: Opacity of the mask overlay (0.0 = transparent, 1.0 = opaque)
        positive_color: RGB color for positive points (default: blue)
        negative_color: RGB color for negative points (default: red)
        point_size: Radius of point circles
        
    Returns:
        (H, W, 3) uint8 numpy array with mask and points visualized
        
    Example:
        >>> mask, point_coords, point_labels = sam_module.generate_mask_with_points(...)
        >>> vis = visualize_mask_with_points(image, mask, point_coords, point_labels)
        >>> Image.fromarray(vis).save("sam_mask_points.png")
    """
    import cv2
    
    # Convert mask to numpy if needed
    if torch.is_tensor(mask):
        mask = mask.detach().cpu().numpy()
    
    # Prepare image
    if isinstance(image, Image.Image):
        image = np.array(image)
    
    # Resize image to target size if needed
    if image.shape[:2] != (size[1], size[0]):
        img_pil = Image.fromarray(image)
        img_pil = img_pil.resize(size, resample=Image.BILINEAR)
        image = np.array(img_pil)
    
    # Create visualization with mask overlay
    img_vis = image.copy().astype(np.float32)
    
    # Resize mask to match image size
    if mask.shape != (size[1], size[0]):
        mask_pil = Image.fromarray((mask * 255).astype(np.uint8))
        mask_pil = mask_pil.resize(size, resample=Image.NEAREST)
        mask = (np.array(mask_pil) > 128).astype(np.float32)
    
    # Apply green mask overlay (like in your image)
    mask_overlay = np.zeros_like(img_vis)
    mask_overlay[:, :, 1] = 255  # Green channel
    img_vis = img_vis * (1 - mask_alpha * mask[..., None]) + mask_overlay * (mask_alpha * mask[..., None])
    img_vis = np.clip(img_vis, 0, 255).astype(np.uint8)
    
    # Draw point prompts if provided
    if point_coords is not None and point_labels is not None:
        # Convert to numpy if needed
        if torch.is_tensor(point_coords):
            point_coords = point_coords.detach().cpu().numpy()
        if torch.is_tensor(point_labels):
            point_labels = point_labels.detach().cpu().numpy()
        
        # Scale points if image was resized
        # Assume points are in original coordinates, scale to visualization size
        scale_x = size[0] / image.shape[1] if image.shape[1] != size[0] else 1.0
        scale_y = size[1] / image.shape[0] if image.shape[0] != size[1] else 1.0
        
        for (x, y), label in zip(point_coords, point_labels):
            # Scale coordinates
            x_scaled = int(x * scale_x)
            y_scaled = int(y * scale_y)
            
            # Choose color based on label
            color = positive_color if label == 1 else negative_color
            
            # Draw circle for point
            cv2.circle(img_vis, (x_scaled, y_scaled), point_size, color, -1)  # Filled circle
            cv2.circle(img_vis, (x_scaled, y_scaled), point_size + 1, (255, 255, 255), 1)  # White border
    
    return img_vis


def visualize_similarity_grid(
    image: Union[np.ndarray, Image.Image],
    similarity_map: Union[torch.Tensor, np.ndarray],
    aggregation: str = "max",
    grid_alpha: float = 0.3,
) -> np.ndarray:
    """
    Visualize similarity as a colored grid overlay showing per-patch scores.
    
    Each patch is filled with a color indicating its similarity score,
    providing a clear view of the patch-level importance.
    
    Args:
        image: Original image (H, W, 3)
        similarity_map: (N_text, N_patches) or (N_patches,) similarity
        aggregation: How to aggregate if 2D
        grid_alpha: Opacity of the grid overlay
        
    Returns:
        (H, W, 3) uint8 numpy array
    """
    # Convert torch tensor to numpy
    if torch.is_tensor(similarity_map):
        sim = similarity_map.detach().cpu().to(torch.float32).numpy()
    else:
        sim = similarity_map.astype(np.float32)
    
    # Aggregate if 2D
    if len(sim.shape) == 2:
        if aggregation == "max":
            sim = sim.max(axis=0)
        elif aggregation == "mean":
            sim = sim.mean(axis=0)
        elif aggregation == "sum":
            sim = sim.sum(axis=0)
    
    if sim.size > 256:
        sim = sim[:256]
    
    # Reshape to grid
    grid_size = int(np.sqrt(sim.size))
    sim = sim.reshape(grid_size, grid_size)
    
    # Normalize
    sim = (sim - sim.min()) / (sim.max() - sim.min() + 1e-8)
    
    # Prepare image
    if isinstance(image, Image.Image):
        image = np.array(image)
    img_h, img_w = image.shape[:2]
    
    # Create colored grid (same resolution as image)
    patch_h = img_h // grid_size
    patch_w = img_w // grid_size
    
    grid_overlay = np.zeros_like(image, dtype=np.float32)
    
    for i in range(grid_size):
        for j in range(grid_size):
            y1, y2 = i * patch_h, (i + 1) * patch_h
            x1, x2 = j * patch_w, (j + 1) * patch_w
            
            # Apply JET colormap to this patch
            v = sim[i, j]
            r = np.clip(4 * v - 2, 0, 1)
            g = np.clip(1.5 - np.abs(4 * v - 2), 0, 1)
            b = np.clip(2 - 4 * v, 0, 1)
            
            grid_overlay[y1:y2, x1:x2] = [r * 255, g * 255, b * 255]
    
    # Blend
    img_vis = image.astype(np.float32)
    combined = img_vis * (1 - grid_alpha) + grid_overlay * grid_alpha
    
    return np.clip(combined, 0, 255).astype(np.uint8)


def create_siglip_guided_sam(
    sam_checkpoint: Optional[str] = None,
    sam_model_type: str = "efficienttam_s",
    device: str = "cuda",
    backend: str = "efficienttam",
    efficienttam_base_dir: Optional[str] = None,
) -> SigLIPGuidedSAM:
    """
    Factory function to create SigLIPGuidedSAM instance.
    
    Args:
        sam_checkpoint: Path to checkpoint file or directory containing checkpoints.
            For EfficientTAM: directory containing .pt files (e.g., "EfficientTAM/checkpoints")
            For SAM: path to .pth file
        sam_model_type: Model type
            For EfficientTAM: "efficienttam_s", "efficienttam_ti", "efficienttam_s_512x512", etc.
            For SAM: "vit_h", "vit_l", "vit_b"
        device: Device to load models on
        backend: "efficienttam" (faster) or "sam" (original)
        efficienttam_base_dir: Path to EfficientTAM repository root
            (e.g., "/path/to/openvla-oft/EfficientTAM")
        
    Returns:
        SigLIPGuidedSAM instance
        
    Example:
        # Use EfficientTAM (recommended - fastest)
        sam_module = create_siglip_guided_sam(
            sam_checkpoint="EfficientTAM/checkpoints",
            sam_model_type="efficienttam_s",
            backend="efficienttam",
            efficienttam_base_dir="EfficientTAM",
        )
        
        # Use original SAM
        sam_module = create_siglip_guided_sam(
            sam_checkpoint="sam_vit_h_4b8939.pth",
            sam_model_type="vit_h",
            backend="sam",
        )
    """
    # Default checkpoint paths
    if sam_checkpoint is None:
        if backend == "efficienttam":
            sam_checkpoint = "EfficientTAM/checkpoints"
        else:
            sam_checkpoint = "sam_vit_h_4b8939.pth"
    
    return SigLIPGuidedSAM(
        sam_checkpoint=sam_checkpoint,
        sam_model_type=sam_model_type,
        device=device,
        backend=backend,
        efficienttam_base_dir=efficienttam_base_dir,
    )
