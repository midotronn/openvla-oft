# SigLIP Text-Image Similarity Guidance for Mask Generation

## Overview

This document describes the new SigLIP text-image similarity guidance approach for mask generation in OpenVLA with EfficientTAM/SAM.

## Architecture

The new approach implements the pipeline shown in the architecture diagram:

```
Instruction Text
    |
    v
SigLIP Text Encoder
    |
    v
Text Embeddings (T1, T2, ..., TN)
    |
    +----------------------+
    |                      |
    v                      v
Image                 Similarity Matrix
    |                 (Text tokens × Image patches)
    v                      |
SigLIP Image Encoder       |
    |                      |
    v                      |
Image Patch Embeddings     |
(I1, I2, ..., IN)         |
    |                      |
    +----------------------+
    |
    v
Top-K Similar Patches
    |
    v
Point Prompts (positions)
    |
    v
EfficientTAM/SAM
    |
    v
Segmentation Mask
    |
    v
Interested Tokens
    |
    v
LLM (Action Prediction)
```

## Key Components

### 1. SigLIP Text Features Extraction
- Encodes task instruction text using SigLIP's text encoder
- Produces text token embeddings with semantic understanding
- Implementation: `SigLIPGuidedSAM.extract_siglip_text_features()`

### 2. SigLIP Image Features Extraction  
- Extracts patch-level image features from VLA's SigLIP vision backbone
- Produces image patch embeddings aligned with text embedding space
- Implementation: `SigLIPGuidedSAM.extract_siglip_image_features()`

### 3. Text-Image Similarity Computation
- Computes cosine similarity between text tokens and image patches
- Results in a similarity matrix (N_text × N_patches)
- Implementation: `SigLIPGuidedSAM.compute_text_image_similarity()`

### 4. Top-K Patch Selection
- Aggregates similarity across text tokens (max/mean/sum)
- Selects K patches with highest similarity scores
- Converts patch indices to spatial coordinates as point prompts
- Implementation: `SigLIPGuidedSAM.select_topk_patches_from_similarity()`

### 5. Mask Generation
- Uses selected points as prompts for EfficientTAM/SAM
- Generates segmentation mask highlighting task-relevant regions
- Implementation: `SigLIPGuidedSAM.generate_mask_with_siglip_similarity()`

## Usage

### Configuration File (config.yaml)

```yaml
# Enable SAM with SigLIP similarity guidance
use_sam: true
use_siglip_similarity: true  # NEW: Enable SigLIP text-image similarity
sam_backend: "efficienttam"
sam_checkpoint: "EfficientTAM/checkpoints"
sam_model_type: "efficienttam_s"
top_k_points: 8  # Number of top similar patches to use as prompts
num_neg_points: 3  # Number of negative points (low similarity)
```

### Command Line

```bash
# Run with SigLIP similarity guidance
python experiments/robot/run_libero_eval_with_sam.py \
    --model_path /path/to/openvla-7b \
    --task_suite_name libero_spatial \
    --task_id 0-9 \
    --use_sam True \
    --use_siglip_similarity True \
    --sam_backend efficienttam \
    --sam_checkpoint EfficientTAM/checkpoints \
    --top_k_points 8

# Run with original attention-based guidance (default)
python experiments/robot/run_libero_eval_with_sam.py \
    --model_path /path/to/openvla-7b \
    --task_suite_name libero_spatial \
    --task_id 0-9 \
    --use_sam True \
    --use_siglip_similarity False \
    --sam_backend efficienttam
```

## Comparison with Attention-Guided Approach

| Aspect | Attention-Guided | SigLIP Similarity-Guided |
|--------|------------------|--------------------------|
| **Point Selection** | Based on VLA's attention weights | Based on text-image similarity |
| **Semantic Alignment** | Implicit (learned) | Explicit (SigLIP alignment) |
| **Text Dependency** | No direct text influence | Direct text-guided selection |
| **Computation** | Reuses VLA forward pass attention | Requires separate feature extraction |
| **Benefits** | No extra computation | Better semantic grounding |

## Implementation Details

### Dependencies

The SigLIP text encoder requires the `transformers` library:

```bash
pip install transformers
```

### Model Loading

- Text encoder: HuggingFace SigLIP model (`google/siglip-so400m-patch14-384`)
- Image encoder: Reuses VLA's existing SigLIP vision backbone
- Models are cached after first load for efficiency

### Fallback Behavior

If SigLIP text encoder is not available or fails:
1. Falls back to using image patch feature magnitudes as scores
2. If that fails, uses uniform attention (all patches equally weighted)
3. Gracefully degrades without crashing the evaluation

## Performance Considerations

- **First Run**: Slower due to downloading SigLIP text encoder (~1.5GB)
- **Subsequent Runs**: Models are cached, minimal overhead
- **Memory**: Additional ~2GB GPU memory for text encoder
- **Speed**: ~10-20ms extra per step for text encoding and similarity computation

## Expected Benefits

1. **Better Task Alignment**: Points are selected based on semantic relevance to instruction
2. **Improved Generalization**: Text-guided selection may generalize better across tasks
3. **Interpretability**: Similarity scores provide insight into model's understanding
4. **Robustness**: Less sensitive to VLA's attention patterns, more grounded in semantics

## Troubleshooting

### Error: "transformers library required"
```bash
pip install transformers
```

### Error: "SigLIP featurizer does not have a text encoder"
- This is expected for TIMM models
- The implementation automatically loads HuggingFace SigLIP instead
- Check internet connection for model download

### High Memory Usage
- Reduce batch size or use smaller SigLIP model variant
- Consider reducing `top_k_points` parameter

### Slow Performance
- First run downloads models (~1.5GB), subsequent runs are faster
- Consider using `efficienttam_ti_512x512` for faster inference
- Reduce image resolution if possible

## Future Improvements

1. Support for cached text features (encode instruction once per task)
2. Batch processing for multiple episodes
3. Alternative text encoders (CLIP, T5, etc.)
4. Adaptive K selection based on similarity distribution
5. Multi-scale patch selection

## References

- SigLIP Paper: [Sigmoid Loss for Language Image Pre-training](https://arxiv.org/abs/2303.15343)
- EfficientTAM: [Efficient Track Anything Model](https://github.com/yformer/EfficientTAM)
- OpenVLA: [Vision-Language-Action Models](https://github.com/openvla/openvla)
