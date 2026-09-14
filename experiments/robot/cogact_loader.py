import sys, os, json, gc, torch
import torch.nn as nn
from typing import Optional, List, Type
sys.path.insert(0, "/workspace/CogACT")
sys.path.insert(0, "/workspace/T3R")

class LLMBackboneWrapper(nn.Module):
    """Wraps LlamaForCausalLM to match prismatic's llm_backbone interface."""
    def __init__(self, llm, tokenizer):
        super().__init__()
        self.llm = llm
        self.tokenizer = tokenizer
        self.identifier = "llama2-7b-pure"
        self.half_precision_dtype = torch.bfloat16
        self.embed_dim = 4096
        self.pad_token_id = 0
        from prismatic.models.backbones.llm.prompting import PurePromptBuilder
        self.prompt_builder_fn = PurePromptBuilder

    def embed_input_ids(self, input_ids):
        return self.llm.get_input_embeddings()(input_ids)

    def forward(self, input_ids=None, attention_mask=None, position_ids=None,
                past_key_values=None, inputs_embeds=None, labels=None,
                use_cache=None, output_attentions=None, output_hidden_states=None,
                return_dict=None):
        return self.llm(
            input_ids=input_ids, attention_mask=attention_mask,
            position_ids=position_ids, past_key_values=past_key_values,
            inputs_embeds=inputs_embeds, labels=labels,
            use_cache=use_cache, output_attentions=output_attentions,
            output_hidden_states=output_hidden_states, return_dict=return_dict,
        )

    def load_state_dict(self, sd, strict=True):
        fixed_sd = {k.removeprefix("llm."): v for k, v in sd.items()}
        return self.llm.load_state_dict(fixed_sd, strict=strict)

    def get_fsdp_wrapping_policy(self):
        from functools import partial
        from torch.distributed.fsdp.wrap import _module_wrap_policy
        from transformers.models.llama.modeling_llama import LlamaDecoderLayer
        return partial(_module_wrap_policy, module_classes={LlamaDecoderLayer})


def load_cogact_for_libero(checkpoint_path, freeze_vlm=True, device="cuda"):
    """Load CogACT-Base. Memory-safe: loads components sequentially in bf16."""
    from prismatic.models.materialize import get_vision_backbone_and_transform
    from prismatic.models.vlms import PrismaticVLM
    from vla.cogactvla import CogACT
    from transformers import LlamaForCausalLM, LlamaConfig, LlamaTokenizerFast
    
    print("[1/6] Loading vision backbone (DINOv2 + SigLIP)...")
    vision_backbone, image_transform = get_vision_backbone_and_transform(
        "dinosiglip-vit-so-224px",
        "resize-naive",
    )
    
    # Faithful to the reference sim_cogact policy: keep the VLM + DiT in fp32
    # (reference CogACTInference defaults use_bf16=False and NEVER casts the DiT
    # action head). The DiT diffusion sampler runs in the action_model's param
    # dtype; bf16 there corrupts the tiny bridge action deltas (~+/-0.02) and
    # degrades all fine-motor tasks. fp32 CogACT-Base is ~30GB, fits the L40S 48GB.
    # Set COGACT_BF16=1 only as a low-memory fallback (accuracy-degrading).
    _bf16 = os.environ.get("COGACT_BF16") == "1"
    print(f"[2/6] Initializing LLaMA-2-7B architecture ({'bf16' if _bf16 else 'fp32'})...")
    tokenizer_path = "/workspace/.cache_huggingface/hub/models--moojink--openvla-7b-oft-finetuned-libero-spatial/snapshots/6d0231af0e48c5985f1ff86908f4674b84bc049b"
    tokenizer = LlamaTokenizerFast.from_pretrained(tokenizer_path)
    tokenizer.pad_token_id = 0
    
    llama_config = LlamaConfig(
        vocab_size=32064, hidden_size=4096, intermediate_size=11008,
        num_hidden_layers=32, num_attention_heads=32,
        max_position_embeddings=4096, rms_norm_eps=1e-5,
    )
    llm = LlamaForCausalLM(llama_config)
    if _bf16:
        llm = llm.to(dtype=torch.bfloat16)
    llm_backbone = LLMBackboneWrapper(llm, tokenizer)
    
    print("[3/6] Building PrismaticVLM (fused-gelu-mlp projector)...")
    vlm = PrismaticVLM(
        "prism-dinosiglip-224px+7b",
        vision_backbone,
        llm_backbone,
        enable_mixed_precision_training=True,
        arch_specifier="no-align+fused-gelu-mlp",
    )
    if _bf16:
        vlm = vlm.to(dtype=torch.bfloat16)
    
    print("[4/6] Loading checkpoint (component-by-component)...")
    checkpoint = torch.load(checkpoint_path, map_location="cpu")
    state_dict = checkpoint["model"] if "model" in checkpoint else checkpoint
    del checkpoint
    gc.collect()
    
    print("  Loading projector...")
    vlm.projector.load_state_dict(state_dict["projector"])
    del state_dict["projector"]
    
    print("  Loading LLM backbone...")
    vlm.llm_backbone.load_state_dict(state_dict["llm_backbone"])
    del state_dict["llm_backbone"]
    
    print("  Loading vision backbone...")
    vlm.vision_backbone.load_state_dict(state_dict["vision_backbone"])
    del state_dict["vision_backbone"]
    gc.collect()

    stats_path = os.path.join(os.path.dirname(os.path.dirname(checkpoint_path)), "dataset_statistics.json")
    norm_stats = None
    if os.path.exists(stats_path):
        with open(stats_path) as f:
            norm_stats = json.load(f)

    print("[5/6] Creating CogACT with DiT-B action head...")
    cogact = CogACT(
        vlm=vlm, action_model_type="DiT-B", token_size=4096,
        action_dim=7, future_action_window_size=15, past_action_window_size=0,
        norm_stats=norm_stats,
    )

    if "action_model" in state_dict:
        cogact.action_model.load_state_dict(state_dict["action_model"])
        print("  DiT loaded from checkpoint")
    del state_dict
    gc.collect()
    
    if _bf16:
        cogact = cogact.to(dtype=torch.bfloat16)

    if freeze_vlm:
        for p in vlm.parameters():
            p.requires_grad = False
        vlm.eval()
        for p in vlm.projector.parameters():
            p.requires_grad = True
        for p in cogact.action_model.parameters():
            p.requires_grad = True
        trainable = sum(p.numel() for p in cogact.parameters() if p.requires_grad)
        total = sum(p.numel() for p in cogact.parameters())
        print(f"  VLM frozen. Total: {total/1e6:.0f}M, Trainable: {trainable/1e6:.1f}M")

    print(f"[6/6] Moving to {device}...")
    if device != "cpu":
        cogact = cogact.to(device)
        print(f"  VRAM: {torch.cuda.memory_allocated()/1e9:.1f}GB")
    
    print("CogACT loaded successfully!")
    return cogact, image_transform, tokenizer

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args()
    
    model, transform, tokenizer = load_cogact_for_libero(
        "/workspace/cogact_checkpoint/checkpoints/CogACT-Base.pt",
        freeze_vlm=True, device=args.device)
    print(f"\nFinal param count: {sum(p.numel() for p in model.parameters())/1e9:.2f}B")
