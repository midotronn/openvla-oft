"""fastv_cogact.py — FAITHFUL FastV baseline for CogACT (arXiv 2403.06764, official
pkunlp-icler/FastV algorithm), adapted to transformers 4.40.1 LlamaModel.

FastV: run the first K LLM layers normally; at layer K, rank image tokens by the attention
they RECEIVE FROM THE LAST TOKEN (averaged over heads) at layer K-1, keep the top-`ATTENTION_RANK`,
prune the rest for all subsequent layers.

We use FastV's TOKEN-MASKING variant (accuracy-identical to inplace token-drop: the last/cognition
token attends to the exact same kept set, and CogACT reads the cognition feature from the last token,
so the SR is unchanged from drop). Masking keeps the sequence length constant, so the per-layer
hidden_states tuple CogACT consumes (`output.hidden_states[...][-1][:, -1, :]`) is structurally
identical to base — only the attention mask changes from layer K onward.

CogACT single-camera layout: [BOS(1)] [vision(256)] [text(T)]  →  SYS_LENGTH=1, IMAGE_TOKEN_LENGTH=256.
Matched-budget config (vs t3r_orig@40% & adp40, both 154 tok): K=1, keep=154 (prune 40%).
Env knobs: FASTV_KEEP (default 154), FASTV_K / agg layer (default 1).
"""
import os
import types
import torch
from transformers.modeling_outputs import BaseModelOutputWithPast


class FastVController:
    def __init__(self, model, keep=None, agg_layer=None, sys_length=1, image_token_length=256):
        self.m = model
        self.keep = int(os.environ.get("FASTV_KEEP", str(keep if keep is not None else 154)))
        self.agg_layer = int(os.environ.get("FASTV_K", str(agg_layer if agg_layer is not None else 1)))
        assert self.agg_layer >= 1, "FastV K (agg layer) must be >= 1"
        self.sys = int(os.environ.get("FASTV_SYS", str(sys_length)))
        self.img = int(os.environ.get("FASTV_IMG", str(image_token_length)))
        self.enabled = True
        self.kept_hist = []
        self.calls = 0
        self.cur = None
        self._orig_forward = None
        self._dbgn = 0

    # ------------------------------------------------------------------
    def reset_episode(self, instruction):
        self.cur = instruction

    # ------------------------------------------------------------------
    @staticmethod
    def _causal_mask(S, dtype, device):
        min_val = torch.finfo(dtype).min
        m = torch.zeros((S, S), dtype=dtype, device=device)
        m = m.masked_fill(torch.triu(torch.ones(S, S, dtype=torch.bool, device=device), diagonal=1), min_val)
        return m[None, None, :, :]                                        # [1,1,S,S]

    def _masked_causal(self, S, keep_cols, dtype, device):
        """Causal mask with the PRUNED image columns additionally blocked for every row."""
        m = self._causal_mask(S, dtype, device).clone()
        block = torch.ones(S, dtype=torch.bool, device=device)
        block[: self.sys] = False                                        # keep system tokens
        block[self.sys + self.img:] = False                              # keep text tokens
        block[keep_cols] = False                                         # keep selected image tokens
        m[..., block] = torch.finfo(dtype).min
        return m

    # ------------------------------------------------------------------
    def attach(self):
        llama = self.m.vlm.llm_backbone.llm.model                        # transformers LlamaModel
        self._llama = llama
        self._orig_forward = llama.forward
        ctrl = self

        def fastv_forward(self2, input_ids=None, attention_mask=None, position_ids=None,
                          past_key_values=None, inputs_embeds=None, use_cache=None,
                          output_attentions=None, output_hidden_states=None,
                          return_dict=None, cache_position=None, **kwargs):
            # Fall back to the original forward for anything that is not a clean single
            # multimodal forward (e.g. cached decode steps, missing embeds).
            if (inputs_embeds is None or past_key_values is not None
                    or inputs_embeds.shape[0] != 1
                    or inputs_embeds.shape[1] < ctrl.sys + ctrl.img + 1
                    or not ctrl.enabled):
                return ctrl._orig_forward(
                    input_ids=input_ids, attention_mask=attention_mask, position_ids=position_ids,
                    past_key_values=past_key_values, inputs_embeds=inputs_embeds, use_cache=use_cache,
                    output_attentions=output_attentions, output_hidden_states=output_hidden_states,
                    return_dict=return_dict, cache_position=cache_position, **kwargs)

            B, S, _ = inputs_embeds.shape
            device = inputs_embeds.device
            dtype = inputs_embeds.dtype
            hidden = inputs_embeds
            pos = (torch.arange(S, device=device).unsqueeze(0) if position_ids is None
                   else position_ids.view(1, -1).long())
            cache_pos = torch.arange(S, device=device)
            cur_mask = ctrl._causal_mask(S, dtype, device)
            K = ctrl.agg_layer
            all_hs = () if output_hidden_states else None

            for idx, layer in enumerate(self2.layers):
                if output_hidden_states:
                    all_hs = all_hs + (hidden,)
                need_attn = (idx == K - 1)
                out = layer(hidden, attention_mask=cur_mask, position_ids=pos,
                            past_key_value=None, output_attentions=need_attn,
                            use_cache=False, cache_position=cache_pos)
                hidden = out[0]
                if idx == K - 1:
                    attn = out[1]                                        # [1, H, S, S]
                    attn_avg = attn.float().mean(dim=1)[0]               # [S, S]
                    last_tok_img = attn_avg[-1][ctrl.sys: ctrl.sys + ctrl.img]   # [IMG]
                    keep_n = min(ctrl.keep, ctrl.img)
                    top_img = last_tok_img.topk(keep_n).indices + ctrl.sys       # image indices to KEEP
                    cur_mask = ctrl._masked_causal(S, top_img, dtype, device)
                    ctrl.kept_hist.append(int(keep_n))
                    if ctrl._dbgn < 6:
                        ctrl._dbgn += 1
                        print(f"[fastv] S={S} prune@K={K} keep_img={keep_n}/{ctrl.img} "
                              f"(prune={100*(1-keep_n/ctrl.img):.0f}%)", flush=True)

            hidden = self2.norm(hidden)
            if output_hidden_states:
                all_hs = all_hs + (hidden,)

            if not return_dict:
                return tuple(v for v in [hidden, None, all_hs, None] if v is not None)
            return BaseModelOutputWithPast(last_hidden_state=hidden, past_key_values=None,
                                           hidden_states=all_hs, attentions=None)

        llama.forward = types.MethodType(fastv_forward, llama)
        self.m._t3r = self
        return self

    def restore(self):
        if self._orig_forward is not None:
            self._llama.forward = self._orig_forward
