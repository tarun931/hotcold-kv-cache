"""Attention monkeypatch for the compression baselines.

Independent of the hot/cold KVController patch: it installs its own
LlamaAttention.forward that projects Q/K/V, applies RoPE, and delegates the
prefill/decode attention + compression to the active PressCache's layers. Only one
attention patch (this one or the hot/cold one) should be installed at a time.
"""

import math
from typing import Optional, Tuple

import torch
from transformers.models.llama import modeling_llama
from transformers.cache_utils import Cache

from .cache import PressCacheLayer

ACTIVE_PRESS_CACHE = None


def set_active_press_cache(cache):
    """Register the PressCache the patched attention should route through."""
    global ACTIVE_PRESS_CACHE
    ACTIVE_PRESS_CACHE = cache


_ORIG = None


def uninstall_press_patch():
    global _ORIG
    if _ORIG is not None:
        modeling_llama.LlamaAttention.forward = _ORIG
        _ORIG = None


def install_press_patch():
    global _ORIG
    if _ORIG is not None:
        return
    _ORIG = modeling_llama.LlamaAttention.forward

    def patched_forward(
        self,
        hidden_states: torch.Tensor,
        position_embeddings: Tuple[torch.Tensor, torch.Tensor],
        attention_mask: Optional[torch.Tensor],
        past_key_values: Optional[Cache] = None,
        cache_position: Optional[torch.LongTensor] = None,
        **kwargs,
    ):
        if ACTIVE_PRESS_CACHE is None or past_key_values is None:
            return _ORIG(
                self,
                hidden_states=hidden_states,
                position_embeddings=position_embeddings,
                attention_mask=attention_mask,
                past_key_values=past_key_values,
                cache_position=cache_position,
                **kwargs,
            )

        num_heads = int(self.config.num_attention_heads)
        num_kv_heads = int(getattr(self.config, "num_key_value_heads", num_heads))
        head_dim = int(self.head_dim) if hasattr(self, "head_dim") else int(self.config.hidden_size // num_heads)
        n_rep = num_heads // num_kv_heads
        scaling = 1.0 / math.sqrt(head_dim)

        bsz, q_len, _ = hidden_states.shape

        q_lin = self.q_proj(hidden_states)
        k_lin = self.k_proj(hidden_states)
        v_lin = self.v_proj(hidden_states)

        q = q_lin.view(bsz, q_len, num_heads, head_dim).transpose(1, 2)       # (B,Hq,T,D)
        k = k_lin.view(bsz, q_len, num_kv_heads, head_dim).transpose(1, 2)    # (B,Hk,T,D)
        v = v_lin.view(bsz, q_len, num_kv_heads, head_dim).transpose(1, 2)    # (B,Hk,T,D)

        cos, sin = position_embeddings
        q_post, k_post = modeling_llama.apply_rotary_pos_emb(q, k, cos, sin)

        layer = past_key_values.layers[self.layer_idx]
        if not isinstance(layer, PressCacheLayer):
            return _ORIG(
                self,
                hidden_states=hidden_states,
                position_embeddings=position_embeddings,
                attention_mask=attention_mask,
                past_key_values=past_key_values,
                cache_position=cache_position,
                **kwargs,
            )

        if layer.get_seq_length() == 0:
            attn_out = layer.prefill(q_post, k_post, v, n_rep, scaling)       # (B,Hq,T,D)
        else:
            attn_out = layer.decode(q_post, k_post, v, n_rep, scaling)        # (B,Hq,1,D)

        attn_out = attn_out.transpose(1, 2).contiguous().view(bsz, q_len, -1)
        attn_out = self.o_proj(attn_out)
        return attn_out, None

    modeling_llama.LlamaAttention.forward = patched_forward
