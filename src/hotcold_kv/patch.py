import math
import torch
import torch.nn.functional as F
from typing import Optional, Tuple
from transformers.models.llama import modeling_llama
from transformers.cache_utils import Cache
from .datatypes import _canon_br_2d, _canon_kv_4d
from .kernels import triton_evict_attention_forward
from .cache import HotColdLayer



ACTIVE_KVCTRL = None
ACTIVE_ROTARY_EMB = None  # set to model.model.rotary_emb

def set_active_kvctrl(kvctrl):
    global ACTIVE_KVCTRL
    ACTIVE_KVCTRL = kvctrl

def set_active_rotary_emb(rotary_emb_module):
    global ACTIVE_ROTARY_EMB
    ACTIVE_ROTARY_EMB = rotary_emb_module


# ============================================================
# RoPE helpers (robust cos/sin normalization)
# ============================================================
def _normalize_rope_cos_sin_to_b1sd(cos: torch.Tensor, sin: torch.Tensor, B: int, S: int):
    """
    Return cos,sin as (B,1,S,D) exactly.
    Handles common variants:
      (S,D), (B,S,D), (1,S,D), (B,1,S,D), (1,1,S,D), (1,1,1,S,D), ...
    """
    while cos.dim() > 4:
        cos = cos.squeeze(0)
        sin = sin.squeeze(0)

    if cos.dim() == 2:  # (S,D)
        cos = cos.unsqueeze(0).unsqueeze(0)  # (1,1,S,D)
        sin = sin.unsqueeze(0).unsqueeze(0)
    elif cos.dim() == 3:  # (B,S,D)
        cos = cos.unsqueeze(1)  # (B,1,S,D)
        sin = sin.unsqueeze(1)
    elif cos.dim() == 4:
        if cos.shape[1] == S and cos.shape[2] == 1:  # (B,S,1,D)
            cos = cos.permute(0, 2, 1, 3).contiguous()
            sin = sin.permute(0, 2, 1, 3).contiguous()
        if cos.shape[0] == 1 and cos.shape[1] == B and cos.shape[2] == S:  # (1,B,S,D)
            cos = cos.permute(1, 0, 2, 3).contiguous()
            sin = sin.permute(1, 0, 2, 3).contiguous()
    else:
        raise RuntimeError(f"Unexpected cos.dim()={cos.dim()}")

    if cos.shape[1] != 1:
        raise RuntimeError(f"Expected cos.shape[1]==1, got {tuple(cos.shape)}")

    if cos.shape[0] == 1 and B > 1:
        cos = cos.expand(B, -1, -1, -1)
        sin = sin.expand(B, -1, -1, -1)

    if cos.shape[0] != B:
        raise RuntimeError(f"Batch mismatch cos.shape[0]={cos.shape[0]} vs B={B}")

    if cos.shape[2] != S:
        if cos.shape[2] > S:
            cos = cos[:, :, :S, :]
            sin = sin[:, :, :S, :]
        else:
            pad = S - cos.shape[2]
            cos = torch.cat([cos, cos[:, :, -1:, :].expand(B, 1, pad, -1)], dim=2)
            sin = torch.cat([sin, sin[:, :, -1:, :].expand(B, 1, pad, -1)], dim=2)

    return cos, sin


def _rope_cos_sin_for_positions(pos_ids: torch.Tensor, dtype: torch.dtype, device: torch.device, head_dim: int):
    if ACTIVE_ROTARY_EMB is None:
        raise RuntimeError("ACTIVE_ROTARY_EMB is None. Call set_active_rotary_emb(model.model.rotary_emb).")

    pos_ids = _canon_br_2d(pos_ids, "pos_ids").to(device=device, dtype=torch.long)
    B, S = pos_ids.shape

    x = torch.empty((B, 1, S, head_dim), device=device, dtype=dtype)

    try:
        cos, sin = ACTIVE_ROTARY_EMB(x, pos_ids)
    except TypeError:
        cos, sin = ACTIVE_ROTARY_EMB(x, position_ids=pos_ids)

    cos, sin = _normalize_rope_cos_sin_to_b1sd(cos, sin, B=B, S=S)
    return cos, sin


def _rotate_k_with_pos(k_pre: torch.Tensor, pos_ids: torch.Tensor, head_dim: int):
    k_pre = _canon_kv_4d(k_pre, "k_pre")
    pos_ids = _canon_br_2d(pos_ids, "pos_ids")
    B, Hk, S, D = k_pre.shape

    cos4, sin4 = _rope_cos_sin_for_positions(pos_ids, dtype=k_pre.dtype, device=k_pre.device, head_dim=head_dim)
    q_dummy = torch.zeros((B, Hk, S, D), device=k_pre.device, dtype=k_pre.dtype)
    _, k_out = modeling_llama.apply_rotary_pos_emb(q_dummy, k_pre, cos4, sin4)
    return k_out


# ============================================================
# Monkeypatch attention.forward
# ============================================================
_ORIG = None

def uninstall_monkeypatch():
    global _ORIG
    if _ORIG is not None:
        modeling_llama.LlamaAttention.forward = _ORIG
        _ORIG = None

def install_monkeypatch(enable_debug: bool = True):
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
        kvctrl = ACTIVE_KVCTRL
        if kvctrl is None:
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

        bsz, q_len, _ = hidden_states.shape
        device = hidden_states.device
        dtype = hidden_states.dtype

        q_lin = self.q_proj(hidden_states)
        k_lin = self.k_proj(hidden_states)
        v_lin = self.v_proj(hidden_states)

        q_pre = q_lin.view(bsz, q_len, num_heads, head_dim).transpose(1, 2)        # (B,Hq,T,D)
        k_pre = k_lin.view(bsz, q_len, num_kv_heads, head_dim).transpose(1, 2)     # (B,Hk,T,D)
        v = v_lin.view(bsz, q_len, num_kv_heads, head_dim).transpose(1, 2)         # (B,Hk,T,D)

        kvctrl._lazy_init(k_pre, n_layers=int(self.config.num_hidden_layers))

        cos, sin = position_embeddings
        q_post, k_post = modeling_llama.apply_rotary_pos_emb(q_pre, k_pre, cos, sin)

        # per-layer retrieval pipeline
        # The query pool is only read when a retrieval launches, every
        # window_size steps. If the pool is shorter than the window, a prefill
        # entry is always pushed out before the first launch, and keeping it
        # would hold a full (B,Hq,T,D) query per layer for nothing.
        if q_len == 1 or kvctrl.query_pool_size >= kvctrl.window_size:
            kvctrl.update_query_window(self.layer_idx, q_pre)
        kvctrl.maybe_launch_async_retrieval(self.layer_idx)
        kvctrl.maybe_prefetch_to_gpu(self.layer_idx, device=device, dtype=dtype)

        hot_prev_len = 0
        if past_key_values is not None and isinstance(past_key_values, Cache):
            hot_prev_len = int(past_key_values.get_seq_length(self.layer_idx))

        if cache_position is not None:
            cp = cache_position.reshape(-1)
            if int(cp.numel()) == q_len:
                pos_new = cp.unsqueeze(0).expand(bsz, -1).to(device=device, dtype=torch.long)
            else:
                pos_new = torch.arange(hot_prev_len, hot_prev_len + q_len, device=device, dtype=torch.long).unsqueeze(0).expand(bsz, -1)
        else:
            pos_new = torch.arange(hot_prev_len, hot_prev_len + q_len, device=device, dtype=torch.long).unsqueeze(0).expand(bsz, -1)

        # This step's own keys/values, captured before the cache update. Prefill
        # attention must use these: the update may compress, and a multi-token
        # forward needs its full uncompressed prompt.
        k_cur, v_cur = k_post, v

        # update hot cache
        if past_key_values is not None and isinstance(past_key_values, Cache):
            layer = past_key_values.layers[self.layer_idx]
            if isinstance(layer, HotColdLayer):
                # Compression scores keys with at most the last recent_size
                # queries; store a copy of just those so the full prefill
                # query is freed after this layer.
                r = kvctrl.recent_size
                layer.set_last_query_post(q_post if q_len <= r else q_post[:, :, -r:, :].clone())

            # NOTE: call our HotColdLayer.update(layer_idx=...)
            if isinstance(layer, HotColdLayer):
                k_post, v = layer.update(
                    layer_idx=self.layer_idx,
                    key_states=k_post,
                    value_states=v,
                    cache_kwargs={
                        "cache_position": cache_position,
                        "pre_key_states": k_pre,
                        "positions": pos_new,
                    },
                )
            else:
                # fallback to Cache.update if some other layer type
                k_post, v = past_key_values.update(
                    k_post, v,
                    layer_idx=self.layer_idx,
                    cache_kwargs={
                        "cache_position": cache_position,
                        "pre_key_states": k_pre,
                        "positions": pos_new,
                    },
                )

        kvctrl.last_injected_len = 0
        inj_len = 0
        rr = None
        mem = None   # (k_mem, v_mem, pos, valid): retrieved tokens attended this step

        # inject only on decode AND only if there are valid retrieved tokens
        if q_len == 1:
            rr = kvctrl.poll_staged_gpu(self.layer_idx)
            if rr.ready and bool(rr.valid.any().item()):
                rr.k_pre = _canon_kv_4d(rr.k_pre, "rr.k_pre")
                rr.v     = _canon_kv_4d(rr.v,     "rr.v")
                rr.pos   = _canon_br_2d(rr.pos,   "rr.pos")
                rr.valid = _canon_br_2d(rr.valid, "rr.valid")

                k_mem = _rotate_k_with_pos(rr.k_pre, rr.pos, head_dim=head_dim)  # (B,Hk,R,D)
                k_mem = _canon_kv_4d(k_mem, "k_mem")
                mem = (k_mem, rr.v, rr.pos, rr.valid)
                if kvctrl.hold_retrieved:
                    # Keep this set for every decode step until the next
                    # retrieval lands, instead of attending to it once.
                    kvctrl.held_mem[self.layer_idx] = mem
                    kvctrl.dbg["injection_new"] += 1
            else:
                if rr.valid.numel() > 0 and not bool(rr.valid.any().item()):
                    kvctrl.dbg["injection_skipped_no_valid"] += 1
                if kvctrl.hold_retrieved:
                    mem = kvctrl.held_mem[self.layer_idx]

            if mem is not None:
                k_mem, v_mem = mem[0], mem[1]
                inj_len = int(k_mem.shape[2])
                kvctrl.last_injected_len = inj_len
                kvctrl.dbg["injection_used"] += 1

                k_post = torch.cat([k_mem, k_post], dim=2)
                v      = torch.cat([v_mem, v],     dim=2)

        n_rep = num_heads // num_kv_heads

        if q_len == 1:
            # Decode: attend over the whole (compressed) cache plus any injected
            # tokens. keep_mask switches off empty retrieval slots and anything
            # not strictly in the past.
            S_total = int(k_post.shape[2])
            S_base = S_total - inj_len
            keep_base = torch.ones((bsz, S_base), device=device, dtype=torch.bool)

            if inj_len > 0:
                if cache_position is not None:
                    cur_pos = int(cache_position.reshape(-1)[-1].item())
                else:
                    cur_pos = hot_prev_len
                mem_pos, mem_valid = mem[2], mem[3]
                keep_mem = mem_valid & (mem_pos <= cur_pos)
                keep_mask = torch.cat([keep_mem, keep_base], dim=1)
            else:
                keep_mask = keep_base

            if hidden_states.is_cuda:
                kvctrl.dbg["triton_decode_calls"] += 1
                attn_out_b1hd, _ = triton_evict_attention_forward(
                    query=q_post,
                    key=k_post,
                    value=v,
                    attention_mask=keep_mask,
                    num_key_value_groups=n_rep,
                    scaling=(1.0 / math.sqrt(head_dim)),
                    dropout=0.0,
                    block_s=64,
                )
                attn_out = attn_out_b1hd.transpose(1, 2).contiguous()  # (B,Hq,1,D)
            else:
                # CPU fallback (debugging only): the same computation via SDPA.
                k_rep = modeling_llama.repeat_kv(k_post, n_rep) if n_rep > 1 else k_post
                v_rep = modeling_llama.repeat_kv(v, n_rep) if n_rep > 1 else v
                attn_out = F.scaled_dot_product_attention(
                    q_post, k_rep, v_rep,
                    attn_mask=keep_mask[:, None, None, :],
                    dropout_p=0.0,
                    is_causal=False,
                )

        else:
            # Prefill: attend over this step's own keys/values (k_cur, v_cur),
            # never the cache the update returned, which may already be
            # compressed. Causality: under SDPA, HuggingFace passes
            # attention_mask=None for an unpadded batch and expects the
            # attention to apply is_causal itself; is_causal=False there would
            # make prefill bidirectional and leak future tokens.
            kvctrl.dbg["triton_prefill_fallback"] += 1
            if hot_prev_len > 0:
                raise NotImplementedError(
                    "multi-token forward on a non-empty hot cache (chunked prefill) "
                    "is not supported by the hot/cold patch"
                )
            k_rep = modeling_llama.repeat_kv(k_cur, n_rep) if n_rep > 1 else k_cur
            v_rep = modeling_llama.repeat_kv(v_cur, n_rep) if n_rep > 1 else v_cur
            attn_out = F.scaled_dot_product_attention(
                q_post, k_rep, v_rep,
                attn_mask=attention_mask,
                dropout_p=0.0,
                is_causal=attention_mask is None,
            )

        attn_out = attn_out.transpose(1, 2).contiguous().view(bsz, q_len, -1)
        attn_out = self.o_proj(attn_out)

        if enable_debug and kvctrl._inited and self.layer_idx == 0 and (kvctrl.step_count[0] % 50 == 0):
            cold0 = kvctrl.cold_size(0)
            hot0 = int(past_key_values.get_seq_length(0)) if past_key_values is not None else -1
            ready_rate = kvctrl.dbg["prefetch_ready"] / max(1, kvctrl.dbg["prefetch_launched"])
            print("[DBG]", "step0", kvctrl.step_count[0], "hot0", hot0, "cold0", cold0, "ready_rate", round(ready_rate, 3))

        return attn_out, None

    modeling_llama.LlamaAttention.forward = patched_forward
