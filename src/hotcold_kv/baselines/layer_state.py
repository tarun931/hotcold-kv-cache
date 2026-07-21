"""Per-layer state machine for the compression baselines.

Torch-only (no transformers dependency) so the compression/attention logic can be
unit-tested in isolation. `PressCacheLayer` in cache.py wraps this to satisfy the
HuggingFace Cache interface.

Convention (matching the hot/cold engine): keys/values stored here are *post-RoPE*,
so a retained token carries its original positional encoding regardless of where it
ends up after compression. Attention is computed inside this class, giving full
control over masking instead of relying on the model's causal-mask machinery.
"""

from typing import Optional

import torch
import torch.nn.functional as F

from .scoring import (
    repeat_kv,
    snapkv_importance,
    h2o_prefill_scores,
    keep_index,
    gather_kv,
    gather_scores,
    _reduce_probs_to_kv,
)


class PressLayerState:
    def __init__(self, method: str, budget: int, window_size: int,
                 kernel_size: int, pooling: str):
        self.method = method
        self.budget = int(budget)
        self.window_size = int(window_size)
        self.kernel_size = int(kernel_size)
        self.pooling = pooling

        self.keys: Optional[torch.Tensor] = None      # (B, Hk, S, D)
        self.values: Optional[torch.Tensor] = None    # (B, Hk, S, D)
        self.scores: Optional[torch.Tensor] = None    # (B, Hk, S) accumulated (h2o only)

    # -- introspection -------------------------------------------------------
    def get_seq_length(self) -> int:
        if self.keys is None or self.keys.numel() == 0:
            return 0
        return int(self.keys.shape[2])

    # -- attention helpers ---------------------------------------------------
    def _attn(self, q, k, v, n_rep, scaling, causal):
        kk = repeat_kv(k, n_rep)
        vv = repeat_kv(v, n_rep)
        out = F.scaled_dot_product_attention(q, kk, vv, is_causal=causal, scale=scaling)
        return out.to(q.dtype)

    def _attn_with_probs(self, q, k, v, n_rep, scaling):
        """Manual attention that also returns per-kv-head attention mass, used to
        accumulate H2O scores on the decode path. q is (B, Hq, 1, D)."""
        Hk = k.shape[1]
        kk = repeat_kv(k, n_rep)
        vv = repeat_kv(v, n_rep)
        logits = torch.matmul(q, kk.transpose(2, 3)) * scaling      # (B, Hq, 1, S)
        probs = torch.softmax(logits.float(), dim=-1)
        out = torch.matmul(probs, vv.float()).to(q.dtype)           # (B, Hq, 1, D)
        col = _reduce_probs_to_kv(probs, Hk).squeeze(2)             # (B, Hk, S)
        return out, col

    # -- compression ---------------------------------------------------------
    def _compress(self, scores: torch.Tensor):
        """Shrink keys/values (and h2o scores) to the budget using `scores`."""
        S = self.keys.shape[2]
        if S <= self.budget:
            if self.method == "h2o":
                self.scores = scores
            return
        keep = keep_index(scores, self.window_size, self.budget)     # (B, Hk, keep)
        self.keys = gather_kv(self.keys, keep).contiguous()
        self.values = gather_kv(self.values, keep).contiguous()
        if self.method == "h2o":
            self.scores = gather_scores(scores, keep).contiguous()

    # -- forward paths -------------------------------------------------------
    def prefill(self, q_post, k_post, v, n_rep, scaling):
        """First call for this layer: attend over the full prompt, then store and
        compress the prompt KV to the budget."""
        out = self._attn(q_post, k_post, v, n_rep, scaling, causal=True)
        self.keys = k_post
        self.values = v
        if self.method == "h2o":
            scores = h2o_prefill_scores(q_post, k_post, scaling)
        else:  # snapkv / pyramidkv
            scores = snapkv_importance(q_post, k_post, self.window_size,
                                       scaling, self.kernel_size, self.pooling)
        self._compress(scores)
        return out

    def decode(self, q_post, k_post, v, n_rep, scaling):
        """Single-token decode: append, attend over the (compressed) cache, and for
        H2O accumulate attention mass and re-evict to the budget."""
        self.keys = torch.cat([self.keys, k_post], dim=2)
        self.values = torch.cat([self.values, v], dim=2)

        if self.method == "h2o":
            out, col = self._attn_with_probs(q_post, self.keys, self.values, n_rep, scaling)
            newlen = self.keys.shape[2]
            if self.scores is None:
                self.scores = torch.zeros(col.shape[0], col.shape[1], newlen,
                                          device=col.device, dtype=torch.float32)
            else:
                pad = torch.zeros(self.scores.shape[0], self.scores.shape[1], 1,
                                  device=self.scores.device, dtype=self.scores.dtype)
                self.scores = torch.cat([self.scores, pad], dim=2)
            self.scores = self.scores + col
            if newlen > self.budget:
                self._compress(self.scores)
        else:  # snapkv / pyramidkv: keep the compressed prompt, append recent tokens
            out = self._attn(q_post, self.keys, self.values, n_rep, scaling, causal=False)
        return out
