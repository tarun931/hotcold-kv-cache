import math
import torch
from typing import Optional, Tuple, Any
from transformers.cache_utils import Cache, CacheLayerMixin
from .datatypes import EvictedKV
from .controller import KVController


# ============================================================
# HotColdLayer (hot cache post-RoPE; keep pre-RoPE+pos for eviction storage)
# ============================================================
class HotColdLayer(CacheLayerMixin):
    is_compileable = False

    def __init__(self, kvctrl: KVController):
        super().__init__()
        self.kvctrl = kvctrl
        self.keys: Optional[torch.Tensor] = None
        self.values: Optional[torch.Tensor] = None
        self.pre_keys: Optional[torch.Tensor] = None
        self.positions: Optional[torch.Tensor] = None  # (B,S)
        self.is_initialized = False
        self._last_q_post: Optional[torch.Tensor] = None

    def set_last_query_post(self, q_post: torch.Tensor):
        self._last_q_post = q_post

    def lazy_initialization(self, key_states: torch.Tensor):
        dev, dt = key_states.device, key_states.dtype
        B, Hk, _, D = key_states.shape
        self.keys = torch.empty((B, Hk, 0, D), device=dev, dtype=dt)
        self.values = torch.empty((B, Hk, 0, D), device=dev, dtype=dt)
        self.pre_keys = torch.empty((B, Hk, 0, D), device=dev, dtype=dt)
        self.positions = torch.empty((B, 0), device=dev, dtype=torch.long)
        self.is_initialized = True

    def _compress_if_needed(self, layer_idx: int):
        if self._last_q_post is None:
            return
        if self.keys is None or self.values is None or self.pre_keys is None or self.positions is None:
            return

        B, Hk, S, D = self.keys.shape
        if S <= self.kvctrl.max_capacity_prompt:
            return

        recent = min(self.kvctrl.recent_size, self.kvctrl.max_capacity_prompt, S)
        prefix_len = S - recent
        if prefix_len <= 0:
            return

        topk_budget = int(self.kvctrl.max_capacity_prompt * self.kvctrl.topk_ratio)
        topk_budget = max(1, min(topk_budget, prefix_len))

        q = self._last_q_post
        tq = min(recent, q.shape[2])
        q_tail = q[:, :, -tq:, :]
        Hq = q_tail.shape[1]

        if Hq == Hk:
            q_for_k = q_tail
        elif Hq % Hk == 0:
            group = Hq // Hk
            q_for_k = q_tail.view(B, Hk, group, tq, D).mean(dim=2)
        else:
            q_for_k = q_tail.mean(dim=1, keepdim=True).expand(B, Hk, tq, D)

        attn_logits = torch.matmul(q_for_k, self.keys.transpose(2, 3)) / math.sqrt(D)
        token_scores = attn_logits.sum(dim=2).mean(dim=1)
        scores_prefix = token_scores[:, :prefix_len]

        top_idx = torch.topk(scores_prefix, k=topk_budget, dim=-1, largest=True).indices
        top_idx_sorted, _ = torch.sort(top_idx, dim=-1)

        recent_idx = torch.arange(prefix_len, S, device=self.keys.device, dtype=torch.long).unsqueeze(0).expand(B, -1)
        keep_idx = torch.cat([top_idx_sorted, recent_idx], dim=1)

        keep_len = keep_idx.shape[1]
        idx_k = keep_idx.unsqueeze(1).unsqueeze(-1).expand(B, Hk, keep_len, D)

        keys_new = self.keys.gather(dim=2, index=idx_k)
        values_new = self.values.gather(dim=2, index=idx_k)
        pre_keys_new = self.pre_keys.gather(dim=2, index=idx_k)
        positions_new = self.positions.gather(dim=1, index=keep_idx)

        # evicted = prefix \ topk
        mask = torch.ones((B, prefix_len), device=self.keys.device, dtype=torch.bool)
        mask.scatter_(dim=1, index=top_idx_sorted, value=False)

        evicted_lists = []
        te_max = 0
        for b in range(B):
            t_ids = torch.nonzero(mask[b], as_tuple=False).squeeze(-1)
            evicted_lists.append(t_ids)
            te_max = max(te_max, int(t_ids.numel()))

        if te_max > 0:
            cpu = torch.device("cpu")
            kpre_cpu = torch.empty((B, Hk, te_max, D), device=cpu, pin_memory=True, dtype=torch.float32)
            v_cpu = torch.empty((B, Hk, te_max, D), device=cpu, pin_memory=True, dtype=torch.float32)
            pos_cpu = torch.empty((B, te_max), device=cpu, pin_memory=True, dtype=torch.long)
            idx_cpu = torch.full((B, te_max), -1, device=cpu, dtype=torch.long)

            for b in range(B):
                t_ids = evicted_lists[b]
                if t_ids.numel() == 0:
                    continue
                te = int(t_ids.numel())
                idx_cpu[b, :te] = t_ids.cpu()

                kpre_src = self.pre_keys[b:b+1, :, t_ids, :].contiguous()
                v_src = self.values[b:b+1, :, t_ids, :].contiguous()
                pos_src = self.positions[b, t_ids].contiguous()

                kpre_cpu[b, :, :te, :].copy_(kpre_src.squeeze(0).float(), non_blocking=True)
                v_cpu[b, :, :te, :].copy_(v_src.squeeze(0).float(), non_blocking=True)
                pos_cpu[b, :te].copy_(pos_src.to(torch.long).cpu(), non_blocking=True)

            self.kvctrl.dbg["compress_calls"] += 1
            self.kvctrl.dbg["evicted_tokens"] += int((idx_cpu >= 0).sum().item())

            self.kvctrl.add_evicted_to_cold_store(
                layer_idx=layer_idx,
                ev=EvictedKV(k_pre_cpu=kpre_cpu, v_cpu=v_cpu, pos_cpu=pos_cpu, evicted_idx=idx_cpu),
            )

        self.keys, self.values = keys_new, values_new
        self.pre_keys, self.positions = pre_keys_new, positions_new

    def update(self, layer_idx: int, key_states: torch.Tensor, value_states: torch.Tensor, cache_kwargs: Optional[dict[str, Any]] = None):
        if not self.is_initialized:
            self.lazy_initialization(key_states)

        pre_k = None
        pos = None
        if cache_kwargs is not None:
            pre_k = cache_kwargs.get("pre_key_states", None)
            pos = cache_kwargs.get("positions", None)

        if pre_k is None or pos is None:
            B, Hk, T, D = key_states.shape
            pre_k = torch.zeros((B, Hk, T, D), device=key_states.device, dtype=key_states.dtype)
            pos = torch.zeros((B, T), device=key_states.device, dtype=torch.long)

        self.keys = torch.cat([self.keys, key_states], dim=-2)
        self.values = torch.cat([self.values, value_states], dim=-2)
        self.pre_keys = torch.cat([self.pre_keys, pre_k], dim=-2)
        self.positions = torch.cat([self.positions, pos.to(torch.long)], dim=-1)

        self._compress_if_needed(layer_idx=layer_idx)
        return self.keys, self.values

    def get_mask_sizes(self, cache_position: torch.Tensor) -> Tuple[int, int]:
        query_length = cache_position.shape[0]
        kv_length = self.get_seq_length() + query_length
        return int(kv_length), 0

    def get_seq_length(self) -> int:
        if (not self.is_initialized) or self.keys is None or self.keys.numel() == 0:
            return 0
        return int(self.keys.shape[-2])

    def get_max_cache_shape(self) -> int:
        return int(self.kvctrl.max_capacity_prompt)


# ============================================================
# Cache builder
# ============================================================
def build_hotcold_cache(model, kvctrl: KVController) -> Cache:
    n_layers = int(model.config.num_hidden_layers)
    layers = [HotColdLayer(kvctrl) for _ in range(n_layers)]
    return Cache(layers=layers)
