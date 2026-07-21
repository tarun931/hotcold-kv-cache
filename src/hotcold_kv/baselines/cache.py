"""HuggingFace Cache wrappers for the compression baselines."""

from typing import Optional, Tuple, Any

import torch
from transformers.cache_utils import Cache, CacheLayerMixin

from .config import PressConfig, pyramid_layer_budget
from .layer_state import PressLayerState


class PressCacheLayer(CacheLayerMixin):
    """A cache layer backed by a PressLayerState. Attention output is produced by
    the state machine (called from the patched attention forward); this class only
    exposes the length/mask bookkeeping the model needs."""

    is_compileable = False

    def __init__(self, cfg: PressConfig, layer_idx: int):
        super().__init__()
        self.cfg = cfg
        self.layer_idx = layer_idx
        if cfg.method == "pyramidkv":
            budget = pyramid_layer_budget(layer_idx, cfg.num_layers,
                                          cfg.max_capacity, cfg.pyramid_beta)
        else:
            budget = cfg.max_capacity
        self.budget = budget
        self.state = PressLayerState(
            method=cfg.method,
            budget=budget,
            window_size=cfg.window_size,
            kernel_size=cfg.kernel_size,
            pooling=cfg.pooling,
        )

    # The patched attention forward calls these directly.
    def prefill(self, q_post, k_post, v, n_rep, scaling):
        return self.state.prefill(q_post, k_post, v, n_rep, scaling)

    def decode(self, q_post, k_post, v, n_rep, scaling):
        return self.state.decode(q_post, k_post, v, n_rep, scaling)

    # -- HuggingFace Cache interface ----------------------------------------
    def update(self, key_states: torch.Tensor, value_states: torch.Tensor,
               layer_idx: int = 0, cache_kwargs: Optional[dict[str, Any]] = None):
        # Unused: the patched attention updates the state directly. Provided so the
        # object is a well-behaved CacheLayerMixin.
        return self.state.keys, self.state.values

    def get_seq_length(self) -> int:
        return self.state.get_seq_length()

    def get_mask_sizes(self, cache_position: torch.Tensor) -> Tuple[int, int]:
        query_length = cache_position.shape[0]
        kv_length = self.get_seq_length() + query_length
        return int(kv_length), 0

    def get_max_cache_shape(self) -> int:
        return int(self.budget)


def build_press_cache(model, cfg: PressConfig) -> Cache:
    n_layers = int(model.config.num_hidden_layers)
    cfg.num_layers = n_layers
    layers = [PressCacheLayer(cfg, layer_idx=i) for i in range(n_layers)]
    return Cache(layers=layers)
