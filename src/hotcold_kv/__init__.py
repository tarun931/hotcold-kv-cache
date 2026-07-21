"""hotcold-kv-cache: two-tier (GPU hot / CPU cold) KV-cache management
with asynchronous look-ahead retrieval for long-context LLM inference."""

from .datatypes import RetrievedKV, EvictedKV
from .kernels import triton_evict_attention_forward
from .controller import KVController
from .cache import HotColdLayer, build_hotcold_cache
from .patch import (
    set_active_kvctrl,
    set_active_rotary_emb,
    install_monkeypatch,
    uninstall_monkeypatch,
)

__all__ = [
    "RetrievedKV",
    "EvictedKV",
    "triton_evict_attention_forward",
    "KVController",
    "HotColdLayer",
    "build_hotcold_cache",
    "set_active_kvctrl",
    "set_active_rotary_emb",
    "install_monkeypatch",
    "uninstall_monkeypatch",
]
