"""KV-cache compression baselines: SnapKV, PyramidKV and H2O.

These are self-contained reference implementations used to benchmark the hot/cold
framework against standard compression methods. Each is a prompt/decode KV
compressor exposed as a HuggingFace Cache plus an attention monkeypatch.
"""

from .config import PressConfig, pyramid_layer_budget
from .layer_state import PressLayerState
from .cache import PressCacheLayer, build_press_cache
from .patch import (
    set_active_press_cache,
    install_press_patch,
    uninstall_press_patch,
)

__all__ = [
    "PressConfig",
    "pyramid_layer_budget",
    "PressLayerState",
    "PressCacheLayer",
    "build_press_cache",
    "set_active_press_cache",
    "install_press_patch",
    "uninstall_press_patch",
]
