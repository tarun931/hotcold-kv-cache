from dataclasses import dataclass


@dataclass
class PressConfig:
    """Configuration for a KV-cache compression baseline.

    method:
        "snapkv"    - observation-window importance + pooling, prompt compression.
        "pyramidkv" - same importance as SnapKV but a per-layer (pyramidal) budget.
        "h2o"       - Heavy-Hitter Oracle: accumulated attention, online eviction.
    max_capacity:
        Total KV budget kept per layer (includes the recent window). For PyramidKV
        this is the *average* budget; per-layer budgets are derived from it.
    window_size:
        Size of the recent window that is always retained. For SnapKV/PyramidKV it
        is also the observation window used to score the prompt.
    kernel_size, pooling:
        1-D pooling applied to importance scores (SnapKV/PyramidKV) to select
        contiguous informative spans. Use an odd kernel so the length is preserved.
    pyramid_beta:
        Controls the spread of PyramidKV's per-layer budgets around max_capacity.
    """

    method: str = "snapkv"
    max_capacity: int = 512
    window_size: int = 32
    kernel_size: int = 7
    pooling: str = "maxpool"      # "maxpool" | "avgpool"
    pyramid_beta: int = 20
    num_layers: int = 32          # overwritten from the model in build_press_cache

    def __post_init__(self):
        assert self.method in ("snapkv", "pyramidkv", "h2o"), f"unknown method {self.method}"
        assert self.pooling in ("maxpool", "avgpool"), f"unknown pooling {self.pooling}"
        assert self.window_size < self.max_capacity, "window_size must be < max_capacity"


def pyramid_layer_budget(layer_idx: int, num_layers: int, max_capacity: int, beta: int) -> int:
    """PyramidKV per-layer budget: large in early layers, small in late layers,
    interpolated linearly so the mean across layers is ~max_capacity."""
    min_num = max(1, max_capacity // beta)
    max_num = max_capacity * 2 - min_num
    if num_layers <= 1:
        return max_capacity
    step = (max_num - min_num) / (num_layers - 1)
    budget = int(round(max_num - layer_idx * step))
    return max(budget, min_num)
