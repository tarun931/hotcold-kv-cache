"""Unit tests for the compression-baseline tensor logic.

These run on CPU with random tensors and no transformers/model dependency: they
exercise the PressLayerState prefill/decode cycle and the scoring primitives to
check shapes, that budgets are respected, that GQA is handled, and that outputs
are finite. They do NOT test the LlamaAttention monkeypatch (that needs a model).
"""

import torch

try:
    # Normal path: full environment with triton/transformers installed.
    from hotcold_kv.baselines.config import PressConfig, pyramid_layer_budget
    from hotcold_kv.baselines.layer_state import PressLayerState
    from hotcold_kv.baselines import scoring
except ModuleNotFoundError:
    # Lightweight path: load only the torch-only submodules (config/scoring/
    # layer_state) without triggering the package __init__ chain that pulls in
    # triton/transformers. Lets these tests run on a CPU-only box.
    import os, sys, types, importlib.util

    _BASE = os.path.join(os.path.dirname(__file__), "..", "src", "hotcold_kv", "baselines")
    for _pkg, _path in [("hotcold_kv", os.path.dirname(_BASE)),
                        ("hotcold_kv.baselines", _BASE)]:
        _m = types.ModuleType(_pkg)
        _m.__path__ = [_path]
        sys.modules[_pkg] = _m

    def _load(name):
        spec = importlib.util.spec_from_file_location(
            f"hotcold_kv.baselines.{name}", os.path.join(_BASE, f"{name}.py"))
        mod = importlib.util.module_from_spec(spec)
        sys.modules[f"hotcold_kv.baselines.{name}"] = mod
        spec.loader.exec_module(mod)
        return mod

    _load("scoring")
    config = _load("config")
    layer_state = _load("layer_state")
    scoring = sys.modules["hotcold_kv.baselines.scoring"]
    PressConfig = config.PressConfig
    pyramid_layer_budget = config.pyramid_layer_budget
    PressLayerState = layer_state.PressLayerState


def _run_cycle(method, B=2, Hq=8, Hk=2, S=200, D=16, budget=64, window=16, decode_steps=40):
    torch.manual_seed(0)
    n_rep = Hq // Hk
    scaling = 1.0 / (D ** 0.5)
    st = PressLayerState(method=method, budget=budget, window_size=window,
                         kernel_size=7, pooling="maxpool")

    q = torch.randn(B, Hq, S, D)
    k = torch.randn(B, Hk, S, D)
    v = torch.randn(B, Hk, S, D)
    out = st.prefill(q, k, v, n_rep, scaling)
    assert out.shape == (B, Hq, S, D), f"prefill out {out.shape}"
    assert torch.isfinite(out).all(), "prefill produced non-finite output"
    assert st.get_seq_length() == budget, f"{method}: post-prefill len {st.get_seq_length()} != {budget}"

    for t in range(decode_steps):
        q1 = torch.randn(B, Hq, 1, D)
        k1 = torch.randn(B, Hk, 1, D)
        v1 = torch.randn(B, Hk, 1, D)
        out1 = st.decode(q1, k1, v1, n_rep, scaling)
        assert out1.shape == (B, Hq, 1, D), f"decode out {out1.shape}"
        assert torch.isfinite(out1).all(), "decode produced non-finite output"

    return st


def test_snapkv_cycle():
    st = _run_cycle("snapkv", budget=64, decode_steps=40)
    # snapkv/pyramid append during decode (no re-eviction) -> grows past budget
    assert st.get_seq_length() == 64 + 40


def test_pyramidkv_cycle():
    _run_cycle("pyramidkv", budget=80, decode_steps=20)


def test_h2o_cycle_holds_budget():
    st = _run_cycle("h2o", budget=64, decode_steps=60)
    # h2o re-evicts every decode step -> length stays pinned at budget
    assert st.get_seq_length() == 64, f"h2o len drifted to {st.get_seq_length()}"
    assert st.scores is not None and st.scores.shape[2] == 64


def test_keep_index_respects_budget_and_recent():
    B, Hk, S = 2, 3, 100
    budget, window = 30, 8
    scores = torch.randn(B, Hk, S)
    keep = scoring.keep_index(scores, window, budget)
    assert keep.shape == (B, Hk, budget), keep.shape
    # recent window (last `window` positions) must always be present
    recent = set(range(S - window, S))
    for b in range(B):
        for h in range(Hk):
            got = set(keep[b, h].tolist())
            assert recent.issubset(got), "recent window not fully retained"


def test_gqa_reduction_shapes():
    B, Hq, Hk, S, D = 1, 8, 2, 50, 16
    q = torch.randn(B, Hq, 5, D)
    reduced = scoring._group_query_to_kv(q, Hk)
    assert reduced.shape == (B, Hk, 5, D)
    probs = torch.randn(B, Hq, 1, S)
    rp = scoring._reduce_probs_to_kv(probs, Hk)
    assert rp.shape == (B, Hk, 1, S)


def test_pyramid_budget_monotone_decreasing():
    n = 16
    budgets = [pyramid_layer_budget(i, n, 512, 20) for i in range(n)]
    assert budgets[0] > budgets[-1], "pyramid budget should shrink with depth"
    assert all(budgets[i] >= budgets[i + 1] for i in range(n - 1)), "not monotone"
    # mean roughly around max_capacity
    assert 300 < sum(budgets) / n < 720


def test_pool_preserves_length_odd_kernel():
    x = torch.randn(2, 3, 57)
    for pooling in ("maxpool", "avgpool"):
        y = scoring.pool1d(x, 7, pooling)
        assert y.shape == x.shape, (pooling, y.shape)


if __name__ == "__main__":
    fns = [v for k, v in sorted(globals().items()) if k.startswith("test_")]
    passed = 0
    for fn in fns:
        fn()
        print(f"PASS {fn.__name__}")
        passed += 1
    print(f"\n{passed}/{len(fns)} tests passed")
