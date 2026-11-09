"""Unit tests for the CPU cold tier in KVController.

They check that the FAISS index and the ring buffer stay in step: every id a
search returns must name the token actually stored in that ring slot, also
after the ring has filled up and started dropping its oldest tokens. Runs on
CPU with random tensors; no model, triton or transformers needed.
"""

import os
import sys
import types
import importlib.util

import numpy as np
import torch

# Load only controller.py and datatypes.py, without the package __init__
# chain that pulls in triton/transformers.
_SRC = os.path.join(os.path.dirname(__file__), "..", "src", "hotcold_kv")
if "hotcold_kv" not in sys.modules:
    _pkg = types.ModuleType("hotcold_kv")
    _pkg.__path__ = [_SRC]
    sys.modules["hotcold_kv"] = _pkg
for _name in ("datatypes", "controller"):
    _spec = importlib.util.spec_from_file_location(f"hotcold_kv.{_name}",
                                                   os.path.join(_SRC, f"{_name}.py"))
    _mod = importlib.util.module_from_spec(_spec)
    sys.modules[f"hotcold_kv.{_name}"] = _mod
    _spec.loader.exec_module(_mod)
KVController = sys.modules["hotcold_kv.controller"].KVController
EvictedKV = sys.modules["hotcold_kv.datatypes"].EvictedKV

if not torch.cuda.is_available():
    # Pinned memory needs CUDA; the cold-store logic does not.
    _zeros = torch.zeros
    torch.zeros = lambda *a, **k: _zeros(*a, **{x: y for x, y in k.items() if x != "pin_memory"})

HK, D = 2, 16


def _controller(cap, lookahead=4):
    ctrl = KVController(cpu_buffer_size=cap, faiss_dim=D, lookahead_k=lookahead,
                        enable_asserts=True, profile_overlap=False)
    ctrl._lazy_init(torch.zeros(1, HK, 1, D), n_layers=1)
    return ctrl


def _tokens(n, seed=0):
    g = torch.Generator().manual_seed(seed)
    return torch.randn(n, HK, D, generator=g)


def _evict(ctrl, keys, positions):
    """Send tokens to the cold tier the way HotColdLayer does."""
    te = keys.shape[0]
    ctrl.add_evicted_to_cold_store(0, EvictedKV(
        k_pre_cpu=keys.permute(1, 0, 2).unsqueeze(0).contiguous(),
        v_cpu=(-keys).permute(1, 0, 2).unsqueeze(0).contiguous(),
        pos_cpu=torch.tensor(positions, dtype=torch.long).unsqueeze(0),
        evicted_idx=torch.arange(te, dtype=torch.long).unsqueeze(0),
    ))


def _top1(ctrl, key):
    """Search with a token's own key; return (position, k_pre, v) of the best hit."""
    q = ctrl._k_token_to_vec_np(key)[None, :]
    ctrl._retrieval_worker(0, q)
    kpre, v, pos, valid = ctrl._retrieval_ready_cpu[0]
    assert bool(valid[0, 0]), "top hit is not valid"
    return int(pos[0, 0]), kpre[0, :, 0, :], v[0, :, 0, :], pos[0][valid[0]].tolist()


def test_no_wrap_returns_queried_token():
    ctrl = _controller(cap=64)
    keys = _tokens(20)
    _evict(ctrl, keys, list(range(100, 120)))
    for t in (0, 7, 19):
        pos, kpre, v, _ = _top1(ctrl, keys[t])
        assert pos == 100 + t, (t, pos)
        assert torch.equal(kpre, keys[t]) and torch.equal(v, -keys[t])


def test_wrap_keeps_ids_aligned():
    cap = 8
    ctrl = _controller(cap)
    keys = _tokens(15, seed=1)
    for lo in (0, 5, 10):                       # three batches of 5: 15 > cap
        _evict(ctrl, keys[lo:lo + 5], list(range(lo, lo + 5)))
    ring = ctrl.cpu_kpre_ring[0][0]
    assert len(ring) == cap and ctrl.index[0][0].ntotal == cap
    live = range(15 - cap, 15)                  # positions 7..14 survive
    for t in live:
        pos, kpre, v, _ = _top1(ctrl, keys[t])
        assert pos == t, (t, pos)
        assert torch.equal(kpre, keys[t]) and torch.equal(v, -keys[t])
    for t in range(15 - cap):                   # dropped tokens never come back
        _, _, _, hits = _top1(ctrl, keys[t])
        assert all(h in live for h in hits), (t, hits)


def test_batch_larger_than_ring():
    cap = 8
    ctrl = _controller(cap)
    keys = _tokens(20, seed=2)
    _evict(ctrl, keys, list(range(20)))         # one batch of 20 > cap
    assert len(ctrl.cpu_kpre_ring[0][0]) == cap and ctrl.index[0][0].ntotal == cap
    for t in range(20 - cap, 20):
        pos, kpre, _, _ = _top1(ctrl, keys[t])
        assert pos == t and torch.equal(kpre, keys[t]), (t, pos)


if __name__ == "__main__":
    fns = [v for k, v in sorted(globals().items()) if k.startswith("test_")]
    passed = 0
    for fn in fns:
        fn()
        print(f"PASS {fn.__name__}")
        passed += 1
    print(f"\n{passed}/{len(fns)} tests passed")
