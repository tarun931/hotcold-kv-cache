# hotcold-kv-cache

A long-context LLM inference framework that decouples the attention KV cache into
two tiers:

- a **hot, compressed cache on the GPU**, kept under a fixed token budget by a
  dynamic eviction policy, and
- a **cold, full-resolution store on the CPU**, indexed with Faiss so that any
  previously evicted token can be recalled by similarity search.

The two tiers are bridged by an **asynchronous look-ahead retrieval** mechanism:
a background worker uses recently generated query states to pre-fetch the KV
pairs that upcoming tokens are likely to attend to, so the CPU→GPU transfer is
overlapped with generation instead of stalling it.

The goal is to hold GPU KV memory well below the cost of a full cache while
keeping accuracy on long-context tasks close to the uncompressed baseline.

## Why

Standard KV-cache offloading moves cold tokens to the CPU and pays a
synchronous transfer penalty every time an old token is needed again. The idea
here is to *predict* which cold tokens will be needed a few steps ahead — using
the trajectory of the last few query states — and stage them onto the GPU
before the model asks for them, hiding the transfer latency behind the GPU's own
decode step.

## Architecture

```
                 ┌─────────────────────────── GPU ───────────────────────────┐
   input token   │                                                            │
  ───────────────▶  Patched LlamaAttention.forward                           │
                 │      │                                                      │
                 │      │  compress when hot cache > budget                    │
                 │      ▼                                                      │
                 │  HotColdLayer  ──evicted (K_pre, V, pos)──┐                 │
                 │  (hot compressed KV)                      │                 │
                 │      ▲                                    │ non-blocking    │
                 │      │ inject retrieved KV (transient)    │ D2H             │
                 └──────┼────────────────────────────────────┼────────────────┘
                        │                                     ▼
                 ┌──────┼──────────────────────────── CPU ────────────────────┐
                 │      │        KVController (background retrieval thread)     │
                 │      │            │                                          │
                 │  look-ahead query │  Faiss index (key cache as vectors)      │
                 │      └────────────▶  ring buffer of (Key, Value) tensors     │
                 └──────────────────────────────────────────────────────────────┘
```

Core components (`src/hotcold_kv/`):

| Module          | Responsibility |
|-----------------|----------------|
| `patch.py`      | Monkeypatches `LlamaAttention.forward`; RoPE re-rotation of retrieved keys; global wiring (`install_monkeypatch`, `set_active_*`). |
| `controller.py` | `KVController` — the cold ring buffer, Faiss index, look-ahead query predictor and the asynchronous retrieval thread. |
| `cache.py`      | `HotColdLayer` (a `CacheLayerMixin`) holding the hot compressed KV and staging evicted tokens; `build_hotcold_cache`. |
| `kernels.py`    | Triton fused attention + eviction-scoring kernel used to pick which tokens to evict. |
| `datatypes.py`  | `RetrievedKV` / `EvictedKV` containers and shape-canonicalisation helpers. |

## Install

```bash
git clone <your-repo-url>
cd hotcold-kv-cache
pip install -r requirements.txt
pip install -e .
```

The framework patches Llama attention and uses Triton kernels, so a CUDA GPU is
required to run generation. It was developed against `transformers==4.57.3` and
`meta-llama/Llama-3.2-1B`; you will need HF access to that model.

## Quick start

```bash
python scripts/demo.py
```

This forces eviction (small hot capacity + a long repeated prompt), installs the
patched attention, and decodes 256 tokens through the hot/cold cache, printing
the hot-cache length and cold-store size at the end.

Using it directly:

```python
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from hotcold_kv import (
    KVController, set_active_kvctrl, set_active_rotary_emb,
    install_monkeypatch, build_hotcold_cache,
)

model = AutoModelForCausalLM.from_pretrained("meta-llama/Llama-3.2-1B",
                                             torch_dtype=torch.float16, device_map="cuda")
tok = AutoTokenizer.from_pretrained("meta-llama/Llama-3.2-1B")

set_active_rotary_emb(model.model.rotary_emb)
kvctrl = KVController(max_capacity_prompt=128, topk_ratio=0.25, recent_size=32,
                      cpu_buffer_size=8192, faiss_dim=64, window_size=8,
                      query_pool_size=4, lookahead_k=32)
set_active_kvctrl(kvctrl)
install_monkeypatch(enable_debug=True)

cache = build_hotcold_cache(model, kvctrl)
inputs = tok("<long prompt>", return_tensors="pt").to(model.device)
out = model.generate(**inputs, max_new_tokens=256, past_key_values=cache, use_cache=True)
print(tok.decode(out[0], skip_special_tokens=True))
```

## Benchmarks

```bash
python benchmarks/run_benchmark.py
```

The harness runs token-by-token decode on LongBench tasks and reports, per task:

- **Perplexity** (token-by-token NLL),
- **Wall-clock latency** (total and per-token),
- **Peak GPU memory allocated**.

Results are written to a CSV. The benchmark expects the LongBench JSONL files at
the path in `LONGBENCH_LOCAL_PATH` (defaults to the Kaggle dataset layout,
`/kaggle/input/longbench-json`); point it at your local copy to run elsewhere.

## Design targets

- GPU KV memory well below a full cache at long context (8k–16k+ tokens),
- minimal accuracy degradation versus the uncompressed baseline,
- retrieval touching only a small fraction of cold tokens per step,

by tuning the hot-cache budget, the eviction top-k ratio and the look-ahead
retrieval window. See [`docs/design.md`](docs/design.md) for the full phased
design and the reference papers (SnapKV, PyramidKV).

## License

MIT — see [LICENSE](LICENSE).
