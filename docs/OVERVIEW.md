# Project overview — hotcold-kv-cache

A complete technical description of this repository: what it does, why it is
built the way it is, how the pieces fit together, and where the sharp edges are.
This document is self-contained — reading it should be enough to understand and
modify the codebase without reading every file first.

---

## 1. What this project is

An inference-time framework that makes the **attention KV cache** of a
long-context LLM cheaper in GPU memory **without permanently discarding
information**.

It does this by splitting the cache into two tiers and bridging them with
asynchronous, predictive retrieval:

- **Hot tier (GPU)** — a small, compressed KV cache held under a fixed token
  budget by a dynamic eviction policy.
- **Cold tier (CPU RAM)** — every token evicted from the GPU, stored at full
  resolution in a ring buffer and indexed with FAISS so it can be found again by
  similarity search.
- **Asynchronous look-ahead retrieval** — a background worker uses the direction
  of the model's recent *query* states to predict which cold tokens upcoming
  steps will need, and stages them onto the GPU **before** they are asked for, so
  the CPU→GPU transfer overlaps with generation instead of stalling it.

It is implemented for **Llama** by replacing `LlamaAttention.forward` at runtime,
with a custom **Triton** kernel on the decode path.

---

## 2. The problem being solved

An autoregressive LLM generates one token at a time. To produce token *n* it must
attend to tokens *1…n−1*. Rather than recompute them, it caches each token's
**Key** and **Value** vectors — the KV cache.

The KV cache grows **linearly with context length and batch size**, and it lives
in GPU memory. At 8k–16k tokens it can exceed the size of the model weights
themselves, and it caps how many concurrent requests a GPU can serve.

Two standard mitigations exist, and each gives something up:

| Approach | Examples | Cost |
|---|---|---|
| **Compression** — score tokens, delete the unimportant ones | SnapKV, PyramidKV, H2O | Deleted tokens are gone permanently. If the model needs one later, accuracy degrades. |
| **Offloading** — move old tokens to CPU RAM | conventional KV offload | The GPU blocks on a PCIe round trip whenever an offloaded token is needed. |

So the usual choice is *lose accuracy* or *lose speed*.

**This project's position:** don't delete, and don't wait. Keep everything
(cheaply, on the CPU), and *predict* what will be needed early enough that the
transfer is free.

The novelty is specifically the **look-ahead**: standard retrieval uses the
current token's query to fetch what the current token needs — which is
inherently synchronous, because you cannot know the query until you are already
at that step. Here, the pooled queries from steps `[t−w, t]` are used to fetch
for step `t+k`, which makes the fetch asynchronous and lets it overlap with
compute.

---

## 3. Environment and model facts

| Item | Value |
|---|---|
| Model | `meta-llama/Llama-3.2-1B` |
| Layers | 16 |
| Hidden size | 2048 |
| Query heads (`Hq`) | 32 |
| KV heads (`Hk`) | 8 (grouped-query attention) |
| `N_REP` = `Hq / Hk` | 4 |
| `head_dim` (`D`) | 64 |
| dtype | float16 on GPU |
| transformers | **4.57.3** (the patch matches this version's attention signature) |
| Other deps | torch, triton, faiss-cpu, numpy, pandas |
| Dev environment | Kaggle GPU notebook; benchmark reads LongBench from `/kaggle/input/longbench-json` |

Everything is **per-layer**: all 16 layers have their own hot cache, cold ring
buffer, FAISS index, background thread and staging slot. Layers never share
state.

---

## 4. Repository layout

```
hotcold-kv-cache/
├── src/hotcold_kv/
│   ├── patch.py          ★ monkeypatch of LlamaAttention.forward — the spine
│   ├── cache.py          ★ HotColdLayer — hot cache + eviction policy
│   ├── controller.py     ★ KVController — cold store, FAISS, async retrieval
│   ├── kernels.py        ★ Triton fused attention + eviction-scoring kernel
│   ├── datatypes.py        RetrievedKV / EvictedKV + shape canonicalisers
│   └── baselines/          SnapKV / PyramidKV / H2O (other papers, for comparison)
├── benchmarks/
│   ├── run_benchmark.py    LongBench harness for the hot/cold framework
│   └── run_baselines.py    same metrics for the three baselines
├── scripts/demo.py         minimal end-to-end example
├── tests/test_baselines.py CPU unit tests for the baseline tensor logic
└── docs/
    ├── design.md           the original phased research plan
    └── OVERVIEW.md         this document
```

★ = the four files that constitute the actual contribution. `baselines/` is
re-implementations of **other people's published methods**, used only as
comparison points — it is not part of the core idea.

---

## 5. The central invariant: four tensors, four destinations

This is the single most important thing to understand. After projection and
RoPE, the attention block holds four tensors, and each goes somewhere different:

| Tensor | RoPE applied? | Destination | Why |
|---|---|---|---|
| `q_pre` | no | FAISS **search vector** | Matching on semantic content, not positional phase |
| `q_post` | yes | attention + **eviction scoring** | Needs true attention geometry |
| `k_pre` | no | **cold store** (CPU) | Must be re-rotated to its original position on retrieval |
| `k_post` | yes | **hot cache** (GPU) | Used directly in attention |

`v` (values) has no pre/post distinction — **RoPE is never applied to values**,
because position only affects query-key matching.

### Why keys are stored pre-RoPE

Two independent reasons, both important:

1. **Correctness.** RoPE bakes absolute position into the key vector. A token
   evicted from position 47 must return *as position 47*, not as whatever index
   it lands on in the freshly concatenated tensor. Storing the unrotated key plus
   its original position lets `_rotate_k_with_pos` restore it exactly.
2. **Search quality.** The FAISS index compares a pre-RoPE query against
   pre-RoPE keys. Post-RoPE vectors are rotated by position, so similarity
   between them is contaminated by *where* tokens are rather than *what they
   are*. Pre-RoPE space gives a content-based match.

---

## 6. End-to-end execution flow

### Setup (see `scripts/demo.py`)

```python
set_active_rotary_emb(model.model.rotary_emb)   # global: RoPE module for re-rotation
kvctrl = KVController(...)                      # the controller (cold store + retrieval)
set_active_kvctrl(kvctrl)                       # global: how the patch finds it
install_monkeypatch(enable_debug=True)          # replaces LlamaAttention.forward
cache = build_hotcold_cache(model, kvctrl)      # Cache(layers=[HotColdLayer × 16])
```

Two module-level globals (`ACTIVE_KVCTRL`, `ACTIVE_ROTARY_EMB`) exist because
HuggingFace controls the `forward` signature — there is no parameter through
which to pass the controller. Consequence: **only one model can be patched at a
time**, and it is not safe for two models in one process.

### Prefill (`q_len > 1`, runs once)

1. Project and reshape to q `(B,32,T,64)`, k/v `(B,8,T,64)`.
2. Apply RoPE → `q_post`, `k_post`.
3. Append everything to the hot cache; compress if over budget; evicted tokens go
   to the CPU cold store and FAISS.
4. Attention via `repeat_kv` + `F.scaled_dot_product_attention` (**not** the
   Triton kernel).

### Decode (`q_len == 1`, runs thousands of times)

1. Project, apply RoPE.
2. **Fire async work** (non-blocking, before the cache update):
   - push `q_pre` into a rolling window of the last 4 queries;
   - every 16 steps, pool those queries and spawn a background thread to search
     FAISS;
   - if an earlier search has finished, start a non-blocking CPU→GPU copy on a
     dedicated CUDA stream.
3. **Update the hot cache**: append, and if over budget, score tokens, keep
   top-k + recent window, ship the rest to the cold store.
4. **Poll for retrieved tokens** with a non-blocking CUDA event check. If ready,
   re-apply RoPE at their original positions and concatenate them *in front of*
   the hot cache — **for this one step only**.
5. **Attention via the Triton kernel**, with a boolean mask marking which slots
   are real and causally valid.
6. `o_proj`, then back to the standard decoder layer (residual → MLP → next
   layer).

The concatenated `[retrieved | hot]` tensor is a local variable. Retrieved tokens
are **never written back** into the hot cache.

---

## 7. File by file

### 7.1 `src/hotcold_kv/patch.py` — the spine (296 lines)

Replaces `LlamaAttention.forward` with `patched_forward`, which orchestrates
everything. Also holds the RoPE re-rotation helpers.

**Key contents:**

- `ACTIVE_KVCTRL`, `ACTIVE_ROTARY_EMB` + setters — the global side-channel.
- `_normalize_rope_cos_sin_to_b1sd(cos, sin, B, S)` — defensive shape
  normaliser. `cos`/`sin` come back in different layouts across transformers
  versions; this forces `(B, 1, S, D)`. The `1` is the head axis, broadcast
  across heads.
- `_rope_cos_sin_for_positions(pos_ids, ...)` — asks the model's own rotary
  module for cos/sin at **arbitrary** positions (retrieved tokens come from
  scattered positions in the past). Passes a dummy `torch.empty` tensor because
  the rotary module reads only its dtype/device, never its values. Has a
  `try/except TypeError` for positional-vs-keyword API differences.
- `_rotate_k_with_pos(k_pre, pos_ids, head_dim)` — **the function that makes
  retrieval correct.** Re-applies RoPE to retrieved keys at their original
  positions. Passes a throwaway zero tensor as `q` because
  `apply_rotary_pos_emb` rotates q and k together and we only want k.
- `install_monkeypatch()` / `uninstall_monkeypatch()` — save and restore the
  original method in `_ORIG`. The `if _ORIG is not None: return` guard makes
  install **idempotent**; without it, a second install would capture the
  *already patched* function as `_ORIG`, and the fallback path would recurse
  infinitely.

**Inside `patched_forward`, the notable parts:**

- Escape hatch: `if kvctrl is None: return _ORIG(...)` — behaves exactly like
  stock Llama when no controller is registered, so the patch is safe to leave
  installed.
- `q_len` is the branch variable: `> 1` → prefill/SDPA, `== 1` → decode/Triton.
- Position handling: positions come from the externally supplied
  `cache_position`, **never** from the cache length — because the cache is
  compressed, its physical length is not the true sequence position. The
  `arange(hot_prev_len, ...)` branches are fallbacks that would be wrong once
  compression begins.
- `layer.set_last_query_post(q_post)` — a workaround. HuggingFace's
  `Cache.update()` receives only keys and values, never the query, but the
  eviction policy needs the query to score tokens. So the query is stashed on the
  layer object first.
- The causality guard: `keep_mem = rr.valid & (rr.pos <= cur_pos)`. A retrieved
  token must be a filled slot **and** from the past. Omitting the second
  condition would leak future information and produce invalid (flatteringly low)
  perplexity.
- Debug probe (layer 0, every 50 steps): prints hot size, cold size, and
  `ready_rate = prefetch_ready / prefetch_launched` — the key metric for whether
  the asynchronous look-ahead is actually landing in time.
- Returns `(attn_out, None)`; attention weights are never materialised, which is
  part of why it is memory-efficient (and why `output_attentions=True` will not
  work).

### 7.2 `src/hotcold_kv/cache.py` — the hot cache and eviction policy (173 lines)

`HotColdLayer(CacheLayerMixin)` holds four parallel tensors per layer:

| Field | Shape | Contents |
|---|---|---|
| `keys` | `(B, Hk, S, D)` | post-RoPE keys (used in attention) |
| `values` | `(B, Hk, S, D)` | values |
| `pre_keys` | `(B, Hk, S, D)` | pre-RoPE keys (shipped to the cold store on eviction) |
| `positions` | `(B, S)` | true absolute position of each cached token |

**`update()`** appends the new k/v/k_pre/positions, then calls
`_compress_if_needed`. Note its signature takes `layer_idx` as the first
argument, which differs from HuggingFace's `Cache.update` — hence the explicit
keyword call in `patch.py`.

**`_compress_if_needed()` is the eviction algorithm:**

1. Return early if `S <= max_capacity_prompt`.
2. `recent = min(recent_size, max_capacity_prompt, S)`; `prefix_len = S - recent`.
   The last `recent` tokens are always retained.
3. `topk_budget = int(max_capacity_prompt * topk_ratio)`, clamped to
   `[1, prefix_len]`.
4. Score tokens using the stashed post-RoPE query:
   - take the last `recent` query positions;
   - reduce query heads into KV groups by averaging (`view(B,Hk,group,tq,D).mean(2)`) — this is the GQA handling;
   - `attn_logits = q_for_k @ keys.transpose(2,3) / sqrt(D)`;
   - `token_scores = attn_logits.sum(dim=2).mean(dim=1)` → `(B, S)` — summed over
     query positions, averaged over heads.
5. Keep `topk_budget` highest-scoring prefix tokens plus the recent window;
   gather `keys`, `values`, `pre_keys`, `positions` down to that set.
6. Everything in the prefix that was not kept is **evicted**: copied into
   **pinned** CPU tensors (padded to the per-batch maximum, with `-1` marking
   unused slots) and handed to `kvctrl.add_evicted_to_cold_store`.

With the benchmark settings (`HOT_CAPACITY=512`, `TOPK_RATIO=0.15`,
`RECENT_SIZE=128`) the cache **oscillates between ~204 and 512 tokens**: it
fills to 512, compresses to 76 + 128 = 204, then refills.

Note: scores are reduced to `(B, S)`, so **the same token indices are evicted
across all KV heads**. This is a deliberate simplification (the baselines in
`baselines/` do per-head selection instead).

`get_seq_length()` returns the **physical** cache length, which after compression
is *not* the sequence position — this is exactly why positions are supplied
externally.

### 7.3 `src/hotcold_kv/controller.py` — cold store and async retrieval (396 lines)

`KVController` owns everything on the CPU side. All state is indexed
`[layer][batch]`:

- `cpu_kpre_ring`, `cpu_v_ring`, `cpu_pos_ring` — `deque(maxlen=cpu_buffer_size)`
  holding evicted tokens.
- `index` — one `faiss.IndexFlatIP` per (layer, batch). `Flat` = exact
  brute-force; `IP` = inner product, which equals cosine similarity because the
  vectors are L2-normalised.
- `faiss_lock` — an `RLock` per (layer, batch), since the background thread and
  the main thread both touch the index.
- `_retrieval_thread`, `_retrieval_ready_cpu` — worker handle and its result slot.
- `_prefetch_stream`, `_staged_gpu`, `_staged_event` — the separate CUDA stream,
  the staged GPU tensors, and the event used to test completion.
- `dbg` — a dictionary of counters used to verify the pipeline is working.

**The methods that matter, in call order:**

| Method | Role |
|---|---|
| `_lazy_init(k_pre, n_layers)` | Allocates all per-layer structures on the first real forward pass (B, Hk, D are only known at runtime). Idempotent. |
| `add_evicted_to_cold_store(layer_idx, ev)` | Appends each evicted token to the rings and adds its vector to FAISS. |
| `update_query_window(layer_idx, q_pre)` | Pushes the pre-RoPE query into a `deque(maxlen=query_pool_size)`. |
| `maybe_launch_async_retrieval(layer_idx)` | Increments `step_count`; every `window_size` steps, pools the query window and **spawns a background thread**. Skips if a previous thread is still alive. |
| `_retrieval_worker(layer_idx, q_vecs)` | Runs off the main thread: searches FAISS, copies the hits into **pinned** CPU tensors, publishes them. |
| `maybe_prefetch_to_gpu(layer_idx, ...)` | Takes a finished result and issues a **non-blocking** host→device copy on `_prefetch_stream`, recording a CUDA event. |
| `poll_staged_gpu(layer_idx)` | **Non-blocking** `event.query()`. Not ready → returns `ready=False` and the step simply proceeds without retrieved tokens. |

**Vector construction** (small but frequently asked about):

- `_k_token_to_vec_np`: take a token's pre-RoPE key `(Hk, D)`, **average over the
  8 KV heads** → `(D,)`, pad or crop to `faiss_dim`, L2-normalise.
- `_q_to_vecs_np`: average the pooled query over heads and over window positions
  → `(B, D)`, pad/crop, L2-normalise.

Because both sides are normalised, inner-product search is cosine similarity.

**How stalling is avoided — four mechanisms:**

1. background Python threads for the FAISS search (FAISS releases the GIL, so
   this is genuinely concurrent);
2. a dedicated CUDA stream for the host→device copy;
3. **pinned (page-locked)** CPU memory, which allows true asynchronous DMA;
4. `event.query()` — a non-blocking completion test, never `synchronize()`.

There is also a `force_wait_for_prefetch_diagnostic` flag that deliberately
blocks, so the benefit of the asynchrony can be measured rather than assumed.

### 7.4 `src/hotcold_kv/kernels.py` — the Triton kernel (137 lines)

`evict_fwd_kernel` computes, **in a single pass over K and V**:

1. the attention output `softmax(q·kᵀ · scale) · V`, and
2. a per-token "contribution" score intended for eviction ranking.

The point of fusing them is that K and V are read from GPU memory **once**
instead of twice.

**Launch geometry:** `grid = (B, Hq)` — one program instance per (batch, query
head); 32 run concurrently for this model. Each walks the whole KV sequence in
blocks of `BLOCK_S = 64`.

**Notable implementation details:**

- `kv_head_idx = pid_h // N_REP` — GQA mapping done by index arithmetic. Unlike
  the SDPA path, this **avoids materialising** a `repeat_kv`-expanded tensor.
- Masked tokens get score `-1e35`, so `exp()` yields exactly 0 and they
  contribute nothing — masking without branching.
- `inb = s < SEQ_LEN_KV` guards the final partial block when the sequence length
  is not a multiple of `BLOCK_S`.
- `acc_o` (numerator) and `l_i` (softmax denominator) accumulate in **float32**
  even when the data is fp16; the single division happens at the very end.
- The kernel is **decode-only** — `query` is `[B, Hq, 1, D]`. Prefill uses SDPA.

The wrapper `triton_evict_attention_forward` allocates the output buffers,
validates that `Hq // N_REP == Hk`, extracts strides, launches the kernel, and
returns `(attn_output, evict_idx)`.

### 7.5 Supporting files

- **`datatypes.py`** — `EvictedKV` (pinned CPU `k_pre`, `v`, `pos`, plus a padded
  index tensor using `-1` for unused slots) and `RetrievedKV` (GPU tensors plus
  `valid` and `ready` flags). Also `_canon_kv_4d` / `_canon_br_2d`, which force
  tensors to `(B,Hk,S,D)` and `(B,S)` by squeezing stray singleton dimensions —
  defensive plumbing that appears throughout.
- **`benchmarks/run_benchmark.py`** — LongBench harness. Measures perplexity via
  **token-by-token NLL** (not one batched forward), wall-clock latency, and peak
  GPU memory. Importantly it drives decoding manually and passes an explicit
  `cache_position` each step, which is what keeps positions correct under
  compression.

---

## 8. Configuration reference

From `benchmarks/run_benchmark.py`:

| Parameter | Value | Meaning |
|---|---|---|
| `HOT_CAPACITY` (`max_capacity_prompt`) | 512 | GPU budget; compression triggers above this |
| `TOPK_RATIO` | 0.15 | Fraction of budget kept as "important" → 76 tokens |
| `RECENT_SIZE` | 128 | Trailing tokens always retained |
| `CPU_BUFFER_SIZE` | 8192 | Ring-buffer capacity per (layer, batch) |
| `FAISS_DIM` | 64 | Search-vector dimension |
| `WINDOW_SIZE` | 16 | Launch a retrieval every N decode steps |
| `QUERY_POOL_SIZE` | 4 | Queries averaged into one search vector |
| `LOOKAHEAD_K` | 32 | Tokens fetched per retrieval; 0 disables retrieval |
| `HOLD_RETRIEVED` | 0 | 1 keeps a retrieved set in attention until the next one replaces it |
| `MAX_INPUT_LENGTH` | 4096 | Prompt truncation |
| `MAX_NEW_TOKENS_LAT` | 64 | Tokens generated for the latency measurement |

---

## 9. Measured results

Full tables live in [`results/`](../results), one per dataset. All runs use
Llama-3.1-8B in fp16 on a single NVIDIA L40S (48 GB), LongBench prompts cut at
8192 tokens, perplexity scored over the last 256 prompt tokens, and latency
measured over 63 decode steps after prefill.

Here is one representative table, `gov_report` (200 samples) at budget 512:

| Method | PPL | Δ vs full | Latency (s) | Peak mem (GiB) |
|---|---|---|---|---|
| full | 5.247 | 0.0% | 1.718 | 18.99 |
| snapkv | 5.942 | +13.2% | 1.562 | 17.15 |
| pyramidkv | 6.352 | +21.1% | 1.560 | 17.16 |
| h2o | 6.430 | +22.5% | 1.751 | 17.33 |
| hotcold (held) | 6.039 | +15.1% | 2.249 | 17.19 |

**Read the memory column carefully:** it is `torch.cuda.max_memory_allocated()`,
the **total** peak device memory, which includes the 14.96 GiB of fp16 model
weights and the prefill activations. It is *not* the KV cache in isolation, and
describing it as a "KV-cache footprint" would be wrong. At 8k the KV cache is
about 1 GiB of that total, which is why the compressed methods all land within
0.6 GiB of each other; at 16k the saving against `full` roughly doubles, from
about 1.8 GiB to about 3.8 GiB (see
[`gov_report_16k_results.md`](../results/gov_report_16k_results.md)).

Across the four datasets measured so far (`multifieldqa_en`, `hotpotqa`,
`gov_report`, `repobench-p`) the ordering is stable: SnapKV is best at every
budget, held hot/cold is second in 11 of 12 configurations, and H2O is last.
Held hot/cold costs about 30% more decode latency than `full`.

The project's main finding to date is the retrieval ablation in
[`results/retrieval_ablation.md`](../results/retrieval_ablation.md): how long a
retrieved token stays in attention matters more than the retrieval itself.

---

## 10. Invariants that must not be broken

Anyone modifying this code should preserve these:

1. **Positions are external.** Never derive a token's position from
   `cache.get_seq_length()`. The cache is compressed, so its length is not the
   sequence position. Positions come from `cache_position` / the stored
   `positions` tensor.
2. **Retrieved keys must be re-rotated to their original positions** before use
   (`_rotate_k_with_pos`).
3. **Causality must be enforced on retrieved tokens** via
   `rr.valid & (rr.pos <= cur_pos)`. Dropping this leaks future information.
4. **Never block on the prefetch.** `poll_staged_gpu` must stay non-blocking; a
   late prefetch degrades gracefully to compression-only behaviour.
5. **Values are never RoPE-rotated.**
6. **`install_monkeypatch` must stay idempotent**, or the fallback path recurses.

---

## 11. Known limitations and next steps

Listed honestly; several are good follow-up work.

1. **The kernel's eviction scores are computed and discarded.** The call site
   does `attn_out, _ = triton_evict_attention_forward(...)`; the actual eviction
   runs separately in `_compress_if_needed` with its own matmul. Wiring the
   kernel's `final_loss` into compression would remove a redundant pass.

2. **The softmax in the Triton kernel is not numerically stabilised.** It uses
   raw `tl.exp(scores)` with no running-maximum subtraction. Fine for typical
   fp32 logits, but a FlashAttention-style online softmax would be the correct
   treatment.

3. **The ring buffer and the FAISS index can desynchronise.** The rings are
   `deque(maxlen=cpu_buffer_size)`, so once full they silently drop the oldest
   entry — but nothing is ever removed from the FAISS index. After the buffer
   wraps, FAISS index *i* no longer corresponds to `deque[i]`, so retrieval would
   return misaligned KV pairs. The `enable_asserts` path checks exactly this
   (`_num_vecs == len(ring)`), and the benchmark runs with asserts disabled.
   **With the current settings it does not trigger** (≈4k evicted tokens per
   layer against a buffer of 8192), but it *would* at the 8k–16k contexts the
   design targets. Fixing it means either an index that supports removal, or a
   mapping from FAISS id to ring slot.

4. **`IndexFlatIP` is exact brute-force search.** Adequate at a few thousand
   vectors; an IVF index is the documented upgrade for scale.

5. **Eviction is uniform across KV heads.** Token scores are averaged to `(B, S)`
   rather than chosen per head, which is coarser than what SnapKV/H2O do.

6. **Global state limits the framework to one patched model per process.**

7. **Attention weights are not returned** (`output_attentions=True` is
   unsupported), because the full attention matrix is never materialised.

8. **The GIL still constrains the retrieval thread.** The FAISS search itself
   releases it, but the surrounding Python bookkeeping does not; a native worker
   would be the next step.

---

## 12. Glossary

| Term | Meaning |
|---|---|
| **Hot cache** | The compressed KV cache on the GPU |
| **Cold store** | Full-resolution evicted KV in CPU RAM, FAISS-indexed |
| **Prefill** | Processing the whole prompt in one forward pass (`q_len > 1`) |
| **Decode** | Generating one token at a time (`q_len == 1`) |
| **RoPE** | Rotary position embedding; rotates q and k to encode position |
| **GQA** | Grouped-query attention; several query heads share one KV head |
| **`N_REP`** | Query heads per KV head (4 here) |
| **Pinned memory** | Page-locked CPU memory enabling asynchronous DMA |
| **`ready_rate`** | `prefetch_ready / prefetch_launched` — whether look-ahead lands in time |
