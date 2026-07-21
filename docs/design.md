# Design

An inference framework that decouples the KV cache into a hot compressed tier
(GPU) and a cold full-retrieval tier (CPU), bridged by an asynchronous
look-ahead retrieval mechanism.

**Novelty.** Unlike standard offloading, the framework uses *previously*
generated query states to pre-fetch the KV pairs that *future* tokens are likely
to attend to, hiding CPU↔GPU transfer latency behind the GPU decode step.

**Targets.** GPU KV memory far below a full cache at long context (8k–16k+
tokens) with minimal accuracy degradation, while retrieving only a small
fraction of cold tokens.

## Phase 1 — Infrastructure & baseline harness

Establish a stable testing ground with unmodified models before introducing
custom kernels.

- **Standard compression baselines** to benchmark against:
  - *SnapKV*: sliding-window + heavy-hitter retention.
  - *PyramidKV*: layer-wise budget allocation.
- **Metric harness** over LongBench: perplexity (PPL), wall-clock latency,
  peak GPU memory allocated. Implemented in `benchmarks/run_benchmark.py`.

> Status: the metric harness and the hot/cold framework are implemented. The
> SnapKV / PyramidKV baselines are on the roadmap for direct side-by-side
> comparison.

## Phase 2 — Core engine

Two decoupled memory components.

### 2.1 GPU "hot" cache manager (modified attention)

`LlamaAttention.forward` is overridden (`patch.py`) rather than using the stock
HuggingFace implementation.

- **Compression policy.** At step *t*, if the hot cache exceeds its budget:
  compute attention scores, keep the top-k tokens, and — instead of deleting the
  evicted tokens — move them to a transfer buffer for offload.
- **KV injection interface.** A hook in the forward pass accepts retrieved KV
  pairs from the CPU and concatenates them with the hot cache *for the current
  step only*.

### 2.2 CPU "cold" store & vector index

- **Ring buffer** on CPU RAM storing `(Key, Value)` tensors for every token
  evicted from the GPU.
- **Indexing.** Faiss over the key cache (keys act as the vector database);
  inner-product search, with room to move to an IVF index for speed.

## Phase 3 — Asynchronous look-ahead retrieval

Standard retrieval uses token *t* to find keys for *t*. Here tokens `[t-w, t]`
are used to find keys for `t+k`.

- **Query predictor.** A background worker triggers every *N* steps; it average-
  pools the last *W* query states and queries the Faiss index with the pooled
  state.
- **Asynchronous pipeline.** GPU generation and CPU retrieval run concurrently;
  non-blocking D2H (evicted) and H2D (retrieved) transfers overlap with decode.
  The forward pass checks an `is_retrieval_ready` flag and, if set, folds the
  retrieved KV pairs into the attention computation.

## Phase 4 — Optimization & refinement

- Nearest-neighbour search instead of raw dot product for retrieval.
- Hyperparameter tuning: compression ratio, retrieval budget, and chunk size
  (how many tokens are fetched at once).

## Phase 5 — Analysis

- **Ablations.** Compression-only (no retrieval) to show the baseline drop;
  synchronous current-token retrieval to show the latency penalty versus the
  asynchronous look-ahead path.
- **Visualisation.** Attention heatmaps confirming retrieved tokens are attended
  to; latency-vs-accuracy Pareto frontier.

## References

- **SnapKV** — *SnapKV: LLM Knows What You are Looking for Before Generation.*
  NeurIPS 2024.
  <https://proceedings.neurips.cc/paper_files/paper/2024/file/28ab418242603e0f7323e54185d19bde-Paper-Conference.pdf>
  · code: <https://github.com/FasterDecoding/SnapKV>
- **PyramidKV** — *Dynamic KV Cache Compression based on Pyramidal Information
  Funneling.* <https://openreview.net/pdf?id=jZVNmDiU86>
