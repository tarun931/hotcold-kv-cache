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
  - *SnapKV*: observation-window importance + pooling; prompt compression.
  - *PyramidKV*: SnapKV importance with a per-layer (pyramidal) budget.
  - *H2O*: Heavy-Hitter Oracle; accumulated attention with online eviction.
- **Metric harness** over LongBench: perplexity (PPL), wall-clock latency,
  peak GPU memory allocated. Implemented in `benchmarks/run_benchmark.py`
  (hot/cold) and `benchmarks/run_baselines.py` (baselines).

> Status: implemented. The hot/cold framework, the metric harness, and all three
> compression baselines (`src/hotcold_kv/baselines/`) are in place; the baseline
> tensor logic is covered by `tests/test_baselines.py`.

### Baseline design notes

All three baselines share one integration path: a per-layer `Cache` whose stored
keys/values are post-RoPE (so a retained token keeps its positional encoding after
compression), plus an attention monkeypatch that computes attention internally —
giving full control over masking rather than depending on the model's causal-mask
machinery. Grouped-query attention is handled by averaging query heads into their
key/value group when scoring, and eviction indices are per-kv-head. Budgets always
retain a recent window; SnapKV/PyramidKV compress once at prefill, while H2O keeps
a running attention accumulator and re-evicts to the budget every decode step.

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
- **Residency of retrieved tokens.** A retrieved set is held per layer on the GPU
  and stays in attention until the next retrieval replaces it, so it is used for
  the whole window rather than only at the step its copy lands
  (`HOLD_RETRIEVED=1`). This is worth far more than the retrieval itself: see
  Phase 5.

> Status: implemented. Retrieval residency is opt-in via `HOLD_RETRIEVED`; the
> default remains attend-once so earlier measurements stay reproducible.

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

> Status: the retrieval ablation is done and is the project's main finding so
> far — see [`results/retrieval_ablation.md`](../results/retrieval_ablation.md).
> Three variants were measured with an identical GPU tier (the evicted-token
> counts match exactly), on LongBench at 8k with Llama-3.1-8B:
>
> | Variant | gov_report, budget 512 |
> |---|---|
> | no retrieval (`LOOKAHEAD_K=0`) | 6.139 |
> | attend once (default) | 6.126 |
> | held (`HOLD_RETRIEVED=1`) | 6.039 |
>
> Attend-once retrieval is nearly worthless: 0.09-0.21% perplexity for about 25%
> more latency, because a retrieved set was used for one decode step out of
> sixteen and then dropped. Holding it for the whole window helps in all twelve
> dataset/budget configurations, by up to 3.18%, and is worth about seven times
> what attend-once was worth. **How long a retrieved token stays in attention
> matters more than the retrieval itself.**
>
> The no-retrieval run also locates the latency cost: with retrieval off,
> hot/cold (1.601 s) is faster than the uncompressed reference (1.718 s), so the
> overhead is the retrieval path, not the Triton kernel or eviction.

## References

- **SnapKV** — *SnapKV: LLM Knows What You are Looking for Before Generation.*
  NeurIPS 2024.
  <https://proceedings.neurips.cc/paper_files/paper/2024/file/28ab418242603e0f7323e54185d19bde-Paper-Conference.pdf>
  · code: <https://github.com/FasterDecoding/SnapKV>
- **PyramidKV** — *Dynamic KV Cache Compression based on Pyramidal Information
  Funneling.* <https://openreview.net/pdf?id=jZVNmDiU86>
- **H2O** — *H2O: Heavy-Hitter Oracle for Efficient Generative Inference of Large
  Language Models.* NeurIPS 2023. <https://arxiv.org/abs/2306.14048>
