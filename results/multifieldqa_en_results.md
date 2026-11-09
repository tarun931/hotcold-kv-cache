# LongBench multifieldqa_en: Llama-3.1-8B, 8k context

| Budget | Method | PPL | Δ vs full | Latency (s) | Peak mem (GiB) |
|---|---|---|---|---|---|
| full | full | 6.257 | 0.0% | 1.711 | 18.99 |
| 512 | snapkv | **7.227** | +15.5% | 1.562 | 17.15 |
| 512 | pyramidkv | 7.839 | +25.3% | 1.561 | 17.16 |
| 512 | h2o | 7.659 | +22.4% | 1.750 | 17.33 |
| 512 | hotcold (attend once) | 7.538 | +20.5% | 2.015 | 17.18 |
| 512 | hotcold (held) | 7.360 | +17.6% | 2.309 | 17.19 |
| 1024 | snapkv | **6.795** | +8.6% | 1.596 | 17.28 |
| 1024 | pyramidkv | 6.987 | +11.7% | 1.591 | 17.29 |
| 1024 | h2o | 7.161 | +14.5% | 1.797 | 17.46 |
| 1024 | hotcold (attend once) | 7.009 | +12.0% | 2.212 | 17.37 |
| 1024 | hotcold (held) | 6.917 | +10.6% | 2.259 | 17.37 |
| 2048 | snapkv | **6.488** | +3.7% | 1.656 | 17.53 |
| 2048 | pyramidkv | 6.637 | +6.1% | 1.652 | 17.55 |
| 2048 | h2o | 6.750 | +7.9% | 1.921 | 17.71 |
| 2048 | hotcold (attend once) | 6.652 | +6.3% | 2.096 | 17.65 |
| 2048 | hotcold (held) | 6.603 | +5.5% | 2.292 | 17.66 |

Bold marks the lowest perplexity at each budget.

## Setup

- **Model:** `meta-llama/Llama-3.1-8B`, fp16, one NVIDIA L40S (48 GB).
- **Data:** all 150 samples. The prompt is the context plus the question, cut at 8192 tokens. Most documents are shorter than that, about 6k tokens on average.
- **Budget:** the most KV tokens each layer keeps on the GPU.
  - SnapKV, PyramidKV and H2O compress the prompt to the budget. H2O also evicts during decoding.
  - Hot/cold keeps between 75% and 100% of the budget in the hot tier and adds up to 64 tokens retrieved from the CPU tier.
- **Hot/cold variants:** see [the retrieval ablation](retrieval_ablation.md).
  - *attend once*: retrieved tokens are attended at the single decode step their copy lands, then dropped.
  - *held*: retrieved tokens stay in attention until the next retrieval replaces them, so they are used for a whole 16-step window.
- **PPL:** prefill all but the last 256 prompt tokens, then score those 256 tokens one at a time against the compressed cache. The protocol is identical for every method.
- **Latency:** wall time of 63 greedy decode steps after prefill; prefill time is not included.
- **Peak memory:** `torch.cuda.max_memory_allocated` over the prefill and the 64-token generation.

## Runs

- **Slurm jobs:**
  - `full`: 61890086
  - Baselines: array 61890087, tasks 0–11
  - Hot/cold, attend once: 61900805, tasks 3, 7 and 11
  - Hot/cold, held: 62374608, tasks 3, 7 and 11
- **Hot/cold re-run:** the attend-once rows come from a re-run, after a fix that stopped the hot/cold patch from holding every layer's full prefill queries. Perplexity matches the first run within 0.001. Peak memory fell from 21.2–21.6 GiB to the values above.
- **Retrieval:** at budgets 512 and 1024 it fired on all 76,800 retrieval windows (150 samples × 32 layers × 16 windows). At 2048 it fired on 70,304, because samples too short to evict anything have an empty CPU tier and skip retrieval.
- **Injection:** attend once injected 72,003, 72,005 and 65,795 times at budgets 512, 1024 and 2048; held injected 1,147,989, 1,148,243 and 1,049,300 times, about 16× more, since each retrieved set is used for a whole window.
- **Same GPU tier:** the two hot/cold variants evicted exactly the same number of tokens, so the only difference between them is how long retrieved tokens stay in attention.
- **Latency variance:** hot/cold latency moved by up to about 10% between the two attend-once runs, because its CPU-side work depends on thread scheduling.
