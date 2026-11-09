# LongBench gov_report: Llama-3.1-8B, 16k context

| Budget | Method | PPL | Δ vs full | Latency (s) | Peak mem (GiB) |
|---|---|---|---|---|---|
| full | full | 5.393 | 0.0% | 1.784 | 23.01 |
| 512 | snapkv | **6.180** | +14.6% | 1.562 | 19.17 |
| 512 | pyramidkv | 6.679 | +23.8% | 1.560 | 19.18 |
| 512 | h2o | 6.706 | +24.4% | 1.747 | 19.57 |
| 512 | hotcold (attend once) | 6.367 | +18.1% | 2.052 | 19.20 |
| 512 | hotcold (held) | 6.278 | +16.4% | 2.232 | 19.21 |
| 1024 | snapkv | **5.911** | +9.6% | 1.596 | 19.30 |
| 1024 | pyramidkv | 5.982 | +10.9% | 1.593 | 19.31 |
| 1024 | h2o | 6.349 | +17.7% | 1.795 | 19.70 |
| 1024 | hotcold (attend once) | 6.058 | +12.3% | 2.037 | 19.38 |
| 1024 | hotcold (held) | 6.000 | +11.3% | 2.274 | 19.40 |
| 2048 | snapkv | **5.681** | +5.3% | 1.658 | 19.55 |
| 2048 | pyramidkv | 5.738 | +6.4% | 1.657 | 19.57 |
| 2048 | h2o | 6.053 | +12.2% | 1.933 | 19.95 |
| 2048 | hotcold (attend once) | 5.814 | +7.8% | 2.125 | 19.66 |
| 2048 | hotcold (held) | 5.773 | +7.0% | 2.307 | 19.68 |

Bold marks the lowest perplexity at each budget.

## What the longer context changes

Against [the same dataset at 8k](gov_report_results.md):

- **Memory saved roughly doubles.** `full` rises from 18.99 to 23.01 GiB because its KV cache
  grows with the prompt, while the compressed methods rise only about 2 GiB, from the longer
  prefill. The saving against `full` goes from about 1.8 GiB to about 3.8 GiB.
- **Latency barely moves.** The compressed methods decode over a fixed-size cache, so their
  latency is unchanged; `full` slows by only 4%, because decoding is bound by reading the
  model weights rather than the cache.
- **Every method loses a little more perplexity**, since a larger share of each prompt is
  evicted.
- **Held retrieval helps by a similar margin as at 8k**: -1.40%, -0.96% and -0.70% at budgets
  512, 1024 and 2048.
- **One ordering change.** At 8k, held hot/cold beat PyramidKV at all three budgets on this
  dataset; at 16k it beats PyramidKV only at budget 512. At 1024 and 2048 PyramidKV is ahead
  (5.982 vs 6.000, and 5.738 vs 5.773). PyramidKV's per-layer budget allocation appears to
  scale better with context length than our flat budget does.

## Setup

The setup is the same as for [gov_report at 8k](gov_report_results.md), including the hot/cold
variants, with these differences:

- **Context:** prompts are cut at 16,384 tokens instead of 8,192.
- **Prompt length:** about 9.4k tokens on average. 35 of the 200 samples are long enough to reach
  the 16k cap; the rest are shorter.

## Runs

- **Slurm jobs:**
  - `full`: 62127873
  - Baselines and hot/cold, attend once: array 62127874, tasks 0–11
  - Hot/cold, held: array 62395234, tasks 3, 7 and 11
- **Retrieval:** at budgets 512 and 1024 it fired on all 102,400 retrieval windows (200 samples ×
  32 layers × 16 windows). At 2048 it fired on 101,888, because a few short samples have nothing
  in the CPU tier.
- **Injection:** attend once injected 96,004, 96,005 and 95,521 times at budgets 512, 1024 and
  2048; held injected 1,530,651, 1,530,987 and 1,523,562 times, about 16× more, since each
  retrieved set is used for a whole window.
- **Same GPU tier:** the two hot/cold variants evicted exactly the same number of tokens, so the
  only difference between them is how long retrieved tokens stay in attention.
- **CPU tier size:** at most about 16k evicted tokens per layer, well under the 32,768-token ring
  buffer, so the ring never filled up.
