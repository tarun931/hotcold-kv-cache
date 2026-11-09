# Retrieval ablation: what the CPU tier contributes

Three hot/cold variants, identical except for what happens to tokens retrieved from the CPU tier.
The GPU hot tier, the eviction policy, the FAISS search and the 16-step retrieval schedule are the
same in all three, and the evicted-token counts match exactly, so nothing else can explain the
differences.

| Variant | What it does | Flag |
|---|---|---|
| **no retrieval** | Tokens are still evicted to the CPU tier, but nothing is ever searched or brought back. | `LOOKAHEAD_K=0` |
| **attend once** | Every 16 decode steps, 64 tokens are fetched and attended at the single step their copy lands, then dropped. | default |
| **held** | The same 64 tokens stay in attention until the next retrieval replaces them, so they are used for the whole 16-step window. | `HOLD_RETRIEVED=1` |

## 1. Does retrieval help at all? (gov_report, 8k)

| Budget | No retrieval | Attend once | Gain | Latency, no retrieval | Latency, attend once |
|---|---|---|---|---|---|
| 512 | 6.139 | 6.126 | -0.21% | 1.601 s | 2.023 s |
| 1024 | 5.818 | 5.810 | -0.13% | 1.645 s | 2.044 s |
| 2048 | 5.578 | 5.573 | -0.09% | 1.726 s | 2.135 s |

Barely. Attend-once retrieval buys 0.09-0.21% perplexity for about 25% more latency.

Two things follow. First, hot/cold's advantage over PyramidKV and H2O came from its GPU tier, not
from the CPU tier: with retrieval switched off it still beats both at budget 512. Second, the whole
latency overhead is the retrieval path. With retrieval off, hot/cold (1.601 s) is *faster* than the
uncompressed `full` run (1.718 s) and close to SnapKV (1.562 s), so neither the Triton decode
kernel nor eviction is responsible.

The cause is that a retrieved set was used for 1 decode step out of 16 and then discarded. The
injection counters confirm it: injections equalled completed copies exactly.

## 2. Does holding the tokens longer help? (all four datasets, 8k)

| Dataset | Budget | Attend once | Held | Change | SnapKV | PyramidKV | H2O |
|---|---|---|---|---|---|---|---|
| multifieldqa_en | 512 | 7.538 | 7.360 | -2.36% | **7.227** | 7.839 | 7.659 |
| multifieldqa_en | 1024 | 7.009 | 6.917 | -1.32% | **6.795** | 6.987 | 7.161 |
| multifieldqa_en | 2048 | 6.652 | 6.603 | -0.75% | **6.488** | 6.637 | 6.750 |
| hotpotqa | 512 | 6.613 | 6.576 | -0.56% | **6.500** | 6.924 | 7.318 |
| hotpotqa | 1024 | 6.415 | 6.397 | -0.27% | **6.374** | 6.411 | 6.983 |
| hotpotqa | 2048 | 6.307 | 6.295 | -0.19% | **6.278** | 6.302 | 6.665 |
| gov_report | 512 | 6.126 | 6.039 | -1.42% | **5.942** | 6.352 | 6.430 |
| gov_report | 1024 | 5.810 | 5.756 | -0.92% | **5.688** | 5.768 | 6.061 |
| gov_report | 2048 | 5.573 | 5.544 | -0.53% | **5.456** | 5.519 | 5.756 |
| repobench-p | 512 | 2.693 | 2.608 | -3.18% | **2.533** | 2.726 | 2.998 |
| repobench-p | 1024 | 2.366 | 2.314 | -2.18% | **2.276** | 2.348 | 2.648 |
| repobench-p | 2048 | 2.162 | 2.141 | -1.00% | **2.125** | 2.165 | 2.374 |

A negative change means lower (better) perplexity. Bold marks the lowest perplexity among the
compressed methods at that budget.

Yes, in all 12 configurations. The gain is largest at the smallest budget, where the GPU tier
evicts the most and the retrieved tokens carry the most context that is otherwise lost: -0.56% to
-3.18% at budget 512, falling to -0.19% to -1.00% at 2048.

On gov_report at budget 512 the three variants line up as 6.139 → 6.126 → 6.039, so holding is
worth about seven times what attend-once retrieval was worth.

Against the baselines, held hot/cold beats PyramidKV in 11 of 12 configurations, against 5 of 12
for attend-once. The single loss is gov_report at budget 2048 (5.544 vs 5.519). SnapKV is still
best everywhere, but the gap narrows from 1.5-4.5% to 0.28-2.95%.

**The finding:** how long a retrieved token stays in attention matters more than the retrieval
itself. Fetching the right tokens is not enough if they are dropped a step later.

## Cost

Held latency is about 2.24-2.32 s, roughly 10% over attend-once and 30% over the uncompressed
`full` run. Peak memory is unchanged, at about 17.19 / 17.38 / 17.66 GiB for budgets 512 / 1024 /
2048: the 64 held tokens are negligible beside the hot tier.

## Setup

Identical to the per-dataset runs ([multifieldqa_en](multifieldqa_en_results.md),
[hotpotqa](hotpotqa_results.md), [gov_report](gov_report_results.md),
[repobench-p](repobench-p_results.md)). Each variant is labelled in its own result files
(`hotcold`, `hotcold_k0`, `hotcold_hold`), so the rows never mix.

## Runs

- **No retrieval:** array 62127664, tasks 3, 7 and 11 (gov_report only).
- **Held:** gov_report budget 512: 62282888 task 3; gov_report budgets 1024 and 2048: 62374590
  tasks 7 and 11; multifieldqa_en: 62374608; hotpotqa: 62374609; repobench-p: 62374622.
- **Injections:** held injects about 16× more often than attend once, as expected when a retrieved
  set is used for a whole window instead of one step. For example gov_report at budget 512:
  96,004 → 1,530,659. The no-retrieval runs inject nothing.
