# LongBench hotpotqa: Llama-3.1-8B, 8k context

| Budget | Method | PPL | Δ vs full | Latency (s) | Peak mem (GiB) |
|---|---|---|---|---|---|
| full | full | 6.181 | 0.0% | 1.743 | 18.99 |
| 512 | snapkv | **6.500** | +5.1% | 1.565 | 17.15 |
| 512 | pyramidkv | 6.924 | +12.0% | 1.560 | 17.16 |
| 512 | h2o | 7.318 | +18.4% | 1.748 | 17.33 |
| 512 | hotcold (attend once) | 6.613 | +7.0% | 2.007 | 17.18 |
| 512 | hotcold (held) | 6.576 | +6.4% | 2.261 | 17.19 |
| 1024 | snapkv | **6.374** | +3.1% | 1.596 | 17.28 |
| 1024 | pyramidkv | 6.411 | +3.7% | 1.591 | 17.29 |
| 1024 | h2o | 6.983 | +13.0% | 1.794 | 17.46 |
| 1024 | hotcold (attend once) | 6.415 | +3.8% | 2.045 | 17.37 |
| 1024 | hotcold (held) | 6.397 | +3.5% | 2.322 | 17.38 |
| 2048 | snapkv | **6.278** | +1.6% | 1.657 | 17.53 |
| 2048 | pyramidkv | 6.302 | +1.9% | 1.658 | 17.55 |
| 2048 | h2o | 6.665 | +7.8% | 1.930 | 17.71 |
| 2048 | hotcold (attend once) | 6.307 | +2.0% | 2.107 | 17.65 |
| 2048 | hotcold (held) | 6.295 | +1.8% | 2.318 | 17.65 |

Bold marks the lowest perplexity at each budget.

## Setup

The setup is the same as for [multifieldqa_en](multifieldqa_en_results.md), including the two
hot/cold variants, with these differences:

- **Samples:** all 200.
- **Prompt length:** about 7.8k tokens on average, so most prompts reach the 8192-token cap.

## Runs

- **Slurm jobs:**
  - `full`: 61934666
  - Baselines and hot/cold, attend once: array 61934667, tasks 0–11. Hot/cold ran with the prefill-query memory fix.
  - Hot/cold, held: 62374609, tasks 3, 7 and 11
- **Retrieval:** at budgets 512 and 1024 it fired on all 102,400 retrieval windows (200 samples × 32 layers × 16 windows). At 2048 it fired on 101,696, because a few short samples have nothing in the CPU tier.
- **Injection:** attend once injected 96,003, 96,004 and 95,328 times at budgets 512, 1024 and 2048; held injected 1,530,593, 1,530,822 and 1,520,472 times, about 16× more, since each retrieved set is used for a whole window.
- **Same GPU tier:** the two hot/cold variants evicted exactly the same number of tokens, so the only difference between them is how long retrieved tokens stay in attention.
