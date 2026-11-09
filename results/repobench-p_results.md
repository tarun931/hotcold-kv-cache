# LongBench repobench-p: Llama-3.1-8B, 8k context

| Budget | Method | PPL | Δ vs full | Latency (s) | Peak mem (GiB) |
|---|---|---|---|---|---|
| full | full | 2.030 | 0.0% | 1.718 | 18.99 |
| 512 | snapkv | **2.533** | +24.7% | 1.560 | 17.16 |
| 512 | pyramidkv | 2.726 | +34.2% | 1.559 | 17.16 |
| 512 | h2o | 2.998 | +47.6% | 1.747 | 17.33 |
| 512 | hotcold (attend once) | 2.693 | +32.6% | 2.002 | 17.18 |
| 512 | hotcold (held) | 2.608 | +28.4% | 2.257 | 17.19 |
| 1024 | snapkv | **2.276** | +12.1% | 1.593 | 17.28 |
| 1024 | pyramidkv | 2.348 | +15.6% | 1.592 | 17.29 |
| 1024 | h2o | 2.648 | +30.4% | 1.792 | 17.46 |
| 1024 | hotcold (attend once) | 2.366 | +16.5% | 2.058 | 17.37 |
| 1024 | hotcold (held) | 2.314 | +14.0% | 2.316 | 17.38 |
| 2048 | snapkv | **2.125** | +4.6% | 1.659 | 17.53 |
| 2048 | pyramidkv | 2.165 | +6.6% | 1.658 | 17.55 |
| 2048 | h2o | 2.374 | +16.9% | 1.931 | 17.71 |
| 2048 | hotcold (attend once) | 2.162 | +6.5% | 2.106 | 17.65 |
| 2048 | hotcold (held) | 2.141 | +5.4% | 2.307 | 17.66 |

Bold marks the lowest perplexity at each budget.

## Setup

The setup is the same as for [multifieldqa_en](multifieldqa_en_results.md), including the two
hot/cold variants, with these differences:

- **Samples:** all 500.
- **Prompt length:** about 7.0k tokens on average. The prompts are code, so perplexity is much lower than on the text datasets.

## Runs

- **Slurm jobs:**
  - `full`: 62017061
  - Baselines and hot/cold, attend once: array 62017063, tasks 0–11. Hot/cold ran with the prefill-query memory fix.
  - Hot/cold, held: 62374622, tasks 3, 7 and 11
- **Retrieval:** it fired on all 256,000 retrieval windows (500 samples × 32 layers × 16 windows) at every budget. Every sample is long enough to send tokens to the CPU tier, even at budget 2048.
- **Injection:** attend once injected 240,012, 240,003 and 240,010 times at budgets 512, 1024 and 2048; held injected 3,826,635, 3,827,226 and 3,828,088 times, about 16× more, since each retrieved set is used for a whole window.
- **Same GPU tier:** the two hot/cold variants evicted exactly the same number of tokens, so the only difference between them is how long retrieved tokens stay in attention.
