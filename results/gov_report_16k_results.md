# LongBench gov_report: Llama-3.1-8B, 16k context

| Budget | Method | PPL | Δ vs full | Latency (s) | Peak mem (GiB) |
|---|---|---|---|---|---|
| full | full | 5.393 | 0.0% | 1.784 | 23.01 |
| 512 | snapkv | **6.180** | +14.6% | 1.562 | 19.17 |
| 512 | pyramidkv | 6.679 | +23.8% | 1.560 | 19.18 |
| 512 | h2o | 6.706 | +24.4% | 1.747 | 19.57 |
| 512 | hotcold (attend once) | 6.367 | +18.1% | 2.052 | 19.20 |
| 1024 | snapkv | **5.911** | +9.6% | 1.596 | 19.30 |
| 1024 | pyramidkv | 5.982 | +10.9% | 1.593 | 19.31 |
| 1024 | h2o | 6.349 | +17.7% | 1.795 | 19.70 |
| 1024 | hotcold (attend once) | 6.058 | +12.3% | 2.037 | 19.38 |
| 2048 | snapkv | **5.681** | +5.3% | 1.658 | 19.55 |
| 2048 | pyramidkv | 5.738 | +6.4% | 1.657 | 19.57 |
| 2048 | h2o | 6.053 | +12.2% | 1.933 | 19.95 |
| 2048 | hotcold (attend once) | 5.814 | +7.8% | 2.125 | 19.66 |

Bold marks the lowest perplexity at each budget.

Hot/cold here runs the *attend once* variant only. The *held* variant, which lowers perplexity on
every 8k dataset (see [the retrieval ablation](retrieval_ablation.md)), has not been run at 16k
yet, so these hot/cold rows understate it.

## Setup

The setup is the same as for [gov_report at 8k](gov_report_results.md), with these differences:

- **Context:** prompts are cut at 16,384 tokens instead of 8,192.
- **Prompt length:** about 9.4k tokens on average. 35 of the 200 samples are long enough to reach the 16k cap; the rest are shorter.

## Runs

- **Slurm jobs:**
  - `full`: 62127873
  - Baselines and hot/cold: array 62127874, tasks 0–11.
- **Retrieval:** at budgets 512 and 1024 it fired on all 102,400 retrieval windows (200 samples × 32 layers × 16 windows). At 2048 it fired on 101,888, because a few short samples have nothing in the CPU tier.
- **Injection:** retrieved tokens were injected 96,004, 96,005 and 95,521 times at budgets 512, 1024 and 2048.
- **CPU tier size:** at most about 16k evicted tokens per layer, well under the 32,768-token ring buffer, so the ring never filled up.
