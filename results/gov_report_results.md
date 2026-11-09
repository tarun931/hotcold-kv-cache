# LongBench gov_report: Llama-3.1-8B, 8k context

| Budget | Method | PPL | Δ vs full | Latency (s) | Peak mem (GiB) |
|---|---|---|---|---|---|
| full | full | 5.247 | 0.0% | 1.718 | 18.99 |
| 512 | snapkv | **5.942** | +13.2% | 1.562 | 17.15 |
| 512 | pyramidkv | 6.352 | +21.1% | 1.560 | 17.16 |
| 512 | h2o | 6.430 | +22.5% | 1.751 | 17.33 |
| 512 | hotcold (no retrieval) | 6.139 | +17.0% | 1.601 | 17.18 |
| 512 | hotcold (attend once) | 6.126 | +16.8% | 2.023 | 17.18 |
| 512 | hotcold (held) | 6.039 | +15.1% | 2.249 | 17.19 |
| 1024 | snapkv | **5.688** | +8.4% | 1.593 | 17.28 |
| 1024 | pyramidkv | 5.768 | +9.9% | 1.591 | 17.29 |
| 1024 | h2o | 6.061 | +15.5% | 1.797 | 17.46 |
| 1024 | hotcold (no retrieval) | 5.818 | +10.9% | 1.645 | 17.37 |
| 1024 | hotcold (attend once) | 5.810 | +10.7% | 2.044 | 17.37 |
| 1024 | hotcold (held) | 5.756 | +9.7% | 2.239 | 17.38 |
| 2048 | snapkv | **5.456** | +4.0% | 1.659 | 17.53 |
| 2048 | pyramidkv | 5.519 | +5.2% | 1.658 | 17.55 |
| 2048 | h2o | 5.756 | +9.7% | 1.931 | 17.71 |
| 2048 | hotcold (no retrieval) | 5.578 | +6.3% | 1.726 | 17.65 |
| 2048 | hotcold (attend once) | 5.573 | +6.2% | 2.135 | 17.65 |
| 2048 | hotcold (held) | 5.544 | +5.6% | 2.298 | 17.66 |

Bold marks the lowest perplexity at each budget.

This is the only dataset with all three hot/cold variants, including *no retrieval*
(`LOOKAHEAD_K=0`), which still evicts to the CPU tier but never searches or brings anything back.
It is the control that isolates what retrieval contributes; see
[the retrieval ablation](retrieval_ablation.md). The same dataset at a longer context is in
[gov_report at 16k](gov_report_16k_results.md).

## Setup

The setup is the same as for [multifieldqa_en](multifieldqa_en_results.md), including the hot/cold
variants, with these differences:

- **Samples:** all 200.
- **Prompt length:** about 7.0k tokens on average.

## Runs

- **Slurm jobs:**
  - `full`: 61959181
  - Baselines and hot/cold, attend once: array 61959182, tasks 0–11. Hot/cold ran with the prefill-query memory fix.
  - Hot/cold, no retrieval: 62127664, tasks 3, 7 and 11
  - Hot/cold, held: 62282888 task 3; 62374590 tasks 7 and 11
- **Retrieval:** at budgets 512 and 1024 it fired on all 102,400 retrieval windows (200 samples × 32 layers × 16 windows). At 2048 it fired on 101,888, because a few short samples have nothing in the CPU tier.
- **Injection:** attend once injected 96,004, 96,006 and 95,524 times at budgets 512, 1024 and 2048; held injected 1,530,659, 1,531,053 and 1,523,586 times, about 16× more, since each retrieved set is used for a whole window. The no-retrieval runs injected nothing.
- **Same GPU tier:** all three hot/cold variants evicted exactly the same number of tokens (41,330,560, 38,047,360 and 33,124,704 at budgets 512, 1024 and 2048), so the only difference between them is what happens to retrieved tokens.
