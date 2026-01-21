# Progress Log

A running log of project progress that is easy for humans and agents to parse.

## Entries

### YYYY-MM-DD
- Summary:
- Changes:
- Blockers:
- Next:

### 2026-01-14
- Summary: Re-ran eval.py on baseline and signature variants for in-sample and OOS ranges using latest checkpoints.
- Changes: Updated eval outputs for baseline, signature-2, signature-3, and signature-4 runs; added OOS outputs under eval_oos_2021_2024; clarified that these results use sell_mode=all with no max_positions cap.
- Blockers: None.
- Next: Compare OOS vs in-sample metrics and decide follow-up experiments.

### 2026-01-17
- Summary: Trained sell-one variants, added eval position/action stats, and ran downtrend evaluations.
- Changes: New runs baseline_sell_one_e8a931, signature2_sell_one_83f00e, signature3_sell_one_9ba2a8 with sell_mode=one and max_positions=5; eval summaries now include position_stats; added eval.py trading_period override; ran 2018 (in-sample) and 2022 (OOS) downtrend evals with trading_period=200; trained rolling_vol and mean+vol signature runs with four evals each; trained sell-all signature runs for vol/mean/mean+vol embeddings with four evals each; trained sell_all_cap runs (max_positions=5) for baseline and signature 2/3 across 4 embeddings with four evals each.
- Blockers: None.
- Next: Review downtrend results and decide whether to retrain on bearish windows.

### 2026-01-18
- Summary: Added sell_one_plus mode and evaluated baseline + signature depth 2/3 across four embeddings.
- Changes: Added sell_mode=one_plus to the trading environment; eval action naming now distinguishes sell_one vs sell_all for 4-action agents; trained and evaluated one_plus runs (max_positions=5) across in-sample, OOS, and bear windows; logged results and action rates in experiments.md.
- Blockers: None.
- Next: Investigate why sell_one usage is near zero and consider reward/action-balance adjustments.

### 2026-01-20
- Summary: Ran OOS evals for existing sell-one checkpoints to compare with sell-one+.
- Changes: Added OOS 2021-2024 eval summaries for baseline_sell_one and signature sell_one runs; logged metrics in experiments.md.
- Blockers: None.
- Next: Compare sell-one vs sell-one+ in bear windows and decide if reward shaping is needed to increase sell usage.
