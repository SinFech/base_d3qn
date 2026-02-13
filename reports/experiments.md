# Experiment Results

Record experiment outcomes in a consistent, readable format.

## Baseline setup

This section summarizes the shared configuration used across recent experiments.

- Data: data/Bitcoin History 2010-2024.csv (sorted by date).
- In-sample window: 2014-01-01 to 2020-12-31.
- OOS window: 2021-01-01 to 2024-02-09.
- Downtrend windows: 2018 (in-sample) and 2022 (OOS). For 1-year windows, trading_period=200.
- Reward: sr (Sharpe-like) unless stated otherwise.
- Eval: 50 episodes, epsilon=0.0, fixed_windows=true, seed=20240101.
- Action space: 0=hold, 1=buy, 2=sell.
- Sell modes:
  - sell_all: sell_mode=all, max_positions=None (unlimited stacking).
  - sell_one: sell_mode=one, max_positions=5.
  - sell_all_cap: sell_mode=all_cap, max_positions=<int> (sell-all with cap).
  - sell_one_plus: sell_mode=one_plus, max_positions=5, action_number=4 (sell_one + sell_all actions).
- Signature observations (logsignature):
  - window_size=24, time_aug=true, lead_lag=false.
  - Depth: degree=2 or degree=3.
  - Embedding components (dict form): log_price, log_return, rolling_mean (window=5), rolling_vol (window=5).
- Model: D3QN with mlp_dueling and hidden_sizes [256, 256].
- Training: 50 episodes, batch_size=40, gamma=0.99, learning_rate=0.00025, eps_steps=200.

## Entries

### YYYY-MM-DD - Short Title
- Goal:
- Hypothesis:
- Config:
- Data:
- Metrics:
- Outcome:
- Notes:
- Artifacts:

### 2026-01-14 - Checkpoint eval refresh
- Goal: Re-run eval.py for baseline and signature variants using latest checkpoints in runs/.
- Hypothesis: N/A (refreshing metrics).
- Config: Used config embedded in each checkpoint; checkpoints: baseline_22575b, signature_e80e94 (signature-2), signature-3_485db8, signature-4_adffa6.
- Data: Per run config in checkpoint (default eval range).
- Metrics:
  - baseline: mean 1423.42, std 1237.34, median 1239.00, min -400.50, max 3971.50
  - signature-2: mean 3853.29, std 1793.90, median 3685.50, min 1062.00, max 7200.00
  - signature-3: mean 2242.12, std 3310.56, median 2742.75, min -3825.50, max 8279.50
  - signature-4: mean 2582.69, std 1533.73, median 2339.75, min 295.00, max 5841.50
- Outcome: Signature-2 has the highest mean return among these runs.
- Notes: Signature-2 leads on mean/median; baseline is lowest with a negative min; signature-3 shows the highest variance and worst min; signature-4 is mid-range with positive min. Eval outputs saved in each run directory. These results use the legacy environment behavior (sell_mode=all, no max_positions cap).
- Artifacts: runs/baseline_22575b/eval_summary.json, runs/signature_e80e94/eval_summary.json, runs/signature-3_485db8/eval_summary.json, runs/signature-4_adffa6/eval_summary.json.

### 2026-01-14 - OOS eval 2021-2024
- Goal: Evaluate baseline and signature variants on an out-of-sample window.
- Hypothesis: N/A (OOS check).
- Config: Used config embedded in each checkpoint; override data range and output directory.
- Data: 2021-01-01 to 2024-02-09 from Bitcoin History 2010-2024.csv.
- Metrics:
  - baseline: mean 727.57, std 338.43, median 662.00, min 204.00, max 1566.00
  - signature-2: mean 2090.21, std 480.29, median 2174.75, min 1397.50, max 3178.00
  - signature-3: mean -1232.17, std 2757.34, median -1835.00, min -4970.00, max 4878.00
  - signature-4: mean 927.13, std 537.63, median 799.50, min 203.00, max 2168.00
- Outcome: Signature-2 leads; signature-3 underperforms with a negative mean.
- Notes: OOS outputs saved under eval_oos_2021_2024 for each run. These results use the legacy environment behavior (sell_mode=all, no max_positions cap).
- Artifacts: runs/baseline_22575b/eval_oos_2021_2024/eval_summary.json, runs/signature_e80e94/eval_oos_2021_2024/eval_summary.json, runs/signature-3_485db8/eval_oos_2021_2024/eval_summary.json, runs/signature-4_adffa6/eval_oos_2021_2024/eval_summary.json.

### 2026-01-17 - Sell-one in-sample eval
- Goal: Evaluate sell_mode=one with max_positions=5 for baseline and signature variants.
- Hypothesis: Partial selling and position caps change action/position dynamics vs sell-all behavior.
- Config: baseline.yaml, signature.yaml (signature-2), signature-3.yaml with env.sell_mode=one and env.max_positions=5.
- Data: 2014-01-01 to 2020-12-31 (in-sample).
- Metrics:
  - baseline: mean 3120.38, std 2267.21, median 3215.75, min -532.50, max 7298.50
  - signature-2: mean 1996.60, std 4329.55, median 2484.50, min -6160.50, max 8008.50
  - signature-3: mean 574.14, std 4483.18, median -417.00, min -7057.50, max 8353.00
- Outcome: Baseline outperforms signature variants under sell-one in-sample metrics.
- Notes: Sell-one runs trained with max_positions=5. Eval summaries include action_counts and position_stats.
- Artifacts: runs/baseline_sell_one_e8a931/eval_summary.json, runs/signature2_sell_one_83f00e/eval_summary.json, runs/signature3_sell_one_9ba2a8/eval_summary.json.

### 2026-01-17 - Position stats (sell-all vs sell-one)
- Goal: Compare position dynamics between sell-all and sell-one setups.
- Hypothesis: sell-one caps lead to lower position variance and fewer large swings.
- Config: sell-all uses legacy env (sell_mode=all, no max_positions cap); sell-one uses max_positions=5 and sell_mode=one.
- Data: 2014-01-01 to 2020-12-31 (in-sample).
- Metrics:
  - sell-all baseline: mean 7.10, median 2, max 92, zero_rate 0.378, change_count 17210, turnover 30406
  - sell-all signature-2: mean 26.83, median 17, max 159, zero_rate 0.124, change_count 21148, turnover 38973
  - sell-all signature-3: mean 82.05, median 61, max 325, zero_rate 0.028, change_count 20611, turnover 36711
  - sell-one baseline: mean 4.58, median 5, max 5, zero_rate 0.033, change_count 2887, turnover 2887
  - sell-one signature-2: mean 4.93, median 5, max 5, zero_rate 0.002, change_count 686, turnover 686
  - sell-one signature-3: mean 4.89, median 5, max 5, zero_rate 0.008, change_count 852, turnover 852
- Outcome: sell-one caps drive near-max steady holdings and lower position volatility; sell-all allows large stacking with higher variance.
- Notes: Position stats aggregated from eval_summary.json position_stats fields.
- Artifacts: runs/baseline_22575b/eval_summary.json, runs/signature_e80e94/eval_summary.json, runs/signature-3_485db8/eval_summary.json, runs/baseline_sell_one_e8a931/eval_summary.json, runs/signature2_sell_one_83f00e/eval_summary.json, runs/signature3_sell_one_9ba2a8/eval_summary.json.

### 2026-01-17 - Downtrend eval 2018 (in-sample)
- Goal: Measure performance in a bearish in-sample window with sell-all vs sell-one.
- Hypothesis: Downtrend windows reduce profitability and highlight differences in sell modes.
- Config: sell-all uses legacy env (sell_mode=all, no max_positions cap); sell-one uses max_positions=5 and sell_mode=one. trading_period override: 200.
- Data: 2018-01-01 to 2018-12-31 (in-sample).
- Metrics:
  - sell-all baseline: mean 226.77, std 33.83, median 231.50, min 152.50, max 299.50
  - sell-all signature-2: mean 760.19, std 237.78, median 870.00, min 371.50, max 1090.50
  - sell-all signature-3: mean -1108.93, std 418.03, median -1013.25, min -1902.00, max -547.50
  - sell-one baseline: mean 443.91, std 74.29, median 426.75, min 311.00, max 624.50
  - sell-one signature-2: mean -981.86, std 514.26, median -928.75, min -2293.50, max 87.00
  - sell-one signature-3: mean -1207.12, std 529.16, median -1221.00, min -2426.50, max 131.00
- Outcome: Signature-2 leads under sell-all; sell-one settings hurt signature variants most in this window.
- Notes: One-year window required trading_period=200 to avoid insufficient data length.
- Artifacts: runs/baseline_22575b/eval_bear_2018_tp200/eval_summary.json, runs/signature_e80e94/eval_bear_2018_tp200/eval_summary.json, runs/signature-3_485db8/eval_bear_2018_tp200/eval_summary.json, runs/baseline_sell_one_e8a931/eval_bear_2018_tp200/eval_summary.json, runs/signature2_sell_one_83f00e/eval_bear_2018_tp200/eval_summary.json, runs/signature3_sell_one_9ba2a8/eval_bear_2018_tp200/eval_summary.json.

### 2026-01-17 - Downtrend eval 2022 (OOS)
- Goal: Measure performance in a bearish out-of-sample window with sell-all vs sell-one.
- Hypothesis: OOS downtrend reduces returns, especially for long-biased strategies.
- Config: sell-all uses legacy env (sell_mode=all, no max_positions cap); sell-one uses max_positions=5 and sell_mode=one. trading_period override: 200.
- Data: 2022-01-01 to 2022-12-31 (OOS).
- Metrics:
  - sell-all baseline: mean 42.46, std 150.84, median 17.75, min -266.00, max 276.00
  - sell-all signature-2: mean 383.58, std 130.67, median 432.75, min 87.50, max 564.50
  - sell-all signature-3: mean -2114.71, std 479.30, median -2087.50, min -2897.00, max -800.00
  - sell-one baseline: mean 287.24, std 274.43, median 305.50, min -261.00, max 914.00
  - sell-one signature-2: mean -1590.41, std 691.44, median -1621.00, min -2528.00, max 88.00
  - sell-one signature-3: mean -2008.18, std 783.13, median -2186.75, min -2897.00, max -150.00
- Outcome: Signature-2 remains positive under sell-all, while sell-one variants are negative across signatures in this OOS window.
- Notes: One-year window required trading_period=200 to avoid insufficient data length.
- Artifacts: runs/baseline_22575b/eval_bear_2022_tp200/eval_summary.json, runs/signature_e80e94/eval_bear_2022_tp200/eval_summary.json, runs/signature-3_485db8/eval_bear_2022_tp200/eval_summary.json, runs/baseline_sell_one_e8a931/eval_bear_2022_tp200/eval_summary.json, runs/signature2_sell_one_83f00e/eval_bear_2022_tp200/eval_summary.json, runs/signature3_sell_one_9ba2a8/eval_bear_2022_tp200/eval_summary.json.

### 2026-01-17 - Rolling vol embedding eval
- Goal: Evaluate rolling_vol embedding for signature depth 2/3 across in-sample, OOS, and downtrend windows.
- Hypothesis: Rolling volatility may improve robustness in bearish windows.
- Config: embedding {log_price, log_return, rolling_vol(window=5)}; depth 2 in signature.yaml and depth 3 in signature-3.yaml; sell_mode=one, max_positions=5; trading_period=200 for bear windows.
- Data: In-sample 2014-01-01 to 2020-12-31; OOS 2021-01-01 to 2024-02-09; bear windows 2018 and 2022.
- Metrics:
  - signature2_vol in-sample: mean 1932.22, std 3209.13, median 2327.00, min -3698.50, max 7222.00
  - signature2_vol OOS: mean 970.56, std 1512.93, median 1062.75, min -1963.00, max 4833.50
  - signature2_vol bear-2018: mean -985.59, std 520.02, median -966.50, min -2273.00, max 310.00
  - signature2_vol bear-2022: mean -7.62, std 346.66, median -42.50, min -539.00, max 708.50
  - signature3_vol in-sample: mean 1194.05, std 3649.17, median 1358.25, min -6136.50, max 7774.00
  - signature3_vol OOS: mean 1528.38, std 1748.87, median 1684.75, min -2101.00, max 5054.00
  - signature3_vol bear-2018: mean -1199.74, std 593.75, median -1167.75, min -2300.50, max 308.00
  - signature3_vol bear-2022: mean 4.07, std 384.18, median -122.75, min -498.00, max 959.00
- Outcome: Rolling_vol did not improve bearish performance; results remain mixed with negative means in 2018 and near-zero in 2022.
- Notes: Runs use new dict-only embedding format.
- Artifacts: runs/signature2_vol_8bf645/eval_summary.json, runs/signature2_vol_8bf645/eval_oos_2021_2024/eval_summary.json, runs/signature2_vol_8bf645/eval_bear_2018_tp200/eval_summary.json, runs/signature2_vol_8bf645/eval_bear_2022_tp200/eval_summary.json, runs/signature3_vol_290495/eval_summary.json, runs/signature3_vol_290495/eval_oos_2021_2024/eval_summary.json, runs/signature3_vol_290495/eval_bear_2018_tp200/eval_summary.json, runs/signature3_vol_290495/eval_bear_2022_tp200/eval_summary.json.

### 2026-01-17 - Mean+vol embedding eval
- Goal: Evaluate price+return+rolling_mean+rolling_vol embedding for signature depth 2/3.
- Hypothesis: Combining trend and volatility features might improve OOS stability.
- Config: embedding {log_price, log_return, rolling_mean(window=5), rolling_vol(window=5)} via overrides; sell_mode=one, max_positions=5; trading_period=200 for bear windows.
- Data: In-sample 2014-01-01 to 2020-12-31; OOS 2021-01-01 to 2024-02-09; bear windows 2018 and 2022.
- Metrics:
  - signature2_meanvol in-sample: mean 2442.21, std 5190.22, median 5266.00, min -6453.00, max 8284.50
  - signature2_meanvol OOS: mean -2681.85, std 3600.48, median -3951.75, min -7384.00, max 4901.00
  - signature2_meanvol bear-2018: mean -1194.89, std 525.49, median -1168.75, min -2314.00, max -270.00
  - signature2_meanvol bear-2022: mean -2006.84, std 723.72, median -2203.50, min -2920.50, max -436.00
  - signature3_meanvol in-sample: mean 1620.35, std 1665.03, median 1311.25, min -715.50, max 5509.50
  - signature3_meanvol OOS: mean 1300.17, std 1504.54, median 593.25, min -817.50, max 4853.50
  - signature3_meanvol bear-2018: mean -186.33, std 174.03, median -131.75, min -546.00, max 214.50
  - signature3_meanvol bear-2022: mean -82.34, std 200.90, median -137.75, min -410.00, max 337.50
- Outcome: Mean+vol performs poorly in OOS for depth-2; depth-3 remains modestly negative in bearish windows.
- Notes: Embedding passed via train overrides.
- Artifacts: runs/signature2_meanvol_16f938/eval_summary.json, runs/signature2_meanvol_16f938/eval_oos_2021_2024/eval_summary.json, runs/signature2_meanvol_16f938/eval_bear_2018_tp200/eval_summary.json, runs/signature2_meanvol_16f938/eval_bear_2022_tp200/eval_summary.json, runs/signature3_meanvol_c5bcc3/eval_summary.json, runs/signature3_meanvol_c5bcc3/eval_oos_2021_2024/eval_summary.json, runs/signature3_meanvol_c5bcc3/eval_bear_2018_tp200/eval_summary.json, runs/signature3_meanvol_c5bcc3/eval_bear_2022_tp200/eval_summary.json.

### 2026-01-17 - Bear windows (sell-all) for rolling vol + mean/vol
- Goal: Compare sell-all behavior on bearish windows for rolling_vol and mean+vol embeddings.
- Hypothesis: Sell-all may improve downside performance by enabling full exits.
- Config: sell_mode=all with no max_positions cap; trading_period=200; embeddings from rolling_vol and mean+vol runs.
- Data: 2018-01-01 to 2018-12-31 and 2022-01-01 to 2022-12-31.
- Metrics:
  - signature2_vol 2018: mean -1209.79, std 290.05, median -1238.50, min -1930.50, max -767.50
  - signature2_vol 2022: mean 58.08, std 482.28, median -304.25, min -434.00, max 823.50
  - signature3_vol 2018: mean -1321.94, std 285.50, median -1246.25, min -2067.50, max -882.00
  - signature3_vol 2022: mean -280.93, std 428.06, median -450.75, min -892.50, max 449.00
  - signature2_meanvol 2018: mean -1351.99, std 254.11, median -1418.75, min -1971.50, max -915.00
  - signature2_meanvol 2022: mean -2048.74, std 519.89, median -2211.00, min -2824.50, max -1111.00
  - signature3_meanvol 2018: mean 357.97, std 162.74, median 458.00, min 108.00, max 541.00
  - signature3_meanvol 2022: mean 312.56, std 118.94, median 346.00, min 45.00, max 527.50
- Outcome: Sell-all improves mean+vol depth-3 in bearish windows but does not rescue rolling_vol.
- Notes: Evaluations run via eval.py with sell_mode override.
- Artifacts: runs/signature2_vol_8bf645/eval_bear_2018_tp200_sell_all/eval_summary.json, runs/signature2_vol_8bf645/eval_bear_2022_tp200_sell_all/eval_summary.json, runs/signature3_vol_290495/eval_bear_2018_tp200_sell_all/eval_summary.json, runs/signature3_vol_290495/eval_bear_2022_tp200_sell_all/eval_summary.json, runs/signature2_meanvol_16f938/eval_bear_2018_tp200_sell_all/eval_summary.json, runs/signature2_meanvol_16f938/eval_bear_2022_tp200_sell_all/eval_summary.json, runs/signature3_meanvol_c5bcc3/eval_bear_2018_tp200_sell_all/eval_summary.json, runs/signature3_meanvol_c5bcc3/eval_bear_2022_tp200_sell_all/eval_summary.json.

### 2026-01-17 - Sell-all embedding sweep (vol/mean/mean+vol)
- Goal: Train sell-all signature depth 2/3 with rolling_vol, rolling_mean, and mean+vol embeddings and evaluate across four windows.
- Hypothesis: Sell-all training may improve OOS/downtrend robustness for these embeddings.
- Config: env.sell_mode=all, env.max_positions=null; embeddings: rolling_vol, rolling_mean, rolling_mean+rolling_vol; trading_period=200 for bear windows.
- Data: In-sample 2014-01-01 to 2020-12-31; OOS 2021-01-01 to 2024-02-09; bear windows 2018 and 2022.
- Metrics:
  - sig2_vol_sell_all: in-sample 3933.23, OOS 1609.31, bear-2018 645.53, bear-2022 282.33
  - sig3_vol_sell_all: in-sample 3526.02, OOS 628.56, bear-2018 -635.35, bear-2022 -530.72
  - sig2_mean_sell_all: in-sample 3553.20, OOS 1906.46, bear-2018 355.93, bear-2022 566.63
  - sig3_mean_sell_all: in-sample 2923.53, OOS 1652.92, bear-2018 539.35, bear-2022 324.11
  - sig2_meanvol_sell_all: in-sample 3035.27, OOS -1627.66, bear-2018 -136.43, bear-2022 -1372.32
  - sig3_meanvol_sell_all: in-sample 3188.17, OOS 1886.64, bear-2018 662.86, bear-2022 345.64
- Outcome: rolling_mean and rolling_vol sell-all runs are mostly positive in bear windows (except sig3_vol); mean+vol depth-2 collapses in OOS and bear.
- Notes: Metrics are mean returns; see artifacts for full stats.
- Artifacts: runs/sig2_vol_sell_all_426adf/eval_summary.json, runs/sig2_vol_sell_all_426adf/eval_oos_2021_2024/eval_summary.json, runs/sig2_vol_sell_all_426adf/eval_bear_2018_tp200/eval_summary.json, runs/sig2_vol_sell_all_426adf/eval_bear_2022_tp200/eval_summary.json, runs/sig3_vol_sell_all_7fc168/eval_summary.json, runs/sig3_vol_sell_all_7fc168/eval_oos_2021_2024/eval_summary.json, runs/sig3_vol_sell_all_7fc168/eval_bear_2018_tp200/eval_summary.json, runs/sig3_vol_sell_all_7fc168/eval_bear_2022_tp200/eval_summary.json, runs/sig2_mean_sell_all_b8396e/eval_summary.json, runs/sig2_mean_sell_all_b8396e/eval_oos_2021_2024/eval_summary.json, runs/sig2_mean_sell_all_b8396e/eval_bear_2018_tp200/eval_summary.json, runs/sig2_mean_sell_all_b8396e/eval_bear_2022_tp200/eval_summary.json, runs/sig3_mean_sell_all_c17b4c/eval_summary.json, runs/sig3_mean_sell_all_c17b4c/eval_oos_2021_2024/eval_summary.json, runs/sig3_mean_sell_all_c17b4c/eval_bear_2018_tp200/eval_summary.json, runs/sig3_mean_sell_all_c17b4c/eval_bear_2022_tp200/eval_summary.json, runs/sig2_meanvol_sell_all_ed2509/eval_summary.json, runs/sig2_meanvol_sell_all_ed2509/eval_oos_2021_2024/eval_summary.json, runs/sig2_meanvol_sell_all_ed2509/eval_bear_2018_tp200/eval_summary.json, runs/sig2_meanvol_sell_all_ed2509/eval_bear_2022_tp200/eval_summary.json, runs/sig3_meanvol_sell_all_fb1f11/eval_summary.json, runs/sig3_meanvol_sell_all_fb1f11/eval_oos_2021_2024/eval_summary.json, runs/sig3_meanvol_sell_all_fb1f11/eval_bear_2018_tp200/eval_summary.json, runs/sig3_meanvol_sell_all_fb1f11/eval_bear_2022_tp200/eval_summary.json.

### 2026-01-18 - Sell-all cap (max_positions=5) embedding sweep
- Goal: Evaluate sell_all_cap (sell_mode=all_cap, max_positions=5) for baseline and signature depth 2/3 across four embeddings.
- Hypothesis: Capped sell-all may stabilize performance vs unlimited stacking.
- Config: sell_mode=all_cap, max_positions=5; embeddings: price_return (log_price+log_return), rolling_mean, rolling_vol, mean+vol; trading_period=200 for bear windows.
- Data: In-sample 2014-01-01 to 2020-12-31; OOS 2021-01-01 to 2024-02-09; bear windows 2018 and 2022.
- Metrics (mean/std/min returns):
  - baseline_allcap: in-sample 2686.79/2268.31/-1771.50, OOS 1157.37/2132.36/-3438.00, bear-2018 -540.36/399.20/-1684.00, bear-2022 -269.16/339.88/-967.00
  - sig2_pr_allcap: in-sample 3723.27/1538.57/1181.50, OOS 2450.98/643.44/1381.00, bear-2018 798.54/260.61/340.50, bear-2022 401.74/380.19/27.50
  - sig2_mean_allcap: in-sample 2508.12/6135.46/-8084.50, OOS -3185.84/4269.41/-8223.50, bear-2018 -1671.50/854.54/-2945.50, bear-2022 -2319.22/893.71/-3225.50
  - sig2_vol_allcap: in-sample 1279.63/3966.59/-7463.00, OOS -4052.82/4413.02/-9352.00, bear-2018 -2168.57/956.65/-3495.50, bear-2022 -2698.00/983.46/-3710.00
  - sig2_meanvol_allcap: in-sample 1533.37/994.28/101.00, OOS 836.34/419.28/2.00, bear-2018 257.45/99.75/29.00, bear-2022 278.14/82.90/104.50
  - sig3_pr_allcap: in-sample 3763.48/2413.38/23.50, OOS 1637.01/804.09/438.50, bear-2018 357.88/431.62/-776.50, bear-2022 65.57/365.76/-372.50
  - sig3_mean_allcap: in-sample 1028.74/4514.85/-8374.50, OOS -4136.76/2699.90/-8741.50, bear-2018 -1949.35/474.71/-2803.50, bear-2022 -2555.83/951.11/-3512.50
  - sig3_vol_allcap: in-sample 1963.18/4185.62/-5789.00, OOS -3675.90/4393.08/-9106.00, bear-2018 -1999.46/1014.97/-3414.00, bear-2022 -2667.96/983.97/-3669.50
  - sig3_meanvol_allcap: in-sample 2333.50/1279.73/866.00, OOS 1296.67/428.50/575.50, bear-2018 382.04/190.34/10.50, bear-2022 -37.44/200.01/-394.00
- Outcome: price_return (sig2/sig3) is strongest under sell_all_cap; rolling_mean/rolling_vol embeddings are unstable in OOS and bear windows.
- Notes: Metrics are mean returns; see artifacts for full stats.
- Artifacts: runs/baseline_allcap_73e0c6/eval_summary.json, runs/baseline_allcap_73e0c6/eval_oos_2021_2024/eval_summary.json, runs/baseline_allcap_73e0c6/eval_bear_2018_tp200/eval_summary.json, runs/baseline_allcap_73e0c6/eval_bear_2022_tp200/eval_summary.json, runs/sig2_pr_allcap_598834/eval_summary.json, runs/sig2_pr_allcap_598834/eval_oos_2021_2024/eval_summary.json, runs/sig2_pr_allcap_598834/eval_bear_2018_tp200/eval_summary.json, runs/sig2_pr_allcap_598834/eval_bear_2022_tp200/eval_summary.json, runs/sig2_mean_allcap_2f1a53/eval_summary.json, runs/sig2_mean_allcap_2f1a53/eval_oos_2021_2024/eval_summary.json, runs/sig2_mean_allcap_2f1a53/eval_bear_2018_tp200/eval_summary.json, runs/sig2_mean_allcap_2f1a53/eval_bear_2022_tp200/eval_summary.json, runs/sig2_vol_allcap_1e1e1f/eval_summary.json, runs/sig2_vol_allcap_1e1e1f/eval_oos_2021_2024/eval_summary.json, runs/sig2_vol_allcap_1e1e1f/eval_bear_2018_tp200/eval_summary.json, runs/sig2_vol_allcap_1e1e1f/eval_bear_2022_tp200/eval_summary.json, runs/sig2_meanvol_allcap_33e912/eval_summary.json, runs/sig2_meanvol_allcap_33e912/eval_oos_2021_2024/eval_summary.json, runs/sig2_meanvol_allcap_33e912/eval_bear_2018_tp200/eval_summary.json, runs/sig2_meanvol_allcap_33e912/eval_bear_2022_tp200/eval_summary.json, runs/sig3_pr_allcap_fcf68f/eval_summary.json, runs/sig3_pr_allcap_fcf68f/eval_oos_2021_2024/eval_summary.json, runs/sig3_pr_allcap_fcf68f/eval_bear_2018_tp200/eval_summary.json, runs/sig3_pr_allcap_fcf68f/eval_bear_2022_tp200/eval_summary.json, runs/sig3_mean_allcap_8b86b5/eval_summary.json, runs/sig3_mean_allcap_8b86b5/eval_oos_2021_2024/eval_summary.json, runs/sig3_mean_allcap_8b86b5/eval_bear_2018_tp200/eval_summary.json, runs/sig3_mean_allcap_8b86b5/eval_bear_2022_tp200/eval_summary.json, runs/sig3_vol_allcap_30e308/eval_summary.json, runs/sig3_vol_allcap_30e308/eval_oos_2021_2024/eval_summary.json, runs/sig3_vol_allcap_30e308/eval_bear_2018_tp200/eval_summary.json, runs/sig3_vol_allcap_30e308/eval_bear_2022_tp200/eval_summary.json, runs/sig3_meanvol_allcap_c50ffe/eval_summary.json, runs/sig3_meanvol_allcap_c50ffe/eval_oos_2021_2024/eval_summary.json, runs/sig3_meanvol_allcap_c50ffe/eval_bear_2018_tp200/eval_summary.json, runs/sig3_meanvol_allcap_c50ffe/eval_bear_2022_tp200/eval_summary.json.

### 2026-01-18 - Sell-one+ (sell_one + sell_all) embedding sweep
- Goal: Evaluate sell_one_plus (sell_mode=one_plus, action_number=4) for baseline and signature depth 2/3 across four embeddings, tracking sell_one vs sell_all trigger rates.
- Hypothesis: Allowing a sell-all action improves liquidation in downtrends and increases sell frequency vs sell_one-only.
- Config: sell_mode=one_plus, max_positions=5, action_number=4; embeddings: price_return, rolling_mean, rolling_vol, mean+vol; trading_period=200 for bear windows.
- Data: In-sample 2014-01-01 to 2020-12-31; OOS 2021-01-01 to 2024-02-09; bear windows 2018 and 2022.
- Metrics (mean/std/min | sell_one_rate/sell_all_rate):
  - baseline_oneplus: in 2496.81/6444.47/-9296.00 | 0.0014/0.0001; oos -3832.38/4874.29/-9589.00 | 0.0000/0.0000; bear2018 -2088.36/1107.56/-3670.00 | 0.0034/0.0000; bear2022 -2846.20/1046.91/-3894.50 | 0.0000/0.0000
  - sig2_oneplus_pr: in 2486.60/6975.35/-9830.00 | 0.0000/0.0000; oos -3860.40/4908.42/-9630.00 | 0.0000/0.0000; bear2018 -2271.40/1125.44/-3745.00 | 0.0000/0.0000; bear2022 -2865.90/1048.44/-3915.00 | 0.0000/0.0000
  - sig2_oneplus_mean: in 2486.60/6975.35/-9830.00 | 0.0000/0.0000; oos -3860.40/4908.42/-9630.00 | 0.0000/0.0000; bear2018 -2271.40/1125.44/-3745.00 | 0.0000/0.0000; bear2022 -2865.90/1048.44/-3915.00 | 0.0000/0.0000
  - sig2_oneplus_vol: in 2356.15/6769.28/-9604.50 | 0.0000/0.0000; oos -3875.35/4888.25/-9630.00 | 0.0000/0.0000; bear2018 -2270.18/1115.43/-3745.00 | 0.0000/0.0000; bear2022 -2865.90/1048.44/-3915.00 | 0.0000/0.0000
  - sig2_oneplus_meanvol: in 339.30/3886.96/-7888.50 | 0.0000/0.0363; oos -2687.67/4367.42/-8414.00 | 0.0000/0.0026; bear2018 -2411.80/1116.22/-3724.50 | 0.0000/0.0012; bear2022 -2708.82/1031.31/-3839.00 | 0.0000/0.0018
  - sig3_oneplus_pr: in 2452.27/6912.42/-9768.50 | 0.0000/0.0000; oos -3876.92/4899.20/-9630.00 | 0.0000/0.0000; bear2018 -2256.70/1120.40/-3724.50 | 0.0000/0.0000; bear2022 -2865.90/1048.44/-3915.00 | 0.0000/0.0000
  - sig3_oneplus_mean: in 2462.91/6778.96/-9379.00 | 0.0000/0.0000; oos -3803.70/4688.52/-9240.50 | 0.0000/0.0000; bear2018 -2109.75/1048.94/-3478.50 | 0.0000/0.0000; bear2022 -2821.59/1000.28/-3853.50 | 0.0000/0.0000
  - sig3_oneplus_vol: in 2312.92/6796.77/-9707.00 | 0.0000/0.0000; oos -3863.76/4905.12/-9630.00 | 0.0000/0.0000; bear2018 -2273.38/1124.91/-3745.00 | 0.0000/0.0000; bear2022 -2865.90/1048.44/-3915.00 | 0.0000/0.0000
  - sig3_oneplus_meanvol: in 1275.70/1907.60/-2942.00 | 0.0000/0.0156; oos 2156.36/809.28/516.50 | 0.0000/0.0219; bear2018 -696.83/559.98/-1871.00 | 0.0000/0.0155; bear2022 164.81/475.76/-287.50 | 0.0000/0.0330
- Outcome: Most one_plus runs collapse to buy-only behavior (sell_one/sell_all near zero). The mean+vol setups are the only ones with noticeable sell_all usage; sig3_meanvol is the only configuration with positive OOS and bear-2022 mean returns.
- Notes: sell_one rates are effectively zero across all runs, indicating the agent rarely chooses the sell_one action even when available.
- Artifacts: runs/baseline_oneplus_45dc83, runs/sig2_oneplus_pr_1eaec9, runs/sig2_oneplus_mean_8bad94, runs/sig2_oneplus_vol_b2ec30, runs/sig2_oneplus_meanvol_69f6c4, runs/sig3_oneplus_pr_d72044, runs/sig3_oneplus_mean_00cef8, runs/sig3_oneplus_vol_6c124e, runs/sig3_oneplus_meanvol_6e51a7.

### 2026-01-20 - Sell-one OOS eval 2021-2024
- Goal: Evaluate sell_mode=one runs on OOS window to compare against sell-one+.
- Hypothesis: sell-one maintains higher sell frequency and better bear/OOS exits than sell-one+.
- Config: sell_mode=one, max_positions=5; baseline and signature depth 2/3 price_return checkpoints.
- Data: 2021-01-01 to 2024-02-09 (OOS).
- Metrics (mean/std/min | sell_rate):
  - baseline_sell_one: 2694.65/1268.12/787.00 | 0.0998
  - signature2_sell_one: -1917.61/3436.70/-6463.50 | 0.0035
  - signature3_sell_one: -2724.34/3555.34/-7063.50 | 0.0022
- Outcome: Baseline remains positive in OOS with non-trivial sell frequency; signature sell-one runs stay negative with very low sell rates.
- Notes: Signature sell-one checkpoints store embedding as a legacy string (price_return); evaluation mapped it to {log_price, log_return} for compatibility.
- Artifacts: runs/baseline_sell_one_e8a931/eval_oos_2021_2024/eval_summary.json, runs/signature2_sell_one_83f00e/eval_oos_2021_2024/eval_summary.json, runs/signature3_sell_one_9ba2a8/eval_oos_2021_2024/eval_summary.json.

### 2026-02-13 - Signature sr_enhanced metric refactor and reruns
- Goal: Make return-rate and periodic eval metrics consistent and diagnosable for signature + `sr_enhanced` experiments.
- Hypothesis: A unified equity-based return-rate formula and eval diagnostics will remove ambiguous metric interpretation and expose true environment behavior.
- Config:
  - Training config: `configs/test_signature.yaml`
  - Reward: `sr_enhanced`
  - `trading_period=500`, `sell_mode=one`, `max_positions=5`
  - Periodic eval: every 20 episodes, 50 fixed windows (`eval_seed=20240101`)
- Data: 2014-01-01 to 2020-12-31.
- Metrics:
  - `test_signature_sr_enhanced_a92a03` (`eval_summary.json`): mean_reward_return `0.0247`, mean_return_rate `11.9890` (about `1198.90%`), sharpe_ratio `0.7070`, diagnostics `initial_state_none_episodes=0`, `zero_step_episodes=0`.
  - `testsignatueenhanced_e8a185` (`eval_history.csv` @episode 200): mean_reward_return `-0.0351`, mean_return_rate `136.5414` (percent in eval_history), sharpe_ratio `0.4743`, win_rate `0.44`.
  - `testsignatueenhanced_e8a185` (`eval_summary.json`, regenerated after fixes): mean_reward_return `-7.7224`, mean_return_rate `12.2015`, mean_return_rate_pct `1220.1461`, sharpe_ratio `0.5938`, diagnostics `initial_state_none_episodes=0`, `zero_step_episodes=0`.
- Outcome: Metrics are now consistently named (`reward_return`) and return-rate is consistently equity-based; periodic eval and standalone eval both expose zero-step diagnostics.
- Notes:
  - `eval_history.csv` stores return-rate in percent; `eval_summary.json` now stores both decimal (`mean_return_rate`) and percent (`mean_return_rate_pct`) fields for compatibility.
  - Reward-logic fix removed hold-reward overwrite for non-`sr_enhanced` modes.
  - Return-rate magnitude remains high because current environment has no explicit cash/margin constraint.
- Artifacts: runs/test_signature_sr_enhanced_a92a03/eval_summary.json, runs/test_signature_sr_enhanced_a92a03/eval_history.csv, runs/testsignatueenhanced_e8a185/eval_history.csv, runs/testsignatueenhanced_e8a185/eval_summary.json, runs/testsignatueenhanced_e8a185/eval_episodes.csv.
