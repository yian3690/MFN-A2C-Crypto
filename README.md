# MFN-A2C Crypto Portfolio Optimization

A research implementation of **Memory Fusion Network (MFN) + Advantage Actor-Critic (A2C)** for cryptocurrency portfolio optimization.

This repository accompanies the paper:

> **Reinforcement Learning for Cryptocurrency Portfolio Optimization with Memory Fusion Network**

The proposed method adapts the original multi-view MFN architecture to two financial modalities:

- **Price-change features**
- **Technical indicators**

The fused temporal representation is then used by an A2C agent to generate portfolio allocation decisions across **BTC, ETH, LTC, BNB, and USDT**.

---

## Overview

```text
Binance K-line Data
        │
        ├── Price-change features
        └── Technical indicators
                │
                ▼
        Two-View Memory Fusion Network
        ├── Modality-specific LSTMs
        ├── Delta-memory Attention
        └── Multi-view Gated Memory
                │
                ▼
        Fused State Representation
                │
                ▼
        Advantage Actor-Critic (A2C)
        ├── Actor
        └── Critic
                │
                ▼
        Portfolio Allocation
        BTC / ETH / LTC / BNB / USDT
```

The repository also includes:

- Standard A2C baseline
- A2C without technical indicators
- DQN baseline
- Buy-and-Hold baseline
- DSR reward vs. portfolio-value reward experiments

---

## Experimental Configuration

| Parameter | Value |
|---|---|
| Assets | USDT, BTC, ETH, LTC, BNB |
| Data source | Binance |
| Downloaded data period | 2018-01-01 to 2025-09-01 00:00 (inclusive), 33,550 raw rows |
| Valid dataset | 2018-01-03 04:00 to 2025-09-01 00:00, 33,524 rows |
| Train period | 2018-01-03 04:00 to 2025-04-19 00:00, 31,904 rows |
| Validation period | 2025-04-19 02:00 to 2025-06-03 00:00, 540 rows (45 days) |
| Test period | 2025-06-03 02:00 to 2025-09-01 00:00, final 1,080 rows |
| Trading interval | 2 hours |
| Historical observation window | 20 timesteps (40 hours) |
| Decision timing | Observe through `t-1`, rebalance at `Open[t]` |
| Return interval | `Open[t]` to `Open[t+1]` |
| Input indicators/features | SMA-20, EMA-20, MACD(12,26,9), RSI-14, Scheme-A RS_14D |
| Feature scaling | Train-only; Validation and Test are always excluded from fitting |
| A2C hidden layers | 2 |
| Units per A2C layer | 64 |
| Learning rate | 7e-4 |
| Discount factor | 0.99 |
| Steps per rollout | 540 |
| Optimizer updates per rollout | Current fair comparison: 1; thesis table: 18 |
| Training reward | `1 × paper-innovation step DSR + 50 × log(1 + portfolio return)` |
| DSR update rate | 0.005 (paper mode uses raw innovations; eta only updates EWMA moments) |
| DSR warm-up | First 5 steps update EWMA moments but return zero reward |
| Training timesteps | Up to 600,000; validate every 50,000 steps |
| Initial portfolio value | 10,000 |
| Transaction fee | 0 |

> All four methods use the same single-stage protocol. Each model trains chronologically on the 31,904-row Train split for at most 600,000 steps and evaluates the independent 540-row (45-day) Validation split every 50,000 steps. The checkpoint with the highest Validation Final PV is saved as the formal model and loaded directly by evaluation. Test remains excluded from scaling, training, feature-horizon choice, and checkpoint selection.

---

## Project Structure

```text
MFN_A2C_Crypto/
│
├── src/
│   ├── mfn_sb3_extractor.py
│   ├── mfn_github_extractor.py
│   ├── simplex_policy.py
│   ├── portfolio_env_sb3.py
│   ├── price_only_env.py
│   └── discrete_action_env.py
│
├── scripts/
│   ├── download_binance_paper.py
│   ├── prepare_paper_features.py
│   ├── check_alignment.py
│   ├── train_mfn_a2c.py
│   ├── evaluate_mfn_a2c.py
│   ├── train_a2c_baseline.py
│   ├── evaluate_a2c_baseline.py
│   ├── train_a2c_without_ti.py
│   ├── evaluate_a2c_without_ti.py
│   ├── train_dqn_baseline.py
│   ├── evaluate_dqn_baseline.py
│   ├── train_experiment3.py
│   ├── evaluate_experiment3.py
│   ├── baseline_buy_hold.py
│   ├── compare_all_methods.py
│   ├── compare_experiment2.py
│   ├── compare_experiment3.py
│   └── final_experiment_summary.py
│
├── original/
│   └── Legacy/original implementation files for reference
│
├── data/
├── models/
├── results/
├── figures/
├── logs/
├── checkpoints/
│
├── requirements.txt
├── setup_env.bat
├── setup_env.ps1
└── README.md
```

---

## Environment Setup

### Windows PowerShell

Create a virtual environment:

```powershell
python -m venv .venv
```

Activate it:

```powershell
.venv\Scripts\Activate.ps1
```

Install dependencies:

```powershell
pip install -r requirements.txt
```

Check CUDA:

```powershell
python -c "import torch; print(torch.cuda.is_available()); print(torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CPU')"
```

---

## Data Preparation

Download Binance 2-hour K-line data:

```powershell
python scripts\download_binance_paper.py
```

Generate aligned price-change and technical-indicator features:

```powershell
python scripts\prepare_paper_features.py
```

此步驟建立5個Close-to-Close price-relative，以及25個SMA/EMA/MACD(DIF)/RSI/RS_14D level特徵。`RS_14D`為各加密貨幣過去14日報酬減去四幣同期平均，只使用當下與過去資料。Scaler只用31,904筆Train擬合；Validation與Test套用相同統計量，不參與擬合。

學長原碼的 `senior_ta` 消融版仍保留於 `src/technical_indicators.py` 與實驗歷史中。該版本對 MACD 做 `pct_change() × 100`，因零點穿越產生數百萬等級尖峰，A2C 600,000步結果下降至23.35%報酬，因此不再作為目前正式資料版本。

Check alignment:

```powershell
python scripts\check_alignment.py
```

---

### No-look-ahead time alignment

The environment follows this decision sequence:

```text
Observation: completed candles [t-20, ..., t-1]
Decision:    rebalance at Open[t]
Reward:      portfolio return from Open[t] to Open[t+1]
```

Features from candle `t` are not included when the action at `Open[t]` is selected. This prevents the agent from using the current candle's Close, High, Low, return, or technical indicators before that candle has completed.

The evaluation scripts use `number_of_rows - lookback - 1` steps because the last decision must still have an available `Open[t+1]`. Buy-and-Hold starts at the same `Open[20]`, so all methods use the same backtest interval.

> Models and checkpoints trained before this time-alignment correction are not valid for the corrected experiment. Train every method again from scratch before comparing results.

### Chronological Train/Test split

The thesis reports 33,524 valid two-hour observations. For feature-horizon and checkpoint selection, the original 32,444-row pre-Test period is split chronologically into 31,904 Train rows and 540 Validation rows (45 days). The final 1,080 rows remain the untouched Test set.

```text
All valid:  2018-01-03 04:00 through 2025-09-01 00:00 (33,524 rows)
Train:      2018-01-03 04:00 through 2025-04-19 00:00 (31,904 rows)
Validation: 2025-04-19 02:00 through 2025-06-03 00:00 (540 rows; 45 days)
Test:       2025-06-03 02:00 through 2025-09-01 00:00 (1,080 rows)
First Test trade after lookback: 2025-06-04 18:00 UTC
```

All methods use the same single-stage protocol: training runs chronologically on 31,904 Train rows for at most 600,000 steps and evaluates the 45-day Validation split every 50,000 steps. The checkpoint with the highest Validation Final PV is loaded directly by evaluation scripts. Test is excluded from scaling, training, horizon choice, and checkpoint selection.

---

## Experiment 1 — Effect of Technical Indicators

Compared methods:

```text
Proposed MFN-A2C
A2C
A2C without Technical Indicators

Buy-and-Hold
```

Train and evaluate the proposed method:

```powershell
python scripts\train_mfn_a2c.py
python scripts\evaluate_mfn_a2c.py
```

Train and evaluate standard A2C:

```powershell
python scripts\train_a2c_baseline.py
python scripts\evaluate_a2c_baseline.py
```

Train and evaluate A2C without technical indicators:

```powershell
python scripts\train_a2c_without_ti.py
python scripts\evaluate_a2c_without_ti.py
```

Generate Buy-and-Hold results:

```powershell
python scripts\baseline_buy_hold.py
```

Compare all methods:

```powershell
python scripts\compare_all_methods.py
```

Typical outputs:

```text
results/experiment1_comparison.csv
figures/experiment1_comparison.png
```

---

### Verified development result

A corrected single-run MFN-A2C backtest trained for `100_000` timesteps produced the following development result:

| Metric | Value |
|---|---:|
| Initial portfolio value | 10,000.00 |
| Final portfolio value | 13,558.43 |
| Total return | 35.58% |
| Peak portfolio value | 14,542.97 |
| Maximum drawdown | -19.94% |
| Sharpe ratio | 1.8268 |

This is an older four-hour, `100_000`-step single-seed development result, rather than the current two-hour, `300_000`-step fair-comparison setting. It is not directly comparable with newly retrained models. Formal reporting should use the same settings for every method and summarize multiple random seeds with mean and standard deviation.

---

## Experiment 2 — Effect of MFN Feature Extraction

Compared methods:

```text
Proposed MFN-A2C
A2C
DQN
Buy-and-Hold
```

Train DQN:

```powershell
python scripts\train_dqn_baseline.py
```

Evaluate DQN:

```powershell
python scripts\evaluate_dqn_baseline.py
```

Compare Experiment 2:

```powershell
python scripts\compare_experiment2.py
```

### DQN note

Stable-Baselines3 DQN requires a discrete action space. This repository therefore uses a discrete portfolio-action wrapper for the DQN baseline. The grid uses 20% increments and limits each risky asset (BTC, ETH, LTC, and BNB) to 60%, while USDT may reach 100%. This leaves 106 valid discrete actions.

The paper does not fully specify the exact DQN action discretization, so this should be treated as an implementation choice in the reconstructed codebase rather than a claim of exact recovery of the original DQN implementation.

---

## Experiment 3 — Effect of Reward Selection

Compared configurations:

```text
MFN-A2C + DSR
MFN-A2C + Portfolio-Value reward
A2C + DSR
A2C + Portfolio-Value reward
```

Train:

```powershell
python scripts\train_experiment3.py
```

Evaluate:

```powershell
python scripts\evaluate_experiment3.py
```

Compare:

```powershell
python scripts\compare_experiment3.py
```

Typical outputs:

```text
results/exp3_mfn_dsr_results.csv
results/exp3_mfn_pv_results.csv
results/exp3_a2c_dsr_results.csv
results/exp3_a2c_pv_results.csv
results/experiment3_comparison.csv
results/experiment3_table.csv
figures/experiment3_comparison.png
```

---

## Final Experiment Summary

After all experiments are complete:

```powershell
python scripts\final_experiment_summary.py
```

---

## Main Components

### `src/mfn_sb3_extractor.py`

Two-view MFN feature extractor used by SB3 A2C. It follows thesis equations 4.5-4.10: adjacent hidden-state differences are concatenated, a single linear mapping plus Softmax produces the attended delta, and two single-linear gates read only that attended delta. Shared memory is updated as `gamma1 * previous_memory + gamma2 * tanh(attended_delta)`.

This thesis-equation extractor is retained as a controlled comparison implementation. It is incompatible with GitHub-style MFN checkpoints and is not used by the current MFN train/evaluate scripts.

### `src/mfn_github_extractor.py`

Current GitHub-style two-view extractor. It uses separate price and indicator LSTM cells, concatenates previous/current cell states, applies a two-layer attention MLP plus Softmax, builds memory through a two-layer candidate MLP, and conditions two-layer retention/update gates on both attended states and previous memory. Final modality hidden states and shared memory are passed directly to SB3 A2C.

The current Scheme-A experiment uses single-stage Validation selection. Training runs for at most 600k steps on Train and evaluates Validation every 50k steps; the checkpoint with the highest Validation Final PV becomes the formal model without a Development retraining stage. It uses Gaussian logits plus environment Softmax, `log_std_init=-2` (initial std about 0.135), `normalize_advantage=True`, one optimizer epoch per 540-step rollout, and a 20-step (40-hour) observation window. The input is 5 price relatives plus 25 level-and-z-score features: the paper's four indicators plus one past-only cross-asset `RS_14D` feature per asset. Raw DSR now uses the published innovations `r-A_old` and `r²-B_old`; `eta=0.005` only updates the EWMA moments. Training reward is `1 × raw DSR + 50 × log(1 + portfolio return)`, while evaluation reports the same unscaled raw DSR. Artifact tags are generated from the actual reward scales, eta, log std and seed. This is an enhancement/ablation and must not be described as the paper's original four-indicator input.

### `src/multi_epoch_a2c.py`

Supports repeating the standard full-batch SB3 A2C optimizer update for every collected 540-step rollout. The thesis table reports 18 update epochs, but the current MFN diagnostic explicitly passes `update_epochs=1` to test whether repeated reuse of on-policy data causes policy collapse. It does not introduce PPO clipping, and models must be retrained whenever this value changes.

### `src/dsr.py`

Single EWMA DSR implementation shared by training, evaluation, and Buy-and-Hold. The default `paper` mode uses the published innovations `delta_A=r-A_old` and `delta_B=r²-B_old`; `eta=0.005` only controls moment updates. The prior eta-scaled implementation remains available as the explicit `ewma_change` ablation, and the archived expanding-mean implementation remains separate. The first five steps update moments but return zero DSR.

### `src/training_diagnostics.py`

The MFN and A2C training scripts write one CSV row per rollout under `logs/training_diagnostics/`. In addition to reward, PV, turnover, allocation and Actor/Critic losses, they record per-asset Gaussian exploration scale, sampled-versus-deterministic allocation distance, Advantage/return-target statistics, Critic target correlation/RMSE, and DSR/log-return sign conflict. MFN additionally exposes extractor parameter/gradient information. Validation scores are saved separately under `logs/validation/`.

All four model evaluators also save per-step `asset_return_*`, `return_contribution_*`, and `pnl_contribution_*` columns. The terminal report shows each underlying asset's compounded return and path-dependent PnL contribution; their PnL total reconciles exactly to `Final PV - Initial PV` in the no-fee environment.

Evaluation also reports whether allocations follow relative strength over 6/12/36/84 bars (12 hours/1 day/3 days/7 days). Trailing returns are shifted by one bar before rolling, so only information available before the action is used. Ex-post next-interval winner/loser weights are explanation-only diagnostics and are never model inputs.

### Resume an interrupted MFN run

Start a new Train/Validation-selection run normally:

```powershell
python scripts\train_mfn_a2c.py
```

If interrupted, resume the newest compatible checkpoint automatically:

```powershell
python scripts\train_mfn_a2c.py --resume
```

The script reads the checkpoint's absolute `num_timesteps`, trains only the remaining amount, keeps TensorBoard's absolute timestep count, and creates a separate diagnostic CSV whose filename records the resume point. It refuses unrelated checkpoint prefixes. Standard SB3 checkpoints do not preserve the environment index, unfinished rollout, DSR moments, or random-number-generator state; therefore resume restarts the Train environment and DSR statistics and is a practical continuation rather than bit-identical reproduction of an uninterrupted run.

### `src/evaluation_metrics.py`

Adds per-step `dsr` and `cumulative_dsr`, reports `Peak Cumulative DSR` and `Final Cumulative DSR`, and summarizes model allocations and turnover.

### `src/portfolio_env_sb3.py`

Portfolio environment for BTC, ETH, LTC, BNB, and USDT. MFN-A2C uses direct simplex weights; a legacy logits-to-Softmax mode remains available for models not yet migrated.

### `src/simplex_policy.py`

Optional Dirichlet actor distribution retained for ablation experiments. It is not used by the current fair comparison, where all three A2C methods use the same standard SB3 Gaussian-logit policy plus environment Softmax.

### `src/price_only_env.py`

Price-only observation wrapper used for the A2C-without-TI ablation.

### `src/discrete_action_env.py`

Discrete action wrapper used by the DQN baseline.

---

## Reproducibility Notes

Reinforcement-learning results can vary with:

- Random seeds
- Training duration
- Reward implementation
- Environment implementation
- Hardware and software versions

Short development runs should therefore not be expected to reproduce the final paper values exactly.

For formal comparison, keep the following consistent across comparable RL methods:

```text
Training timesteps
Chronological train/test split
Historical observation window
Learning rate
Discount factor
Random seed
Initial portfolio value
Backtesting interval
```

Run at least five random seeds for formal experiments and report the mean and standard deviation. Do not compare a newly corrected model with result CSVs or models produced by the earlier time-leaking environment. Models trained without the restored five-step DSR warm-up must also be retrained.

---

## Important Notes

1. Transaction fees and market slippage are not modeled in the current environment.
2. USDT is treated as the stable/cash asset for RL agents. Following the thesis definition, Buy-and-Hold allocates 25% each to BTC, ETH, LTC, and BNB, assigns 0% to USDT, and does not rebalance.
3. The formal reproduction uses paper-style EWMA moment changes with `eta = 0.005` and a five-step warm-up. The numerical variance guard remains enabled, and reported Peak/Final values refer to cumulative DSR. Canonical-innovation results remain an ablation and cannot be compared numerically with paper-style DSR.
4. Technical indicators form the second temporal modality.
5. All three A2C methods use the same SB3 Gaussian-logit action followed by environment Softmax. DQN uses a discrete simplex grid and bypasses the Softmax because its wrapper already emits valid weights.
6. DQN uses a discrete action-space adaptation.
7. `original/` is retained for reference and is not part of the main execution pipeline.
8. Large model files, checkpoints, TensorBoard logs, and intermediate datasets may be excluded from Git.
9. Every formal model must be retrained after changes to the data period, feature scaling, environment timing, or reward calculation.

---

## Recommended `.gitignore`

```gitignore
.venv/
__pycache__/
*.pyc

logs/
checkpoints/
checkpoints_*/
models/

*.zip
```

If datasets are regenerated locally, you may also add:

```gitignore
data/*.csv
```

---

## Citation

If you use this repository or the associated work, please cite the corresponding paper:

```bibtex
@article{wu2026mfna2c,
  title  = {Reinforcement Learning for Cryptocurrency Portfolio Optimization with Memory Fusion Network},
  author = {Wu, Cheng-Hsian and Chen, Yi-An and Ju, Ming-Yi},
  year   = {2026}
}
```

Update the BibTeX entry with the final conference publication information once available.

---

## Disclaimer

This repository is intended for **academic research and experimental evaluation**.

It is not financial advice, and the trading strategies in this repository should not be interpreted as recommendations for real-world investment.
