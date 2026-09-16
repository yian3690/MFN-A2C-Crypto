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
| Train period | First 32,444 valid rows; through 2025-06-03 00:00 |
| Validation | None |
| Test period | 2025-06-03 02:00 to 2025-09-01 00:00, final 1,080 rows |
| Trading interval | 2 hours |
| Historical observation window | 20 timesteps (40 hours) |
| Decision timing | Observe through `t-1`, rebalance at `Open[t]` |
| Return interval | `Open[t]` to `Open[t+1]` |
| Technical indicators | SMA-20, EMA-20, MACD(12,26,9), RSI-14 |
| Feature scaling | Per-column z-score fitted on Train only |
| A2C hidden layers | 2 |
| Units per A2C layer | 64 |
| Learning rate | 7e-4 |
| Discount factor | 0.99 |
| Steps per rollout | 540 |
| Optimizer updates per rollout | 18 |
| DSR update rate | 0.005 (paper mode uses EWMA moment changes) |
| DSR warm-up | First 5 steps update EWMA moments but return zero reward |
| Training timesteps | Current A2C warm-up comparison: 1,800,000; final model is used |
| Initial portfolio value | 10,000 |
| Transaction fee | 0 |

> This configuration has no Validation selection. The current A2C warm-up comparison saves the final model after 1,800,000 steps. GitHub-style two-view MFN-A2C is temporarily configured for a 300,000-step, one-update-epoch, 5+20-feature diagnostic; A2C without TI and DQN retain their own current step settings.

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

此步驟依論文式 4.2 建立五個 Close-to-Close price-relative 特徵，並建立每項資產的 SMA-20、EMA-20、MACD DIF 與 RSI-14，共 20 個技術指標特徵。四個 USDT 指標為中性常數，標準化後為 0。MACD 採封存原碼使用的 `MACD_12_26_9`（DIF/MACD line），而非九期 signal line。依論文保留最新 33,524 筆有效資料，前 32,444 筆為 Train、最後 1,080 筆為 Test。每欄 z-score 的 mean/std 只由 Train 計算；Test 僅套用同一組參數，統計量會存於 `data/feature_scaler.csv`。特徵處理改變後，既有模型與結果不可混用，必須重新訓練與評估。

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

The thesis reports 33,524 valid two-hour observations. The first 32,444 rows are used for training and the final 1,080 rows are held out for Test. There is no Validation split. Public Binance archives may include a few additional early candles, so preprocessing retains the most recent 33,524 aligned observations to reproduce the reported row counts and fixed final Test period.

```text
All valid:  2018-01-03 04:00 through 2025-09-01 00:00 (33,524 rows)
Train:      2018-01-03 04:00 through 2025-06-03 00:00 (32,444 rows)
Validation: none
Test:       2025-06-03 02:00 through 2025-09-01 00:00 (1,080 rows)
First Test trade after lookback: 2025-06-04 18:00 UTC
```

All methods train on the same 32,444 Train rows. GitHub-style two-view MFN-A2C currently uses a temporary 300,000-step 5+20-feature diagnostic. MFN keeps `n_steps=540` for rollout collection but treats the complete chronological Train set as one episode, so DSR moments persist across rollouts. Periodic checkpoints are diagnostic only. Test is excluded from scaling and training and must not be used to choose a checkpoint.

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

This is an older four-hour, `100_000`-step single-seed development result, rather than the current two-hour, `600_000`-step paper-reproduction setting. It is not directly comparable with newly retrained models. Formal reporting should use the same settings for every method and summarize multiple random seeds with mean and standard deviation.

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

The current paper-reproduction run uses 300,000 timesteps, a complete chronological Train episode, one optimizer epoch per 540-step rollout, the 5-price + 20-indicator input schema, and the archived paper-style EWMA moment-change DSR. It uses the `MFN_A2C_GITHUB2_5X20_PAPER_DSR_E1_300K_*` checkpoint prefix and saves the evaluation model as `models/mfn_a2c_github2_5x20_300k_paper_dsr.zip`; the thesis-reported setting remains 18 update epochs.

### `src/multi_epoch_a2c.py`

Supports repeating the standard full-batch SB3 A2C optimizer update for every collected 540-step rollout. The thesis table reports 18 update epochs, but the current MFN diagnostic explicitly passes `update_epochs=1` to test whether repeated reuse of on-policy data causes policy collapse. It does not introduce PPO clipping, and models must be retrained whenever this value changes.

### `src/dsr.py`

Single EWMA DSR implementation shared by training, evaluation, and Buy-and-Hold. The default `paper_legacy` mode uses `delta_A=A_new-A_old` and `delta_B=B_new-B_old`, matching the archived project behavior and paper-scale output. `eta=0.005` is retained. The canonical innovation form remains selectable for ablation. The first five steps update moments but return zero DSR.

### `src/training_diagnostics.py`

The MFN training script writes one CSV row per rollout under `logs/training_diagnostics/`. It records reward and DSR scale, returns, portfolio value, turnover, allocation entropy/concentration, Dirichlet concentrations, deterministic weight variation, MFN parameter/gradient norms, Actor/Critic losses, and explained variance. `evaluate_mfn_a2c.py` automatically reads the newest matching run and prints threshold-based possible causes of weak training.

### Resume an interrupted MFN run

Start a new 300,000-step run normally:

```powershell
python scripts\train_mfn_a2c.py
```

If it is interrupted after a periodic checkpoint has been written, resume the newest checkpoint that exactly matches the current MFN architecture, 5+20 features, paper DSR, one update epoch, and 300,000-step experiment:

```powershell
python scripts\train_mfn_a2c.py --resume
```

The script reads the checkpoint's absolute `num_timesteps`, trains only the remaining amount, keeps TensorBoard's absolute timestep count, and creates a separate diagnostic CSV whose filename records the resume point. It refuses unrelated checkpoint prefixes. Standard SB3 checkpoints do not preserve the environment index, unfinished rollout, DSR moments, or random-number-generator state; therefore resume restarts the Train environment and DSR statistics and is a practical continuation rather than bit-identical reproduction of an uninterrupted run.

### `src/evaluation_metrics.py`

Adds per-step `dsr` and `cumulative_dsr`, reports `Peak Cumulative DSR` and `Final Cumulative DSR`, and summarizes model allocations and turnover.

### `src/portfolio_env_sb3.py`

Portfolio environment for BTC, ETH, LTC, BNB, and USDT. MFN-A2C uses direct simplex weights; a legacy logits-to-Softmax mode remains available for models not yet migrated.

### `src/simplex_policy.py`

Dirichlet actor distribution for MFN-A2C. Positive concentration parameters are produced with Softplus; stochastic training actions and deterministic mean actions are both non-negative and sum directly to one, without `[-5, 5]` clipping or an environment Softmax.

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
5. MFN-A2C directly samples continuous simplex portfolio weights from a Dirichlet distribution. Older A2C experiments still use the legacy Gaussian-logit action mode until migrated separately.
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
