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
| Data period | 2018-01-01 to 2025-09-01 |
| Trading interval | 4 hours |
| Historical observation window | 20 timesteps |
| Technical indicators | SMA-20, EMA-20, MACD(12,26,9), RSI-14 |
| A2C hidden layers | 2 |
| Units per A2C layer | 64 |
| Learning rate | 7e-4 |
| Discount factor | 0.99 |
| Steps per rollout | 540 |
| DSR update rate | 0.005 |
| Formal training timesteps | 1,800,000 |
| Initial portfolio value | 10,000 |
| Transaction fee | 0 |

> Development runs may use fewer timesteps such as `100_000`. Set `TOTAL_TIMESTEPS = 1_800_000` for the formal experiment configuration.

---

## Project Structure

```text
MFN_A2C_Crypto/
│
├── src/
│   ├── mfn_sb3_extractor.py
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

Download Binance 4-hour K-line data:

```powershell
python scripts\download_binance_paper.py
```

Generate aligned price-change and technical-indicator features:

```powershell
python scripts\prepare_paper_features.py
```

Check alignment:

```powershell
python scripts\check_alignment.py
```

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

Stable-Baselines3 DQN requires a discrete action space. This repository therefore uses a discrete portfolio-action wrapper for the DQN baseline.

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

Two-view MFN feature extractor used by SB3 A2C.

### `src/portfolio_env_sb3.py`

Portfolio environment for BTC, ETH, LTC, BNB, and USDT.

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
Train/test split
Historical observation window
Learning rate
Discount factor
Random seed
Initial portfolio value
Backtesting interval
```

---

## Important Notes

1. Transaction fees and market slippage are not modeled in the current environment.
2. USDT is treated as the stable/cash asset.
3. Technical indicators form the second temporal modality.
4. A2C uses continuous portfolio-allocation actions.
5. DQN uses a discrete action-space adaptation.
6. `original/` is retained for reference and is not part of the main execution pipeline.
7. Large model files, checkpoints, TensorBoard logs, and intermediate datasets may be excluded from Git.

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
