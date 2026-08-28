
# 下一階段：MFN 串接 SB3 A2C

這一版不是把原本的 MFN `train_mfn()` 當成獨立 supervised learning
模型，而是改成 Stable-Baselines3 的 `BaseFeaturesExtractor`。

流程：

Binance data
-> prepare_paper_features.py
-> 20 x 32 observation
-> TwoViewMFN
-> SB3 A2C
-> Actor/Critic
-> 5-action portfolio allocation
-> environment
-> DSR reward

## 執行順序

```powershell
.\.venv\Scripts\Activate.ps1

pip install -r requirements.txt

python download_binance_paper.py
python prepare_paper_features.py
python train_mfn_a2c_sb3.py
python evaluate_mfn_a2c.py
```

第一次訓練只跑 10,000 timesteps，是為了確認 pipeline 可以運作。
確認無誤後，把 `train_mfn_a2c_sb3.py` 的 `total_timesteps=10_000`
提高，再做正式實驗。

## 重要設計

- MFN 改成 2-view：Price Change + Technical Indicators
- 第三 view 及其對應結構不再保留
- 20 timesteps
- 4-hour data
- A2C `learning_rate=7e-4`
- A2C `gamma=0.99`
- A2C `n_steps=540`
- A2C policy hidden layers = [64, 64]
- DSR eta = 0.005
- 5 actions = BTC, ETH, LTC, BNB, USDT
- transaction fee = 0，符合論文目前環境
- backtest = final 1,080 rows

## 與原始 MFN.py 的差別

原始檔案是從多 view MFN 改到一半的版本，還有 `lstm_l`、
`.cuda()`、舊的 supervised `train_mfn()` 等結構。

新的 `mfn_sb3_extractor.py` 直接以 SB3 `BaseFeaturesExtractor`
為介面，因此 A2C 的 rollout/training loop 由 SB3 處理，
MFN 只負責把 `(20, 32)` observation 轉成 feature vector。

這比較符合你描述的「最後應該是用 SB3 feature extractor 串 MFN」的方向。
