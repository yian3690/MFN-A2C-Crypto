# 程式碼用途導覽

本專案重現論文的 MFN-A2C 加密貨幣投資組合流程。每個決策點使用前 20 個 4 小時 K 線（80 小時）作為觀察值，將資金配置到 BTC、ETH、LTC、BNB 與 USDT。USDT 視為報酬率 0 的穩定資產。

## 核心模組

- `src/portfolio_env_sb3.py`：Gymnasium 交易環境；讀取資料、把模型輸出轉成投資權重，在 `Open[t]` 再平衡，根據 `Open[t]` 到 `Open[t+1]` 更新投資組合並回傳 reward。
- `src/mfn_sb3_extractor.py`：兩視角 MFN；16 個價格變化與 16 個技術指標各經 LSTM，再以 DMAN 類注意力與 MGM 類記憶閘門融合成 128 維特徵給 A2C。
- `src/price_only_env.py`：實驗 1 的 A2C w/o TI，只保留價格變化特徵。
- `src/discrete_action_env.py`：DQN 專用包裝；將離散動作轉為近乎全倉單一資產的權重。

## 資料流程

1. `download_binance_paper.py`：下載四種加密貨幣的 Binance 4 小時 K 線並依時間戳合併。
2. `prepare_paper_features.py`：建立 16 個價格變化、16 個 SMA/EMA/MACD/RSI 技術指標特徵，最後保留 1,080 列測試資料。
3. `check_alignment.py`：驗證 train/test 的時間戳、原始價格及兩類特徵逐列對齊。

## 正確的決策時間軸

環境不可以讓模型看到尚未完成的 K 線。每一步的順序如下：

```text
觀測資料：第 t-20 到第 t-1 根已完成 K 線
執行動作：在 Open[t] 配置 BTC、ETH、LTC、BNB、USDT 權重
計算報酬：使用 Open[t] 到 Open[t+1] 的價格變化
下一狀態：觀測窗口向前移動一根 K 線
```

`CryptoPortfolioEnv.step()` 的索引關係為：

```python
decision_idx = start_idx + counter + n_previous_timesteps
next_idx = decision_idx + 1
```

觀測窗口最後一列是 `decision_idx - 1`。若模型在 `Open[t]` 交易時已看到第 `t` 根 K 線的 Close、High、Low、報酬率或技術指標，就會形成未來資料洩漏。

完整測試期間應設定：

```python
max_episode_steps = len(test_data) - LOOKBACK - 1
```

最後的 `-1` 用來保留 `Open[t+1]`。Buy-and-Hold 也從 `Open[20]` 開始，才能公平比較。

## 訓練與實驗

- `train_mfn_a2c.py` / `evaluate_mfn_a2c.py`：主方法 MFN-A2C。
- `train_a2c_baseline.py` / `evaluate_a2c_baseline.py`：含兩種輸入、但不使用 MFN 的 A2C。
- `train_a2c_without_ti.py` / `evaluate_a2c_without_ti.py`：僅價格特徵的 A2C，對應實驗 1。
- `train_dqn_baseline.py` / `evaluate_dqn_baseline.py`：DQN 基準，對應實驗 2。
- `train_experiment3.py` / `evaluate_experiment3.py`：比較 DSR 與 portfolio-value reward，對應實驗 3。

## 建議執行順序

先檢查資料：

```powershell
python scripts\check_alignment.py
```

再從頭訓練及評估：

```powershell
python scripts\train_mfn_a2c.py
python scripts\evaluate_mfn_a2c.py

python scripts\train_a2c_baseline.py
python scripts\evaluate_a2c_baseline.py

python scripts\train_a2c_without_ti.py
python scripts\evaluate_a2c_without_ti.py

python scripts\train_dqn_baseline.py
python scripts\evaluate_dqn_baseline.py
```

`Wrapping the env in a DummyVecEnv.` 是 Stable-Baselines3 自動包裝單一環境的正常提示，不是錯誤。

時間軸修正後不可沿用舊模型、checkpoint 或結果 CSV，所有方法都必須重新訓練與評估。

## 結果整理

- `baseline_buy_hold.py`：四種幣等權重買入持有。
- `compare_all_methods.py`：實驗 1 的 MFN-A2C、A2C、A2C w/o TI、Buy-and-Hold。
- `compare_experiment2.py`：實驗 2 的 MFN-A2C、A2C、DQN、Buy-and-Hold。
- `compare_experiment3.py`：實驗 3 的四個模型/獎勵組合。
- `compare_mfn_buyhold.py`：只比較 MFN-A2C 與 Buy-and-Hold。

## 論文設定

- 4 小時資料、20 步觀察窗口、五種資產。
- A2C：兩層 `[64, 64]`、learning rate `7e-4`、gamma `0.99`、rollout `540`。
- DSR 更新率 `eta = 0.005`；正式訓練為 1,800,000 steps。
- 環境不含交易手續費與滑價，這是論文的實驗假設。

## 目前已驗證的開發結果

修正時間軸後，MFN-A2C 單次訓練 `100,000` 步的測試結果為：

| 指標 | 結果 |
|---|---:|
| Initial PV | 10,000.00 |
| Final PV | 13,558.43 |
| Total Return | 35.58% |
| Peak PV | 14,542.97 |
| Max Drawdown | -19.94% |
| Sharpe Ratio | 1.8268 |

這是流程驗證與單一 seed 的開發結果，不是最終論文數據。正式比較應固定資料切分、訓練步數、測試期間與 seed，至少執行 5 個 seeds，並報告平均值與標準差。
