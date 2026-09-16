# 程式碼用途導覽

本專案重現論文的 MFN-A2C 加密貨幣投資組合流程。每個決策點使用前 20 個 2 小時 K 線（40 小時）作為觀察值，將資金配置到 BTC、ETH、LTC、BNB 與 USDT。USDT 視為報酬率 0 的穩定資產。

## 核心模組

- `src/dsr.py`：全專案唯一的 EWMA DSR 實作；正式重現預設採封存程式的`paper_legacy`定義，`delta_A=A_new-A_old`、`delta_B=B_new-B_old`、`eta=0.005`；canonical innovation保留供消融，前5 steps更新統計量但回傳0。
- `src/training_diagnostics.py`：每個rollout記錄reward/DSR尺度、PV、return、turnover、配置entropy/集中度、Dirichlet concentration、deterministic權重變化、MFN參數/梯度、Actor/Critic loss與explained variance；MFN evaluate會自動讀取最新CSV並提示可能原因。
- `src/evaluation_metrics.py`：加入逐步 DSR、累積 DSR、配置比例與 turnover；表格名稱使用 Peak/Final Cumulative DSR。
- `src/portfolio_env_sb3.py`：Gymnasium 交易環境；MFN 使用 simplex 模式直接接收總和為 1 的投資權重，尚未遷移的模型可繼續使用舊 logits＋Softmax 模式；在 `Open[t]` 再平衡，根據 `Open[t]` 到 `Open[t+1]` 更新投資組合並回傳 reward。
- `src/simplex_policy.py`：MFN-A2C 專用 Dirichlet action distribution；以 Softplus 產生正的 concentration，訓練抽樣與 deterministic mean 都直接位於五資產 simplex，不再經過 `[-5,5]` 硬裁切或環境 Softmax。
- `src/mfn_sb3_extractor.py`：兩視角 MFN；5 個資產價格相對值與 20 個技術指標各經獨立 LSTM。DMAN 依式 4.5～4.8 對相鄰 hidden state 差分 `Δh_t` 做單層 Linear＋Softmax attention；MGM 依式 4.9～4.10 讓兩個單層 gate 只讀取 attended delta，並直接以 `tanh(attended)` 更新共享記憶，最後交給 A2C。
- `src/mfn_github_extractor.py`：目前 MFN 診斷版本；將原始 GitHub 三模態 MFN 改為價格與技術指標兩模態，保留前後 cell-state 串接、兩層 attention/candidate MLP，以及讀取 previous memory 的兩層 retention/update gates。
- `src/multi_epoch_a2c.py`：論文表 2 的 A2C 更新方式；每收集一個 540-step rollout，對同一份 rollout 重複執行 18 次標準 A2C optimizer update。
- `src/feature_scaling.py`：逐欄 z-score 標準化；只允許使用 Train 統計量，避免 Test 洩漏。
- `src/price_only_env.py`：實驗 1 的 A2C w/o TI，只保留價格變化特徵。
- `src/discrete_action_env.py`：DQN 專用包裝；使用 20% 權重網格，BTC/ETH/LTC/BNB 單一資產最高 60%，USDT 可達 100%，共 106 個動作。

## 資料流程

1. `download_binance_paper.py`：下載四種加密貨幣自 2018-01-01 至 2025-09-01 00:00（含）的 Binance 2 小時 K 線，共33,550筆原始資料並依時間戳合併。
2. `prepare_paper_features.py`：依式4.2建立 BTC、ETH、LTC、BNB、USDT 共5個Close price-relative特徵，並建立5資產×4指標共20個SMA/EMA/MACD/RSI特徵。MACD依封存原碼採`MACD_12_26_9`（DIF/MACD line）；USDT為中性常數，標準化後為0。保留最新33,524筆有效資料；前32,444筆為Train、最後1,080筆為Test。標準化只用Train擬合，Test套用同一組統計量。
3. `check_alignment.py`：驗證 train/test 的時間戳、原始價格及兩類特徵逐列對齊。
4. `src/experiment_periods.py`：集中定義2H間隔、Train/Test日期、20步lookback與預期筆數；本版本沒有Validation。

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

所有方法均使用完整32,444筆Train。GitHub-style雙模態MFN-A2C目前以300,000步測試5＋20新特徵；MFN的A2C rollout仍為540 steps，完整Train為一個episode，DSR moments會跨rollout延續。每100,000步保留診斷checkpoint，正式模型使用訓練結束時的最後狀態，不依Test挑選checkpoint。

`src/mfn_sb3_extractor.py`保留論文式4.5～4.10的簡化MFN作為對照；目前train/evaluate使用`GitHubStyleTwoViewMFN`。5＋20、paper DSR、300k模型儲存為`models/mfn_a2c_github2_5x20_300k_paper_dsr.zip`，checkpoint使用`MFN_A2C_GITHUB2_5X20_PAPER_DSR_E1_300K_*`前綴，避免覆蓋canonical模型。

資料切分、時間軸、DSR reward、特徵公式、特徵標準化或動作分布變更後，不可沿用舊模型、checkpoint或結果CSV。paper DSR現在是共享預設，因此MFN、A2C、A2C w/o TI與DQN若使用DSR reward都必須重新訓練；Buy-and-Hold只需重新評估。

目前依論文筆數切分：33,524筆有效2小時資料中，前32,444筆為Train、最後1,080筆為Test，沒有Validation。Test固定為2025-06-03 02:00至2025-09-01 00:00，前20根只作觀察，第一筆交易為2025-06-04 18:00 UTC。

## 結果整理

- `baseline_buy_hold.py`：依論文文字定義，BTC、ETH、LTC、BNB 期初各配置 25%，USDT 為 0%，買入後不再平衡；輸出 return、Sharpe、逐步 DSR、累積 DSR 與隨價格漂移的實際權重。
- `compare_all_methods.py`：實驗 1 的 MFN-A2C、A2C、A2C w/o TI、Buy-and-Hold。
- `compare_experiment2.py`：實驗 2 的 MFN-A2C、A2C、DQN、Buy-and-Hold。
- `compare_experiment3.py`：實驗 3 的四個模型/獎勵組合。

## 論文設定

- 2 小時資料、20 步（40小時）觀察窗口、五種資產。
- A2C：兩層 `[64, 64]`、learning rate `7e-4`、gamma `0.99`、rollout `540`。
- A2C 更新：論文設定為每個 rollout 重複 18 epochs；目前 GitHub-style MFN-A2C 的 300,000-step 特徵消融維持 1 epoch，以隔離資料前處理變因。`MultiEpochA2C` 沒有加入 PPO clipping。
- DSR統一由`src/dsr.py`計算；正式重現預設為`paper_legacy`，使用`delta_A=A_new-A_old`、`delta_B=B_new-B_old`與`eta=0.005`；前5 steps僅更新統計量，第6 step起產生DSR。canonical innovation模式保留供消融。
- `Peak Cumulative DSR` 與 `Final Cumulative DSR` 是累積值，不是單一步驟 DSR；目前 A2C warm-up 比較使用1,800,000 steps的最後模型。
- 環境不含交易手續費與滑價，這是論文的實驗假設。

目前正式設定已切回論文/封存程式的EWMA moment-change尺度。canonical模型、結果及checkpoint均保留但不可與paper DSR混用；所有使用DSR reward的正式模型都必須重新訓練。

## MFN 中斷後續訓

正常開始使用`python scripts\train_mfn_a2c.py`；中斷後可使用`python scripts\train_mfn_a2c.py --resume`。程式只搜尋目前5＋20、paper DSR、1 epoch、300k實驗前綴的週期checkpoint，選擇絕對步數最大者，並以`reset_num_timesteps=False`完成剩餘步數。若正式模型已存在，`--resume`不會重複訓練。

目前是實用型續訓：模型參數、optimizer及絕對步數可恢復，但SB3 checkpoint沒有保存環境所在列、未完成rollout、DSR moments與完整RNG狀態；恢復後Train與DSR會重新初始化。因此可避免從零訓練，但不宣稱與不中斷執行逐位元一致。每次恢復會建立含`from_已完成步數`的新診斷CSV。

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

這是舊資料頻率與切分下的流程驗證結果，不能與目前2小時流程直接比較。正式比較應固定Train/Test、訓練步數、測試期間與seed，至少執行5個seeds並報告平均值與標準差。
