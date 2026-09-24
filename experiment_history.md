# 實驗與程式修改歷史

> 最後更新：2026-09-15  
> 本文件整理本專案至今的重要程式修改、實驗設定與單次執行結果。強化學習結果會受隨機種子、資料版本及環境實作影響；除非明確標示為相同設定，表中的不同結果不可直接當作正式模型排名。

## 1. 研究目的與方法

本專案重現以 Memory Fusion Network（MFN）結合 Advantage Actor-Critic（A2C）的加密貨幣投資組合方法，並比較：

- MFN-A2C（Proposed method）
- 一般 A2C baseline
- A2C without Technical Indicators（A2C w/o TI）
- DQN baseline
- Buy-and-Hold

RL 模型配置 BTC、ETH、LTC、BNB、USDT 五種資產；USDT 視為報酬率為 0 的現金／避險資產。論文文字中的 Buy-and-Hold 則只投資四種加密貨幣，不含 USDT。

## 2. 目前採用的正式實驗設定

### 2.1 資料與切分

目前依論文所列筆數重建為 2 小時 K 線：

| 集合 | 期間（UTC） | 筆數 |
|---|---|---:|
| 原始下載資料 | 2018-01-01 00:00 ～ 2025-09-01 00:00 | 33,550 |
| 全部有效資料 | 2018-01-03 04:00 ～ 2025-09-01 00:00 | 33,524 |
| Train | 2018-01-03 04:00 ～ 2025-06-03 00:00 | 32,444 |
| Validation | 無 | 0 |
| Test | 2025-06-03 02:00 ～ 2025-09-01 00:00 | 1,080 |

- Test 固定使用完整有效資料的最後 1,080 筆。
- 1,080 根 2 小時 K 線約為 90 天，不是 180 天。
- 先使用 Test 前 20 根 K 線作為觀察，因此第一筆實際交易時間為 2025-06-04 18:00 UTC。
- 目前沒有 Validation，也不以 Test 選擇 checkpoint；正式模型使用固定步數訓練完成後的最後模型。

### 2.2 無未來資料洩漏的時間軸

目前環境統一使用以下決策順序：

```text
Observation：已完成的 K 線 [t-20, ..., t-1]
Decision：   在 Open[t] 產生新投資權重
Return：     使用 Open[t] → Open[t+1] 的價格變化
```

已檢查的重點：

- `pct_change()` 只使用當期與前一期資料。
- rolling 技術指標採 trailing window，沒有 `center=True`。
- observation 沒有 `shift(-1)` 或 future backfill。
- 特徵標準化只以 Train 的 mean/std 擬合，Test 只套用既有統計量。
- Test 不應反覆用來挑選超參數、步數或隨機種子，以免產生實驗層級的測試集洩漏。

### 2.3 共同環境與超參數

| 項目 | 目前設定 |
|---|---|
| 資產 | BTC、ETH、LTC、BNB、USDT |
| 初始投資組合價值 | 10,000 USDT |
| 資料頻率 | 2 小時 |
| Observation window | 20 根 K 線（40 小時） |
| Episode steps | 540 |
| Reward | Differential Sharpe Ratio（DSR） |
| DSR eta | 0.005 |
| DSR warm-up | 前 5 steps 回傳 0，但持續更新 EWMA moments |
| A2C learning rate | 7e-4 |
| Gamma | 0.99 |
| Seed | 123 |
| 交易成本／滑價 | 不計入 |

目前訓練步數尚未完全一致：

| 模型 | 腳本目前設定 |
|---|---:|
| MFN-A2C | 300,000 |
| A2C baseline | 1,800,000 |
| A2C w/o TI | 600,000 |
| DQN | 600,000 |

因此，現階段不可把四個模型的最新版結果當成完全公平的正式比較。若要製作最終論文表格，應先統一訓練步數並重新訓練。

## 3. 主要程式修改歷史

### 3.1 專案結構與檔名

- 將正式 MFN 腳本重新命名：
  - `scripts/train_formal.py` → `scripts/train_mfn_a2c.py`
  - `scripts/evaluate_formal.py` → `scripts/evaluate_mfn_a2c.py`
- 訓練、評估、比較程式集中於 `scripts/`。
- 核心環境、MFN extractor、DSR 與評估工具集中於 `src/`。
- 統一使用 `Path(__file__).resolve().parents[1]` 取得專案根目錄。
- 統一由 `data/`、`models/`、`results/`、`figures/`、`logs/` 與各 checkpoint 資料夾讀寫。
- README 與 CODE_GUIDE 已更新腳本名稱、執行方式、資料切分與研究限制。

### 3.2 資料處理

- 資料下載改為 Binance 2 小時 K 線，結束時間包含 2025-09-01 00:00 UTC。
- 依論文筆數保留最新 33,524 筆有效且對齊的資料。
- Train 固定 32,444 筆，Test 固定最後 1,080 筆，不設 Validation。
- 新增 `src/experiment_periods.py`，集中管理資料頻率、日期、筆數、lookback 與年化係數。
- 新增 `src/feature_scaling.py`，只用 Train 計算 z-score，避免 Test 統計量洩漏。
- MACD 特徵改為 `MACD / Close * 100`，避免 MACD 穿越零軸時用 `pct_change()` 造成極端值。
- 新增／更新 alignment 檢查，確保 raw、price-change 與 technical-indicator 資料逐列對齊。

### 3.3 交易環境與 PV

- 投資組合價值定義統一為：

  ```text
  V[t+1] = V[t] × Σ_i w_i[t] × (1 + r_i[t+1])
  ```

- 模型在 `Open[t]` 配置權重，使用 `Open[t] → Open[t+1]` 計算下一期投資組合價值。
- 權重包含 BTC、ETH、LTC、BNB 與 USDT，且總和為 1。
- USDT 報酬率視為 0。
- 評估步數使用 `len(test) - lookback - 1`，保留最後一次決策所需的 `Open[t+1]`。

### 3.4 DSR 統一

- 新增 `src/dsr.py`，讓 training environment、所有 evaluate 腳本與 Buy-and-Hold 共用相同 EWMA DSR 實作。
- `eta = 0.005`。
- 恢復原始實驗程式採用的 5-step warm-up：前 5 步更新統計量但 reward／DSR increment 為 0。
- 評估輸出中的 `Peak DSR`、`Final DSR` 已更名為：
  - `Peak Cumulative DSR`
  - `Final Cumulative DSR`
- 這兩項是逐步 DSR 的累積值，不是單一步驟的 DSR。

> 注意：論文公式描述 EWMA DSR，但專案封存的早期程式曾使用 expanding mean，且 reward 可能使用 cumulative DSR。現行主程式採用「論文公式的 EWMA + 原始程式的 5-step warm-up」，屬於目前選定的重現方式。

### 3.5 評估輸出

- 三支主要 evaluate 程式均加入：
  - Test data period 與 first trade time
  - Initial、Final、Peak PV
  - Total Return、Max Drawdown、Sharpe Ratio
  - Peak／Final Cumulative DSR
  - 各資產平均與最大配置比例
  - BTC+ETH 平均配置與高度集中步數比例
  - Average／Cumulative turnover
- 新增共同的 `src/evaluation_metrics.py`，避免各支程式的公式漂移。

### 3.6 Buy-and-Hold

- 依論文文字改為初始資金平均分配給四種加密貨幣：

  ```text
  BTC 25%、ETH 25%、LTC 25%、BNB 25%、USDT 0%
  ```

- 買入後不再平衡，實際權重會隨各幣價格漲跌而漂移。
- 與 RL 方法共用 Test、20-step lookback、第一筆交易時間及 DSR 計算。
- 補上 return、Sharpe、DSR、cumulative DSR 與漂移後配置比例。

### 3.7 DQN 動作空間

- Stable-Baselines3 DQN 只支援離散動作，因此加入離散投資組合包裝器。
- 權重網格間隔為 20%。
- 原始五資產完整網格共有 126 個動作。
- 為降低單一上漲幣種過度集中，BTC、ETH、LTC、BNB 各自最高限制為 60%；USDT 可到 100%。
- 套用限制後剩 106 個離散動作。

> 論文沒有完整說明 DQN 的動作離散方式與單幣上限，因此 60% 上限是目前專案的實作選擇，不能聲稱是論文原始設定。

### 3.8 模型、checkpoint 與紀錄

- 模型統一保存至 `models/`。
- TensorBoard 統一寫入 `logs/tensorboard/`。
- 每 100,000 steps 可保存診斷 checkpoint。
- 在目前「無 Validation」設定下，正式評估使用固定訓練步數的最後模型，不用 Test 挑最佳 checkpoint。
- `.gitignore` 曾調整為允許將指定的訓練模型提交至 Git；checkpoint 與 logs 仍建議排除。

### 3.9 論文公式版 DMAN 與 MGM

- 2026-09-13 將 DMAN 從舊版的前後 LSTM cell-state 串接，改成論文式 4.5 至 4.8 的 hidden-state difference：`Δh_t = concat(h_t^m - h_(t-1)^m)`。
- Attention 以論文式 4.7 的單一線性映射直接作用於兩模態的 `Δh_t`，不再使用 256 維的 `[c_(t-1), c_t]` 或額外的兩層 ReLU MLP。
- MGM 的 retention/update gates 改為式 4.9 的單一線性映射，只接收當期 attended delta，並依式 4.10 更新：`u_t = gamma1 * u_(t-1) + gamma2 * tanh(interaction_t)`。
- 開發訓練維持 600,000 steps，checkpoint 改用 `MFN_A2C_PAPER_*` 前綴。
- 此次修改改變模型參數形狀；修改前的 MFN 模型與 checkpoint 不可沿用，必須重新訓練。

### 3.10 每個 rollout 重複 18 次更新

- 2026-09-14 新增 `src/multi_epoch_a2c.py`，對應論文表 2 的 `Number of epochs for updating = 18`。
- 每收集 540 steps，對同一份 rollout 執行 18 次標準 SB3 A2C optimizer update；沒有改成 PPO，也沒有加入 clipping。
- 套用於 MFN-A2C、A2C baseline、A2C w/o TI，以及 Experiment 3 的 A2C 系列；DQN 不適用。
- 本次保持 learning rate、DSR reward 尺度及其他參數不變，以單獨檢驗 update epochs 的影響。
- 先前使用單次更新訓練的所有 A2C 系列模型均屬舊設定；正式比較前需要重新訓練。
- 600,000-step 論文公式版 MFN 在單次更新下的結果為 Final PV 12,377.10、Return 23.77%、Peak PV 13,416.90、Max Drawdown -12.24%、Sharpe 2.6408、Peak/Final Cumulative DSR 0.2059/0.0653；配置仍近似五資產等權。
- 同一版 MFN 改為每個 rollout 重複更新 18 次後，Final PV 11,969.79、Return 19.70%、Peak PV 13,383.71、Max Drawdown -13.06%、Sharpe 2.1957、Peak/Final Cumulative DSR 0.1966/0.0458。策略約 96.22% 時間為 BTC／ETH／LTC／USDT 各 25%、BNB 近 0%，其餘約 3.78% 時間 BNB 接近 100%，顯示直接重複完整 SB3 A2C 更新可能造成動作飽和。

### 3.11 恢復原始 MFN 的 candidate network 與 memory-conditioned gates

- 2026-09-14 參照原始 MFN repository，保留 Softmax attention，將單層 attention 改為兩層 MLP。
- 在 attended hidden-state difference 後新增兩層 candidate-memory MLP，不再把被 Softmax 縮小的 interaction 直接寫入共享記憶。
- Retention gate 與 update gate 改為兩層 MLP，輸入統一為 `[attended_delta, previous_memory]`，使閘門能依既有記憶決定保留與更新比例。
- 仍保留本論文改編版的兩模態 hidden-state difference；未改回原始情緒辨識 MFN 使用的前後 cell-state 串接。
- DSR、`eta=0.005`、rollout 540、18 update epochs、learning rate 與目前 600,000-step 開發設定均未改動。
- 此次修改再次改變 MFN 參數名稱與形狀；所有既有 MFN 模型及 checkpoints 均不可沿用，必須重新訓練。

### 3.12 分離 rollout 與 episode

- 2026-09-14 將 MFN-A2C 的開發訓練暫時改為 300,000 steps。
- 保留 `n_steps=540`、18 update epochs、`eta=0.005`、learning rate 7e-4、gamma 0.99 與 seed 123。
- 移除訓練環境的 `max_episode_steps=540` 與隨機起點；改為從 Train 開頭依時間順序走完整 32,444 筆資料後才結束 episode。
- 目的為避免把 540-step rollout 誤當成 episode，使 DSR EWMA moments 可跨相鄰 rollout 延續，而不是每 540 steps 歸零。
- 100,000／200,000／300,000 checkpoints 只供診斷策略何時飽和，不可使用 Test 選擇正式模型。

### 3.13 完整 Train episode 下改回論文 MFN

- 2026-09-14 將 MFN extractor 從 3.11 的原始 MFN 混合版改回學長論文式 4.5～4.10。
- DMAN 使用 hidden-state difference、單層 Linear attention 與 Softmax；MGM 的兩個單層 gate 只讀取 attended delta，並直接以 `tanh(attended)` 作為候選更新。
- 完整 Train episode、300,000 steps、540-step rollout、18 update epochs、`eta=0.005`、learning rate 7e-4、gamma 0.99 與 seed 123 全部維持不變，使本次與 3.11 的主要變因只有 MFN 架構。
- 3.11 混合版的 300,000-step 結果為 Final PV 13,076.21、Return 30.76%、Peak PV 13,806.44、Max Drawdown -12.69%、Sharpe 3.5686、Peak/Final Cumulative DSR 0.2496/0.1231；策略退化為固定 50% ETH＋50% USDT。
- 架構改變後既有 `mfn_a2c_formal.zip` 與 checkpoints 不可用於新程式，必須重新訓練。
- 新論文版 checkpoints 改用 `MFN_A2C_THESIS_*` 前綴，避免覆蓋 3.11 混合版留下的 `MFN_A2C_PAPER_300000_final.zip`。

### 3.14 MFN 改用真正的 simplex action distribution

- 2026-09-15 移除 MFN-A2C 的 Gaussian logits、`[-5,5]` Box 硬裁切與環境 Softmax，新增 `src/simplex_policy.py` 的 Dirichlet policy。
- Actor 輸出先經 Softplus 轉為正的 Dirichlet concentration；訓練時直接抽樣 simplex 權重，deterministic evaluation 使用 Dirichlet mean。兩者均非負且五項總和為 1。
- Dirichlet 沒有 Gaussian `log_std`；已移除 SB3 相容層產生的空 `log_std` 屬性，避免訓練紀錄出現無意義的 `std=NaN`。新版探索程度應改看 distribution entropy 與 concentration。
- `CryptoPortfolioEnv` 新增 `action_mode`：MFN train/evaluate 明確使用 `simplex`，環境只驗證 simplex 並修正浮點誤差；預設 `logits` 模式保留給尚未遷移的 A2C baseline／A2C w/o TI及其他既有實驗，DQN仍使用獨立離散環境。
- 此改動針對先前 Gaussian logits 碰到 `-5/+5` 後產生固定 25/25/25/25、單一資產近 100%，或固定兩資產 50/50 的飽和問題。
- MFN 架構仍為學長論文式 4.5～4.10；完整 Train episode、300,000 steps、rollout 540、18 update epochs、`eta=0.005`、learning rate 7e-4、gamma 0.99及seed 123均不變。
- 動作分布已改變，所有舊 MFN 模型與 checkpoints 都不能作為新版正式結果，必須重新訓練。舊檔可保留作實驗歷史。

## 4. 論文表格目標值

論文 Experiment 1 表格所列數據如下，供重現時比較：

| Method | Peak PV | Peak DSR | Final PV | Final DSR |
|---|---:|---:|---:|---:|
| Proposed method | 14,902 | 0.237 | 13,929 | 0.127 |
| A2C | 14,378 | 0.232 | 13,357 | 0.120 |
| A2C w/o TI | 13,120 | 0.225 | 11,959 | 0.112 |
| Buy-and-Hold | 11,861 | 0.130 | 10,431 | -0.030 |

論文同時存在兩項待釐清的描述矛盾：

1. 論文寫 4 小時 K 線、20 steps 為 80 小時；但 2018-01 至 2025-09 的 33,550 筆更接近 2 小時 K 線。
2. 論文稱 1,080 筆 Test 約 180 天；若為 2 小時資料，實際只有約 90 天，只有 4 小時資料才約 180 天。

因此，目前 2 小時版本是依「資料總筆數」重建，不代表已完全確定論文原始執行頻率。

## 5. 已執行的實驗結果

### 5.1 MFN-A2C 歷史結果

| 訓練步數 | Final PV | Return | Peak PV | Max Drawdown | Sharpe | 備註 |
|---:|---:|---:|---:|---:|---:|---|
| 100,000 | 13,558.43 | 35.58% | 14,542.97 | -19.94% | 1.8268 | 修正時間軸後的舊 4H 開發結果 |
| 600,000 | 13,821.15 | 38.21% | 14,780.09 | -18.96% | 1.9084 | 舊結果；曾接近論文，但與目前 2H／warm-up 設定不可直接混用 |
| 1,000,000 | 103,284.11 | 932.84% | 103,697.47 | -3.91% | 16.0160 | 已判定異常，來自舊環境／時間對齊問題，不可使用 |
| 600,000 | 12,336.44 | 23.36% | 13,372.16 | -12.28% | 2.6022 | 2H、5-step warm-up；舊 cell-state DMAN/MGM，配置近似五資產等權，已由論文公式版結構取代 |
| 600,000 | 11,969.79 | 19.70% | 13,383.71 | -13.06% | 2.1957 | 論文 hidden-difference MFN、短 540-step episode、18 updates；策略飽和為四資產各25%與偶發BNB近100% |
| 600,000 | 11,214.80 | 12.15% | 12,373.27 | -10.02% | 1.5503 | 原始MFN混合版、短540-step episode、18 updates；固定50% LTC＋50% USDT |
| 300,000 | 13,076.21 | 30.76% | 13,806.44 | -12.69% | 3.5686 | 原始MFN混合版、完整Train episode、18 updates；固定50% ETH＋50% USDT，Peak/Final Cum. DSR 0.2496/0.1231 |

100,000-step 的另一個較早結果曾為 Final PV 13,719.59、Return 37.20%、Peak PV 14,693.85、Max Drawdown -20.68%、Sharpe 1.8438；它同樣屬於舊流程記錄。

### 5.2 A2C baseline 歷史結果

| 資料／DSR 設定 | Steps | Final PV | Return | Peak PV | Max DD | Sharpe | Peak Cum. DSR | Final Cum. DSR |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| 舊流程 | 100,000 | 13,086.66 | 30.87% | 14,168.04 | -20.62% | 1.6397 | — | — |
| 舊流程 | 1,000,000 | 13,721.78 | 37.22% | 14,756.42 | -19.59% | 1.8824 | — | — |
| 舊 4H／600k | 600,000 | 13,483.44 | 34.83% | 14,516.20 | -20.30% | 1.7645 | 0.1430 | 0.0570 |
| 舊 4H／600k（調整後） | 600,000 | 13,156.20 | 31.56% | 13,889.28 | -23.28% | 1.6306 | 0.1688 | 0.0860 |
| 舊 4H／1.8M | 1,800,000 | 12,774.87 | 27.75% | 13,404.94 | -24.54% | 1.4360 | 0.1662 | 0.0941 |
| 2H、無 warm-up | 600,000 | 11,377.80 | 13.78% | 12,243.27 | -12.51% | 1.6553 | 0.2771 | 0.1377 |
| 2H、無 warm-up | 1,800,000 | 11,556.01 | 15.56% | 12,285.47 | -10.88% | 1.8988 | 0.3857 | 0.2549 |
| **目前 2H、5-step warm-up** | **1,800,000** | **14,882.13** | **48.82%** | **16,055.85** | **-12.61%** | **3.9599** | **0.2764** | **0.1448** |

目前 A2C 1.8M 單次結果的配置：

| 資產 | 平均權重 | 最大權重 |
|---|---:|---:|
| BTC | 14.25% | 56.58% |
| ETH | 33.36% | 71.99% |
| LTC | 22.95% | 80.29% |
| BNB | 14.96% | 58.42% |
| USDT | 14.48% | 33.19% |

- BTC+ETH 平均配置：47.61%。
- BTC+ETH 合計至少 80% 的步數：0.00%。
- Average turnover：9.18%。
- Cumulative turnover：97.24x。
- 此結果高於論文 A2C 的 Final PV 13,357 與 Final DSR 0.120，但測試行情、資料頻率及部分特徵處理不同，不能只用數值接近程度判定是否成功重現。

### 5.3 DQN 歷史結果

| 設定 | Steps | Final PV | Return | Peak PV | Max DD | Sharpe | Peak Cum. DSR | Final Cum. DSR |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| 舊 126-action 動作空間 | 1,000,000 | 16,156.72 | 61.57% | 17,049.02 | -19.44% | 2.6051 | — | — |
| 106 actions、單一 crypto 上限 60% | 600,000 | 14,080.78 | 40.81% | 15,519.77 | -21.92% | 1.9169 | 0.2298 | 0.1093 |

106-action DQN 的平均配置為 BTC 21.95%、ETH 14.62%、LTC 24.76%、BNB 17.73%、USDT 20.94%；BTC+ETH 平均為 36.56%，平均 turnover 46.21%，累積 turnover 489.40x。限制後已減少極端集中，但換手率仍遠高於 A2C；在不含交易成本的環境中，DQN 報酬可能因此偏樂觀。

此 DQN 結果產生於目前最後一次資料／DSR 修正之前，需重新訓練與評估後才能和最新版 A2C 比較。

### 5.4 Buy-and-Hold

目前 2H Test 區間、從第一筆實際交易時間進場時：

| Metric | Value |
|---|---:|
| Initial PV | 10,000.00 |
| Final PV | 13,011.44 |
| Total Return | 30.11% |
| Peak PV | 約 14,403.92 |
| Max Drawdown | 約 -15.17% |
| Sharpe Ratio | 約 2.6085 |

這個區間的 Buy-and-Hold 報酬顯著高於論文的 4.31%，代表論文表格很可能不是用目前 2025-06 至 2025-09 的相同行情，或資料頻率／進場點／價格欄位仍與原始實驗不同。

### 5.5 曾測試但已撤回的期間

曾為尋找接近論文 Buy-and-Hold Final PV 10,431 的區間，測試過或討論過：

- 2023-01-27 ～ 2023-07-26。
- 約 2023-02-06 ～ 2023-08-02 的實際持有區間，四幣等權報酬約 4.34%。
- 2019-04-24 ～ 2019-10-21，四幣等權報酬約 4.318%。
- 一度出現 2026-03-02 ～ 2026-08-28 的未來測試期間，已確認不符合原始資料收集範圍並撤回。

不能只因某段 Buy-and-Hold 剛好等於 4.31% 就認定它是論文原始 Test；還必須同時符合資料範圍、筆數、K 線頻率、lookback、進場價格以及所有 RL 方法的共同測試期間。

## 6. 目前判斷與已知限制

1. **最新版結果尚未齊全**：MFN 已改為論文架構＋Dirichlet simplex policy，尚未產生新版結果；其他模型也尚未全部在完全相同設定下重跑。
2. **訓練步數不一致**：目前 MFN 開發設定為300k、A2C為1.8M、A2C w/o TI與DQN為600k，不適合直接做正式排名。
3. **論文頻率敘述矛盾**：總筆數支持 2H，但文字、圖軸與 180 天 Test 支持 4H。
4. **DSR 實作仍有重現選擇**：目前採 EWMA step DSR reward + 5-step warm-up，未完全照搬封存原始環境的 expanding／cumulative reward 寫法。
5. **DQN 並非論文原始動作空間的確切復原**：106-action 與 60% 上限為合理化後的本專案設定。
6. **未計交易成本**：DQN 的高 turnover 尤其可能使無成本回測偏樂觀。
7. **單一 seed 不足以證明穩健性**：正式結果應至少執行 5 個 seeds，報告平均值與標準差。
8. **MFN 訓練較慢**：目前 extractor 以 Python 迴圈逐一處理 20 個時間步，兩個 `LSTMCell` 及注意力／記憶閘門都重複執行；直接改成批次 `nn.LSTM` 可能加速，但若改用 hidden-state differences，模型語義會改變，舊模型與新模型不可直接比較。

## 7. 後續正式實驗建議

在製作最終論文表格前，建議依序完成：

1. 最終決定採 2H 還是 4H，並在論文中解釋原文矛盾。
2. 將 MFN-A2C、A2C、A2C w/o TI、DQN 的訓練步數統一。
3. 刪除或封存舊設定產生的模型與 CSV，避免 evaluate 誤讀。
4. 對每個模型使用相同 Train、Test、lookback、reward、warm-up、seed 與訓練步數重新訓練。
5. 先執行 Buy-and-Hold，再依序評估所有模型並產生比較表。
6. 至少執行 5 個 seeds，正式報告 mean ± standard deviation。
7. 除無交易成本的論文重現表外，可額外補一組含手續費／滑價的現實性測試。

## 8. 目前執行指令

```powershell
# 資料與對齊
python scripts\download_binance_paper.py
python scripts\prepare_paper_features.py
python scripts\check_alignment.py

# MFN-A2C
python scripts\train_mfn_a2c.py
python scripts\evaluate_mfn_a2c.py

# A2C baseline
python scripts\train_a2c_baseline.py
python scripts\evaluate_a2c_baseline.py

# A2C without technical indicators
python scripts\train_a2c_without_ti.py
python scripts\evaluate_a2c_without_ti.py

# DQN
python scripts\train_dqn_baseline.py
python scripts\evaluate_dqn_baseline.py

# Buy-and-Hold 與比較
python scripts\baseline_buy_hold.py
python scripts\compare_all_methods.py
python scripts\compare_experiment2.py
python scripts\compare_experiment3.py
```

## 9. 結果使用規則

- 任何資料期間、頻率、特徵公式、標準化、時間對齊、DSR、warm-up 或動作空間的變更，都必須重新訓練受影響的模型。
- 舊模型即使能成功載入，也不代表可以和新結果公平比較。
- `Wrapping the env with a Monitor wrapper` 與 `Wrapping the env in a DummyVecEnv` 是 Stable-Baselines3 的正常訊息，不是錯誤。
- 正式表格應同時保留：程式 commit、資料日期、模型步數、seed、模型檔名及結果 CSV，以便完整追溯。

## 10. 2026-09-15：MFN update epochs 18 → 1 消融實驗

- 目的：檢查每個 540-step rollout 重複更新 18 次，是否造成 Actor 的 Tanh 飽和、Dirichlet concentration 極端化及全測試期固定配置。
- 僅將 `scripts/train_mfn_a2c.py` 的 `UPDATE_EPOCHS` 從 18 改為 1。
- 保留 300,000 timesteps、完整 chronological Train episode、`n_steps=540`、learning rate `7e-4`、gamma `0.99`、DSR `eta=0.005`、seed 123、論文式 MFN 與 Dirichlet simplex policy。
- 新 checkpoint 前綴為 `MFN_A2C_THESIS_E1_*`，最終 checkpoint 為 `MFN_A2C_THESIS_E1_300000_final.zip`，避免覆蓋 18-epoch checkpoint。
- `models/mfn_a2c_formal.zip` 仍作為目前 evaluate script 的預設正式模型；重新訓練完成後會更新為本次 1-epoch 模型。
- 此設定是診斷用消融實驗，不是論文表格所列的 18-epoch 正式設定；改變 update epochs 後必須從頭訓練。
- 300,000-step、1-epoch 評估結果：Final PV 12,138.96、Return 21.39%、Peak PV 13,177.15、Max Drawdown -11.69%、Sharpe 2.5231、Peak/Final Cumulative DSR 0.1960/0.0559、Cumulative Turnover 4.44x。
- 平均配置為 BTC 24.44%、ETH 17.68%、LTC 18.58%、BNB 17.97%、USDT 21.32%；相較 18 epochs 的固定 BTC/LTC 配置，極端策略塌縮已解除，但配置仍偏平滑。
## 11. 2026-09-15：1 epoch 診斷延伸至 600,000 steps

- 將 `TOTAL_TIMESTEPS` 從 300,000 提高為 600,000，`UPDATE_EPOCHS` 維持 1。
- `n_steps=540`、完整 chronological Train episode、learning rate `7e-4`、gamma `0.99`、DSR `eta=0.005`、seed 123、論文式 MFN 與 Dirichlet simplex policy 均不變。
- 週期 checkpoint 前綴改為 `MFN_A2C_THESIS_E1_600K_*`，最終 checkpoint 為 `MFN_A2C_THESIS_E1_600000_final.zip`，保留既有 300,000-step 實驗檔案。
- 本輪用來判斷 300,000 steps 是否更新不足；不得與 18-epoch 結果混稱為相同訓練設定。
- 600,000-step、1-epoch 評估結果：Final PV 12,130.19、Return 21.30%、Peak PV 13,181.69、Max Drawdown -11.72%、Sharpe 2.4758、Peak/Final Cumulative DSR 0.1959/0.0561、Cumulative Turnover 5.97x。
- 平均配置為 BTC 23.41%、ETH 16.38%、LTC 20.49%、BNB 19.42%、USDT 20.30%。相較 300,000-step 結果幾乎沒有改善，顯示目前論文公式版 MFN 已收斂至偏平均、低動態配置，訓練步數不是主要限制。

## 12. 2026-09-15：完整 GitHub-style 雙模態 MFN

- 新增 `src/mfn_github_extractor.py` 的 `GitHubStyleTwoViewMFN`，保留 `src/mfn_sb3_extractor.py` 的論文公式版作為對照。
- 價格與技術指標各使用一個獨立 `LSTMCell`；DMAN 將兩模態的 previous/current cell states 串接為 `cStar`。當 `lstm_hidden=64` 時，`cStar` 為 256 維。
- Attention 使用兩層 MLP＋ReLU＋Dropout＋Softmax；attended `cStar` 再經兩層 candidate MLP 產生128維候選記憶。
- Retention/update gates 均為兩層 MLP，輸入為 `[attended, previous_memory]`，依原始 MFN 形式更新 shared memory。
- 最終將 price hidden 64、indicator hidden 64與shared memory 128直接串接成256維特徵，交給SB3 A2C的Actor/Critic `[64,64]`；不保留原始情緒預測用的scalar output head。
- Attention、candidate與gate hidden units固定為64，dropout為0；這是因學長最終版本遺失而採用的明確重建選擇，不宣稱是論文未記載的原始數值。
- 本輪設定為300,000 steps、`n_steps=540`、1 update epoch、learning rate `7e-4`、gamma `0.99`、DSR `eta=0.005`、完整Train episode、Dirichlet simplex policy與seed 123。
- 模型輸出為 `models/mfn_a2c_github2.zip`；checkpoint 前綴為 `MFN_A2C_GITHUB2_E1_*`，最終checkpoint為`MFN_A2C_GITHUB2_E1_300000_final.zip`。評估CSV使用`mfn_github2_backtest_results.csv`與`mfn_github2_metrics.csv`，避免覆蓋論文公式版結果。
- 新增結構、維度、forward與梯度測試。
- 300,000-step 評估結果：Final PV 12,251.99、Return 22.52%、Peak PV 13,283.83、Max Drawdown -11.74%、Sharpe 2.5901、Peak/Final Cumulative DSR 0.1995/0.0603、Cumulative Turnover 10.63x。
- 平均配置為 BTC 22.95%、ETH 17.83%、LTC 19.59%、BNB 19.07%、USDT 20.56%。相較同為300,000 steps、1 epoch的論文公式版，Return提升1.13個百分點、Sharpe提升0.0670，且配置動態增加，但單一seed尚不足以證明穩定優勢。

## 13. 2026-09-15：GitHub-style MFN 延伸至 600,000 steps

- 將 `TOTAL_TIMESTEPS` 從300,000提高至600,000；`UPDATE_EPOCHS=1`及其他資料、環境、DSR、Dirichlet、A2C與MFN參數全部不變。
- 週期checkpoint前綴改為`MFN_A2C_GITHUB2_E1_600K_*`，最終checkpoint為`MFN_A2C_GITHUB2_E1_600000_final.zip`，保留300,000-step checkpoint。
- `models/mfn_a2c_github2.zip`仍是evaluate預設模型，完成600,000-step訓練後會更新；正式結果待重新評估後補入。

## 14. 2026-09-15：依學長論文調整為 5＋20 特徵，先測試 300,000 steps

- 價格模態由4幣×OHLC共16維，改為BTC、ETH、LTC、BNB、USDT各一個Close-to-Close price relative，共5維；USDT price relative固定為1，Train-only z-score後為0。
- 技術指標模態由16維改為5資產×SMA-20、EMA-20、MACD、RSI-14，共20維。四種加密貨幣改用指標level，不再對SMA/EMA取`pct_change()`；RSI不再預先中心化；MACD不再除以Close。
- MACD採`pandas_ta`的`MACD_12_26_9`，即DIF/MACD line。原因是封存原碼明確選用此欄位，且26期warm-up與論文33,550減為33,524筆相符；論文文字所寫九期signal line仍屬待確認歧義。
- USDT的SMA、EMA、MACD、RSI使用中性常數1、1、0、50，標準化後均為0，不向模型提供虛假趨勢。
- 保留逐欄Train-only z-score：只用32,444筆Train擬合mean/std，1,080筆Test完全不參與fit。
- 新增`src/feature_schema.py`作為5＋20維度的單一來源，並同步`CryptoPortfolioEnv`、兩種MFN extractor、A2C w/o TI、Experiment 3及測試。
- `scripts/train_mfn_a2c.py`改為300,000 steps、`UPDATE_EPOCHS=1`；learning rate、gamma、rollout、DSR、完整Train episode、Dirichlet simplex與seed均不變，以隔離資料前處理變因。
- 新模型為`models/mfn_a2c_github2_5x20_300k.zip`；checkpoint前綴為`MFN_A2C_GITHUB2_5X20_E1_300K_*`；評估CSV也使用`mfn_github2_5x20_300k_*`名稱，避免覆蓋舊16＋16實驗。
- 重新產生資料後驗證：總有效33,524、Train 32,444、Test 1,080；price shape為5、technical shape為20，timestamp alignment、有限值及USDT中性欄位均通過。
- 完整23項unittest通過；實際GitHub-style MFN＋Dirichlet smoke test得到`(20,25)` observation、合法sum-to-one action與finite reward。
- 因輸入維度及特徵公式已改變，所有舊MFN、A2C、A2C w/o TI與DQN模型均不可搭配新CSV作正式比較；各方法必須以新資料重新訓練。

## 15. 2026-09-15：修正 DSR innovation 定義並加入訓練診斷

- 修正共用的 `src/dsr.py`，將 DSR 的 innovation 定義為 `delta_A = r - A_old`、`delta_B = r**2 - B_old`，再以 `A_new = A_old + eta * delta_A`、`B_new = B_old + eta * delta_B` 更新 EWMA moments。
- `eta=0.005` 現在只控制一階與二階矩的更新速度，不再額外把送入 A2C 的 DSR reward 縮小約 200 倍；DSR 公式使用更新前的 `A_old`、`B_old` 與 variance。
- 保留 `warmup_steps=5` 與數值穩定檢查；訓練環境及所有評估程式仍共用同一個 DSR 實作，避免公式漂移。
- 這項修改改變 reward 尺度與訓練目標，因此 MFN-A2C、A2C、A2C w/o TI 與 DQN 的既有模型都不能當作新設定的正式結果，必須重新訓練；Buy-and-Hold 不需訓練，但需重新評估 DSR。
- MFN 300,000-step 新模型命名為 `models/mfn_a2c_github2_5x20_300k_canonical_dsr.zip`，checkpoint 使用 `MFN_A2C_GITHUB2_5X20_CANONICAL_DSR_E1_300K` 前綴，避免覆蓋舊尺度模型。
- 新增 `src/training_diagnostics.py`，每個 rollout 記錄 reward、portfolio return、DSR、DSR variance、PV、turnover、資產配置、Dirichlet concentration、Critic explained variance、loss，以及 MFN gradient/parameter norm。
- 診斷 CSV 儲存在 `logs/training_diagnostics/`；同時將主要診斷值寫入 TensorBoard。`evaluate_mfn_a2c.py` 會自動讀取最新一份相符紀錄，列出可能原因。
- 自動判讀涵蓋：reward 太弱、reward 尖峰、Critic 未學到、MFN 梯度過小、近平均配置、配置塌縮、策略對不同狀態反應太小，以及換手率過高。
- 已完成 64-step 實際 MFN smoke training，確認診斷 CSV 可產生且主要欄位為有限值；正式 300,000-step 結果尚待重新訓練與評估。

## 16. 2026-09-15：新增 MFN `--resume` 中斷續訓

- `python scripts\train_mfn_a2c.py --resume`會自動尋找目前canonical-DSR、5＋20、1-epoch、300k實驗前綴中絕對步數最大的週期checkpoint。
- 從checkpoint的`model.num_timesteps`計算剩餘步數，並以`reset_num_timesteps=False`延續SB3計步與TensorBoard橫軸；不相容的舊MFN或舊DSR checkpoint不會被載入。
- 恢復訓練會建立`from_已完成步數`命名的新診斷CSV，避免覆寫中斷前的紀錄；完成模型已存在時不會重複訓練。
- 現階段checkpoint可保存模型與optimizer，但不保存環境資料位置、未完成rollout、DSR moments及完整RNG狀態；因此屬實用型續訓，不是逐位元一致的無縫恢復。正式研究應記錄是否曾resume。

## 17. 2026-09-16：正式重現切回 paper-style DSR

- `src/dsr.py`新增`paper_legacy`與`canonical`兩種明確模式；共享預設切回`paper_legacy`。
- paper模式保留`eta=0.005`與5-step warm-up，但DSR numerator使用`delta_A=A_new-A_old`、`delta_B=B_new-B_old`，重現封存程式約為canonical乘上eta的尺度。
- canonical模式仍保留供理論對照與消融，不刪除既有模型或結果。
- MFN正式模型改名為`models/mfn_a2c_github2_5x20_300k_paper_dsr.zip`；checkpoint、診斷CSV及結果CSV同步使用`PAPER_DSR`/`paper_dsr`名稱，避免續訓或評估時誤載canonical模型。
- 這是reward尺度與訓練目標變更，不能從canonical checkpoint使用`--resume`；必須重新開始paper DSR訓練。

## 18. 2026-09-16：建立四模型共用公平比較設定

- 新增含中文註釋的`src/experiment_config.py`，集中管理資料路徑、Train/Test環境、300,000環境步、seed123、learning rate、gamma、paper DSR、初始資產與checkpoint週期。
- MFN-A2C、A2C baseline、A2C without TI統一使用SB3 Gaussian logits＋環境Softmax、`n_steps=540`及`UPDATE_EPOCHS=1`；MFN僅多出MFN特徵擷取器，w/o TI僅移除技術指標。
- 四種方法都改為完整chronological Train：`random_start=False`且不把540-step rollout誤當episode。
- DQN共用相同資料、步數、seed與DSR，但保留replay buffer、探索率及離散動作等專屬設定；`UPDATE_EPOCHS`不適用於DQN。
- 修正DQN wrapper已輸出simplex權重後，底層環境又Softmax一次的問題；DQN底層環境現在使用`action_mode="simplex"`直接執行離散配置。
- 新模型與結果使用`fair_300k_paper_gaussian_e1`run tag，舊模型不覆蓋；Experiment 1與2比較腳本改讀新結果。
- MFN診斷擴充為支援Gaussian policy，記錄deterministic權重變化、logit絕對值與碰觸±5邊界的比例，用來辨識Softmax配置飽和。

## 19. 2026-09-16：學長原碼技術指標前處理消融

- 新增`src/technical_indicators.py`，以中文註釋集中實作與測試學長封存原碼的技術指標轉換。
- 四種加密貨幣的SMA-20、EMA-20與MACD(DIF)改為`pct_change(fill_method=None) × 100`；RSI-14改為`(RSI-50) × 0.1`。
- MACD仍取封存原碼的`MACD_12_26_9` DIF／MACD line，不改成九期signal line，避免同時改變兩個變因。
- 技術指標不再做z-score；價格模態仍保留Train-only z-score，使本輪主要隔離「技術指標前處理」的影響。
- 學長資料產生器只有四種加密貨幣；目前5＋20架構所需的四個USDT技術指標固定補0，不提供虛假訊號。
- 訓練步數維持600,000，其他資料期間、DSR、Gaussian＋Softmax、A2C epoch與seed不變。
- 新run tag為`fair_600k_paper_gaussian_e1_senior_ta`；四種模型、結果及checkpoint都使用此標籤，避免覆蓋先前level＋z-score實驗。
- 新增技術指標公式與USDT中性欄位的回歸測試。資料重新產生後，所有受技術指標影響的模型都必須重新訓練；A2C without TI理論上不受此變更影響，但若要留下完整同批次實驗紀錄仍可使用新標籤重跑。
- 重新產生資料後確認Train/Test分別為32,444/1,080筆、20個技術欄、無NaN/Inf且timestamp alignment通過。由於MACD會穿越0，`MACD.pct_change() × 100`在Train產生數百萬等級尖峰；這是封存原碼公式的固有問題，本輪不裁切並由資料腳本主動警告。

## 20. 2026-09-16：正式流程恢復指標 level＋Train-only z-score

- `senior_ta` A2C baseline 600,000步結果為Final PV 12,334.61、Return 23.35%、Peak PV 13,241.62、Max Drawdown -12.65%、Sharpe 2.5734、Peak/Final Cumulative DSR 0.1859/0.0548。
- 平均配置接近五資產各20%，但Cumulative Turnover達94.05x；相較前一版level＋z-score的Return 27.84%與Turnover 17.28x，報酬下降且交易明顯更不穩定。
- 判斷主要原因是`MACD.pct_change() × 100`在零點附近產生最高約6,742,828.88的極端值，使MACD尺度壓過其他指標。
- 正式資料流程因此恢復SMA、EMA、MACD(DIF)、RSI原始level，再對價格與技術指標使用Train-only z-score；Test不參與mean/std擬合。
- 學長原碼函式、測試與結果保留作為消融證據，不刪除；目前新run tag為`fair_600k_paper_gaussian_e1_level_zscore`，避免覆蓋`senior_ta`實驗。

## 21. 2026-09-16：paper-style DSR training reward 放大200倍

- 保留`eta=0.005`、5-step warm-up、EWMA moments與paper-style DSR公式不變。
- 新增共用`DSR_REWARD_SCALE=200.0`；環境僅對送入A2C／DQN的training reward計算`reward=200×raw DSR`。
- `info["DSR"]`、評估程式重新計算的逐步DSR、Peak/Final Cumulative DSR全部維持未縮放的論文尺度，避免表格數字被錯誤放大200倍。
- 四種模型共用相同倍率；A2C update epochs仍為1、learning rate仍為`7e-4`，其餘資料、架構、步數與seed不變，以隔離reward尺度變因。
- 新run tag為`fair_600k_paper_dsr200_gaussian_e1_level_zscore`，舊倍率模型、結果與checkpoint不覆蓋。
- 新增回歸測試，逐步驗證環境回傳reward等於`200×info["DSR"]`，同時保留raw DSR供評估。

## 22. 2026-09-16：切換為學長封存 expanding DSR＋cumulative reward

- 新增`legacy_expanding` DSR模式，不刪除既有`paper_legacy` EWMA與`canonical`模式。
- 每一步先用當期以前的全部episode報酬計算`A=mean(r)`與`B=mean(r²)`，再依封存公式計算`Dt`並回傳`eta × Dt`；`eta=0.005`與5-step warm-up維持不變。
- 訓練reward改為截至當步的cumulative DSR，重現封存環境`reward=self.cumDSR`；前一輪`DSR_REWARD_SCALE=200`停用並恢復為1。
- `info["DSR"]`保留單步值，`info["cumulative_DSR"]`記錄累積值；A2C、MFN-A2C、A2C w/o TI、DQN共用同一設定。
- A2C、A2C w/o TI、DQN及MFN正式evaluate都明確傳入環境的DSR formula；Buy-and-Hold也改用共用legacy公式，避免表格算法漂移。
- 新run tag為`fair_600k_legacyexp_cumdsr_gaussian_e1_level_zscore`，舊EWMA、DSR×200模型與結果不覆蓋，且不可用`--resume`混接。
- 完整37項測試通過，包含封存公式數值回歸與環境reward等於cumulative DSR的逐步驗證。
- 不重新訓練、只對上一輪A2C回測報酬重算legacy DSR時，Peak/Final Cumulative DSR為0.2830/0.1548；同區間Buy-and-Hold為0.2789/0.1491。這證明DSR算法本身會明顯改變表格數字，但不會改變既有PV軌跡。

## 23. 2026-09-17：legacy expanding＋cumulative reward實驗結果與正式設定復原

- A2C baseline以600,000 steps、level＋Train-only z-score、Gaussian＋Softmax、1 update epoch、`eta=0.005`、legacy expanding DSR及cumulative DSR reward重新訓練。
- 評估結果：Initial PV 10,000.00、Final PV 12,002.36、Return 20.02%、Peak PV 12,661.61、Max Drawdown -10.35%、Sharpe 2.5549、Peak/Final Cumulative DSR 0.2458/0.1356。
- 平均配置為BTC 28.31%、ETH 9.82%、LTC 2.86%、BNB 44.33%、USDT 14.68%；BTC＋ETH平均38.13%，Average/Cumulative Turnover為0.82%/8.70x。
- 與論文A2C的Peak/Final DSR 0.232/0.120已相當接近，但Peak/Final PV仍比論文14,378/13,357低1,716.39/1,354.64，顯示DSR計算方式能解釋DSR尺度，不能單獨解釋PV差異。
- 與目前同區間Buy-and-Hold約30.11%相比，legacy A2C報酬低約10.09個百分點；主要因模型平均只配置ETH 9.82%，卻配置BNB 44.33%與BTC 28.31%，錯過Test期間ETH約66.67%的漲幅。
- TensorBoard `A2C_59`顯示Critic不穩：explained variance多數接近0或為負，最低-6.6726、最後-0.0065；value loss約6.52～4,224.95，policy loss約-452.77～343.50。cumulative reward使過去DSR在後續每一步重複計入，造成非平穩reward與信用分配困難。
- 此實驗證實legacy公式可使DSR表格尺度接近論文，但不利目前A2C的PV與訓練穩定性，因此保留為消融，不作為目前正式設定。
- 正式共用設定恢復到本實驗之前：`DSR_FORMULA=paper_legacy` EWMA moment changes、`DSR_REWARD_MODE=step`、`DSR_REWARD_SCALE=200`、`eta=0.005`。run tag恢復為`fair_600k_paper_dsr200_gaussian_e1_level_zscore`。
- `legacy_expanding`實作、回歸測試、模型、結果與checkpoint全部保留，沒有刪除；不得用legacy checkpoint接續目前EWMA DSR×200訓練。

## 24. 2026-09-17：加入獨立Validation與最佳模型選擇

- 保留最後1,080筆Test完全不動，從原32,444筆Train尾端切出1,080筆Validation；新切分為Train 31,364、Validation 1,080、Test 1,080。
- 時間範圍：Train為2018-01-03 04:00至2025-03-05 00:00，Validation為2025-03-05 02:00至2025-06-03 00:00，Test為2025-06-03 02:00至2025-09-01 00:00（UTC）。
- z-score的mean/std改為只使用31,364筆Train擬合；Validation與Test都只套用Train統計量，避免特徵前處理洩漏。
- 四種方法每50,000環境步完整回測一次Validation，依Validation Final PV保存最佳模型；同時記錄episode reward與mean step reward到`logs/validation/`。
- `models/<model_name>.zip`代表Validation最佳模型；訓練到600,000步的最後狀態另存為`models/<model_name>_final.zip`，正式Test評估預設讀取最佳模型。
- 新run tag加入`val1080_pv`，避免與先前沒有Validation、使用完整32,444筆Train的模型混用。這是資料切分與Scaler變更，四種模型都必須重新訓練。

## 25. 2026-09-17：改為兩階段訓練、100k Validation與300k上限

- Stage 1使用31,364筆Train，最多訓練300,000步；Validation頻率由50,000改為100,000，因此候選步數為100k、200k、300k。
- 以1,080筆Validation的Final PV選出最佳訓練步數，不直接沿用Stage 1權重。
- Stage 2以相同seed重新初始化模型，使用Train＋Validation共32,444筆Development，訓練Stage 1選出的步數；正式Test只評估Stage 2模型。
- 建立兩套無洩漏Scaler：Stage 1的Train-only scaler供Train/Validation使用；Stage 2的Development-only scaler供Development/Test使用。Test均不參與mean/std擬合。
- 四種方法皆使用相同兩階段規則。MFN的`--resume`會分辨Stage 1與Stage 2 checkpoint，避免跨階段或跨Scaler續訓。
- 新run tag為`twostage_300k_paper_dsr200_gaussian_e1_level_zscore_val1080_pv100k`，不與先前單階段600k模型混用。
- 實際計算量為Stage 1的300k，加Stage 2的100k／200k／300k，因此總環境互動為400k～600k。

## 26. 2026-09-17：Hybrid-50 reward消融

- 保留兩階段流程、Stage 1上限300,000步、每100,000步Validation、Stage 2重新初始化、seed123、A2C epoch1及所有資料／模型架構不變。
- 將訓練reward由純`200×DSR`改為`200×step DSR + 50×log(1+portfolio_return)`，使訓練目標同時重視風險調整收益與Final PV。
- `eta=0.005`與5-step warm-up不變；評估輸出的逐步DSR、Peak/Final Cumulative DSR仍是未乘200的原始尺度。
- 環境info新增`log_portfolio_return`、`scaled_DSR_reward`與`scaled_return_reward`；MFN診斷CSV分別記錄兩個reward分量，以便判斷哪一項主導訓練。
- 新run tag為`twostage_300k_hybrid_dsr200_ret50_gaussian_e1_level_zscore_val1080_pv100k`，不覆蓋Pure DSR兩階段模型與結果。
- Hybrid-50屬改良方法／消融實驗，不應宣稱為論文原始純DSR設定。模型必須重新訓練，不能從Pure DSR checkpoint續訓。

## 27. 2026-09-17：Hybrid-100 reward消融

- 將log-return倍率由50提高為100；正式訓練reward改為`200×step DSR + 100×log(1+portfolio_return)`。
- DSR倍率200、`eta=0.005`、5-step warm-up、兩階段300k流程、每100k Validation與其他超參數全部不變，以隔離portfolio growth權重的影響。
- 新run tag為`twostage_300k_hybrid_dsr200_ret100_gaussian_e1_level_zscore_val1080_pv100k`，保留Hybrid-50與Pure DSR模型、結果及checkpoint。
- 依Hybrid-50 Test軌跡事後估算，Hybrid-100的淨累積reward約為DSR 35.8%、log return 64.2%；這只是尺度參考，正式比較仍應以新模型的Validation/Test及多seed結果判斷。

## 28. 2026-09-17：恢復Hybrid-50、Window 40與可解釋性診斷

- 正式訓練reward由Hybrid-100改回`200×step DSR + 50×log(1+portfolio_return)`；`eta=0.005`、5-step warm-up、兩階段300k流程、每100k Validation與A2C update epoch 1維持不變。
- 歷史觀察窗由20根提高為40根2小時K線，即80小時；Train、Validation、Test與Buy-and-Hold共用相同lookback，Test第一筆交易改為2025-06-06 10:00 UTC。
- 新run tag為`twostage_300k_hybrid_dsr200_ret50_win40_gaussian_e1_level_zscore_val1080_pv100k`。因observation shape已改變，Window-20模型與checkpoint不可載入或續訓，四種方法需重新訓練後再公平比較。
- A2C baseline的Stage 1與Stage 2均接上`TrainingDiagnosticsCallback`，每個540-step rollout保存Actor policy loss、Critic value loss、explained variance、總reward及DSR/log-return兩分量、PV、turnover、配置entropy／HHI、各資產平均與最大權重、deterministic配置變化與Gaussian logit裁切比例。
- 四支正式model evaluator新增逐資產歸因：`asset_return_*`保存資產當期報酬，`return_contribution_*`保存`weight_i×return_i`，`pnl_contribution_*`保存交易前PV乘上該報酬貢獻。
- 回測終端會列出BTC、ETH、LTC、BNB、USDT各自標的複利漲跌幅、累積PnL、占總PnL比例與簡單報酬貢獻；PnL歸因總和精確核對`Final PV - Initial PV`。
- 新增回歸測試驗證每一步五資產return contribution等於portfolio return，且完整期間PnL contribution總和等於PV變化；完整39項測試通過。

## 29. 2026-09-17：改為完整Train單階段300k與Window 20

- 四種正式方法統一取消Validation checkpoint選擇與Stage 2，直接在論文原始的32,444筆完整Train（程式中的`development`切分）依時間順序訓練。
- 固定訓練300,000 steps，最後模型直接作為正式Test模型；最後1,080筆Test仍完全隔離，不參與Scaler擬合、訓練或模型選擇。
- 歷史觀察窗由40根改回20根2小時K線，即40小時；Test第一筆交易由2025-06-06 10:00 UTC改回2025-06-04 18:00 UTC。
- Hybrid-50 reward、`eta=0.005`、5-step warm-up、Gaussian logits＋Softmax、A2C rollout 540、update epoch 1與其他超參數維持不變。
- 新run tag為`fulltrain_300k_hybrid_dsr200_ret50_win20_gaussian_e1_level_zscore`，不覆蓋先前two-stage／Window-40模型、checkpoint、結果與診斷紀錄。
- MFN `--resume`保留，但只尋找此單階段run tag相容的checkpoint並接續至總計300,000步，不會讀取舊Stage 1或Stage 2 checkpoint。

## 30. 2026-09-17：加入探索、Critic、reward衝突與相對強勢診斷

- `TrainingDiagnosticsCallback`新增逐資產Gaussian `log_std/std`、抽樣配置與deterministic配置的half-L1距離、兩種配置的turnover與entropy，用來判斷訓練探索是否遠比部署策略吵雜。
- 新增Critic／Advantage診斷：value prediction與return target的mean/std、相關係數、RMSE，以及Advantage mean/std/absolute mean/p01/p99。
- 新增Hybrid reward診斷：DSR與log-return分量的absolute magnitude占比、相關係數與符號衝突比例。
- A2C baseline與A2C w/o TI均輸出上述rollout診斷；MFN另保留特徵擷取器梯度。修正無參數`FlattenExtractor`被誤報為MFN零梯度的問題。
- 四支正式evaluate加入6／12／36／84步（12小時／1日／3日／7日）的相對強勢配置診斷。Trailing return先`shift(1)`再rolling，確保不含當期未來報酬；下一期winner exposure明確標示為事後解釋，不作模型輸入。
- 現有A2C 300k回測顯示權重與相對強勢只有弱正相關：12h／1d／3d／7d rank correlation約0.060／0.140／0.214／0.086；下一期贏家與輸家平均權重約18.95%／18.70%，顯示策略幾乎沒有短期贏家辨識能力。
- 540-step smoke training成功寫入所有新欄位：Gaussian std約1.0、抽樣與deterministic配置差距約32.07%、deterministic turnover約0.26%、value/return correlation約-0.066、reward符號衝突約0.56%。正式判讀需重新訓練；舊診斷CSV缺少的新欄位會顯示`n/a`。

## 31. 2026-09-17：Gaussian低探索噪音300k消融

- 將四種方法的共用訓練預算由暫測600,000步恢復為300,000步；完整Train、Window 20、Hybrid-50、seed123、learning rate、gamma、rollout 540與A2C update epoch 1皆不變。
- 三個A2C系列的`policy_kwargs`新增`log_std_init=-1.0`，初始Gaussian std由約1.0降為`exp(-1)≈0.368`，用來檢驗訓練抽樣配置與deterministic部署配置差距約30%的問題。
- DQN不使用Gaussian policy，因此不受`log_std_init`影響；正式A2C比較需從頭訓練，不能沿用std約1的checkpoint。
- 新run tag為`fulltrain_300k_hybrid_dsr200_ret50_win20_gaussian_logstdm1_e1_level_zscore`，舊300k／600k Gaussian std約1模型、結果與診斷檔均保留。

## 32. 2026-09-17：方案A－加入單一RS_7D相對強勢特徵

- 保留完整chronological Development、300,000 steps、Window 20、Hybrid-50、`eta=0.005`、5-step warm-up、A2C rollout 540、update epoch 1、`log_std_init=-1`及seed123，只改變輸入特徵。
- 每種加密貨幣新增`RS_7D`：過去84根2小時K線的資產報酬，減去BTC／ETH／LTC／BNB同期平均報酬；USDT設為中性0。計算只使用當下與過去價格，不使用下一期或Test未來資料。
- 原始資料最前段未滿84根時，使用截至當時可取得的最長歷史作基準；滿84根後固定使用完整7日，藉此維持論文33,524筆有效資料與原Train／Validation／Test期間。
- 技術模態由20維增為25維，完整observation由`20×25`改成`20×30`；MFN指標LSTM輸入同步改為25維。舊模型與checkpoint因shape不同不可續訓。
- 新run tag為`fulltrain_300k_hybrid_dsr200_ret50_win20_gaussian_logstdm1_e1_level_zscore_rs7d`，不覆蓋原四指標模型與結果。
- 資料重建驗證：總計33,524筆、Train 31,364筆、Validation 1,080筆、Test 1,080筆、5個價格特徵與25個技術／相對強勢特徵；alignment通過。
- 完整42項unittest通過。此方案屬改良／消融實驗，不宣稱為論文原始SMA／EMA／MACD／RSI四指標設定。

## 33. 2026-09-17：Advantage標準化300k消融

- RS_7D版本由600,000步改回300,000步，其他資料、Window 20、Hybrid-50、`eta=0.005`、5-step warm-up、rollout 540、update epoch 1、`log_std_init=-1`及seed123不變。
- 三個A2C系列共用`normalize_advantage=True`；每個rollout進行Actor更新前，將Advantage標準化為近似零均值與單位標準差，降低後期單一rollout使策略突然集中到LTC的風險。DQN不使用A2C Advantage，因此不受此參數影響。
- 變更原因：RS_7D 600k最後實際跑至600,480步，最後rollout出現Advantage mean 10.523、policy loss 23.182、value loss 144.747，最終LTC平均配置升至48.58%，Final PV降至11,997.98。
- 新run tag為`fulltrain_300k_hybrid_dsr200_ret50_win20_gaussian_logstdm1_normadv_e1_level_zscore_rs7d`，不覆蓋未標準化Advantage的300k／600k模型、checkpoint、結果與診斷CSV。
- 此設定是訓練穩定性消融，並非論文表格明確列出的超參數；正式比較時三個A2C方法必須一致使用。

## 34. 2026-09-17：RS_14D與Validation Final-PV選模

- 相對強勢期限由7日改為14日，即168根2小時K線；公式仍為各加密貨幣同期報酬減去BTC／ETH／LTC／BNB平均報酬，USDT為中性0，且只使用當下與過去資料。
- 原32,444筆pre-Test資料依時間切成Train 31,364筆與Validation 1,080筆；最後1,080筆Test維持完全隔離。正式Scaler只以Train擬合，Validation與Test套用相同統計量。
- 四種方法最多訓練300,000步，每100,000步完整回測Validation，依Final PV保存最佳checkpoint；evaluate載入最佳Validation模型。訓練結束狀態另存`*_final.zip`供診斷，不覆蓋最佳模型。
- 本輪不做Stage 2；`normalize_advantage=True`只適用三個A2C系列，DQN維持其原演算法設定。其餘Window 20、Hybrid-50、`eta=0.005`、warm-up 5、A2C rollout 540、epoch 1、`log_std_init=-1`與seed123不變。
- 新run tag為`valselect_300k_hybrid_dsr200_ret50_win20_gaussian_logstdm1_normadv_e1_level_zscore_rs14d_val1080_pv100k`，與RS_7D full-Train模型、checkpoint、結果及診斷檔完全分離。
- 資料重建結果：33,524筆有效資料、5個價格特徵、25個指標／相對強勢特徵；Train／Validation／Development／Test逐列對齊全部PASS。42項unittest與未訓練模型的完整Validation smoke backtest通過。
- 要正式判定RS_7D或RS_14D較佳，RS_7D也必須以完全相同的Train／Validation流程重跑，再比較Validation Final PV；既有RS_7D full-Train結果已看過Validation期間，不能作公平期限選擇。

## 35. A2C baseline 實驗帳本與後續調整依據

本節把先前散落於各章的A2C結果集中整理。所有比較都必須先確認資料期間、特徵、reward、模型選擇方式與seed是否一致；若不一致，只能當作歷史線索，不能直接宣稱某個參數較好。

### 35.1 論文與Buy-and-Hold參考值

| 參考方法 | Peak PV | Final PV | Return | Peak Cum. DSR | Final Cum. DSR | 備註 |
|---|---:|---:|---:|---:|---:|---|
| 論文A2C | 14,378 | 13,357 | 33.57% | 0.232 | 0.120 | 僅作重現目標；封存程式、切分與目前流程未完全確認一致 |
| 目前Test的四幣等權Buy-and-Hold | 待正式輸出 | 約13,011 | 約30.11% | — | — | 2025-06-03至2025-09-01；不含USDT，需以相同lookback的正式baseline輸出為準 |

### 35.2 舊流程結果（不可與目前Validation流程直接混比）

| 實驗 | Steps | Final PV | Return | Peak PV | Max DD | Sharpe | Peak/Final Cum. DSR | 主要差異或觀察 |
|---|---:|---:|---:|---:|---:|---:|---:|---|
| 舊4H流程 | 100k | 13,086.66 | 30.87% | 14,168.04 | -20.62% | 1.6397 | — | 4小時資料，不能與目前2H Test直接比較 |
| 舊4H流程 | 1.0M | 13,721.78 | 37.22% | 14,756.42 | -19.59% | 1.8824 | — | 4小時資料 |
| 舊4H／600k | 600k | 13,483.44 | 34.83% | 14,516.20 | -20.30% | 1.7645 | 0.1430/0.0570 | 舊特徵與舊切分 |
| 舊4H／600k調整後 | 600k | 13,156.20 | 31.56% | 13,889.28 | -23.28% | 1.6306 | 0.1688/0.0860 | DSR／環境曾調整 |
| 舊4H／1.8M | 1.8M | 12,774.87 | 27.75% | 13,404.94 | -24.54% | 1.4360 | 0.1662/0.0941 | 增加步數反而下降 |
| 2H、無warm-up | 600k | 11,377.80 | 13.78% | 12,243.27 | -12.51% | 1.6553 | 0.2771/0.1377 | 無固定warm-up |
| 2H、無warm-up | 1.8M | 11,556.01 | 15.56% | 12,285.47 | -10.88% | 1.8988 | 0.3857/0.2549 | 高DSR沒有轉成高PV |
| 歷史最佳2H、warm-up 5 | 1.8M | 14,882.13 | 48.82% | 16,055.85 | -12.61% | 3.9599 | 0.2764/0.1448 | ETH平均33.36%；舊隨機episode／特徵／reward流程，尚未精確重現 |
| Legacy 16+16、random540、pure DSR | 1.8M | 11,778.93 | 17.79% | 12,939.17 | -13.04% | 2.0988 | 0.2431/0.0911 | turnover 176.21x，證明照舊輸入並未重現14,882 |
| Two-stage、Hybrid-50、Window40 | 1.8M | 13,173.26 | 31.73% | 14,515.86 | -13.65% | 3.1838 | 0.0863/-0.0829 | BTC／ETH平均28.21%／25.40%；流程與目前單階段不同 |

### 35.3 目前2H同一Test期間的主要消融

下表皆使用2025-06-03 02:00至2025-09-01 00:00 UTC的Test，但仍須看「設定」欄確認是否能做單變因比較。

| ID | 設定／本輪重點 | Steps | Final PV | Return | Peak PV | Max DD | Sharpe | BTC/ETH/LTC/BNB/USDT平均權重 | 診斷摘要 |
|---|---|---:|---:|---:|---:|---:|---:|---|---|
| A01 | 2H完整Train早期版本 | 600k | 13,355.57 | 33.56% | 14,380.80 | -13.32% | 3.1988 | 17.36/28.79/17.12/19.97/16.76% | ETH配置較高；舊reward／特徵版本 |
| A02 | Legacy expanding cumulative DSR | 600k | 12,002.36 | 20.02% | 12,661.61 | -10.35% | 2.5549 | 28.31/9.82/2.86/44.33/14.68% | DSR接近論文，但Critic不穩且PV低 |
| A03 | Validation實驗 | 600k | 11,970.66 | 19.71% | 12,981.74 | -12.12% | 2.4579 | 38.06/22.54/6.65/12.29/20.46% | 不同選模流程，不能只用Test判斷Validation無效 |
| A04 | Hybrid-50 | 300k | 12,520.38 | 25.20% | 13,538.31 | -12.02% | 2.6915 | 20.58/15.04/22.57/27.59/14.21% | 比純風險目標更重視PV |
| A05 | Hybrid-100 | 300k | 12,507.97 | 25.08% | 13,586.13 | -12.21% | 2.5914 | 17.39/13.24/26.60/30.16/12.61% | 增加return倍率沒有提升Final PV |
| A06 | Hybrid-50、Gaussian std約1 | 600k | 12,314.40 | 23.14% | 13,656.66 | -12.73% | 2.2088 | 10.35/8.66/40.20/30.01/10.79% | sampled/deterministic gap 29.92%；探索過強 |
| A07 | `log_std_init=-1`、無RS | 300k | 12,560.05 | 25.60% | 13,616.02 | -12.60% | 2.6256 | 21.13/16.45/21.56/31.18/9.68% | gap 12.36%；EV 0.2494、value/return corr 0.6227 |
| A08 | RS_7D、未normalize advantage | 300k | 12,467.20 | 24.67% | 13,610.96 | -13.10% | 2.4549 | 17.75/14.90/25.33/33.71/8.31% | 7D rank corr 0.108；EV 0.3221 |
| A09 | RS_7D、未normalize advantage | 600k | 11,997.98 | 19.98% | 13,500.58 | -13.66% | 1.8272 | 9.35/5.44/48.58/29.89/6.74% | 後期崩向LTC；7D corr 0.008，增加步數惡化 |
| A10 | RS_7D、`normalize_advantage=True` | 300k | 12,597.24 | 25.97% | 13,705.64 | -12.94% | 2.5109 | 12.25/14.78/27.07/37.37/8.53% | 7D corr 0.162；EV 0.3266，為目前RS_7D較穩版本 |
| A11 | RS_14D＋Validation選模 | 100k最佳／300k上限 | 13,091.19 | 30.91% | 14,155.03 | -14.81% | 2.9658 | 24.04/37.05/8.63/17.68/12.61% | Validation在100k最佳；ETH貢獻70.48%，7D corr 0.194 |
| A12 | RS_7D＋Validation選模 | 100k最佳／300k上限 | 13,003.11 | 30.03% | 14,042.25 | -13.99% | 3.0020 | 24.65/32.92/10.84/16.78/14.81% | Validation在100k最佳；ETH貢獻69.99%，7D corr 0.137 |
| A13 | A2C w/o TI、RS_14D資料版本 | 100k最佳／300k上限 | 12,489.84 | 24.90% | 13,521.56 | -12.44% | 2.7180 | 20.17/19.70/20.86/19.69/19.58% | Price-only wrapper移除TI與RS_14D；配置近等權、turnover 75.91x |
| A14 | RS_14D標準兩階段、Validation每50k | Stage 1選100k／Stage 2實跑100,440 | 12,498.82 | 24.99% | 13,598.79 | -12.54% | 2.5917 | 17.61/17.70/22.76/27.70/14.23% | Stage 2重新初始化後趨近平均且偏BNB，未保留Stage 1的ETH配置 |

### 35.4 可以較有把握得到的因果線索

1. **增加steps不保證變好。** RS_7D未標準化Advantage由300k增至600k時，Final PV由12,467.20降至11,997.98，ETH由14.90%降至5.44%，LTC升至48.58%。因此目前問題不是單純「訓練不夠」。
2. **降低Gaussian探索噪音是有效方向。** `log_std_init=-1`後，sampled／deterministic配置差距約由29.92%降至12%，sampled turnover約由41%降至17%；但比較步數並非完全相同，仍應以相同Validation流程再次驗證。
3. **Advantage標準化改善RS_7D穩定性。** 在同為300k RS_7D條件下，Final PV由12,467.20升至12,597.24（+130.04），7D rank correlation由0.108升至0.162，且未出現600k後期LTC崩塌。
4. **RS_7D提高了部分強勢相關性，但尚未提升PV。** 無RS低噪音版本Final PV為12,560.05；RS_7D未標準化版本為12,467.20。加入RS後Critic指標改善，但不代表訊號具有下一期預測力。
5. **提高log-return倍率沒有幫助。** Hybrid-100相較Hybrid-50，Final PV由12,520.38微降至12,507.97，Sharpe亦下降；目前不建議繼續單純放大return項。
6. **不能強迫ETH權重作為訓練目標。** Test期間ETH事後漲最多，但用Test的ETH表現調參會造成Test洩漏。應以Validation Final PV與多seed穩定性選模型。

### 35.5 RS_7D與RS_14D的公平判定流程

1. 固定Train 31,364、Validation 1,080、Test 1,080、Train-only scaler、Window 20、Hybrid-50、warm-up 5、`log_std_init=-1`、`normalize_advantage=True`、epoch 1與300k上限。
2. RS_7D與RS_14D都在100k、200k、300k評估Validation Final PV，僅保存Validation最佳checkpoint。
3. 至少使用3個seed，例如123、456、789；先比較各seed的Validation最佳PV、平均值與標準差。
4. 只在期限與checkpoint確定後，各模型對Test評估一次。禁止依Test的ETH權重、Final PV或Sharpe回頭選RS期限。
5. 若RS_14D只在單一seed領先或平均提升低於約1%，先視為不確定；若平均Validation PV提升且標準差沒有明顯擴大，才採用RS_14D。
6. 若rank correlation提高但Validation PV下降，代表特徵較會描述「過去強勢」，未必能預測下一期；此時應拒絕該期限。

### 35.6 建議的下一輪順序

1. 先完成目前RS_14D的3-seed Validation實驗，記錄100k／200k／300k各checkpoint的Validation PV。
2. 把單一常數改回RS_7D，以完全相同Validation流程與相同3個seed重跑。
3. 再跑「不含RS＋normalize advantage」作為控制組，隔離改善究竟來自RS還是Advantage標準化。
4. 三組都決定後才做一次Test；若最佳Validation模型仍低於Buy-and-Hold，優先檢查預測訊號與reward，而不是直接增加到600k或1.8M。
5. 只有在Validation曲線到300k仍持續上升且配置未崩塌時，才延長訓練；若Validation在100k／200k已見頂，就使用早期checkpoint。

### 35.7 每次實驗追加格式

往後每次完成訓練與評估，請在本檔追加一列並保留以下欄位：

```text
日期／Experiment ID：
Git commit／branch：
唯一變更：
固定不變設定：
資料切分與Test期間：
Seed與實際訓練steps：
Validation 100k／200k／300k PV：
最佳checkpoint：
Test Initial／Final／Peak PV、Return、Max DD、Sharpe、Peak／Final Cum. DSR：
五資產平均／最大權重：
Turnover、explained variance、value-return correlation、Gaussian std與sampled-deterministic gap：
12h／1d／3d／7d／14d rank correlation：
相較控制組的變化：
結論（接受／拒絕／不確定）：
下一輪只改的一個變因：
```

此格式能避免只保存「最好的一次」，也能從數值變化判斷問題位於探索噪音、Critic、reward、相對強勢訊號、過度訓練或模型選擇，而不是反覆憑Test結果調參。

## 36. 2026-09-18：A2C RS_14D＋Validation選模結果

- 設定為RS_14D、Train 31,364筆、Validation 1,080筆、Test 1,080筆、Train-only z-score、Window 20、Hybrid-50、paper-style step DSR×200、log return×50、5-step warm-up、Gaussian `log_std_init=-1`、`normalize_advantage=True`、epoch 1、seed123與300,000步上限。
- Validation Final PV依序為：100k步11,644.55、200k步11,062.37、300k步10,902.87。100k checkpoint為最佳模型，顯示繼續訓練出現明顯Validation退化；本次Test使用100k最佳checkpoint，而非300k final模型。
- Test結果：Initial PV 10,000.00、Final PV 13,091.19、Return 30.91%、Peak PV 14,155.03、Max Drawdown -14.81%、Sharpe 2.9658、Peak／Final Cumulative DSR 0.1995／0.0650。
- 平均配置：BTC 24.04%、ETH 37.05%、LTC 8.63%、BNB 17.68%、USDT 12.61%；BTC＋ETH合計61.09%。Average／Cumulative Turnover為0.96%／10.17x。
- PnL歸因：ETH +2,178.71 USDT，占總獲利70.48%；BNB +601.89、LTC +351.45、BTC -40.85 USDT。模型成功提高Test期間最強資產ETH的配置，是Final PV改善的主要來源。
- 相對強勢配置關係為12h 0.070、1d 0.054、3d 0.103、7d 0.194；7日winner／loser平均權重26.17%／16.97%。但事後下一期winner／loser僅19.74%／18.83%，表示模型較能跟隨中期趨勢，尚未證明能預測下一根K線贏家。
- 相較RS_7D＋normalize advantage的舊full-Train 300k結果，Final PV由12,597.24升至13,091.19（+493.95，約+3.92%），Return增加4.94個百分點，ETH平均權重由14.78%升至37.05%；但兩者模型選擇流程不同，因此只能視為RS_14D的正面訊號，不能作最終因果結論。
- 相較論文A2C，Final PV低265.81（約-1.99%）、Peak PV低222.97（約-1.55%）、Return低2.66個百分點；Peak／Final Cumulative DSR則低0.0325／0.0550。PV已相當接近論文，DSR仍有尺度或策略路徑差異。
- 相較目前同期間四幣等權Buy-and-Hold約30.11%，本次Return約高0.80個百分點、Final PV約高80 USDT，屬小幅領先；仍不足以用單一seed斷言穩定勝出。
- 訓練診斷CSV記錄到完整300k訓練結束，Recent explained variance 0.2561、value/return correlation 0.6178、Gaussian std 0.3694、sampled/deterministic gap 12.14%、reward sign conflict 1.48%、sampled turnover 17.26%。這些recent數值描述300k末段，不是被Test評估的100k最佳checkpoint，後續應按checkpoint分段摘要，避免誤判。
- NumPy `Mean of empty slice`警告出現在診斷摘要階段，通常表示某個可選診斷欄位全為NaN；本次PV、配置與歸因已完整輸出，沒有證據顯示它影響回測計算，但後續應定位並消除該警告。
- 結論：本輪結果「有希望但尚未定案」。下一步不是增加steps，而是用相同Validation流程重跑RS_7D與無RS控制組，並對RS_7D、RS_14D至少各跑seed123／456／789；模型期限只看Validation平均PV與穩定性決定，Test不再用於選擇。

## 37. 2026-09-18：A2C RS_7D與RS_14D同流程比較

- RS_7D使用與RS_14D相同的Train／Validation／Test切分、Train-only scaler、Window 20、Hybrid-50、warm-up 5、`log_std_init=-1`、`normalize_advantage=True`、epoch 1、seed123、100k Validation頻率及300k上限；本次可視為單一RS期限變因比較。
- RS_7D Validation Final PV為100k 11,525.71、200k 11,189.07、300k 11,163.30，最佳為100k checkpoint。RS_14D相同三點為11,644.55、11,062.37、10,902.87，也在100k最佳。
- 以模型選擇唯一依據Validation判斷，RS_14D在最佳checkpoint領先118.85（約1.03%）；但RS_7D在200k與300k分別高126.70與260.43，表示RS_14D早期較好、後期退化較快。
- RS_7D Test結果：Final PV 13,003.11、Return 30.03%、Peak PV 14,042.25、Max Drawdown -13.99%、Sharpe 3.0020、Peak／Final Cumulative DSR 0.2020／0.0666。
- RS_7D平均配置：BTC 24.65%、ETH 32.92%、LTC 10.84%、BNB 16.78%、USDT 14.81%；ETH貢獻2,101.76 USDT，占總獲利69.99%。Average／Cumulative Turnover為1.05%／11.16x。
- 相較RS_7D，RS_14D的Final PV高88.08、Return高0.88個百分點、Peak PV高112.78，且ETH平均權重高4.13個百分點；RS_7D則Max Drawdown改善0.82個百分點、Sharpe高0.0362、Peak／Final DSR高0.0025／0.0016。
- RS_14D的7日rank correlation為0.194，高於RS_7D的0.137；RS_14D 7日winner／loser權重差為9.20個百分點，RS_7D為6.68個百分點。兩者下一期winner與loser權重差都不到1個百分點，仍以中期趨勢跟隨為主。
- 依目前同期間四幣等權Buy-and-Hold約30.11%，RS_7D的30.03%大致持平、略低約0.08個百分點；RS_14D的30.91%則小幅領先約0.80個百分點。
- RS_7D 300k末段診斷為explained variance 0.2400、value/return correlation 0.5844、Gaussian std 0.3686、sampled/deterministic gap 12.07%、reward sign conflict 1.02%、sampled turnover 17.08%。和RS_14D一樣，這些recent數值來自300k final訓練狀態，不是Test載入的100k最佳checkpoint。
- seed123的Validation結果暫時支持RS_14D，但1.03%的優勢位於接近判定門檻的範圍，不能只靠一個seed定案。下一步應先跑seed456與789；RS_28D只能作第三候選，且同樣只能依Validation平均PV選擇，不能因Test期間ETH事後大漲而採用。
- 100k／200k／300k診斷確認模型後期仍在更新：`n_updates`約185／370／556，policy loss持續非零；RS_7D的explained variance為0.274／0.353／0.328，RS_14D為0.289／0.267／0.382。Critic仍能擬合Train rollout，但Validation PV持續下降，屬於「仍在學習、泛化卻惡化」，不是停止學習。
- Gaussian探索亦持續存在：RS_7D的std約0.3670／0.3680／0.3691，RS_14D約0.3668／0.3683／0.3699；sampled與deterministic配置差距約11.8%～12.7%，sampled turnover約16.5%～18.0%。探索沒有隨訓練衰減，且明顯高於deterministic turnover約1.4%～4.1%；Validation與Test採deterministic action，因此探索只直接作用於訓練。

## 38. 2026-09-18：A2C baseline與A2C w/o TI的RS_14D資料版本比較

- 兩個模型使用相同Train／Validation／Test、Window 20、Hybrid-50、warm-up 5、`log_std_init=-1`、`normalize_advantage=True`、epoch 1、seed123與100k Validation頻率；兩者最佳checkpoint皆為100k。
- 名稱需精確解讀：A2C baseline接收5個價格特徵加25個SMA／EMA／MACD／RSI／RS_14D特徵；A2C w/o TI經`PriceOnlyObservation`只保留5個價格特徵，RS_14D雖出現在共用run tag中，實際沒有送入模型。因此本實驗衡量的是完整技術／強勢模態的整體貢獻，不能單獨歸因於RS_14D。
- A2C w/o TI Validation Final PV為100k 10,650.87、200k 10,620.11、300k 10,532.47；100k最佳。A2C baseline相同100k最佳Validation PV為11,644.55，高993.69（約9.33%）。
- Test結果：A2C baseline Final／Peak PV為13,091.19／14,155.03，A2C w/o TI為12,489.84／13,521.56；完整特徵使Final增加601.35、Return增加6.01個百分點、Peak增加633.47、Sharpe由2.7180升至2.9658。
- 風險面不完全一致：A2C w/o TI的Max Drawdown -12.44%，優於baseline的-14.81%；Peak／Final Cumulative DSR 0.2205／0.0779也高於baseline的0.1995／0.0650。完整特徵提高報酬與Sharpe，但沒有提高目前的Cumulative DSR，且承擔較深回撤。
- A2C w/o TI平均配置接近等權：BTC 20.17%、ETH 19.70%、LTC 20.86%、BNB 19.69%、USDT 19.58%；baseline則把ETH提高至37.05%、USDT降至12.61%。ETH PnL由1,261.96升至2,178.71，是兩者Final PV差異的主要來源。
- A2C w/o TI的Average／Cumulative Turnover為7.17%／75.91x，遠高於baseline的0.96%／10.17x；然而它並未形成有效趨勢配置，7日rank correlation為-0.002，baseline為0.194。完整特徵同時降低deterministic換倉並提高中期強勢辨識。
- Critic亦支持完整特徵有用：300k末段A2C w/o TI explained variance／value-return correlation為0.0677／0.2715，baseline為0.2561／0.6178。這些是300k final診斷，不是被Test評估的100k checkpoint，但顯示price-only Critic較難解釋return target。
- 對照論文：目前baseline相較論文A2C的Final／Peak僅低265.81／222.97；目前w/o TI卻比論文A2C w/o TI高530.84／401.56。論文中TI使Final提升1,398，現在只提升601.35，因此目前技術模態的相對增益約小於論文一半，且DSR方向相反。
- 相對當期Buy-and-Hold約30.11%，baseline Return 30.91%小幅領先約0.80個百分點，w/o TI Return 24.90%落後約5.21個百分點。結論是完整特徵確實有幫助，但單一seed尚不足以宣稱穩定效果；兩模型應使用相同額外seeds重跑。

## 39. 2026-09-18：Validation縮短為50k並恢復標準兩階段訓練

- 四種方法的Stage 1上限維持300,000步，Validation頻率由100,000縮短為50,000，因此候選步數改為50k、100k、150k、200k、250k、300k，可搜尋先前100k附近更精確的最佳點。
- Stage 1只使用31,364筆Train與Train-only scaler；每50k完整回測固定1,080筆Validation，Final PV只用來選擇訓練步數。Stage 1最佳checkpoint與300k final診斷模型分別保存，不作正式Test模型。
- Stage 2以相同seed重新初始化，不載入Stage 1權重或optimizer狀態；使用Train＋Validation共32,444筆Development與Development-only scaler，訓練Stage 1選出的固定步數。正式evaluate載入Stage 2模型。
- Test改用Development-only scaler，且完全不參與mean/std擬合、特徵期限、步數或模型選擇。這是恢復Stage 2不可缺少的配套，避免Development訓練與Test輸入尺度不一致。
- A2C baseline、A2C w/o TI、MFN-A2C與DQN全部套用相同流程。MFN的`--resume`只續跑Stage 1；Stage 1完成後，Stage 2仍固定重新初始化。
- 新run tag為`twostage_300k_hybrid_dsr200_ret50_win20_gaussian_logstdm1_normadv_e1_level_zscore_rs14d_val1080_pv50k`，不會覆蓋先前`valselect...pv100k`模型、結果、Validation CSV或診斷紀錄。
- 資料重新產生後維持33,524筆：Train 31,364、Validation 1,080、Test 1,080；5個價格特徵、25個TI／RS特徵、時間對齊全部通過。42項unittest與所有修改腳本的語法檢查通過。
- 這次只改模型選擇解析度與最終資料使用方式；RS_14D、Hybrid-50、warm-up 5、`log_std_init=-1`、`normalize_advantage=True`、epoch 1、rollout 540、learning rate 7e-4及seed123均不變。

## 40. 2026-09-18：A2C RS_14D兩階段100k結果

- Stage 1每50k的Validation Final PV依序為：50k 11,437.20、100k 11,644.55、150k 11,265.10、200k 11,062.37、250k 10,530.57、300k 10,902.87；最佳點仍為100k。縮短間隔增加了解析度，但本seed沒有找到優於原100k的新checkpoint。
- Stage 2以相同seed重新初始化，使用Development-only scaler與32,444筆Development，指定100,000步；因A2C每個rollout為540步，實際完成100,440步（186個完整rollout），屬SB3 on-policy正常行為。
- Test結果：Final PV 12,498.82、Return 24.99%、Peak PV 13,598.79、Max Drawdown -12.54%、Sharpe 2.5917、Peak／Final Cumulative DSR 0.2071／0.0663。
- 相較單階段Validation最佳RS_14D模型，Final PV由13,091.19降592.37、Return下降5.92個百分點、Peak PV下降556.24、Sharpe下降0.3741；Max Drawdown則改善2.27個百分點，Peak／Final DSR微升0.0076／0.0013。
- 主要差異來自配置：ETH平均權重由37.05%降至17.70%，BNB由17.68%升至27.70%，ETH PnL由2,178.71降至1,094.10。Stage 2策略更分散且風險較低，但錯過Test的ETH大漲，因此Final PV下降。
- Stage 2的Average／Cumulative Turnover為0.71%／7.54x，explained variance 0.3224、value-return correlation 0.6797、Gaussian std 0.3662、sampled/deterministic gap 12.75%。Critic並未失效；模型是學到不同且較保守的局部解，而非沒有學習。
- Validation期間BTC／ETH／LTC／BNB本身報酬約+19.44%／+19.00%／-12.72%／+12.47%，並沒有明顯支持Test期間ETH應被重押。Stage 2加入近期資料後仍偏BNB，較可能源自重新初始化、Scaler與完整Development梯度路徑改變，而非單一近期行情。
- 對照同期間Buy-and-Hold約30.11%，兩階段A2C的24.99%落後約5.12個百分點；相較論文A2C Final PV 13,357則低858.18。以PV為目標，本次單seed結果確實變差。
- 但不能因已看到Test較差就直接用Test選擇恢復單階段，否則形成實驗層級洩漏。方法選擇應以多seed、rolling Validation或新的未觸碰holdout決定；目前應把兩階段結果視為「方法較嚴謹但單seed Test表現較差」。

## 41. 2026-09-18：保留50k Validation並恢復單階段訓練

- 依本輪實驗需求，四種方法由第39節的兩階段流程改回單階段；第39、40節仍保留為歷史消融，不回寫或刪除。
- A2C baseline、A2C w/o TI、MFN-A2C與DQN均只在31,364筆Train訓練最多300,000步，每50,000步完整回測固定1,080筆Validation。
- Validation Final PV最高的checkpoint直接寫入正式模型路徑並由evaluate載入；跑到300k的模型另存為`*_final.zip`，只供診斷，不會取代Validation最佳模型。
- 取消使用32,444筆Development重新初始化訓練的Stage 2；Test的特徵標準化也恢復為Train-only scaler，Validation與Test都不參與mean／std擬合。
- 新run tag為`valselect_300k_hybrid_dsr200_ret50_win20_gaussian_logstdm1_normadv_e1_level_zscore_rs14d_val1080_pv50k`，與先前100k Validation頻率以及兩階段50k模型分開保存。
- 這次不改RS_14D、Hybrid-50、5-step warm-up、`log_std_init=-1`、`normalize_advantage=True`、epoch 1、rollout 540、learning rate 7e-4、seed123、Window 20或資料時間範圍。
- 恢復單階段是使用Test結果後做出的實驗流程選擇，因此若要將後續結果作為正式無偏估計，仍須使用多seed Validation、rolling Validation或新的未觸碰holdout確認，不能把本次Test改善直接當成泛化證據。
- 已重新產生33,524筆特徵資料，確認Train 31,364、Validation 1,080、Test 1,080，Test使用Train-only scaler；四支訓練程式通過語法檢查，完整42項unittest全數通過。

## 42. 2026-09-18：raw DSR改為論文innovation並分離reward scale

- 正式raw DSR改用論文與Moody–Saffell原式：`delta_A=r-A_old`、`delta_B=r²-B_old`；`A_new=A_old+eta*delta_A`、`B_new=B_old+eta*delta_B`，因此`eta=0.005`只控制EWMA記憶，不再隱含縮小raw DSR。
- 舊的`delta_A=A_new-A_old`、`delta_B=B_new-B_old`實作沒有刪除，改名保留為`ewma_change`消融；它的輸出仍等於論文raw DSR的`eta`倍。封存原碼的`legacy_expanding`模式也維持不變。
- 為避免把canonical raw DSR再放大100或200倍，正式`DSR_REWARD_SCALE`改為1.0；`RETURN_REWARD_SCALE=50`與`REWARD_TYPE=hybrid`保留，所以新reward為`raw DSR + 50×log(1+portfolio_return)`。
- 保留使用者目前的seed456、`log_std_init=-2`、300k上限、50k Validation、Window 20、RS_14D、Advantage normalization與epoch 1；模型必須重新訓練，不可沿用舊reward公式checkpoint。
- run tag改由實際reward scale、eta、log std與seed自動生成，修正先前實驗雖修改參數卻仍被錯標為`dsr200`、`logstdm1`並覆蓋檔案的問題。
- 評估、Buy-and-Hold與訓練共用同一raw DSR，因此後續Cumulative DSR尺度會約為舊eta-scaled報告的`1/eta`倍；新舊DSR數字不可直接比較，PV、Return與Drawdown則仍使用相同定義。
- 驗證結果：相關訓練／評估腳本均通過`py_compile`，完整43項unittest通過；回歸測試亦確認`ewma_change = eta × paper raw DSR`，以及環境reward確實等於`1 × raw DSR + 50 × log return`。

## 43. 2026-09-18：A2C paper raw DSR、seed456結果

- 設定：單階段Validation選模、300k上限、每50k驗證、seed456、RS_14D、Window 20、`log_std_init=-2`、`normalize_advantage=True`、epoch 1，以及`raw paper DSR + 50×log return`。
- Validation Final PV依序為：50k 10,740.96、100k 10,605.04、150k 11,164.20、200k 11,412.19、250k 11,429.32、300k 11,521.83；最佳checkpoint為300k，顯示本輪100k後仍持續改善。
- Test結果：Final PV 13,613.76、Return 36.14%、Peak PV 14,629.87、Max Drawdown -15.75%、Sharpe 3.2062；raw paper Peak／Final Cumulative DSR為43.2104／16.7559。
- 若只為舊eta-scaled顯示尺度作換算，Peak／Final DSR約為0.2161／0.0838；此換算不改變訓練，只用於理解尺度，不能據此宣稱與論文評估定義完全相同。
- 平均配置為BTC 17.49%、ETH 47.21%、LTC 2.93%、BNB 18.94%、USDT 13.42%。ETH貢獻2,939.28 USDT，占總獲利81.34%，是本輪PV提高的主要來源。
- 相對強勢對齊明顯改善：3日／7日rank correlation為0.177／0.283，7日winner與loser平均權重為30.66%／16.50%；策略主要學到中期強勢，而不是下一期預知，事後下一期winner／loser僅20.98%／19.52%。
- 相較同期間四幣各25%的Buy-and-Hold Final PV約13,011.44，本輪高602.32 USDT、報酬高約6.02個百分點；相較包含20% USDT的五資產Buy-and-Hold 12,409.15則高1,204.61 USDT。
- 相較論文A2C Final／Peak PV 13,357／14,378，本輪分別高256.76／251.87；相較論文Proposed Final／Peak PV 13,929／14,902，仍低315.24／272.13。
- 診斷未顯示單一失敗：reward std 1.5618、p99 5.0325，沒有因改用raw DSR而爆炸；explained variance 0.2605、value-return correlation 0.6280，Critic可用但仍有改善空間；Gaussian std 0.1382、sampled/deterministic gap 4.18%、sampled turnover 6.32%，探索已受控制。
- 這仍是單一seed且Test已在多輪研究中被查看，不能單獨當成無偏泛化證據；後續應固定全部設定，只更換seed123與789，以Validation統計與多seed Test摘要評估穩定性。

## 44. 2026-09-18：A2C paper raw DSR、seed456延長至600k

- 除訓練上限由300k延長為600k外，其餘設定與第43節相同。50k至300k的Validation數字逐項一致，符合相同seed與相同前300k訓練路徑的預期。
- Validation Final PV在300k達全程最高11,521.83；350k／400k／450k／500k／550k／600k依序為11,330.97／11,038.83／11,169.45／11,214.43／11,210.85／11,384.10，均未超越300k。
- 因正式模型以Validation Final PV選擇，600k實驗仍載入300k checkpoint，所以Test結果與第43節完全相同：Final PV 13,613.76、Return 36.14%、Peak PV 14,629.87、Max Drawdown -15.75%、Sharpe 3.2062。
- 這不是600k模型碰巧產生完全相同配置，而是模型選擇機制正確阻止350k後的Validation退化覆蓋最佳模型。`*_final.zip`另存600k末端模型，只供診斷；無`_final`檔名者是正式300k最佳checkpoint。
- 600k末段explained variance／value-return correlation提升至0.3847／0.7036，表示Critic仍在學習；但較好的value fitting沒有轉化為較高Validation PV。Gaussian std 0.1377與gap 4.60%仍受控制，reward conflict 0.83%，沒有數值或探索崩潰。
- 結論：對seed456，本設定的泛化最佳步數是300k。延長到600k只增加訓練成本並在後半段出現Validation退化；後續公平比較可維持300k上限，但其他seed仍應各自依Validation選checkpoint，不能假設都固定300k最佳。

## 45. 2026-09-18：Validation由90天縮短為45天

- 為讓模型納入更多Test前近期資料，Validation由1,080筆（90天）縮短為540筆（45天）；Test仍固定1,080筆與原日期，沒有移動或縮短。
- 被釋出的540筆併回Train，因此切分由31,364／1,080／1,080改為31,904／540／1,080；Train結束時間改為2025-04-19 00:00 UTC，Validation為2025-04-19 02:00至2025-06-03 00:00 UTC。
- Train-only scaler將以31,904筆重新擬合；Validation與Test依然不參與Scaler擬合。此資料切分改變會使舊模型與舊結果失去直接相容性，必須重新產生特徵並重新訓練。
- run tag改含`val540`，與先前`val1080`模型、Validation紀錄及結果CSV分開保存。Validation頻率仍為每50k，並保留修改當下的600k訓練上限；其餘A2C與reward設定不變。
- 45天Validation更貼近Test且提供較新Train資料，但樣本減半後選模方差可能增加；此項應視為資料切分消融，不應用同一Test反覆挑選45天或90天版本。

## 46. 2026-09-18：45天Validation、seed456、600k結果

- Validation Final PV於50k至600k依序為：12,208.70、12,249.07、12,694.17、12,797.99、12,674.14、12,862.87、13,228.55、13,149.68、14,119.44、14,018.61、12,988.36、12,850.58；最佳checkpoint為450k，正式Test未使用600k final模型。
- Test結果：Final PV 14,108.97、Return 41.09%、Peak PV 15,347.13、Max Drawdown -17.72%、Sharpe 3.2067；raw paper Peak／Final Cumulative DSR為44.9194／18.0314，若僅乘eta換算舊顯示尺度約0.2246／0.0902。
- 平均配置為BTC 15.00%、ETH 56.21%、LTC 9.92%、BNB 8.81%、USDT 10.05%；ETH PnL 3,562.24 USDT，占總獲利86.69%，是PV提高的主要來源。Average／Cumulative Turnover為0.45%／4.77x。
- 相較90天Validation版本（同seed、同600k上限、正式選到300k），Final PV增加495.21、Return增加4.95個百分點、Peak增加717.26，但Max Drawdown惡化1.97個百分點，Sharpe幾乎不變（3.2062→3.2067）。ETH平均權重增加9.00個百分點，USDT下降3.37個百分點。
- 被併回Train的前45天（2025-03-05 02:00至04-19 00:00）BTC／ETH／LTC／BNB報酬為-3.62%／-27.45%／-27.17%／+0.22%；新Validation後45天則為+25.21%／+63.98%／+18.05%／+12.40%。新Validation明確偏好高ETH策略，而Test的ETH隨後再漲66.67%，形成有利的連續市場regime。
- 這不構成逐步未來資訊洩漏：45天Validation嚴格早於Test，且Test沒有進入Scaler或梯度。但Validation縮短後對單一近期行情更敏感，450k高ETH模型在Test繼續受益可能含有regime延續的幸運成分，不能只以本次Test宣稱45天必然優於90天。
- 600k末段診斷（不是正式450k checkpoint）為explained variance 0.4518、value-return correlation 0.7280、Gaussian std 0.1348、sampled/deterministic gap 4.64%、sampled turnover 6.87%；Critic仍在學習，但500k後Validation退化，顯示繼續訓練沒有改善泛化。
- 相較四幣各25%的Buy-and-Hold Final PV約13,011.44，本輪高1,097.53 USDT、報酬高約10.98個百分點；相較論文A2C Final 13,357高751.97，相較論文Proposed Final 13,929高179.97，但測試資料與流程若不完全相同，不可直接宣稱超越論文。

## 47. 2026-09-18：A2C w/o TI、45天Validation、300k結果

- 設定與45天Validation資料切分一致，但模型經`PriceOnlyWrapper`只取得5個價格特徵，不使用SMA、EMA、MACD、RSI或RS_14D。訓練上限為300k，和第46節A2C baseline的600k上限不同，尚非完全相同預算比較。
- Validation Final PV於50k／100k／150k／200k／250k／300k為12,264.72／12,347.53／12,381.01／12,402.04／12,431.01／12,419.49；最佳checkpoint為250k，正式Test未使用300k final模型。
- Test結果：Final PV 12,171.27、Return 21.71%、Peak PV 13,143.43、Max Drawdown -12.27%、Sharpe 2.4673；raw paper Peak／Final Cumulative DSR為40.6025／12.9357，乘eta僅作舊尺度換算約0.2030／0.0647。
- 平均配置接近等權：BTC 19.92%、ETH 19.32%、LTC 19.27%、BNB 21.52%、USDT 19.97%；Average／Cumulative Turnover卻高達10.39%／109.99x。雖然環境無交易成本，高換手仍顯示策略頻繁且無效地偏離再回到等權。
- 相對強勢rank correlation在12小時／1日／3日／7日僅-0.002／0.013／0.033／0.036，下一期winner權重19.80%甚至低於loser的19.88%；符合模型沒有RS_14D或TI可用，未形成有效強勢配置。
- 本輪低於四幣各25% Buy-and-Hold約13,011.44，也低於五資產各20% Buy-and-Hold 12,409.15與每期固定再平衡20%的12,348.98。主要原因是約20%資金長期留在USDT、ETH維持約19%而未隨上漲漂移，且動態偏移時機沒有產生額外報酬。
- 相較第46節完整特徵A2C，Final PV低1,937.70、Return低19.38個百分點；完整模型ETH平均權重56.21%且turnover 4.77x，w/o TI則ETH 19.32%且turnover 109.99x，顯示技術／RS模態對本輪配置具有重大影響。但因兩者訓練上限不同，正式消融結論仍應把w/o TI也跑到600k並由相同Validation規則選模。
- 300k末段診斷（不是正式250k checkpoint）explained variance僅0.0787、value-return correlation 0.2886，遠弱於完整A2C；Gaussian std 0.1358與gap 4.82%正常，因此瓶頸較像觀測資訊不足與Critic難以預測，而非探索噪音失控。
- 對照論文A2C w/o TI Final／Peak PV 11,959／13,120，本輪高212.27／23.43，數值接近但資料期間與Buy-and-Hold基準不一致，不能視為完整重現。

## 48. 2026-09-18：MFN-A2C GitHub-style、45天Validation、300k結果

- 使用GitHub-style兩模態MFN、paper raw DSR Hybrid reward、seed456、Window 20、`log_std_init=-2`、Advantage normalization、epoch 1與45天Validation；訓練上限300k。
- Validation Final PV於50k／100k／150k／200k／250k／300k為12,185.90／12,268.87／12,105.10／12,248.52／12,226.47／12,318.26；300k為全程最佳，正式Test與300k末段診斷對應同一checkpoint。最佳點位於預算邊界，與w/o TI在250k見頂不同，不能排除MFN尚未充分訓練。
- Test結果：Final PV 12,697.96、Return 26.98%、Peak PV 13,533.65、Max Drawdown -12.47%、Sharpe 3.0245；raw paper Peak／Final Cumulative DSR為41.2654／14.9189，乘eta僅作舊尺度換算約0.2063／0.0746。
- 平均配置接近等權：BTC 18.78%、ETH 20.46%、LTC 19.04%、BNB 20.42%、USDT 21.29%；Average／Cumulative Turnover為0.95%／10.11x。Test期間ETH上漲66.67%，但MFN沒有形成高ETH結構性配置，ETH只貢獻1,366.29 USDT。
- 相對強勢rank correlation在12小時／1日尚有0.125／0.105，但3日／7日降至0.032／0.023；模型主要對短期變化有微弱反應，沒有充分利用RS_14D的中期橫截面訊號。
- 診斷顯示MFN並未斷線：gradient norm 0.492、explained variance 0.3228、value-return correlation 0.5973均為有效訊號；瓶頸是deterministic allocation entropy 0.9861、temporal variation 0.0170，策略輸出過度分散且變化小。這較像複雜編碼器的樣本效率／過度平滑問題，而非梯度完全消失。
- 同一45天Validation下，完整A2C在300k的Validation PV為12,862.87，MFN為12,318.26，差544.61；兩者50k僅差22.80，之後baseline持續學到高ETH配置而MFN停滯，支持端到端MFN收斂較慢的判斷。
- 相較四幣各25% Buy-and-Hold約13,011.44，MFN低313.48；但高於五資產各20% Buy-and-Hold 12,409.15約288.81，也高於A2C w/o TI 12,171.27約526.69。MFN不是完全無效，而是偏保守、沒有捕捉本期ETH集中報酬。
- 下一個最小變因實驗應維持所有設定，只把MFN上限擴至600k並繼續每50k由Validation選模；若Validation仍停滯且配置維持等權，再增加MFN feature activation variance、attention entropy與memory update幅度診斷，而不是立即改reward或架構。

## 49. 2026-09-18：MFN改用CPU並降低診斷頻率

- 本機為RTX 3060 Laptop GPU，但GitHub-style MFN含20步`LSTMCell`／memory Python loop，rollout採樣時每次只推論batch 1。實測純前向batch 1為CPU 12.86ms、GPU 36.35ms；batch 540前向＋反向則CPU 369.50ms、GPU 120.10ms。由於每540次batch-1採樣才做一次更新，整體MFN特徵擷取計算估計CPU較有利。
- MFN的新模型與resume固定`device="cpu"`；A2C baseline、A2C w/o TI與DQN仍維持原共用`device="auto"`，不受影響。Evaluation本來就是CPU。
- MFN診斷由每個rollout改為每5個rollout記錄一次；callback在未記錄的rollout直接跳過info蒐集、policy distribution、Critic與MFN gradient統計，但A2C仍每個rollout照常執行環境採樣與optimizer更新。
- Validation仍每50k執行、checkpoint仍依既有頻率保存，reward、模型輸入、Actor/Critic、MFN架構與隨機種子均不變。代價是診斷時間解析度降為原本五分之一，短暫尖峰可能不會被CSV捕捉，且摘要最後20列代表約100個rollout而非20個。
- 為避免覆蓋GPU／逐rollout診斷實驗，MFN專屬tag追加`cpu_diag5`；其他方法沿用原`RUN_TAG`。

## 50. 2026-09-18：MFN-A2C支援300k延長至600k續訓

- 將正式輸出`RUN_TAG`與checkpoint相容標籤拆開：正式模型、結果與診斷仍保留`300k`／`600k`，但`MFN_RESUME_TAG`不包含目標總步數，因此只改`TOTAL_TIMESTEPS`不會使舊checkpoint失效。
- 續訓相容標籤仍包含MFN架構版本、reward與scale、eta、Window、Gaussian log std、Advantage normalization、epochs、seed、特徵版本與Validation切分；這些條件不同的checkpoint不會混用。CPU/GPU與診斷頻率不影響模型張量，因此不列入相容性判斷。
- `python scripts/train_mfn_a2c.py --resume`會讀取不超過目前目標步數的最新相容checkpoint。例如設定600k並載入300k時，顯示`Completed=300,000`、`Remaining=300,000`，並以`reset_num_timesteps=False`延續模型、optimizer及全域步數。
- 新checkpoint改用不含目標步數的固定prefix；同時向下相容既有`MFN_A2C_VALSELECT_300K_..._300000_steps.zip`，不必重新訓練或手動重新命名舊檔。
- Validation CSV與Validation最佳模型改用跨目標步數的固定續訓名稱。首次載入舊300k checkpoint時，會複製舊Validation紀錄與最佳模型；正式完成後再輸出一份含目前300k／600k標籤的模型，避免實驗結果互相覆蓋。
- 若使用`--resume`卻找不到相容checkpoint，程式現在會直接停止並報錯，不再悄悄從零開始；超過目前目標步數的checkpoint也不會被載入。
- 驗證：實際載入目前既有300k相容checkpoint，確認`num_timesteps=300000`且optimizer含37筆狀態；完整48項unittest全數通過，包含舊Validation紀錄／最佳模型搬移測試，`py_compile`與`git diff --check`亦通過。此次只新增續訓能力，`TOTAL_TIMESTEPS`仍維持300,000，reward與模型架構沒有改動。

## 51. 2026-09-19：A2C baseline純paper DSR、45天Validation、300k結果

- 設定與第46節相同的2H、Window 20、RS_14D、seed456、`log_std_init=-2`、Advantage normalization、epoch 1與45天Validation；只將`RETURN_REWARD_SCALE`由50改為0。雖然`REWARD_TYPE`字串仍為`hybrid`，數值上reward等於`1×paper raw DSR`。
- Validation Final PV於50k／100k／150k／200k／250k／300k為12,250.95／12,228.93／12,479.77／12,545.32／12,511.38／12,853.37；最佳checkpoint為300k，顯示純DSR在預算邊界仍有改善。
- Test結果：Final PV 13,230.84、Return 32.31%、Peak PV 14,176.63、Max Drawdown -13.45%、Sharpe 3.2169；raw paper Peak／Final Cumulative DSR為43.0646／15.5344。
- 平均配置為BTC 19.63%、ETH 32.24%、LTC 13.11%、BNB 17.53%、USDT 17.49%；ETH貢獻2,011.78 USDT，占總獲利62.27%。Average／Cumulative Turnover為1.03%／10.93x。
- 相對強勢rank correlation在12小時／1日／3日／7日為0.116／0.110／-0.015／0.088，較偏短期強勢；下一期winner／loser權重20.73%／19.72%，沒有顯示明顯未來資訊利用。
- 與第46節Hybrid-50的600k Validation選模結果相比，純DSR Final PV低878.13、Return低8.78個百分點，但Max Drawdown改善4.27個百分點，Sharpe略高0.0102。兩者訓練上限與正式選中步數不同，這不是完全公平的單變因比較。
- 在相同300k節點，Hybrid-50的Validation PV為12,862.87，純DSR為12,853.37，只差9.50（約0.07%）；Validation幾乎無法判定兩者優劣。Test差異主要來自純DSR的ETH平均權重較低、USDT較高，符合DSR偏好風險調整後報酬而非單純追求資產成長。
- 純DSR仍高於四幣各25% Buy-and-Hold約219.40 USDT，也高於五資產各20% Buy-and-Hold約821.69 USDT；但低於論文A2C Final PV 13,357約126.16，低於論文Proposed 13,929約698.16。
- 訓練診斷正常：reward std 1.0710、p99 3.5556、explained variance 0.3440、value-return correlation 0.6929、Gaussian std 0.1351、sampled/deterministic gap 4.71%、entropy 0.9551、sampled turnover 7.01%。沒有單一失敗指標；主要是策略較分散、較保守，而非Actor或Critic失效。
