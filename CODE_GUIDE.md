# 程式碼用途導覽

本專案重現論文的 MFN-A2C 加密貨幣投資組合流程。目前設定在每個決策點使用前20個2小時K線（40小時）作為觀察值，將資金配置到BTC、ETH、LTC、BNB與USDT。USDT視為報酬率0的穩定資產。

## 核心模組

- `src/dsr.py`：全專案唯一的EWMA DSR實作；目前Hybrid-50訓練使用`200×step DSR + 50×log return`，評估DSR仍輸出未放大的原始值。
- `src/training_diagnostics.py`：MFN與A2C系列每個rollout記錄兩個reward分量、符號衝突率、PV、return、抽樣／deterministic配置差距、Gaussian逐資產std、turnover、配置entropy/集中度、Advantage分布、Critic target/prediction相關與RMSE、Actor/Critic loss及explained variance；MFN另有特徵擷取器參數／梯度資訊。
- `src/experiment_config.py`：集中管理Train 31,364筆、Validation 1,080筆、300k上限、每100k依Validation Final PV選模、資料路徑、seed、paper DSR與四種方法共用設定。
- `src/evaluation_metrics.py`：加入逐步DSR、累積DSR、配置比例、turnover與逐資產PV損益歸因；五項資產PnL加總會核對`Final PV - Initial PV`。
- `src/evaluation_metrics.py`亦會以已完成報酬計算12小時／1日／3日／7日相對強勢，輸出權重與動量rank correlation、近期贏家／輸家權重及事後下一期贏家權重；動量先`shift(1)`，不會把未來報酬放入訊號。
- `src/portfolio_env_sb3.py`：Gymnasium 交易環境；MFN 使用 simplex 模式直接接收總和為 1 的投資權重，尚未遷移的模型可繼續使用舊 logits＋Softmax 模式；在 `Open[t]` 再平衡，根據 `Open[t]` 到 `Open[t+1]` 更新投資組合並回傳 reward。
- `src/simplex_policy.py`：保留作Dirichlet消融；目前公平比較不使用，三個A2C方法統一採SB3 Gaussian logits＋環境Softmax。
- `src/mfn_sb3_extractor.py`：兩視角MFN；5個資產價格相對值與25個指標／相對強勢特徵各經獨立LSTM。DMAN依式4.5～4.8對相鄰hidden state差分`Δh_t`做單層Linear＋Softmax attention；MGM依式4.9～4.10讓兩個單層gate只讀取attended delta，並直接以`tanh(attended)`更新共享記憶，最後交給A2C。
- `src/mfn_github_extractor.py`：目前 MFN 診斷版本；將原始 GitHub 三模態 MFN 改為價格與技術指標兩模態，保留前後 cell-state 串接、兩層 attention/candidate MLP，以及讀取 previous memory 的兩層 retention/update gates。
- `src/multi_epoch_a2c.py`：支援重複更新同一份540-step rollout；目前公平比較統一1 epoch，論文表列18 epochs保留作另一套重現設定。
- `src/feature_scaling.py`：逐欄 z-score 標準化；只允許使用 Train 統計量，避免 Test 洩漏。
- `src/price_only_env.py`：實驗 1 的 A2C w/o TI，只保留價格變化特徵。
- `src/discrete_action_env.py`：DQN 專用包裝；使用 20% 權重網格，BTC/ETH/LTC/BNB 單一資產最高 60%，USDT 可達 100%，共 106 個動作。

## 資料流程

1. `download_binance_paper.py`：下載四種加密貨幣自 2018-01-01 至 2025-09-01 00:00（含）的 Binance 2 小時 K 線，共33,550筆原始資料並依時間戳合併。
2. `prepare_paper_features.py`：建立5個價格相對值、原四項指標與方案A的`RS_14D`，共5＋25維。正式Scaler只以31,364筆Train擬合，Validation與Test均不參與。
3. `check_alignment.py`：驗證train/validation/test的時間戳、原始價格及兩類特徵逐列對齊。
4. `src/experiment_periods.py`：集中定義2H間隔、Train／Validation／Test、20步lookback與預期筆數。正式切分為31,364／1,080／1,080筆。

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

最後的`-1`用來保留`Open[t+1]`。Buy-and-Hold也從`Open[20]`開始，才能公平比較。

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

四種方法均在31,364筆Train依時間順序訓練最多300,000步，每100,000步完整回測獨立Validation，保存Final PV最高的checkpoint。Test不參與Scaler、訓練、特徵期限或checkpoint選擇；本輪不做Stage 2。

`src/mfn_sb3_extractor.py`保留論文簡化MFN作為對照；目前使用`GitHubStyleTwoViewMFN`。本輪Gaussian `log_std_init=-1`（初始std約0.368）、`normalize_advantage=True`，並以`RS_14D`跨資產相對強勢特徵做Validation消融。輸入為`5+25`，run tag為`valselect_300k_hybrid_dsr200_ret50_win20_gaussian_logstdm1_normadv_e1_level_zscore_rs14d_val1080_pv100k`。

資料切分、時間軸、DSR reward、特徵公式、特徵標準化或動作分布變更後，不可沿用舊模型、checkpoint或結果CSV。paper DSR現在是共享預設，因此MFN、A2C、A2C w/o TI與DQN若使用DSR reward都必須重新訓練；Buy-and-Hold只需重新評估。

目前將論文原32,444筆Train尾端1,080筆保留為Validation，實際Train為31,364筆；Test仍固定為2025-06-03 02:00至2025-09-01 00:00且最後才評估一次。

## 結果整理

- `baseline_buy_hold.py`：依論文文字定義，BTC、ETH、LTC、BNB 期初各配置 25%，USDT 為 0%，買入後不再平衡；輸出 return、Sharpe、逐步 DSR、累積 DSR 與隨價格漂移的實際權重。
- `compare_all_methods.py`：實驗 1 的 MFN-A2C、A2C、A2C w/o TI、Buy-and-Hold。
- `compare_experiment2.py`：實驗 2 的 MFN-A2C、A2C、DQN、Buy-and-Hold。
- `compare_experiment3.py`：實驗 3 的四個模型/獎勵組合。

## 論文設定

- 2小時資料、目前20步（40小時）觀察窗口、五種資產。
- A2C：兩層 `[64, 64]`、learning rate `7e-4`、gamma `0.99`、rollout `540`。
- A2C 更新：目前三個A2C公平比較均為1 epoch；論文表列18 epochs，但屬另一套重現實驗。`MultiEpochA2C`沒有加入PPO clipping。
- DSR統一由`src/dsr.py`計算；正式重現預設為`paper_legacy`，使用`delta_A=A_new-A_old`、`delta_B=B_new-B_old`與`eta=0.005`；前5 steps僅更新統計量，第6 step起產生DSR。canonical innovation模式保留供消融。
- `Peak Cumulative DSR` 與 `Final Cumulative DSR` 是累積值，不是單一步驟 DSR；目前正式比較使用300,000 steps的最後模型。
- 環境不含交易手續費與滑價，這是論文的實驗假設。

目前正式設定已切回論文/封存程式的EWMA moment-change尺度。canonical模型、結果及checkpoint均保留但不可與paper DSR混用；所有使用DSR reward的正式模型都必須重新訓練。

## MFN 中斷後續訓

正常開始使用`python scripts\train_mfn_a2c.py`；中斷後使用`python scripts\train_mfn_a2c.py --resume`。程式只會讀取目前單階段實驗標籤相容的checkpoint，並接續到總計300,000步。

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
