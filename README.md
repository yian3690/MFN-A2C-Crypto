# DMAN Temporal Attention A2C Crypto Portfolio（4H）

本專案目前統一使用Binance 4小時K線，資產為BTC、ETH、LTC、BNB與USDT。
所有執行腳本直接放在`scripts/`，所有產物直接放在根目錄的`data/`、
`models/`、`results/`、`logs/`、`checkpoints/`與`figures/`，不再建立
`experiment1_4h`子資料夾。

## 共用實驗設定

- 資料期間：2018-01-01至2025-09-01。
- Test：最後1,080根4H資料，約180天。
- Train：Test以前的完整資料；不使用Validation。
- Train episode：隨機抽取180天。
- Window：`LOOKBACK=20`，即80小時；可在`scripts/config_4h.py`修改。
- 特徵：5維price-relative＋25維SMA20、EMA20、MACD、RSI14、RS_14D。
- Reward：由`scripts/config_4h.py`的`DSR_REWARD_SCALE`、`RETURN_REWARD_SCALE`與`DSR_ETA`統一管理。
- A2C：`n_steps=540`、epoch=1、normalize advantage、`log_std_init=-2`。
- 目標步數：由`scripts/config_4h.py`的`TOTAL_TIMESTEPS`統一管理。
- Test只用於最終評估，不參與Scaler或訓練。

## 資料準備

```powershell
python scripts\download_data_4h.py
python scripts\prepare_features_4h.py
```

## Experiment 1：技術指標與架構比較

目前比較DMAN＋Temporal Self-Attention A2C、A2C、A2C without TI與Buy-and-Hold。
舊MFN/MGM訓練路徑已移除，DMAN與雙LSTM仍保留在目前的自訂特徵擷取器中。

```powershell
python scripts\train_dman_temporal_attention_a2c.py
python scripts\train_a2c.py
python scripts\train_a2c_without_ti.py

python scripts\evaluate_dman_temporal_attention_a2c.py
python scripts\evaluate_a2c.py
python scripts\evaluate_a2c_without_ti.py
python scripts\evaluate_buy_and_hold.py
python scripts\compare_experiment1.py
```

三種A2C方法皆可使用`--resume`，例如：

```powershell
python scripts\train_dman_temporal_attention_a2c.py --resume
```

## Experiment 2：時間特徵擷取效果

比較DMAN＋Temporal Self-Attention A2C、一般A2C、離散動作DQN與Buy-and-Hold。
DQN使用20%權重網格，每項風險資產上限60%，共106種合法配置。

```powershell
python scripts\train_dqn.py
python scripts\evaluate_dqn.py
python scripts\compare_experiment2.py
```

繪圖腳本會讀取同一`TOTAL_TIMESTEPS`標籤下的Temporal Attention、A2C與DQN結果，輸出：

- `results/experiment2_4h_<steps>_comparison.csv`
- `results/experiment2_4h_<steps>_table.csv`
- `figures/experiment2_4h_<steps>_comparison.png`

## Experiment 3：Reward選擇

目前比較一般A2C在共用DSR/return reward與absolute Portfolio Value reward下的結果。
除了reward外，其餘4H資料、episode、Actor/Critic及訓練步數保持一致。

```powershell
python scripts\train_a2c.py
python scripts\train_a2c_pv.py
python scripts\evaluate_a2c.py
python scripts\evaluate_a2c_pv.py
python scripts\compare_experiment3.py
```

PV版本也支援`--resume`。繪圖輸出：

- `results/experiment3_4h_<steps>_comparison.csv`
- `results/experiment3_4h_<steps>_table.csv`
- `figures/experiment3_4h_<steps>_comparison.png`

## 主要架構

```text
Price LSTM ─┐
            ├→ DMAN → Temporal Self-Attention → Attention Pooling → A2C
TI LSTM ────┘
```

- `src/dman_temporal_attention_extractor.py`是目前唯一的自訂雙模態特徵擷取器。
- DMAN在每個時間點融合價格與技術指標LSTM狀態。
- Temporal Self-Attention取代舊MGM/shared memory，建模LOOKBACK內跨時間關係。
- 一般A2C與A2C without TI仍使用Stable-Baselines3預設特徵流程。

## 注意事項

- `original/`只作為學長封存原碼參考，不參與目前4H流程。
- 模型ZIP、checkpoint、logs與大型資料不應提交Git。
- 變更資料頻率、LOOKBACK、reward或特徵後，舊模型不可直接比較。
- 正式結果應使用多個random seed報告平均值與標準差。
- 目前未建模交易手續費與滑價。

模型、結果與checkpoint標籤會依`config_4h.py`主要參數動態生成。