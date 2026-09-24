# Experiment 1：4H 版本

此資料夾將4小時實驗與既有2H的資料、模型、紀錄及結果分開。

## 共用設定

- Binance Spot 4H，BTC／ETH／LTC／BNB＋USDT。
- Window=20（80小時）。
- 最近1,080根為Test（180天），其餘資料為Train；不使用Validation。
- Train episode隨機抽取1,080根（180天）。
- 價格模態5維；每資產5個指標：SMA20、EMA20、MACD、RSI14、RS_14D。
- RS_14D在4H下為84根，只使用當下及過去資料。
- Train-only z-score；Test不參與scaler或模型選擇。
- `n_steps=540`、`update_epochs=1`、eta=0.005。
- Reward=`1 × paper DSR + 0 × log return`（目前為純DSR）。
- Gaussian logits→Softmax、`log_std_init=-2`、normalize advantage。
- 六種方法的訓練診斷統一每5個rollout記錄一次；不改變梯度或環境步進。

`config_4h.py`的`TOTAL_TIMESTEPS`是唯一目標步數設定。正式模型、結果及
診斷名稱中的`300k`／`600k`／`1800k`會自動更新；checkpoint的相容prefix
刻意不含目標步數，因此可跨目標步數續訓。

## 續訓

所有六種方法都支援`--resume`。例如先完成300k後：

1. 將`config_4h.py`的`TOTAL_TIMESTEPS`改為`600_000`。
2. 對同一方法加上`--resume`：

```powershell
python scripts\experiment1_4h\train_a2c.py --resume
python scripts\experiment1_4h\train_a2c_without_ti.py --resume
python scripts\experiment1_4h\train_mfn_a2c.py --resume
python scripts\experiment1_4h\train_dman_temporal_attention_a2c.py --resume
python scripts\experiment1_4h\train_asset_attention_mfn_a2c.py --resume
python scripts\experiment1_4h\train_self_attention_a2c.py --resume
```

程式會讀取該方法不超過目標步數的最新相容checkpoint，保留模型、optimizer
與累計步數，只訓練剩餘步數。找不到相容checkpoint時會停止，不會偷偷從零
開始。舊版名稱內含`300K`／`600K`的相同設定checkpoint也可讀取。

不加`--resume`永遠從零開始：

```powershell
python scripts\experiment1_4h\train_a2c.py
```

## 新增的三種架構

1. **Asset-wise Attention＋原MFN/MGM**
   - 每根K線先把五種資產各自的5個技術指標嵌入。
   - 對五個資產做Self-Attention。
   - 再送進原本雙LSTM、DMAN、MGM/shared memory與A2C。

2. **雙LSTM＋DMAN＋Temporal Self-Attention（乾淨MGM消融）**
   - Price與技術指標仍分別通過原本的兩個LSTM。
   - 每個時間點仍以原本DMAN融合兩個模態的前後cell state。
   - 只移除MGM的candidate memory與兩個gate。
   - 將全部20／40個DMAN融合向量送進跨時間步Self-Attention。
   - 以可學習attention pooling彙整時間，再送進A2C Actor/Critic。
   - 輸出維度維持256，讓下游A2C與原MFN盡量公平比較。

3. **Asset-temporal Self-Attention A2C（簡化無MGM版）**
   - 每個資產使用`[price, 5 indicators]`的20-step序列。
   - 五個資產共用同一個LSTM做時間編碼。
   - 對五個資產token做Self-Attention，再送進A2C。
   - 不使用原本的雙LSTM、DMAN、MGM或shared memory。

## 從零訓練與評估

```powershell
python scripts\experiment1_4h\download_data_4h.py
python scripts\experiment1_4h\prepare_features_4h.py

python scripts\experiment1_4h\train_a2c.py
python scripts\experiment1_4h\train_a2c_without_ti.py
python scripts\experiment1_4h\train_mfn_a2c.py
python scripts\experiment1_4h\train_dman_temporal_attention_a2c.py
python scripts\experiment1_4h\train_asset_attention_mfn_a2c.py
python scripts\experiment1_4h\train_self_attention_a2c.py

python scripts\experiment1_4h\evaluate_a2c.py
python scripts\experiment1_4h\evaluate_a2c_without_ti.py
python scripts\experiment1_4h\evaluate_mfn_a2c.py
python scripts\experiment1_4h\evaluate_dman_temporal_attention_a2c.py
python scripts\experiment1_4h\evaluate_asset_attention_mfn_a2c.py
python scripts\experiment1_4h\evaluate_self_attention_a2c.py
python scripts\experiment1_4h\evaluate_buy_and_hold.py
python scripts\experiment1_4h\compare_results.py
```

輸出位置：

- `data/experiment1_4h`
- `models/experiment1_4h`
- `logs/experiment1_4h`
- `results/experiment1_4h`
- `checkpoints/experiment1_4h`

`evaluate_buy_and_hold.py`採靜態Buy-and-Hold：第一筆交易時五項資產各投入
20%，之後固定持幣數量，不在每根K線重新調回等權。
