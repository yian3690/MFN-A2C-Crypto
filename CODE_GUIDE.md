# 4H程式指南

## 設定與資料

- `scripts/config_4h.py`：4H資料期間、Train/Test、reward、模型名稱及輸出路徑。
- `scripts/download_data_4h.py`：下載Binance 4H K線。
- `scripts/prepare_features_4h.py`：產生Train-only z-score特徵與最後1,080根Test。
- `src/portfolio_env_sb3.py`：共用投資組合環境；動作作用於下一期報酬。

## 訓練與評估

- `scripts/train_common.py`：A2C、A2C without TI與DMAN Temporal Attention共用訓練及跨步數續訓。
- `scripts/evaluate_common.py`：上述三種A2C方法共用回測、資產歸因與診斷。
- `scripts/train_dman_temporal_attention_a2c.py`／`evaluate_dman_temporal_attention_a2c.py`：目前的雙LSTM＋DMAN＋Temporal Self-Attention架構。
- `scripts/train_dqn.py`／`evaluate_dqn.py`：Experiment 2的4H DQN。
- `scripts/experiment3_common.py`：Experiment 3的一般A2C absolute PV reward流程。

## 比較圖

- `scripts/compare_experiment1.py`：三種A2C架構與Buy-and-Hold表格。
- `scripts/compare_experiment2.py`：Temporal Attention、A2C、DQN、Buy-and-Hold曲線與表格。
- `scripts/compare_experiment3.py`：一般A2C的DSR/return與PV reward曲線及表格。

## 目錄規則

後續不再建立`experiment1_4h`子目錄：

```text
data/        4H資料
models/      4H最終模型
results/model_result/       各模型4H逐步結果與metrics
results/experiment_result/  Experiment彙整CSV
logs/        TensorBoard與訓練診斷
checkpoints/ 續訓checkpoint
figures/     Experiment比較圖
```