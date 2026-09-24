"""4H Experiment 1 的單一設定來源（與既有2H輸出隔離）。"""

from __future__ import annotations

from datetime import datetime, timedelta, timezone
from pathlib import Path

from src.dsr import PAPER_FORMULA
from src.portfolio_env_sb3 import CryptoPortfolioEnv

ROOT = Path(__file__).resolve().parents[2]
DATA = ROOT / "data" / "experiment1_4h"
MODELS = ROOT / "models" / "experiment1_4h"
LOGS = ROOT / "logs" / "experiment1_4h"
RESULTS = ROOT / "results" / "experiment1_4h"
CHECKPOINTS = ROOT / "checkpoints" / "experiment1_4h"

UTC = timezone.utc
BAR_HOURS = 4
BAR_INTERVAL = timedelta(hours=BAR_HOURS)
BARS_PER_DAY = 24 // BAR_HOURS
PERIODS_PER_YEAR = BARS_PER_DAY * 365
LOOKBACK = 20
DATA_START = datetime(2018, 1, 1, tzinfo=UTC)
DATA_END_INCLUSIVE = datetime(2025, 9, 1, tzinfo=UTC)
TEST_END_EXCLUSIVE = DATA_END_INCLUSIVE + BAR_INTERVAL
TEST_ROWS = 1080
TEST_START = TEST_END_EXCLUSIVE - TEST_ROWS * BAR_INTERVAL

# Train不使用Validation，每個episode隨機抽取180天。
TRAIN_EPISODE_DAYS = 180
TRAIN_EPISODE_STEPS = TRAIN_EPISODE_DAYS * BARS_PER_DAY

CRYPTO_ASSETS = ("BTC", "ETH", "LTC", "BNB")
PORTFOLIO_ASSETS = (*CRYPTO_ASSETS, "USDT")
RELATIVE_STRENGTH_DAYS = 14
RELATIVE_STRENGTH_BARS = RELATIVE_STRENGTH_DAYS * BARS_PER_DAY
RELATIVE_STRENGTH_NAME = "RS_14D"
TECHNICAL_INDICATORS = (
    "SMA20", "EMA20", "MACD", "RSI14", RELATIVE_STRENGTH_NAME,
)
PRICE_DIM = len(PORTFOLIO_ASSETS)
INDICATORS_PER_ASSET = len(TECHNICAL_INDICATORS)
INDICATOR_DIM = len(PORTFOLIO_ASSETS) * INDICATORS_PER_ASSET

# 修改此數值後，正式模型／結果標籤會自動更新；續訓相容標籤不含步數。
TOTAL_TIMESTEPS = 300_000
STEP_TAG = f"{TOTAL_TIMESTEPS // 1000}k"
SEED = 456
LEARNING_RATE = 7e-4
GAMMA = 0.99
INITIAL_BALANCE = 10_000.0
DSR_ETA = 0.005
DSR_FORMULA = PAPER_FORMULA
DSR_REWARD_SCALE = 1.0
RETURN_REWARD_SCALE = 0.0
DSR_REWARD_MODE = "step"
REWARD_TYPE = "hybrid"
CHECKPOINT_FREQUENCY = 100_000
DIAGNOSTICS_EVERY_ROLLOUTS = 5
NETWORK_ARCHITECTURE = (64, 64)

A2C_N_STEPS = 540
A2C_UPDATE_EPOCHS = 1
A2C_ACTION_MODE = "logits"
A2C_NORMALIZE_ADVANTAGE = True
A2C_LOG_STD_INIT = -2
MFN_DEVICE = "cpu"

# LOOKBACK也納入標籤：win20與win40的模型／結果／checkpoint完全分離。
RUN_TAG = (
    f"4h_noval_random180d_{STEP_TAG}_hybrid_paperdsr1_ret0_eta0p005_"
    f"win{LOOKBACK}_gaussian_logstdm2_normadv_e1_seed456_"
    "level_zscore_rs14d"
)
MODEL_NAMES = {
    "a2c": f"a2c_baseline_{RUN_TAG}",
    "without_ti": f"a2c_without_ti_{RUN_TAG}",
    "mfn": f"mfn_a2c_{RUN_TAG}_cpu_diag5",
    "dman_attention": f"dman_temporal_attention_a2c_{RUN_TAG}_cpu_diag5",
    "asset_mfn": f"asset_attention_mfn_a2c_{RUN_TAG}_cpu_diag5",
    "self_attention": f"asset_temporal_self_attention_a2c_{RUN_TAG}_cpu_diag5",
}
RESULT_NAMES = {
    key: f"{value}_results.csv" for key, value in MODEL_NAMES.items()
}


def data_paths(split: str) -> dict[str, Path]:
    """取得獨立4H Train/Test資料。"""
    if split not in {"train", "test"}:
        raise ValueError(f"未知資料切分：{split}；目前僅使用train/test。")
    return {
        "pct": DATA / f"pct_change_output_{split}.csv",
        "ta": DATA / f"ta_test_{split}.csv",
        "raw": DATA / f"merged_output_{split}.csv",
    }


def make_portfolio_env(split: str, *, action_mode: str = A2C_ACTION_MODE):
    """Train隨機抽180天，Test固定使用完整保留期間。"""
    paths = data_paths(split)
    is_train = split == "train"
    return CryptoPortfolioEnv(
        pct_csv=str(paths["pct"]),
        ta_csv=str(paths["ta"]),
        raw_csv=str(paths["raw"]),
        n_previous_timesteps=LOOKBACK,
        max_episode_steps=(TRAIN_EPISODE_STEPS if is_train else None),
        reward_type=REWARD_TYPE,
        eta=DSR_ETA,
        dsr_reward_scale=DSR_REWARD_SCALE,
        return_reward_scale=RETURN_REWARD_SCALE,
        dsr_reward_mode=DSR_REWARD_MODE,
        dsr_formula=DSR_FORMULA,
        initial_balance=INITIAL_BALANCE,
        random_start=is_train,
        action_mode=action_mode,
    )


def a2c_policy_kwargs() -> dict:
    layers = list(NETWORK_ARCHITECTURE)
    return {
        "net_arch": {"pi": layers.copy(), "vf": layers.copy()},
        "log_std_init": A2C_LOG_STD_INIT,
    }


def a2c_algorithm_kwargs() -> dict:
    return {
        "learning_rate": LEARNING_RATE,
        "gamma": GAMMA,
        "n_steps": A2C_N_STEPS,
        "normalize_advantage": A2C_NORMALIZE_ADVANTAGE,
        "update_epochs": A2C_UPDATE_EPOCHS,
        "policy_kwargs": a2c_policy_kwargs(),
        "verbose": 1,
        "device": "auto",
        "seed": SEED,
        "tensorboard_log": str(LOGS / "tensorboard"),
    }
