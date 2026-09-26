"""所有4H實驗的單一設定來源。"""

from __future__ import annotations

from datetime import datetime, timedelta, timezone
from pathlib import Path

from src.dsr import EWMA_CHANGE_FORMULA, PAPER_FORMULA
from src.portfolio_env_sb3 import CryptoPortfolioEnv

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data"
MODELS = ROOT / "models"
LOGS = ROOT / "logs"
RESULTS = ROOT / "results"
MODEL_RESULTS = RESULTS / "model_result"
EXPERIMENT_RESULTS = RESULTS / "experiment_result"
CHECKPOINTS = ROOT / "checkpoints"
FIGURES = ROOT / "figures"

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
RELATIVE_STRENGTH_NAME = f"RS_{RELATIVE_STRENGTH_DAYS}D"
TECHNICAL_INDICATORS = (
    "SMA20", "EMA20", "MACD", "RSI14", RELATIVE_STRENGTH_NAME,
)
PRICE_DIM = len(PORTFOLIO_ASSETS)
INDICATORS_PER_ASSET = len(TECHNICAL_INDICATORS)
INDICATOR_DIM = len(PORTFOLIO_ASSETS) * INDICATORS_PER_ASSET

# 修改此數值後，模型、結果與checkpoint標籤會自動更新。
TOTAL_TIMESTEPS = 900_000
STEP_TAG = f"{TOTAL_TIMESTEPS // 1000}k"
SEED = 456
LEARNING_RATE = 7e-4
GAMMA = 0.99
INITIAL_BALANCE = 10_000.0
DSR_ETA = 0.005
# 訓練仍沿用目前模型所使用的innovation DSR，避免只為重畫圖而改變模型。
DSR_FORMULA = PAPER_FORMULA
# 正式評估依論文方法章：把EWMA moments的實際變化代入DSR。
EVALUATION_DSR_FORMULA = EWMA_CHANGE_FORMULA
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
ATTENTION_DEVICE = "cpu"

# Experiment 2的DQN採離散simplex網格。
DQN_ACTION_MODE = "simplex"
DQN_LEARNING_RATE = 7e-4
DQN_BUFFER_SIZE = 100_000
DQN_LEARNING_STARTS = 10_000
DQN_BATCH_SIZE = 64
DQN_TRAIN_FREQUENCY = 4
DQN_GRADIENT_STEPS = 1
DQN_TARGET_UPDATE_INTERVAL = 10_000
DQN_EXPLORATION_FRACTION = 0.3
DQN_EXPLORATION_FINAL_EPS = 0.05
DQN_WEIGHT_STEP = 0.05
DQN_MAX_CRYPTO_WEIGHT = 0.35
DQN_DIAGNOSTICS_FREQUENCY = 10_000

def number_tag(value: int | float) -> str:
    """把數值轉為安全檔名標籤，例如-2→m2、0.005→0p005。"""
    numeric = float(value)
    if not numeric == numeric or numeric in {float("inf"), float("-inf")}:
        raise ValueError(f"標籤數值必須為有限值：{value}")
    text = f"{abs(numeric):g}".replace(".", "p")
    return f"m{text}" if numeric < 0 else text


ADVANTAGE_TAG = "normadv" if A2C_NORMALIZE_ADVANTAGE else "rawadv"

# 所有會影響實驗身分的主要設定均動態納入標籤。
RUN_TAG = (
    f"{BAR_HOURS}h_noval_random{TRAIN_EPISODE_DAYS}d_{STEP_TAG}_"
    f"{REWARD_TYPE}_paperdsr{number_tag(DSR_REWARD_SCALE)}_"
    f"ret{number_tag(RETURN_REWARD_SCALE)}_eta{number_tag(DSR_ETA)}_"
    f"win{LOOKBACK}_gaussian_logstd{number_tag(A2C_LOG_STD_INIT)}_"
    f"{ADVANTAGE_TAG}_e{A2C_UPDATE_EPOCHS}_seed{SEED}_"
    "level_zscore_rs14d"
)
MODEL_NAMES = {
    "a2c": f"a2c_baseline_{RUN_TAG}",
    "without_ti": f"a2c_without_ti_{RUN_TAG}",
    "dman_attention": f"dman_temporal_attention_a2c_{RUN_TAG}_cpu_diag5",
}
RESULT_NAMES = {
    key: f"{value}_results.csv" for key, value in MODEL_NAMES.items()
}
def compact_int_tag(value: int) -> str:
    """將DQN大數超參數縮短成檔名標籤。"""
    if value >= 1000 and value % 1000 == 0:
        return f"{value // 1000}k"
    return str(value)


# DQN不沿用A2C的Gaussian與advantage標籤；離散網格、replay與探索設定
# 都會改變實驗身分，因此必須出現在模型與結果檔名中。
DQN_RUN_TAG = (
    f"{BAR_HOURS}h_noval_random{TRAIN_EPISODE_DAYS}d_{STEP_TAG}_"
    f"{REWARD_TYPE}_paperdsr{number_tag(DSR_REWARD_SCALE)}_"
    f"ret{number_tag(RETURN_REWARD_SCALE)}_"
    f"eta{number_tag(DSR_ETA)}_win{LOOKBACK}_seed{SEED}_level_zscore_rs14d_"
    f"ws{number_tag(DQN_WEIGHT_STEP)}_cap{number_tag(DQN_MAX_CRYPTO_WEIGHT)}_"
    f"buf{compact_int_tag(DQN_BUFFER_SIZE)}_"
    f"start{compact_int_tag(DQN_LEARNING_STARTS)}_b{DQN_BATCH_SIZE}_"
    f"tf{DQN_TRAIN_FREQUENCY}_g{DQN_GRADIENT_STEPS}_"
    f"tgt{compact_int_tag(DQN_TARGET_UPDATE_INTERVAL)}_"
    f"ex{number_tag(DQN_EXPLORATION_FRACTION)}to"
    f"{number_tag(DQN_EXPLORATION_FINAL_EPS)}_lr{number_tag(DQN_LEARNING_RATE)}"
)
DQN_MODEL_NAME = f"dqn_baseline_{DQN_RUN_TAG}"
DQN_RESULT_NAME = f"{DQN_MODEL_NAME}_results.csv"
PORTFOLIO_RETURN_RUN_TAG = (
    f"{BAR_HOURS}h_noval_random{TRAIN_EPISODE_DAYS}d_{STEP_TAG}_"
    f"reward_portfolio_return_win{LOOKBACK}_"
    f"gaussian_logstd{number_tag(A2C_LOG_STD_INIT)}_"
    f"{ADVANTAGE_TAG}_e{A2C_UPDATE_EPOCHS}_seed{SEED}_"
    "level_zscore_rs14d"
)
A2C_RETURN_MODEL_NAME = f"a2c_return_{PORTFOLIO_RETURN_RUN_TAG}"
A2C_RETURN_RESULT_NAME = f"{A2C_RETURN_MODEL_NAME}_results.csv"
PROPOSED_RETURN_MODEL_NAME = (
    "dman_temporal_attention_a2c_return_"
    f"{PORTFOLIO_RETURN_RUN_TAG}_cpu_diag5"
)
PROPOSED_RETURN_RESULT_NAME = f"{PROPOSED_RETURN_MODEL_NAME}_results.csv"


def data_paths(split: str) -> dict[str, Path]:
    """取得根data資料夾中的4H Train/Test資料。"""
    if split not in {"train", "test"}:
        raise ValueError(f"未知資料切分：{split}；目前僅使用train/test。")
    return {
        "pct": DATA / f"pct_change_output_{split}.csv",
        "ta": DATA / f"ta_test_{split}.csv",
        "raw": DATA / f"merged_output_{split}.csv",
    }


def make_portfolio_env(
    split: str,
    *,
    action_mode: str = A2C_ACTION_MODE,
    reward_type: str = REWARD_TYPE,
):
    """Train隨機抽180天，Test固定使用完整保留期間。"""
    paths = data_paths(split)
    is_train = split == "train"
    return CryptoPortfolioEnv(
        pct_csv=str(paths["pct"]),
        ta_csv=str(paths["ta"]),
        raw_csv=str(paths["raw"]),
        n_previous_timesteps=LOOKBACK,
        max_episode_steps=(TRAIN_EPISODE_STEPS if is_train else None),
        reward_type=reward_type,
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


def dqn_algorithm_kwargs() -> dict:
    """回傳4H Experiment 2的DQN超參數。"""
    return {
        "learning_rate": DQN_LEARNING_RATE,
        "gamma": GAMMA,
        "buffer_size": DQN_BUFFER_SIZE,
        "learning_starts": DQN_LEARNING_STARTS,
        "batch_size": DQN_BATCH_SIZE,
        "train_freq": DQN_TRAIN_FREQUENCY,
        "gradient_steps": DQN_GRADIENT_STEPS,
        "target_update_interval": DQN_TARGET_UPDATE_INTERVAL,
        "exploration_fraction": DQN_EXPLORATION_FRACTION,
        "exploration_final_eps": DQN_EXPLORATION_FINAL_EPS,
        "policy_kwargs": {"net_arch": list(NETWORK_ARCHITECTURE)},
        "verbose": 1,
        "device": "auto",
        "seed": SEED,
        "tensorboard_log": str(LOGS / "tensorboard"),
    }
