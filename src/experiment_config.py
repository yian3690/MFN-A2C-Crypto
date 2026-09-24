"""所有正式方法共用的Train/Test、環境與訓練設定。"""

from __future__ import annotations

from pathlib import Path

from src.dsr import PAPER_FORMULA
from src.experiment_periods import EXPECTED_TRAIN_ROWS, LOOKBACK
from src.feature_schema import RELATIVE_STRENGTH_NAME
from src.portfolio_env_sb3 import CryptoPortfolioEnv


ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data"
MODELS = ROOT / "models"
LOGS = ROOT / "logs"
RESULTS = ROOT / "results"

# 論文重現採固定訓練預算，使用最後一步模型；Test不參與模型選擇。
TOTAL_TIMESTEPS = 300_000
SEED = 456
LEARNING_RATE = 7e-4
GAMMA = 0.99
INITIAL_BALANCE = 10_000.0
DSR_ETA = 0.005
DSR_FORMULA = PAPER_FORMULA
DSR_REWARD_SCALE = 1.0
RETURN_REWARD_SCALE = 50.0
DSR_REWARD_MODE = "step"
REWARD_TYPE = "hybrid"
CHECKPOINT_FREQUENCY = 100_000
NETWORK_ARCHITECTURE = (64, 64)

MFN_DEVICE = "cpu"
MFN_DIAGNOSTICS_EVERY_ROLLOUTS = 5

A2C_N_STEPS = 540
A2C_UPDATE_EPOCHS = 1
A2C_ACTION_MODE = "logits"
A2C_NORMALIZE_ADVANTAGE = True
A2C_LOG_STD_INIT = -2

DQN_ACTION_MODE = "simplex"
DQN_BUFFER_SIZE = 100_000
DQN_LEARNING_STARTS = 10_000
DQN_BATCH_SIZE = 64
DQN_TRAIN_FREQUENCY = 4
DQN_GRADIENT_STEPS = 1
DQN_TARGET_UPDATE_INTERVAL = 10_000
DQN_EXPLORATION_FRACTION = 0.1
DQN_EXPLORATION_FINAL_EPS = 0.05
DQN_WEIGHT_STEP = 0.2
DQN_MAX_CRYPTO_WEIGHT = 0.60

FEATURE_VARIANT = f"level_zscore_{RELATIVE_STRENGTH_NAME.lower().replace('_', '')}"
STEP_TAG = f"{TOTAL_TIMESTEPS // 1000}k"


def _number_tag(value: float | int) -> str:
    """將數值轉成穩定檔名片段，例如-2→m2、0.005→0p005。"""
    number = float(value)
    sign = "m" if number < 0 else ""
    magnitude = f"{abs(number):g}".replace(".", "p")
    return f"{sign}{magnitude}"


RUN_COMPAT_TAG = (
    f"{REWARD_TYPE}_paperdsr"
    f"{_number_tag(DSR_REWARD_SCALE)}_ret"
    f"{_number_tag(RETURN_REWARD_SCALE)}_eta{_number_tag(DSR_ETA)}_"
    f"win20_gaussian_logstd{_number_tag(A2C_LOG_STD_INIT)}_"
    f"normadv_e{A2C_UPDATE_EPOCHS}_seed{SEED}_"
    f"{FEATURE_VARIANT}_train{EXPECTED_TRAIN_ROWS}_fixedfinal"
)
RUN_TAG = f"fixedfinal_{STEP_TAG}_{RUN_COMPAT_TAG}"
MFN_RUN_TAG = f"{RUN_TAG}_{MFN_DEVICE}_diag{MFN_DIAGNOSTICS_EVERY_ROLLOUTS}"
MFN_RESUME_TAG = f"github2_lstm64_mem128_{RUN_COMPAT_TAG}"

MFN_MODEL_NAME = f"mfn_a2c_{MFN_RUN_TAG}"
A2C_MODEL_NAME = f"a2c_baseline_{RUN_TAG}"
A2C_WITHOUT_TI_MODEL_NAME = f"a2c_without_ti_{RUN_TAG}"
DQN_MODEL_NAME = f"dqn_baseline_{RUN_TAG}"

MFN_RESULT_NAME = f"mfn_a2c_{MFN_RUN_TAG}_results.csv"
A2C_RESULT_NAME = f"a2c_baseline_{RUN_TAG}_results.csv"
A2C_WITHOUT_TI_RESULT_NAME = f"a2c_without_ti_{RUN_TAG}_results.csv"
DQN_RESULT_NAME = f"dqn_baseline_{RUN_TAG}_results.csv"


def data_paths(split: str) -> dict[str, Path]:
    """回傳32,444筆Train或1,080筆Test的三份對齊資料。"""
    if split not in {"train", "test"}:
        raise ValueError("split必須是'train'或'test'。")
    return {
        "pct": DATA / f"pct_change_output_{split}.csv",
        "ta": DATA / f"ta_test_{split}.csv",
        "raw": DATA / f"merged_output_{split}.csv",
    }


def make_portfolio_env(split: str, *, action_mode: str) -> CryptoPortfolioEnv:
    """建立完整chronological Train或Test環境。"""
    paths = data_paths(split)
    return CryptoPortfolioEnv(
        pct_csv=str(paths["pct"]),
        ta_csv=str(paths["ta"]),
        raw_csv=str(paths["raw"]),
        n_previous_timesteps=LOOKBACK,
        max_episode_steps=None,
        reward_type=REWARD_TYPE,
        eta=DSR_ETA,
        dsr_reward_scale=DSR_REWARD_SCALE,
        return_reward_scale=RETURN_REWARD_SCALE,
        dsr_reward_mode=DSR_REWARD_MODE,
        dsr_formula=DSR_FORMULA,
        initial_balance=INITIAL_BALANCE,
        random_start=False,
        action_mode=action_mode,
    )


def a2c_policy_kwargs() -> dict:
    """回傳新的A2C Actor/Critic網路設定。"""
    layers = list(NETWORK_ARCHITECTURE)
    return {
        "net_arch": {"pi": layers.copy(), "vf": layers.copy()},
        "log_std_init": A2C_LOG_STD_INIT,
    }


def dqn_policy_kwargs() -> dict:
    """回傳DQN專屬MLP設定。"""
    return {"net_arch": list(NETWORK_ARCHITECTURE)}


def a2c_algorithm_kwargs() -> dict:
    """回傳三個A2C方法完全相同的演算法參數。"""
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
    """回傳DQN專屬演算法參數。"""
    return {
        "learning_rate": LEARNING_RATE,
        "gamma": GAMMA,
        "buffer_size": DQN_BUFFER_SIZE,
        "learning_starts": DQN_LEARNING_STARTS,
        "batch_size": DQN_BATCH_SIZE,
        "train_freq": DQN_TRAIN_FREQUENCY,
        "gradient_steps": DQN_GRADIENT_STEPS,
        "target_update_interval": DQN_TARGET_UPDATE_INTERVAL,
        "exploration_fraction": DQN_EXPLORATION_FRACTION,
        "exploration_final_eps": DQN_EXPLORATION_FINAL_EPS,
        "policy_kwargs": dqn_policy_kwargs(),
        "verbose": 1,
        "seed": SEED,
        "tensorboard_log": str(LOGS / "tensorboard"),
        "device": "auto",
    }
