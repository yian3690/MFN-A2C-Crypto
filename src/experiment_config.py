"""四種正式比較方法共用的實驗設定。

此檔案只集中管理「應該相同」的資料、環境與訓練預算。
A2C 與 DQN 的演算法專屬參數仍分開保存，避免把不適用於
DQN 的 rollout 或 update epochs 強行套用。
"""

from __future__ import annotations

from pathlib import Path

from src.dsr import PAPER_FORMULA
from src.experiment_periods import LOOKBACK, VALIDATION_ROWS
from src.feature_schema import RELATIVE_STRENGTH_NAME
from src.portfolio_env_sb3 import CryptoPortfolioEnv


# ---------------------------------------------------------------------------
# 專案路徑：所有 train/evaluate script 都從這裡取得相同資料位置。
# ---------------------------------------------------------------------------
ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data"
MODELS = ROOT / "models"
LOGS = ROOT / "logs"
RESULTS = ROOT / "results"


# ---------------------------------------------------------------------------
# 四種方法共同設定：相同市場資料、隨機種子與環境互動步數。
# 模型只使用Train訓練，並以Validation Final PV選最佳checkpoint；
# 不再重新初始化或執行Stage 2。
# ---------------------------------------------------------------------------
TOTAL_TIMESTEPS = 600_000
SEED = 456
LEARNING_RATE = 7e-4
GAMMA = 0.99
INITIAL_BALANCE = 10_000.0
DSR_ETA = 0.005
DSR_FORMULA = PAPER_FORMULA
# raw DSR直接使用論文innovation公式；eta只控制EWMA moments。
# reward scale與公式分離，1.0代表不再用1/eta補償或隱含縮放。
DSR_REWARD_SCALE = 1.0
RETURN_REWARD_SCALE = 50.0
DSR_REWARD_MODE = "step"
REWARD_TYPE = "hybrid"
CHECKPOINT_FREQUENCY = 100_000
VALIDATION_FREQUENCY = 50_000
NETWORK_ARCHITECTURE = (64, 64)


# ---------------------------------------------------------------------------
# A2C 共用設定：MFN、baseline、without-TI 都使用同一更新方式。
# Gaussian policy 先輸出 logits，再由環境 Softmax 成合法投資比例。
# ---------------------------------------------------------------------------
A2C_N_STEPS = 540
A2C_UPDATE_EPOCHS = 1
A2C_ACTION_MODE = "logits"
# 將每個540-step rollout的Advantage標準化為近似零均值與單位標準差，
# 避免單一後期rollout因Advantage整體偏正／偏負而造成Actor劇烈漂移。
A2C_NORMALIZE_ADVANTAGE = True
# SB3預設為0（std=1）；本輪降低探索噪音，測試訓練抽樣配置與
# deterministic部署配置差距過大的問題。
A2C_LOG_STD_INIT = -2


# ---------------------------------------------------------------------------
# DQN 專屬設定：DQN 沒有 A2C rollout epochs，使用 replay buffer。
# DiscretePortfolioWrapper 已輸出 simplex 權重，所以底層環境不可再 Softmax。
# ---------------------------------------------------------------------------
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


# 特徵版本寫入模型與結果檔名。目前恢復指標level＋Train-only z-score；
# 學長原碼的senior_ta消融模型與結果仍保留，不會被本輪覆蓋。
FEATURE_VARIANT = (
    f"level_zscore_{RELATIVE_STRENGTH_NAME.lower().replace('_', '')}"
)
STEP_TAG = f"{TOTAL_TIMESTEPS // 1000}k"


def _number_tag(value: float | int) -> str:
    """將實驗數值轉成穩定檔名片段，例如-2→m2、0.005→0p005。"""
    number = float(value)
    sign = "m" if number < 0 else ""
    magnitude = f"{abs(number):g}".replace(".", "p")
    return f"{sign}{magnitude}"


RUN_TAG = (
    f"valselect_{STEP_TAG}_{REWARD_TYPE}_paperdsr"
    f"{_number_tag(DSR_REWARD_SCALE)}_ret"
    f"{_number_tag(RETURN_REWARD_SCALE)}_eta{_number_tag(DSR_ETA)}_"
    f"win20_gaussian_logstd{_number_tag(A2C_LOG_STD_INIT)}_"
    f"normadv_e{A2C_UPDATE_EPOCHS}_seed{SEED}_"
    f"{FEATURE_VARIANT}_val{VALIDATION_ROWS}_pv50k"
)
MFN_MODEL_NAME = f"mfn_a2c_{RUN_TAG}"
A2C_MODEL_NAME = f"a2c_baseline_{RUN_TAG}"
A2C_WITHOUT_TI_MODEL_NAME = f"a2c_without_ti_{RUN_TAG}"
DQN_MODEL_NAME = f"dqn_baseline_{RUN_TAG}"

MFN_RESULT_NAME = f"mfn_a2c_{RUN_TAG}_results.csv"
A2C_RESULT_NAME = f"a2c_baseline_{RUN_TAG}_results.csv"
A2C_WITHOUT_TI_RESULT_NAME = f"a2c_without_ti_{RUN_TAG}_results.csv"
DQN_RESULT_NAME = f"dqn_baseline_{RUN_TAG}_results.csv"


def data_paths(split: str) -> dict[str, Path]:
    """回傳完整Train、保留切分與Test所需的三份已對齊資料路徑。"""
    if split not in {"train", "validation", "development", "test"}:
        raise ValueError(
            "split必須是'train'、'validation'、'development'或'test'。"
        )
    return {
        "pct": DATA / f"pct_change_output_{split}.csv",
        "ta": DATA / f"ta_test_{split}.csv",
        "raw": DATA / f"merged_output_{split}.csv",
    }


def make_portfolio_env(
    split: str,
    *,
    action_mode: str,
) -> CryptoPortfolioEnv:
    """建立所有方法共用的時間順序投資環境。

    Development不把540-step rollout誤當episode，會沿完整Train依序前進；
    Test固定使用扣除20步觀察窗後的完整保留區間。
    """
    paths = data_paths(split)
    max_episode_steps = None
    return CryptoPortfolioEnv(
        pct_csv=str(paths["pct"]),
        ta_csv=str(paths["ta"]),
        raw_csv=str(paths["raw"]),
        n_previous_timesteps=LOOKBACK,
        max_episode_steps=max_episode_steps,
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
    """每次回傳新的 A2C Actor/Critic 網路設定，避免共享可變 dict。"""
    layers = list(NETWORK_ARCHITECTURE)
    return {
        "net_arch": {"pi": layers.copy(), "vf": layers.copy()},
        "log_std_init": A2C_LOG_STD_INIT,
    }


def dqn_policy_kwargs() -> dict:
    """回傳 DQN 專屬 MLP 網路設定。"""
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
    """回傳DQN專屬參數；不包含不存在於DQN的update epochs。"""
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
