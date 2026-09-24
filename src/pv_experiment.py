"""Portfolio Value reward消融的Train/Test共用設定。

除reward改為投資組合絕對總值外，資料、A2C超參數、動作轉換與
固定最後一步訓練方式皆沿用正式DSR／Hybrid實驗。
"""

from src.experiment_config import (
    A2C_ACTION_MODE,
    DSR_ETA,
    DSR_FORMULA,
    DSR_REWARD_MODE,
    DSR_REWARD_SCALE,
    INITIAL_BALANCE,
    RETURN_REWARD_SCALE,
    RUN_TAG,
    data_paths,
)
from src.experiment_periods import LOOKBACK
from src.portfolio_env_sb3 import CryptoPortfolioEnv


PV_RUN_TAG = f"{RUN_TAG}_reward_pv"
A2C_PV_MODEL_NAME = f"a2c_baseline_PV_{PV_RUN_TAG}"
MFN_PV_MODEL_NAME = f"mfn_a2c_PV_{PV_RUN_TAG}"
A2C_PV_RESULT_NAME = "exp3_a2c_pv_results.csv"
MFN_PV_RESULT_NAME = "exp3_mfn_pv_results.csv"


def make_pv_env(split: str) -> CryptoPortfolioEnv:
    """建立Train或Test環境；唯一消融變因是reward_type='pv'。"""
    paths = data_paths(split)
    return CryptoPortfolioEnv(
        pct_csv=str(paths["pct"]),
        ta_csv=str(paths["ta"]),
        raw_csv=str(paths["raw"]),
        n_previous_timesteps=LOOKBACK,
        max_episode_steps=None,
        reward_type="pv",
        eta=DSR_ETA,
        dsr_reward_scale=DSR_REWARD_SCALE,
        return_reward_scale=RETURN_REWARD_SCALE,
        dsr_reward_mode=DSR_REWARD_MODE,
        dsr_formula=DSR_FORMULA,
        initial_balance=INITIAL_BALANCE,
        random_start=False,
        action_mode=A2C_ACTION_MODE,
    )
