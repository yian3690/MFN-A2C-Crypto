"""以4H Train資料訓練Experiment 2的離散動作DQN。"""

from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from stable_baselines3 import DQN
from stable_baselines3.common.callbacks import CheckpointCallback
from stable_baselines3.common.monitor import Monitor

import config_4h as cfg
from src.discrete_action_env import DiscretePortfolioWrapper


def main() -> None:
    """使用與其他4H方法相同的Train/Test切分與純DSR reward。"""
    for directory in (cfg.MODELS, cfg.LOGS / "tensorboard", cfg.CHECKPOINTS / "dqn"):
        directory.mkdir(parents=True, exist_ok=True)
    base_env = cfg.make_portfolio_env("train", action_mode=cfg.DQN_ACTION_MODE)
    env = Monitor(DiscretePortfolioWrapper(
        base_env,
        weight_step=cfg.DQN_WEIGHT_STEP,
        max_crypto_weight=cfg.DQN_MAX_CRYPTO_WEIGHT,
    ))
    model = DQN("MlpPolicy", env, **cfg.dqn_algorithm_kwargs())
    checkpoint = CheckpointCallback(
        save_freq=cfg.CHECKPOINT_FREQUENCY,
        save_path=str(cfg.CHECKPOINTS / "dqn"),
        name_prefix=cfg.DQN_MODEL_NAME.upper(),
    )
    print("=" * 72)
    print("4H EXPERIMENT 2 - DQN BASELINE")
    print("=" * 72)
    print(f"Total timesteps : {cfg.TOTAL_TIMESTEPS:,}")
    print(f"Train episode   : {cfg.TRAIN_EPISODE_DAYS} days, random start")
    print(f"Window          : {cfg.LOOKBACK} ({cfg.LOOKBACK * cfg.BAR_HOURS} hours)")
    print(f"Discrete actions: {env.action_space.n}")
    print("Validation      : disabled")
    model.learn(total_timesteps=cfg.TOTAL_TIMESTEPS, callback=checkpoint, progress_bar=True)
    output = cfg.MODELS / cfg.DQN_MODEL_NAME
    model.save(str(output))
    env.close()
    print(f"Saved final model: {output}.zip")


if __name__ == "__main__":
    main()
