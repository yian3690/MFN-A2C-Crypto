import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from stable_baselines3 import A2C
from stable_baselines3.common.callbacks import CheckpointCallback
from stable_baselines3.common.monitor import Monitor

from src.portfolio_env_sb3 import CryptoPortfolioEnv


DATA = ROOT / "data"
MODELS = ROOT / "models"
LOGS = ROOT / "logs"
CHECKPOINTS = ROOT / "checkpoints_a2c"


TOTAL_TIMESTEPS = 100_000


def make_env():

    env = CryptoPortfolioEnv(
        pct_csv=str(
            DATA / "pct_change_output_train.csv"
        ),

        ta_csv=str(
            DATA / "ta_test_train.csv"
        ),

        raw_csv=str(
            DATA / "merged_output_train.csv"
        ),

        n_previous_timesteps=20,

        max_episode_steps=540,

        reward_type="dsr",

        eta=0.005,

        initial_balance=10000,

        random_start=True,
    )

    return Monitor(env)


def main():
    MODELS.mkdir(parents=True, exist_ok=True)
    (LOGS / "tensorboard").mkdir(parents=True, exist_ok=True)
    CHECKPOINTS.mkdir(parents=True, exist_ok=True)

    env = make_env()

    policy_kwargs = dict(

        net_arch=dict(
            pi=[64, 64],
            vf=[64, 64],
        )
    )

    model = A2C(

        policy="MlpPolicy",

        env=env,

        learning_rate=7e-4,

        gamma=0.99,

        n_steps=540,

        policy_kwargs=policy_kwargs,

        verbose=1,

        device="auto",

        seed=123,

        tensorboard_log=str(
            LOGS / "tensorboard"
        ),
    )

    checkpoint_callback = CheckpointCallback(
        save_freq=100_000,
        save_path=str(
            CHECKPOINTS
        ),
        name_prefix="A2C",
    )

    print("=" * 60)
    print("A2C BASELINE TRAINING")
    print("=" * 60)

    print(
        f"Total timesteps : {TOTAL_TIMESTEPS}"
    )

    print("MFN             : NO")
    print("Feature extractor: SB3 MLP")
    print("Window          : 20")
    print("Interval        : 4H")
    print("DSR eta         : 0.005")
    print("A2C gamma       : 0.99")
    print("A2C n_steps     : 540")

    print("=" * 60)

    model.learn(

        total_timesteps=TOTAL_TIMESTEPS,

        progress_bar=True,

        callback=checkpoint_callback,
    )

    model.save(
        str(MODELS / "a2c_baseline")
    )

    print()
    print("=" * 60)
    print("A2C BASELINE FINISHED")
    print("=" * 60)

    env.close()


if __name__ == "__main__":
    main()