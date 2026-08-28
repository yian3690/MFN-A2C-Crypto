from pathlib import Path

from stable_baselines3 import A2C
from stable_baselines3.common.callbacks import CheckpointCallback
from stable_baselines3.common.monitor import Monitor

from mfn_sb3_extractor import TwoViewMFN
from portfolio_env_sb3 import CryptoPortfolioEnv


ROOT = Path(__file__).resolve().parent
DATA = ROOT / "data"


#測試先用18000，正式訓練用1,800,000
TOTAL_TIMESTEPS = 1_800_0


def make_train_env():
    env = CryptoPortfolioEnv(
        pct_csv=str(DATA / "pct_change_output_train.csv"),
        ta_csv=str(DATA / "ta_test_train.csv"),
        raw_csv=str(DATA / "merged_output_train.csv"),

        # Paper setting
        n_previous_timesteps=20,

        # Episode length
        max_episode_steps=540,

        # DSR
        reward_type="dsr",
        eta=0.005,

        initial_balance=10000,

        random_start=True,
    )

    return Monitor(env)


def main():

    env = make_train_env()

    policy_kwargs = dict(

        features_extractor_class=TwoViewMFN,

        features_extractor_kwargs=dict(
            price_dim=16,
            indicator_dim=16,

            lstm_hidden=64,
            memory_dim=128,

            att_hidden=64,
            gate_hidden=64,

            output_dim=128,
            dropout=0.0,
        ),

        # Actor / Critic
        net_arch=dict(
            pi=[64, 64],
            vf=[64, 64],
        ),
    )

    model = A2C(

        policy="MlpPolicy",

        env=env,

        # SB3 A2C
        learning_rate=7e-4,
        gamma=0.99,
        n_steps=540,

        policy_kwargs=policy_kwargs,

        verbose=1,

        # RTX 3060
        device="auto",

        seed=123,

        tensorboard_log=str(
            ROOT / "tensorboard"
        ),
    )

    # 每 100,000 timestep 儲存一次
    checkpoint_callback = CheckpointCallback(
        save_freq=100_000,
        save_path=str(ROOT / "checkpoints"),
        name_prefix="MFN_A2C",
    )

    print("=" * 60)
    print("MFN-A2C FORMAL TRAINING")
    print("=" * 60)

    print(f"Total timesteps : {TOTAL_TIMESTEPS}")
    print("Device          : auto")
    print("MFN             : 2-view")
    print("Window          : 20")
    print("Interval        : 4H")
    print("DSR eta         : 0.005")
    print("A2C gamma       : 0.99")
    print("A2C n_steps     : 540")
    print("=" * 60)

    model.learn(
        total_timesteps=TOTAL_TIMESTEPS,

        # ★ 顯示進度條
        progress_bar=True,

        callback=checkpoint_callback,
    )

    model.save(
        str(ROOT / "mfn_a2c_formal")
    )

    print()
    print("=" * 60)
    print("FORMAL TRAINING FINISHED")
    print("=" * 60)

    env.close()


if __name__ == "__main__":
    main()