"""Train or resume the GitHub-style two-view MFN-A2C diagnostic model."""

import argparse
import re
import sys
from datetime import datetime
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from stable_baselines3.common.callbacks import CheckpointCallback
from stable_baselines3.common.monitor import Monitor

from src.dsr import PAPER_FORMULA
from src.experiment_periods import LOOKBACK
from src.feature_schema import INDICATOR_DIM, PRICE_DIM
from src.mfn_github_extractor import GitHubStyleTwoViewMFN
from src.multi_epoch_a2c import MultiEpochA2C
from src.portfolio_env_sb3 import CryptoPortfolioEnv
from src.simplex_policy import SimplexActorCriticPolicy
from src.training_diagnostics import TrainingDiagnosticsCallback


DATA = ROOT / "data"
MODELS = ROOT / "models"
LOGS = ROOT / "logs"
CHECKPOINTS = ROOT / "checkpoints"


# Short controlled run for the GitHub-style two-view MFN.
TOTAL_TIMESTEPS = 300_000
UPDATE_EPOCHS = 1
CHECKPOINT_PREFIX = "MFN_A2C_GITHUB2_5X20_PAPER_DSR_E1_300K"
MODEL_NAME = "mfn_a2c_github2_5x20_300k_paper_dsr"


def parse_args():
    """Parse command-line options without changing the paper parameters."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--resume",
        action="store_true",
        help="Continue from the newest compatible periodic checkpoint.",
    )
    return parser.parse_args()


def checkpoint_timesteps(path: str | Path) -> int | None:
    """Extract the absolute timestep from one compatible checkpoint name."""
    name = Path(path).name
    match = re.fullmatch(
        rf"{re.escape(CHECKPOINT_PREFIX)}_(\d+)_steps\.zip",
        name,
    )
    return int(match.group(1)) if match else None


def find_latest_checkpoint(directory: str | Path) -> Path | None:
    """Find the compatible checkpoint with the greatest saved timestep."""
    candidates = []
    for path in Path(directory).glob(f"{CHECKPOINT_PREFIX}_*_steps.zip"):
        timesteps = checkpoint_timesteps(path)
        if timesteps is not None:
            candidates.append((timesteps, path))
    return max(candidates, default=(None, None), key=lambda item: item[0])[1]


def make_train_env(paths, *, random_start, max_episode_steps):
    """Create the configured Gymnasium environment used by this script."""
    env = CryptoPortfolioEnv(
        pct_csv=str(paths["pct"]),
        ta_csv=str(paths["ta"]),
        raw_csv=str(paths["raw"]),

        # Paper setting
        n_previous_timesteps=LOOKBACK,

        # Episode length
        max_episode_steps=max_episode_steps,

        # DSR
        reward_type="dsr",
        eta=0.005,
        dsr_formula=PAPER_FORMULA,

        initial_balance=10000,

        random_start=random_start,
        action_mode="simplex",
    )

    return Monitor(env)


def main():
    """主程式入口：依序執行此腳本定義的完整流程。"""
    args = parse_args()
    MODELS.mkdir(parents=True, exist_ok=True)
    (LOGS / "tensorboard").mkdir(parents=True, exist_ok=True)
    CHECKPOINTS.mkdir(parents=True, exist_ok=True)

    train_paths = {
        "pct": DATA / "pct_change_output_train.csv",
        "ta": DATA / "ta_test_train.csv",
        "raw": DATA / "merged_output_train.csv",
    }
    env = make_train_env(
        train_paths,
        # A rollout is an optimizer collection interval, not an episode.
        # Traverse the complete chronological Train set so DSR EWMA moments
        # continue across consecutive 540-step rollouts.
        random_start=False,
        max_episode_steps=None,
    )

    policy_kwargs = dict(

        features_extractor_class=GitHubStyleTwoViewMFN,

        features_extractor_kwargs=dict(
            price_dim=PRICE_DIM,
            indicator_dim=INDICATOR_DIM,

            lstm_hidden=64,
            memory_dim=128,
            attention_hidden=64,
            candidate_hidden=64,
            gate_hidden=64,
            dropout=0.0,
        ),

        # Actor / Critic
        net_arch=dict(
            pi=[64, 64],
            vf=[64, 64],
        ),
    )

    output = MODELS / MODEL_NAME
    final_model_path = output.with_suffix(".zip")
    resume_path = None
    if args.resume:
        if final_model_path.exists():
            print(f"Training is already complete: {final_model_path}")
            env.close()
            return
        resume_path = find_latest_checkpoint(CHECKPOINTS)
        if resume_path is None:
            env.close()
            raise SystemExit(
                "No compatible checkpoint was found. Run without --resume "
                "to start a new training run."
            )
        model = MultiEpochA2C.load(
            str(resume_path),
            env=env,
            device="auto",
            tensorboard_log=str(LOGS / "tensorboard"),
        )
        completed_timesteps = int(model.num_timesteps)
    else:
        model = MultiEpochA2C(

            policy=SimplexActorCriticPolicy,

            env=env,

            # SB3 A2C
            learning_rate=7e-4,
            gamma=0.99,
            n_steps=540,
            update_epochs=UPDATE_EPOCHS,

            policy_kwargs=policy_kwargs,

            verbose=1,

            # RTX 3060
            device="auto",

            seed=123,

            tensorboard_log=str(
                LOGS / "tensorboard"
            ),
        )
        completed_timesteps = 0

    remaining_timesteps = max(TOTAL_TIMESTEPS - completed_timesteps, 0)
    if remaining_timesteps == 0:
        print(
            f"Checkpoint already reached {completed_timesteps:,} timesteps; "
            "nothing remains to train."
        )
        env.close()
        return

    checkpoint_callback = CheckpointCallback(
        save_freq=100_000,
        save_path=str(CHECKPOINTS),
        name_prefix=CHECKPOINT_PREFIX,
    )
    run_stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    diagnostics_path = (
        LOGS
        / "training_diagnostics"
        / (
            "mfn_a2c_5x20_paper_dsr_300k_"
            f"from_{completed_timesteps}_{run_stamp}.csv"
        )
    )
    diagnostics_callback = TrainingDiagnosticsCallback(diagnostics_path)

    print("=" * 60)
    print("MFN-A2C GITHUB-STYLE TRAINING")
    print("=" * 60)

    print(f"Total timesteps : {TOTAL_TIMESTEPS}")
    print(f"Completed       : {completed_timesteps}")
    print(f"Remaining       : {remaining_timesteps}")
    print(f"Resume          : {resume_path or 'no'}")
    print("Device          : auto")
    print("MFN             : 2-view")
    print("DMAN            : previous/current cell-state attention")
    print("MGM             : candidate MLP + memory-conditioned gates")
    print("Window          : 20")
    print("Interval        : 2H")
    print("Episode         : complete chronological Train set")
    print("Action policy   : Dirichlet simplex")
    print("DSR eta         : 0.005")
    print("DSR formula     : paper legacy EWMA moment changes")
    print("A2C gamma       : 0.99")
    print("A2C n_steps     : 540")
    print(f"Update epochs   : {UPDATE_EPOCHS}")
    print(f"Diagnostics CSV : {diagnostics_path}")
    if args.resume:
        print("Resume note     : environment/DSR state restarts from Train beginning")
    print("=" * 60)

    model.learn(
        total_timesteps=remaining_timesteps,
        reset_num_timesteps=not args.resume,

        # ★ 顯示進度條
        progress_bar=True,

        callback=[checkpoint_callback, diagnostics_callback],
    )

    model.save(str(output))
    model.save(
        str(
            CHECKPOINTS
            / "MFN_A2C_GITHUB2_5X20_PAPER_DSR_E1_300000_final"
        )
    )

    print()
    print("=" * 60)
    print("FORMAL TRAINING FINISHED")
    print("=" * 60)

    env.close()


if __name__ == "__main__":
    main()
