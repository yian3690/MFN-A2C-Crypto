"""評估以Portfolio Value作reward訓練的A2C baseline。"""

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.pv_evaluation import evaluate_pv_model
from src.pv_experiment import (
    A2C_PV_MODEL_NAME,
    A2C_PV_RESULT_NAME,
    PV_RUN_TAG,
)


def main():
    """執行A2C-PV完整Test回測。"""
    evaluate_pv_model(
        title="A2C BASELINE PV BACKTEST",
        model_name=A2C_PV_MODEL_NAME,
        result_name=A2C_PV_RESULT_NAME,
        diagnostics_pattern=f"a2c_baseline_PV_{PV_RUN_TAG}_*.csv",
    )


if __name__ == "__main__":
    main()
