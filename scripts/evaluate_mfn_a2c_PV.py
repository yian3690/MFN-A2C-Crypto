"""評估以Portfolio Value作reward訓練的MFN-A2C。"""

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.pv_evaluation import evaluate_pv_model
from src.pv_experiment import (
    MFN_PV_MODEL_NAME,
    MFN_PV_RESULT_NAME,
    PV_RUN_TAG,
)


def main():
    """執行MFN-A2C-PV完整Test回測。"""
    evaluate_pv_model(
        title="MFN-A2C PV BACKTEST",
        model_name=MFN_PV_MODEL_NAME,
        result_name=MFN_PV_RESULT_NAME,
        diagnostics_pattern=f"mfn_a2c_PV_{PV_RUN_TAG}_*.csv",
    )


if __name__ == "__main__":
    main()
