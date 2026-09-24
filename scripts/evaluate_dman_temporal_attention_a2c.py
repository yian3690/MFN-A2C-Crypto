"""評估保留雙LSTM/DMAN、以Temporal Self-Attention取代MGM的A2C。"""

from evaluate_common import evaluate


if __name__ == "__main__":
    evaluate("dman_attention")
