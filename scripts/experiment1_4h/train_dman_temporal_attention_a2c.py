"""訓練保留雙LSTM/DMAN、以Temporal Self-Attention取代MGM的A2C。"""

from train_common import train


if __name__ == "__main__":
    train("dman_attention")
