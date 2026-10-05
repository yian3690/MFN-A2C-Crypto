"""訓練 4H Original MFN-A2C（雙 LSTM＋DMAN＋MGM、DSR reward）。"""

from train_common import train


if __name__ == "__main__":
    train("original_mfn")
