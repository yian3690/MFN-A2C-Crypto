"""技術指標前處理版本。

此模組刻意把「學長封存原碼」的轉換集中在同一處，讓資料產生
腳本與單元測試共用完全相同的公式，避免日後再次出現公式漂移。
"""

from __future__ import annotations

import pandas as pd
import pandas_ta as ta


def build_senior_original_indicators(close: pd.Series) -> pd.DataFrame:
    """依學長封存原碼，從一項資產的 Close 建立四個技術特徵。

    學長原碼不是直接輸入指標水準，而是：

    - SMA-20、EMA-20、MACD(DIF)：相鄰指標值的百分比變化 × 100。
    - RSI-14：先以 50 為中心，再縮小為原來的十分之一。

    MACD 保留封存原碼使用的 ``MACD_12_26_9``，也就是
    EMA-12 減 EMA-26 的 DIF／MACD line，而不是九期 signal line。
    """
    close = pd.to_numeric(close, errors="coerce")
    sma = ta.sma(close, length=20)
    ema = ta.ema(close, length=20)
    rsi = ta.rsi(close, length=14)
    macd = ta.macd(close, fast=12, slow=26, signal=9)

    if any(value is None for value in (sma, ema, rsi, macd)):
        raise RuntimeError("技術指標計算失敗。")
    if "MACD_12_26_9" not in macd.columns:
        raise RuntimeError("找不到學長原碼使用的 MACD_12_26_9 欄位。")

    # 明確指定 fill_method=None，避免 pandas 自動補值改變學長公式。
    return pd.DataFrame(
        {
            "SMA20": sma.pct_change(fill_method=None) * 100.0,
            "EMA20": ema.pct_change(fill_method=None) * 100.0,
            "MACD": (
                macd["MACD_12_26_9"].pct_change(fill_method=None) * 100.0
            ),
            "RSI14": (rsi - 50.0) * 0.1,
        },
        index=close.index,
    )


def build_neutral_usdt_indicators(index: pd.Index) -> pd.DataFrame:
    """建立四個全為零的 USDT 中性欄位，以維持既定 5＋20 維度。

    學長封存的資料產生器只有四種加密貨幣；目前模型則已統一採用
    5 資產 × 4 指標。USDT 沒有獨立市場趨勢，因此補零比人工製造
    SMA、EMA 或 RSI 變化更合理，也不會向模型提供虛假訊號。
    """
    return pd.DataFrame(
        0.0,
        index=index,
        columns=["SMA20", "EMA20", "MACD", "RSI14"],
    )
