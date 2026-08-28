# MFN-A2C Crypto Reproduction Environment

## 1. Create the virtual environment

### Windows PowerShell
```powershell
Set-ExecutionPolicy -Scope Process -ExecutionPolicy Bypass
.\setup_env.ps1
```

### Windows CMD
```bat
setup_env.bat
```

## 2. Activate

PowerShell:
```powershell
.\.venv\Scripts\Activate.ps1
```

CMD:
```bat
.venv\Scripts\activate
```

## 3. Download the paper's Binance data

```powershell
python download_binance_paper.py
```

The script downloads:

- BTCUSDT
- ETHUSDT
- LTCUSDT
- BNBUSDT

at 4-hour resolution from 2018-01-01 through 2025-09-01.

It creates:

```text
data/
  BTCUSDT.csv
  ETHUSDT.csv
  LTCUSDT.csv
  BNBUSDT.csv

merged_output.csv
```

`merged_output.csv` is compatible with the old project convention:

```text
Open0 High0 Low0 Close0
Open1 High1 Low1 Close1
Open2 High2 Low2 Close2
Open3 High3 Low3 Close3
```

The order is:

```text
0 = BTC
1 = ETH
2 = LTC
3 = BNB
```

USDT is treated as cash and is added by the environment with price=1.

## 4. Important

The original code uses `get_historical_klines()` and the same date range/4-hour interval, but it puts the symbols in the order BNB, BTC, LTC, ETH. This replacement follows the paper's stated portfolio order more closely: BTC, ETH, LTC, BNB.

Do not put a Binance API key in this downloader. Public historical K-line data does not require one.

The downloader intentionally merges by `Open Time` rather than row number, so the four assets cannot silently become misaligned.
