@echo off
setlocal

echo Creating Python virtual environment...
py -3 -m venv .venv

echo Upgrading pip...
.venv\Scripts\python.exe -m pip install --upgrade pip

echo Installing requirements...
.venv\Scripts\python.exe -m pip install -r requirements.txt

echo.
echo Environment ready.
echo Activate with:
echo   .venv\Scripts\activate
echo.
echo Then run:
echo   python download_binance_paper.py

endlocal
