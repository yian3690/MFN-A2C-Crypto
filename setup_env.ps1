$ErrorActionPreference = "Stop"

Write-Host "Creating Python virtual environment..." -ForegroundColor Cyan
py -3 -m venv .venv

Write-Host "Upgrading pip..." -ForegroundColor Cyan
.\.venv\Scripts\python.exe -m pip install --upgrade pip

Write-Host "Installing requirements..." -ForegroundColor Cyan
.\.venv\Scripts\python.exe -m pip install -r requirements.txt

Write-Host ""
Write-Host "Environment ready." -ForegroundColor Green
Write-Host "Activate with:"
Write-Host "  .\.venv\Scripts\Activate.ps1"
Write-Host ""
Write-Host "Then download Binance data with:"
Write-Host "  python download_binance_paper.py"
