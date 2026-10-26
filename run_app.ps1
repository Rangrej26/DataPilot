# DataPilot AI PowerShell Launch Script
Write-Host "===================================================" -ForegroundColor Cyan
Write-Host "Starting DataPilot AI - Analyst Workspace" -ForegroundColor Green
Write-Host "===================================================" -ForegroundColor Cyan

Set-Location -Path $PSScriptRoot

if (Test-Path ".\.venv\Scripts\python.exe") {
    Write-Host "Using project virtual environment (.venv)..." -ForegroundColor Yellow
    & ".\.venv\Scripts\python.exe" -m streamlit run app.py --server.port 8501 --server.headless false
} else {
    Write-Host "Using system Python..." -ForegroundColor Yellow
    python -m streamlit run app.py --server.port 8501 --server.headless false
}
