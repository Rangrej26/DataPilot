@echo off
title DataPilot AI - Analyst Workspace
echo ===================================================
echo Starting DataPilot AI - Workspace
echo ===================================================

cd /d "%~dp0"

if exist ".venv\Scripts\python.exe" (
    echo Using project virtual environment (.venv)...
    ".venv\Scripts\python.exe" -m streamlit run app.py --server.port 8501 --server.headless false
) else (
    echo Using system Python...
    python -m streamlit run app.py --server.port 8501 --server.headless false
)

if %ERRORLEVEL% NEQ 0 (
    echo.
    echo [ERROR] Application encountered an issue or was stopped.
    pause
)
