@echo off
chcp 65001 >nul

echo =========================================
echo DayTrade Terminal v3 Startup Script
echo =========================================
echo.

set "PYTHON_EXE=python"
if exist "%~dp0.venv\Scripts\python.exe" set "PYTHON_EXE=%~dp0.venv\Scripts\python.exe"

echo [1/3] Checking libraries...
"%PYTHON_EXE%" -m pip install flask flask-cors yfinance pandas numpy requests --quiet 2>nul

echo [2/3] Starting backend server...
cd /d "%~dp0files"
start "DayTrade Backend" "%PYTHON_EXE%" server.py

echo [3/3] Opening app in browser...
timeout /t 3 /nobreak >nul
start index.html

echo.
echo Startup complete.
pause
