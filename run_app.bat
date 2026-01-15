@echo off
echo ===================================================
echo   Cancer Detection AI App - Startup Script
echo ===================================================
echo.

:: Check if Python is installed
python --version >nul 2>&1
if %errorlevel% neq 0 (
    echo Error: Python is not installed or not in PATH.
    pause
    exit /b
)

:: Check if Node is installed
call npm --version >nul 2>&1
if %errorlevel% neq 0 (
    echo Error: Node.js is not installed or not in PATH.
    pause
    exit /b
)

echo [1/3] Starting Backend Server...
start "CancerAI Backend" /D "backend" cmd /k "..\.venv\Scripts\python.exe main.py || pause"

echo [2/3] Waiting for backend to initialize...
timeout /t 5 /nobreak >nul

echo [3/3] Starting Frontend UI...
echo.
echo The application will open in your default browser shortly.

cd frontend
call npm run dev

pause
