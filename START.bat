@echo off
title MedPredict AI - Setup & Launch
color 0A

echo.
echo  ============================================
echo   MedPredict AI - Medical Expense Predictor
echo   One-Click Setup ^& Launch
echo  ============================================
echo.

:: Check if Python is installed
python --version >nul 2>&1
if %errorlevel% neq 0 (
    echo  [ERROR] Python is not installed or not in PATH!
    echo  Please install Python from https://www.python.org/downloads/
    echo  Make sure to check "Add Python to PATH" during installation.
    echo.
    pause
    exit /b 1
)

echo  [OK] Python found:
python --version
echo.

:: Navigate to project directory
cd /d "%~dp0"
echo  [INFO] Working directory: %cd%
echo.

:: Install dependencies
echo  [1/3] Installing dependencies...
echo  ----------------------------------------
pip install flask pandas numpy scikit-learn >nul 2>&1
if %errorlevel% neq 0 (
    echo  [WARN] pip install had issues. Trying with --user flag...
    pip install --user flask pandas numpy scikit-learn >nul 2>&1
)
echo  [OK] Dependencies installed!
echo.

:: Train models if they don't exist
if not exist "model\rf_model.pkl" (
    echo  [2/3] Training ML models (first-time setup)...
    echo  ----------------------------------------
    python model\train_model.py
    echo  [OK] Models trained!
) else (
    echo  [2/3] Models already trained. Skipping...
    echo  [OK] rf_model.pkl found!
)
echo.

:: Start server and open browser
echo  [3/3] Starting Flask server...
echo  ----------------------------------------
echo.
echo  ============================================
echo   Dashboard: http://127.0.0.1:5000
echo   Press Ctrl+C to stop the server
echo  ============================================
echo.

:: Wait 2 seconds then open Chrome
start "" cmd /c "timeout /t 2 /nobreak >nul && start chrome http://127.0.0.1:5000"

:: Start Flask server (this blocks until Ctrl+C)
python app\app.py

echo.
echo  Server stopped. Goodbye!
pause
