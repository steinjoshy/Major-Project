@echo off
REM ==========================================
REM  AI Demand Forecasting System - Launcher
REM ==========================================
REM This script activates the virtual environment,
REM validates dependencies and project imports,
REM then starts the Streamlit dashboard.

echo.
echo ==========================================
echo  AI Demand Forecasting System
echo ==========================================
echo.

cd /d "%~dp0"

REM Check Python version
echo [1/5] Checking Python version...
python --version
if errorlevel 1 (
    echo ERROR: Python not found in PATH
    echo Install Python 3.10-3.12 from https://python.org
    pause
    exit /b 1
)

REM Check virtual environment
echo.
echo [2/5] Checking virtual environment...
if not exist "venv\Scripts\python.exe" (
    echo Virtual environment not found.
    echo Creating venv with Python 3.12...
    py -3.12 -m venv venv
    if errorlevel 1 (
        echo ERROR: Failed to create virtual environment
        echo Make sure Python 3.12 is installed: py -3.12 --version
        pause
        exit /b 1
    )
    echo Virtual environment created.
)

REM Activate and upgrade pip
echo.
echo [3/5] Upgrading pip...
call venv\Scripts\activate
python -m pip install --upgrade pip --quiet

REM Install dependencies
echo.
echo [4/5] Installing/verifying dependencies...
pip install -r requirements.txt --quiet
if errorlevel 1 (
    echo ERROR: Failed to install dependencies
    echo Try: pip install -r requirements.txt --verbose
    pause
    exit /b 1
)

REM Validate imports
echo.
echo [5/5] Validating project imports...
venv\Scripts\python.exe test_imports.py
if errorlevel 1 (
    echo.
    echo ERROR: Project import validation failed
    echo Check test_imports.py output above
    pause
    exit /b 1
)

echo.
echo ==========================================
echo  All checks passed! Starting dashboard...
echo ==========================================
echo.
echo Opening at http://localhost:8501
echo Press Ctrl+C to stop the server
echo.

REM Start Streamlit
venv\Scripts\python.exe -m streamlit run dashboard/app.py

pause