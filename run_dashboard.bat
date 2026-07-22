@echo off
REM Start the AI Demand Forecasting Dashboard
REM This script activates the virtual environment and starts Streamlit

echo.
echo ========================================
echo  AI Demand Forecasting System
echo ========================================
echo.

cd /d C:\Users\stein\copilot-worktrees\major-project\steinjoshy-super-pancake

echo Verifying installation...
venv\Scripts\python.exe test_imports.py

echo.
echo Starting Streamlit Dashboard...
echo Opening at http://localhost:8501
echo.
echo Press Ctrl+C to stop the server
echo.

venv\Scripts\python.exe -m streamlit run dashboard/app.py

pause
