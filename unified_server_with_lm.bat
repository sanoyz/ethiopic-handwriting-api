@echo off
title Ethiopic Handwriting Recognition Server (with LM)
echo ========================================
echo  Starting Unified Server with Language Model
echo ========================================
echo.

REM Change to the directory where this batch file is located
cd /d "%~dp0"

REM Activate the virtual environment
if exist "venv\Scripts\activate.bat" (
    echo Activating virtual environment...
    call venv\Scripts\activate.bat
) else (
    echo ERROR: Virtual environment not found at venv\
    echo Please make sure you have created a venv and installed dependencies.
    pause
    exit /b 1
)

REM Check if the Python script exists
if not exist "unified_server_with_lm.py" (
    echo ERROR: unified_server_with_lm.py not found in the current directory.
    pause
    exit /b 1
)

echo.
echo Starting server...
echo Press Ctrl+C to stop.
echo.
echo Open http://localhost:8081 in your browser.
echo.

python unified_server_with_lm.py

pause