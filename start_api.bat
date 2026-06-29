@echo off
echo ========================================
echo  Ethiopic Handwriting Recognition API
echo  MAX_STROKES=60
echo ========================================
echo.

REM Check if Python is installed
python --version >nul 2>&1
if errorlevel 1 (
    echo Python is not installed or not in PATH
    echo Please install Python 3.8 or higher
    pause
    exit /b 1
)

REM Check if model exists
if not exist "models\API_ready_model\best_model.pt" (
    echo ERROR: Model file not found!
    echo Please place your model at: models\API_ready_model\best_model.pt
    pause
    exit /b 1
)

REM Check deployment data
if exist "deployment_data" (
    echo ✅ Deployment data found
) else (
    echo ⚠️  Deployment data not found. Using default corrections.
)

REM Install dependencies if needed
echo.
echo Checking dependencies...
pip show torch >nul 2>&1
if errorlevel 1 (
    echo Installing dependencies...
    pip install -r requirements.txt
)

echo.
echo Starting API server...
echo Press Ctrl+C to stop
echo.

python start_api.py --host 0.0.0.0 --port 8000

pause
