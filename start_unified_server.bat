@echo off
echo ========================================
echo  Ethiopic Handwriting Recognition
echo  Unified Server (MAX_STROKES=60)
echo ========================================
echo.

cd /d C:\YonAPIAPI

REM Activate virtual environment
call venv\Scripts\activate.bat

REM Check if model exists
if not exist "C:\YonAPI\models\API_ready_model\best_model.pt" (
    echo WARNING: Model file not found!
    echo Please place your model at: C:\YonAPI\models\API_ready_model\best_model.pt
    echo.
)

REM Start the server
echo Starting server...
python unified_server.py

pause