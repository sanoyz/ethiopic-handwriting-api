# Ethiopic Handwriting Recognition API Startup Script
# MAX_STROKES=60

Write-Host "========================================" -ForegroundColor Cyan
Write-Host "  Ethiopic Handwriting Recognition API" -ForegroundColor Cyan
Write-Host "  MAX_STROKES=60" -ForegroundColor Cyan
Write-Host "========================================" -ForegroundColor Cyan
Write-Host ""

# Check Python
$pythonVersion = python --version 2>&1
if ($LASTEXITCODE -ne 0) {
    Write-Host "❌ Python is not installed or not in PATH" -ForegroundColor Red
    Write-Host "Please install Python 3.8 or higher" -ForegroundColor Yellow
    Read-Host "Press Enter to exit"
    exit 1
}
Write-Host "✅ Python found: $pythonVersion" -ForegroundColor Green

# Check model
if (-not (Test-Path "models\API_ready_model\best_model.pt")) {
    Write-Host "❌ Model file not found!" -ForegroundColor Red
    Write-Host "Please place your model at: models\API_ready_model\best_model.pt" -ForegroundColor Yellow
    Read-Host "Press Enter to exit"
    exit 1
}
Write-Host "✅ Model found: models\API_ready_model\best_model.pt" -ForegroundColor Green

# Check deployment data
if (Test-Path "deployment_data") {
    Write-Host "✅ Deployment data found" -ForegroundColor Green
} else {
    Write-Host "⚠️  Deployment data not found. Using default corrections." -ForegroundColor Yellow
}

# Check dependencies
Write-Host ""
Write-Host "Checking dependencies..." -ForegroundColor Yellow
$depsCheck = python -c "import torch, fastapi, uvicorn, scipy" 2>&1
if ($LASTEXITCODE -ne 0) {
    Write-Host "⚠️  Missing dependencies. Installing..." -ForegroundColor Yellow
    pip install -r requirements.txt
}
Write-Host "✅ Dependencies OK" -ForegroundColor Green

# Start API
Write-Host ""
Write-Host "🚀 Starting API server on http://localhost:8000" -ForegroundColor Green
Write-Host "📚 Documentation: http://localhost:8000/docs" -ForegroundColor Green
Write-Host ""
Write-Host "Press Ctrl+C to stop" -ForegroundColor Yellow
Write-Host "========================================" -ForegroundColor Cyan
Write-Host ""

python start_api.py --host 0.0.0.0 --port 8000
