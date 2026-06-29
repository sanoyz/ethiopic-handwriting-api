#!/usr/bin/env python
"""
Start script for the Ethiopic Handwriting Recognition API
MAX_STROKES=60
"""

import os
import sys
import subprocess
from pathlib import Path


def check_model():
    """Check if model file exists"""
    model_path = Path("models/API_ready_model/best_model.pt")
    if not model_path.exists():
        print("❌ Model file not found!")
        print(f"   Please place your model file at: {model_path.absolute()}")
        print("   You can also set the MODEL_PATH environment variable.")
        return False
    print(f"✅ Model found at: {model_path}")

    # Check if deployment data exists
    deployment_path = Path("deployment_data")
    if deployment_path.exists():
        print(f"✅ Deployment data found at: {deployment_path}")
    else:
        print(f"⚠️  Deployment data not found. Using default corrections.")

    return True


def check_dependencies():
    """Check if dependencies are installed"""
    try:
        import torch
        import fastapi
        import uvicorn
        from scipy.interpolate import interp1d
        print("✅ Dependencies installed")
        print(f"   PyTorch: {torch.__version__}")
        print(f"   FastAPI: {fastapi.__version__}")
        print(f"   CUDA available: {torch.cuda.is_available()}")
        return True
    except ImportError as e:
        print(f"❌ Missing dependency: {e}")
        print("\n   Install dependencies with:")
        print("   pip install -r requirements.txt")
        return False


def start_api(host="0.0.0.0", port=8000, reload=True):
    """Start the API server"""
    print("\n" + "="*60)
    print("  Ethiopic Handwriting Recognition API")
    print("  MAX_STROKES=60")
    print("="*60)

    if not check_dependencies():
        sys.exit(1)

    if not check_model():
        sys.exit(1)

    print(f"\n🚀 Starting API server on {host}:{port}")
    print(f"   Documentation: http://localhost:{port}/docs")
    print(f"   Health check: http://localhost:{port}/health")
    print("\n   Press Ctrl+C to stop")
    print("="*60 + "\n")

    # Start the server
    cmd = [
        sys.executable, "-m", "uvicorn",
        "api.app:app",
        f"--host={host}",
        f"--port={port}",
        "--reload" if reload else ""
    ]
    cmd = [c for c in cmd if c]

    try:
        subprocess.run(cmd, check=True)
    except KeyboardInterrupt:
        print("\n\n👋 Shutting down...")
    except Exception as e:
        print(f"❌ Error: {e}")
        sys.exit(1)


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Start Ethiopic Handwriting Recognition API")
    parser.add_argument("--host", default="0.0.0.0", help="Host to bind to")
    parser.add_argument("--port", type=int, default=8000, help="Port to bind to")
    parser.add_argument("--no-reload", action="store_true", help="Disable auto-reload")

    args = parser.parse_args()

    start_api(host=args.host, port=args.port, reload=not args.no_reload)
