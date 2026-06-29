"""
Test client for the Ethiopic Handwriting Recognition API
"""

import requests
import json
from pathlib import Path


def test_health(base_url="http://localhost:8000"):
    """Test health endpoint"""
    response = requests.get(f"{base_url}/health")
    print(f"Health: {response.json()}")
    return response.status_code == 200


def test_predict(base_url="http://localhost:8000", json_path=None):
    """Test prediction endpoint"""
    if json_path is None:
        json_path = Path("sample_handwriting.json")
        if not json_path.exists():
            print("No sample file found. Please provide a JSON file.")
            return False

    with open(json_path, "rb") as f:
        files = {"file": (json_path.name, f, "application/json")}
        response = requests.post(f"{base_url}/predict", files=files)

    print(f"Prediction: {response.json()}")
    return response.status_code == 200


def test_info(base_url="http://localhost:8000"):
    """Test info endpoint"""
    response = requests.get(f"{base_url}/info")
    print(f"Model Info: {response.json()}")
    return response.status_code == 200


if __name__ == "__main__":
    import sys

    base_url = "http://localhost:8000"

    print("Testing API endpoints...")

    if test_health(base_url):
        print("✅ Health check passed")
    else:
        print("❌ Health check failed")

    if test_info(base_url):
        print("✅ Info endpoint passed")
    else:
        print("❌ Info endpoint failed")

    if len(sys.argv) > 1:
        json_path = Path(sys.argv[1])
        if json_path.exists():
            if test_predict(base_url, json_path):
                print("✅ Prediction passed")
            else:
                print("❌ Prediction failed")
        else:
            print(f"❌ File not found: {json_path}")
    else:
        print("ℹ️  To test prediction, provide a JSON file path:")
        print("   python test_client.py handwriting.json")
