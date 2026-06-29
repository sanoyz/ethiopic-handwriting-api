"""
Utility functions for the API
"""

import json
import hashlib
from pathlib import Path
from datetime import datetime
from typing import Dict, Any, Optional


def generate_file_hash(filename: str, content: bytes) -> str:
    """Generate SHA-256 hash of file content"""
    hash_obj = hashlib.sha256(content)
    return hash_obj.hexdigest()


def save_uploaded_file(content: bytes, filename: str, upload_dir: str = "uploads") -> Path:
    """Save uploaded file with timestamp"""
    upload_path = Path(upload_dir)
    upload_path.mkdir(exist_ok=True)

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    name, ext = Path(filename).stem, Path(filename).suffix
    new_filename = f"{name}_{timestamp}{ext}"

    file_path = upload_path / new_filename
    file_path.write_bytes(content)

    return file_path


def format_response(success: bool, data: Any = None, error: str = None) -> Dict[str, Any]:
    """Format API response"""
    response = {
        "success": success,
        "timestamp": datetime.now().isoformat()
    }

    if data is not None:
        response["data"] = data

    if error is not None:
        response["error"] = error

    return response


def validate_json_structure(data: dict) -> bool:
    """Validate that the JSON has the expected structure"""
    if not isinstance(data, dict):
        return False

    if "strokes" not in data:
        return False

    if not isinstance(data["strokes"], list):
        return False

    if len(data["strokes"]) == 0:
        return False

    if "points" not in data["strokes"][0]:
        return False

    return True
