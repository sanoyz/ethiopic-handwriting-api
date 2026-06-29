"""
Batch prediction script for multiple handwriting files
"""

import os
import json
import requests
from pathlib import Path
from datetime import datetime
from tqdm import tqdm


def batch_predict(input_dir: str, output_file: str = "batch_results.json", 
                  base_url: str = "http://localhost:8000"):
    """
    Process all JSON files in a directory
    """

    input_path = Path(input_dir)
    if not input_path.exists():
        print(f"❌ Directory not found: {input_dir}")
        return

    json_files = list(input_path.glob("*.json"))
    if not json_files:
        print(f"❌ No JSON files found in {input_dir}")
        return

    print(f"📁 Found {len(json_files)} JSON files")

    results = []

    for json_file in tqdm(json_files, desc="Processing"):
        try:
            with open(json_file, "rb") as f:
                files = {"file": (json_file.name, f, "application/json")}
                response = requests.post(f"{base_url}/predict", files=files)

            result = {
                "file": json_file.name,
                "status": "success" if response.status_code == 200 else "error",
                "response": response.json() if response.status_code == 200 else response.text
            }
            results.append(result)

        except Exception as e:
            results.append({
                "file": json_file.name,
                "status": "error",
                "error": str(e)
            })

    output_data = {
        "timestamp": datetime.now().isoformat(),
        "total_files": len(json_files),
        "successful": sum(1 for r in results if r["status"] == "success"),
        "results": results
    }

    with open(output_file, "w", encoding="utf-8") as f:
        json.dump(output_data, f, indent=2, ensure_ascii=False)

    print(f"\n✅ Results saved to: {output_file}")
    print(f"   Success: {output_data['successful']}/{len(json_files)}")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Batch prediction")
    parser.add_argument("input_dir", help="Directory containing JSON files")
    parser.add_argument("--output", default="batch_results.json", help="Output file")
    parser.add_argument("--url", default="http://localhost:8000", help="API URL")

    args = parser.parse_args()

    batch_predict(args.input_dir, args.output, args.url)
