"""
Sends a JSON batch of prediction requests (produced by generate_batch.py)
to your running POST /predict endpoint.

Reads the API key from --api-key, or falls back to the API_KEY environment
variable (works if you've either exported it in your shell per the .env
loading snippet, or run this script with `python -m dotenv run -- python
scripts/send_predictions.py ...` if you have python-dotenv's CLI installed).

Usage:
    python scripts/send_predictions.py --file heavy_drift_batch.json --url http://localhost:8001/predict
    python scripts/send_predictions.py --file heavy_drift_batch.json --url http://localhost:8001/predict --api-key abc123
"""

import argparse
import json
import os
import sys
import time

import requests
from dotenv import load_dotenv

load_dotenv()  # reads backend/.env automatically — no more manual PowerShell loader needed

def main():
    parser = argparse.ArgumentParser(description="Send a batch of prediction requests")
    parser.add_argument("--file", required=True, help="JSON file produced by generate_batch.py")
    parser.add_argument("--url", required=True, help="full URL of the /predict endpoint")
    parser.add_argument("--api-key", default=None, help="X-API-Key value; falls back to $API_KEY env var")
    parser.add_argument("--delay", type=float, default=0.0, help="seconds to sleep between requests")
    args = parser.parse_args()

    api_key = args.api_key or os.environ.get("API_KEY")
    if not api_key:
        print(
            "WARNING: no API key provided (--api-key not set and $API_KEY env var is empty). "
            "If your /predict endpoint requires X-API-Key, every request below will 401.",
            file=sys.stderr,
        )

    headers = {"Content-Type": "application/json"}
    if api_key:
        headers["X-API-Key"] = api_key

    with open(args.file, "r") as f:
        batch = json.load(f)

    success = 0
    failed = 0

    for i, payload in enumerate(batch, start=1):
        try:
            response = requests.post(args.url, json=payload, headers=headers, timeout=10)
            if response.status_code == 200:
                success += 1
            else:
                failed += 1
                print(f"[{i}/{len(batch)}] HTTP {response.status_code}: {response.text}", file=sys.stderr)
        except requests.RequestException as exc:
            failed += 1
            print(f"[{i}/{len(batch)}] request failed: {exc}", file=sys.stderr)

        if args.delay:
            time.sleep(args.delay)

        if i % 100 == 0:
            print(f"  ...sent {i}/{len(batch)}")

    print(f"Done. success={success} failed={failed} total={len(batch)}")
    if failed and success == 0:
        sys.exit(1)


if __name__ == "__main__":
    main()