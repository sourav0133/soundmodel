#!/usr/bin/env python3
"""
Launcher script for the Environmental Noise Live Prediction FastAPI backend.
Usage:
    python -m backend.run
    python -m backend.run --port 8080 --host 127.0.0.1 --reload
"""

import argparse
import sys
import uvicorn
from backend.config import SERVER_HOST, SERVER_PORT

# Force UTF-8 encoding on standard output for Windows console compatibility
if sys.stdout.encoding != "utf-8":
    try:
        sys.stdout.reconfigure(encoding="utf-8")
    except Exception:
        pass


def main():
    parser = argparse.ArgumentParser(
        description="Run Environmental Noise Live Prediction FastAPI Server"
    )
    parser.add_argument(
        "--host",
        type=str,
        default=SERVER_HOST,
        help=f"Host interface to bind (default: {SERVER_HOST})"
    )
    parser.add_argument(
        "--port",
        type=int,
        default=SERVER_PORT,
        help=f"Port number to bind (default: {SERVER_PORT})"
    )
    parser.add_argument(
        "--reload",
        action="store_true",
        help="Enable auto-reload on code change"
    )
    args = parser.parse_args()

    print(f"[*] Starting Environmental Noise Prediction API on http://{args.host}:{args.port}")
    print(f"[*] Swagger Interactive Documentation: http://{args.host}:{args.port}/docs")
    uvicorn.run("backend.main:app", host=args.host, port=args.port, reload=args.reload)


if __name__ == "__main__":
    main()
