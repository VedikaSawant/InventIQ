"""
scripts/run_api.py

Start the InventIQ FastAPI server.

Usage:
    python scripts/run_api.py
    python scripts/run_api.py --host 0.0.0.0 --port 8000 --reload
"""

import argparse
import uvicorn

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Start InventIQ API")
    parser.add_argument("--host",   default="127.0.0.1")
    parser.add_argument("--port",   type=int, default=8000)
    parser.add_argument("--reload", action="store_true",
                        help="Enable auto-reload for development")
    args = parser.parse_args()

    uvicorn.run(
        "src.api.main:app",
        host    = args.host,
        port    = args.port,
        reload  = args.reload,
        log_level = "info",
    )