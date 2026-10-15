"""Initialise RailSentinel: create the database, train every model on the digital
twin, register prompt/agent configurations, seed demo PNRs and run warm-up
traffic so the monitors have a baseline.

Usage:
    python scripts/bootstrap.py              # first-time setup (skips if already done)
    python scripts/bootstrap.py --reset      # delete data/ and start again
    python scripts/bootstrap.py --scale 0.3  # smaller training sets for slow machines
"""
from __future__ import annotations

import argparse
import shutil
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from railsentinel.config import load_settings  # noqa: E402
from railsentinel.core import Platform  # noqa: E402


def main() -> int:
    parser = argparse.ArgumentParser(description="Initialise RailSentinel")
    parser.add_argument("--reset", action="store_true", help="delete the data directory first")
    parser.add_argument("--scale", type=float, default=None, help="training-set size multiplier (default 1.0)")
    parser.add_argument("--warmup", type=int, default=2, help="warm-up traffic batches (default 2)")
    args = parser.parse_args()
    settings = load_settings()
    if args.reset and settings.data_dir.exists():
        shutil.rmtree(settings.data_dir)
        print(f"Removed {settings.data_dir}")
    platform = Platform(settings)
    if platform.is_bootstrapped():
        print(f"Already initialised at {settings.data_dir}. Use --reset to start again.")
        return 0
    print(f"Initialising RailSentinel in {settings.data_dir}")
    print(f"Gemini: {'enabled (' + settings.llm_model + ')' if settings.llm_enabled else 'not configured, offline mode'}")
    t0 = time.time()
    platform.bootstrap(scale=args.scale, warmup_batches=args.warmup)
    print(f"Done in {time.time() - t0:.0f}s. Start the API with: uvicorn railsentinel.api.main:app --port 8000")
    return 0


if __name__ == "__main__":
    sys.exit(main())
