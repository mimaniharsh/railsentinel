"""Measure serving performance on this machine.

For every governed model it times the full serving path (data contract, model,
prediction log, signed passport) for single requests and for batches, then
converts batch throughput into passenger interactions per day. With --api it
also times real HTTP round trips against a running API.

Usage:
    python scripts/benchmark.py                 # in-process, all models
    python scripts/benchmark.py --requests 300  # more single-request samples
    python scripts/benchmark.py --api http://localhost:8000

Predictions are logged with source "benchmark", which the monitors count for
latency but exclude from drift and accuracy checks.
"""
from __future__ import annotations

import argparse
import os
import statistics
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
os.environ.setdefault("RS_SUPERVISOR_INTERVAL", "0")

import numpy as np  # noqa: E402

from railsentinel.core import Platform  # noqa: E402

SECONDS_PER_DAY = 86_400


def pct(values: list[float], q: float) -> float:
    return float(np.percentile(values, q)) if values else float("nan")


def bench_in_process(p: Platform, n_single: int, batch: int) -> None:
    print(f"In-process serving path, one CPU thread per call ({os.cpu_count()} CPUs on this machine)\n")
    print(f"{'model':24s} {'single p50 ms':>14s} {'single p95 ms':>14s} {'batch rec/s':>12s} {'per day (1 core)':>17s}")
    rng = np.random.default_rng(0)
    for name, spec in p.specs.items():
        records = spec.frame_to_records(spec.generate(p.network, max(n_single, batch), rng, p.world.params()))
        runtime = p.runtime(name)
        runtime.predict(records[:5], source="benchmark")  # warm caches
        lat = []
        for rec in records[:n_single]:
            t0 = time.perf_counter()
            runtime.predict([rec], source="benchmark")
            lat.append((time.perf_counter() - t0) * 1000)
        t0 = time.perf_counter()
        runtime.predict(records[:batch], source="benchmark")
        rate = batch / (time.perf_counter() - t0)
        print(f"{name:24s} {pct(lat, 50):14.1f} {pct(lat, 95):14.1f} {rate:12,.0f} {rate * SECONDS_PER_DAY:17,.0f}")
    planner_q = "I want to travel from Pune to Delhi tomorrow under ₹1500 and reach before 10:00 AM."
    lat = []
    for _ in range(5):
        t0 = time.perf_counter()
        p.planner.plan(planner_q, source="benchmark", use_llm=False, create_intent=False)
        lat.append((time.perf_counter() - t0) * 1000)
    print(f"{'planner agent (offline)':24s} {pct(lat, 50):14.1f} {pct(lat, 95):14.1f} {'-':>12s} {'-':>17s}")
    lat = []
    for q in ["When is the reservation chart prepared?", "Can I get a refund on a confirmed Tatkal ticket?"] * 5:
        t0 = time.perf_counter()
        p.support.answer(q, source="benchmark", use_llm=False)
        lat.append((time.perf_counter() - t0) * 1000)
    print(f"{'support (offline)':24s} {pct(lat, 50):14.1f} {pct(lat, 95):14.1f} {'-':>12s} {'-':>17s}")
    print("\nBatch throughput is per process on one thread; the API is stateless apart from the database, so "
          "capacity grows with worker processes and machines.")


def bench_api(url: str, key: str, n: int) -> None:
    import httpx

    body = {"train_number": "RS101", "origin": "PUNE", "destination": "NZM", "travel_class": "SL",
            "journey_date": time.strftime("%Y-%m-%d", time.localtime(time.time() + 15 * 86400)),
            "booking_status": "WL", "queue_position": 30}
    lat = []
    with httpx.Client(base_url=url.rstrip("/"), headers={"X-API-Key": key}, timeout=30) as c:
        c.post("/predict/confirmation", json=body).raise_for_status()
        for _ in range(n):
            t0 = time.perf_counter()
            c.post("/predict/confirmation", json=body).raise_for_status()
            lat.append((time.perf_counter() - t0) * 1000)
    print(f"HTTP POST /predict/confirmation (with Shapley explanation), {n} sequential requests: "
          f"p50 {statistics.median(lat):.1f} ms, p95 {pct(lat, 95):.1f} ms")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--requests", type=int, default=100, help="single-request samples per model")
    parser.add_argument("--batch", type=int, default=1000, help="batch size for throughput")
    parser.add_argument("--api", help="base URL of a running API to time over HTTP")
    parser.add_argument("--key", default="rs-admin-demo", help="API key for --api")
    args = parser.parse_args()
    if args.api:
        bench_api(args.api, args.key, args.requests)
        return 0
    p = Platform()
    if not p.is_bootstrapped():
        print("Run python scripts/bootstrap.py first.")
        return 1
    bench_in_process(p, args.requests, args.batch)
    return 0


if __name__ == "__main__":
    sys.exit(main())
