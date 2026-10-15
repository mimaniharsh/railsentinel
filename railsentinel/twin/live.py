"""Live feeds of the twin: weather and running status of today's trains.

Both are deterministic for a given train, date and world, so repeated calls
agree with each other (a train does not jump around between page refreshes).
"""
from __future__ import annotations

from datetime import date, datetime, timedelta

import numpy as np

from railsentinel.security import stable_hash
from railsentinel.twin import generators
from railsentinel.twin.network import Network, Train
from railsentinel.twin.world import WorldParams


def weather_feed(region: str, day: date, world: WorldParams) -> float:
    rng = np.random.default_rng(stable_hash("weather", region, day.isoformat()))
    return round(generators.sample_weather(rng, region, day.month, world), 4)


def run_delays(network: Network, train: Train, run_date: date, world: WorldParams) -> tuple[np.ndarray, float]:
    rng = np.random.default_rng(stable_hash("run", train.number, run_date.isoformat(),
                                            round(world.weather_shift, 3), round(world.weather_effect_mult, 3)))
    return generators.simulate_run(network, train, rng, run_date.month, world)


def running_status(network: Network, train: Train, run_date: date, now: datetime, world: WorldParams) -> dict:
    """Where a train run that started on `run_date` is at `now` (naive IST datetimes)."""
    delays, w = run_delays(network, train, run_date, world)
    start = datetime.combine(run_date, datetime.min.time())
    last_idx = None
    for i, stop in enumerate(train.stops):
        t = stop.dep_min if stop.dep_min is not None else stop.arr_min
        actual = start + timedelta(minutes=float(t) + float(delays[i]))
        if actual <= now:
            last_idx = i
    if last_idx is None:
        return {"started": False, "last_station": None, "delay_min": 0.0, "weather": w, "completed": False}
    return {"started": True, "last_station": train.stops[last_idx].code, "last_index": last_idx,
            "delay_min": round(float(delays[last_idx]), 1), "weather": w,
            "completed": last_idx == len(train.stops) - 1}
