"""World state: the knobs the chaos engine turns and the operational flags the
supervisor sets (quarantined feeds, circuit breakers, thresholds).

Scenario parameters describe how the simulated world behaves. Operational
flags describe what the platform has done about it. Keeping them separate lets
the supervisor fix the platform without "fixing" the world: a quarantined fare
feed switches traffic to the backup feed while the glitch keeps happening on
the primary one.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass, field, fields

from railsentinel.db import Database

STATE_KEY = "world_state"


@dataclass
class WorldParams:
    cancel_mult: float = 1.0          # multiplier on cancellation propensity (concept drift for confirmation)
    demand_mult: float = 1.0          # multiplier on demand (waitlist depth, occupancy)
    festival_share: float = 0.0       # extra probability that sampled journeys fall in the festival season
    weather_shift: float = 0.0        # added to weather severity in affected regions
    weather_regions: tuple = ("North", "East", "Central")
    weather_effect_mult: float = 1.0  # how strongly weather turns into delay (concept drift for delays)
    fare_glitch_rate: float = 0.0     # share of corrupted quotes on the primary fare feed
    bot_rate: float = 0.04            # share of bot sessions
    stealth_bots: bool = False        # bots imitate human pacing
    legit_fast_share: float = 0.0     # share of genuine passengers using autofill during Tatkal rush
    search_failure_rate: float = 0.0  # failure probability of the timetable search tool
    search_latency_ms: int = 0        # added latency on the timetable search tool

    @classmethod
    def from_dict(cls, data: dict | None) -> "WorldParams":
        data = data or {}
        known = {f.name for f in fields(cls)}
        clean = {k: (tuple(v) if k == "weather_regions" else v) for k, v in data.items() if k in known}
        return cls(**clean)


@dataclass
class OpsFlags:
    quarantined_feeds: list = field(default_factory=list)
    circuit_breakers: list = field(default_factory=list)
    fraud_threshold: float = 0.5
    support_strict_mode: bool = False

    @classmethod
    def from_dict(cls, data: dict | None) -> "OpsFlags":
        data = data or {}
        known = {f.name for f in fields(cls)}
        return cls(**{k: v for k, v in data.items() if k in known})


class World:
    def __init__(self, db: Database):
        self.db = db

    def _state(self) -> dict:
        return self.db.get_kv(STATE_KEY, {"scenario": None, "params": {}, "ops": {}})

    def _save(self, state: dict) -> None:
        self.db.set_kv(STATE_KEY, state)

    @property
    def scenario(self) -> str | None:
        return self._state().get("scenario")

    def params(self) -> WorldParams:
        return WorldParams.from_dict(self._state().get("params"))

    def ops(self) -> OpsFlags:
        return OpsFlags.from_dict(self._state().get("ops"))

    def effective_params(self) -> WorldParams:
        """World as seen by the platform after operational mitigations."""
        p = self.params()
        ops = self.ops()
        if "fare_feed" in ops.quarantined_feeds:
            p.fare_glitch_rate = 0.0  # traffic is served from the backup fare feed
        return p

    def set_scenario(self, scenario: str | None, params: dict) -> None:
        state = self._state()
        state["scenario"] = scenario
        state["params"] = asdict(WorldParams.from_dict(params))
        self._save(state)

    def update_ops(self, **changes) -> OpsFlags:
        state = self._state()
        ops = asdict(self.ops())
        ops.update(changes)
        state["ops"] = ops
        self._save(state)
        return OpsFlags.from_dict(ops)

    def reset(self) -> None:
        self._save({"scenario": None, "params": asdict(WorldParams()), "ops": asdict(OpsFlags())})

    def snapshot(self) -> dict:
        return {"scenario": self.scenario, "params": asdict(self.params()), "ops": asdict(self.ops())}
