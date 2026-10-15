"""Chaos engine: inject realistic failures into the running platform.

Each scenario either changes how the simulated world behaves (data drift,
concept drift, corrupted feeds, adversaries, outages) or ships a bad release
through the registry, exactly as a faulty deployment would arrive. The
supervisor is not told which scenario is active; it has to detect, diagnose
and fix the problem from evidence.
"""
from __future__ import annotations

import copy
import time

from railsentinel.agents.support import PROMPT_V2_FAULTY
from railsentinel.governance.monitors import set_window_floor
from railsentinel.twin.world import OpsFlags

SCENARIOS: dict[str, dict] = {
    "fog_surge": {
        "title": "Dense fog and signal failures in the north and east",
        "targets": ["delay_predictor"],
        "params": {"weather_shift": 0.4, "weather_effect_mult": 2.2},
        "story": "Weather severity jumps on northern, eastern and central routes and each unit of bad weather now "
                 "causes twice the delay.",
        "expected": "Drift on weather_severity and a jump in delay error. RCA points at the weather feed; the "
                    "supervisor retrains on post-change data, shadows the candidate and promotes it.",
    },
    "cancellation_shift": {
        "title": "Passengers stop cancelling (concept drift)",
        "targets": ["confirmation_predictor"],
        "params": {"cancel_mult": 0.45},
        "story": "Cancellations fall by more than half, so waitlists clear far less often. Booking inputs look "
                 "exactly as before: only the outcome changes.",
        "expected": "Accuracy falls with no input drift. RCA classifies concept drift, localises the change point "
                    "and retrains on post-change outcomes before shadow promotion.",
    },
    "festival_rush": {
        "title": "Festival travel rush",
        "targets": ["confirmation_predictor", "delay_predictor", "fare_forecaster"],
        "params": {"festival_share": 0.75},
        "story": "Three quarters of journeys now fall in the festival season, with deeper waitlists and fuller trains.",
        "expected": "Input drift without loss of accuracy. RCA marks it a benign shift and the supervisor watches "
                    "instead of retraining needlessly.",
    },
    "fare_feed_glitch": {
        "title": "Corrupted fare feed",
        "targets": ["fare_forecaster"],
        "params": {"fare_glitch_rate": 0.25},
        "story": "A quarter of fare quotes arrive as zero, negative or 10-25x the real price.",
        "expected": "Data contracts block bad quotes at inference; RCA traces the violations to fare_feed and the "
                    "supervisor quarantines it, serving the backup feed until probes pass.",
    },
    "stealth_bots": {
        "title": "Stealth booking-bot swarm",
        "targets": ["fraud_detector"],
        "params": {"stealth_bots": True, "bot_rate": 0.15},
        "story": "Bots copy human pacing, rotate payment instruments and residential IPs during Tatkal.",
        "expected": "Recall collapses (false negatives). The supervisor retrains on investigated sessions, shadows "
                    "and promotes the new detector.",
    },
    "tatkal_rush_false_positives": {
        "title": "Genuine Tatkal rush flagged as bots",
        "targets": ["fraud_detector"],
        "params": {"legit_fast_share": 0.3},
        "story": "Genuine passengers use autofill and shared counter PCs during Tatkal opening.",
        "expected": "False positives rise. RCA identifies a behaviour change and proposes a threshold change, which "
                    "affects passengers and therefore waits for human approval.",
    },
    "recommender_bad_release": {
        "title": "Bad recommender release (sponsored boost)",
        "targets": ["train_recommender"],
        "params": {},
        "deploy": {"asset": "train_recommender", "patch": {"premium_boost": 2.5},
                   "notes": "release 'premium promotion' (misconfigured boost)"},
        "story": "A release adds a large boost to premium trains for every passenger.",
        "expected": "Exposure bias and relevance drop right after a deployment. RCA blames the release and the "
                    "supervisor rolls back to the last healthy version.",
    },
    "prompt_regression": {
        "title": "Ungrounded support prompt released",
        "targets": ["support_assistant"],
        "params": {},
        "deploy": {"asset": "support_assistant", "artifact": PROMPT_V2_FAULTY,
                   "notes": "prompt release 'faster answers' (drops grounding rules)"},
        "story": "A prompt change tells the assistant to answer from memory. Answers start inventing numbers.",
        "expected": "Hallucination rate spikes after the release. RCA blames the prompt version and the supervisor "
                    "rolls it back.",
    },
    "search_api_outage": {
        "title": "Timetable search API outage",
        "targets": ["travel_planner_agent"],
        "params": {"search_failure_rate": 0.65, "search_latency_ms": 2500},
        "story": "The timetable search service times out on most calls and is slow on the rest.",
        "expected": "The planner agent loops on retries. The agent monitor flags abnormal behaviour; RCA isolates the "
                    "failing tool and the supervisor opens a circuit breaker to the cached timetable.",
    },
}


class ChaosEngine:
    def __init__(self, platform):
        self.p = platform

    def scenarios(self) -> list[dict]:
        active = self.p.world.scenario
        return [{"id": k, **{x: y for x, y in v.items() if x not in ("deploy",)}, "deploys_release": "deploy" in v,
                 "active": k == active} for k, v in SCENARIOS.items()]

    def activate(self, scenario_id: str, actor: str) -> dict:
        if scenario_id not in SCENARIOS:
            raise KeyError(f"unknown scenario {scenario_id}")
        sc = SCENARIOS[scenario_id]
        self.p.world.set_scenario(scenario_id, sc.get("params", {}))
        result = {"scenario": scenario_id, "title": sc["title"], "params": sc.get("params", {})}
        dep = sc.get("deploy")
        if dep:
            asset = dep["asset"]
            prod = self.p.registry.production(asset)
            if "artifact" in dep:
                artifact = copy.deepcopy(dep["artifact"])
            else:
                artifact = copy.deepcopy(self.p.registry.load(asset, prod["version"]))
                artifact.update(dep["patch"])
            version = self.p.registry.add_version(asset, artifact, prod["metrics"], params=dep.get("patch", {}),
                                                  stage="production", notes=dep["notes"], parent=prod["version"],
                                                  actor="release-pipeline")
            set_window_floor(self.p.db, asset)
            result["deployed"] = {"asset": asset, "version": version}
        self.p.audit.append(actor, "chaos_scenario_activated", scenario_id, result)
        return result

    def clear(self, actor: str) -> dict:
        previous = self.p.world.scenario
        self.p.world.set_scenario(None, {})
        self.p.audit.append(actor, "chaos_scenario_cleared", previous or "none", {})
        return {"cleared": previous}

    def reset(self, actor: str) -> dict:
        """Return the demo to its launch state: baseline world, launch model versions, no open incidents."""
        self.p.world.set_scenario(None, {})
        self.p.world.update_ops(**{k: v for k, v in OpsFlags().__dict__.items()})
        restored = {}
        for asset in self.p.asset_names():
            versions = self.p.registry.versions(asset)
            first = min(v["version"] for v in versions)
            prod = self.p.registry.production(asset)
            for v in versions:
                if v["stage"] == "shadow":
                    self.p.registry.set_stage(asset, v["version"], "rejected", actor=actor, reason="demo reset")
            if prod["version"] != first:
                self.p.registry.set_stage(asset, first, "production", actor=actor, reason="demo reset to launch version")
                restored[asset] = first
            set_window_floor(self.p.db, asset)
        self.p.db.execute("DELETE FROM kv_state WHERE key LIKE 'drift_reference:%'")
        now = time.time()
        self.p.db.execute("UPDATE incidents SET status='closed_by_reset', closed_ts=? WHERE status IN "
                          "('open','mitigating','awaiting_approval','monitoring')", (now,))
        self.p.db.execute("UPDATE approvals SET status='cancelled', decided_ts=?, decided_by=? WHERE status='pending'",
                          (now, actor))
        self.p.audit.append(actor, "demo_reset", "platform", {"restored_versions": restored})
        return {"reset": True, "restored_versions": restored}

    def autopilot(self, scenario_id: str, actor: str, max_rounds: int = 6, scale: float = 1.0) -> dict:
        """Activate a scenario and alternate traffic and supervisor cycles until the
        incident it causes is resolved, waits for a human, or rounds run out."""
        # The background supervisor loop is paused so every event of this run lands in its own timeline.
        self.p.supervisor.pause()
        try:
            return self._autopilot(scenario_id, actor, max_rounds, scale)
        finally:
            self.p.supervisor.resume()

    def _autopilot(self, scenario_id: str, actor: str, max_rounds: int, scale: float) -> dict:
        started = time.time()
        activation = self.activate(scenario_id, actor)
        targets = SCENARIOS[scenario_id]["targets"]
        timeline = []
        outcome = "rounds_exhausted"
        for rnd in range(1, max_rounds + 1):
            traffic = self.p.traffic.run(scale=scale)
            cycle = self.p.supervisor.run_cycle()
            timeline.append({"round": rnd, "traffic": traffic, "events": cycle["events"],
                             "status": {a: cycle["fleet"][a]["status"] for a in targets}})
            incidents = [self.p.supervisor._decode(i) for i in
                         self.p.db.query("SELECT * FROM incidents WHERE opened_ts>=?", (started,))
                         if i["model"] in targets]
            if any(i["status"] == "awaiting_approval" for i in incidents):
                outcome = "awaiting_human_approval"
                break
            if incidents and all(i["status"] == "resolved" for i in incidents):
                outcome = "resolved"
                break
            # Stop at "monitoring" only when the supervisor decided to watch (a benign shift). After a real
            # remediation, a watch on a new transient finding is not the end state: keep verifying.
            watch_only = all(a["action"] == "watch" for i in incidents for a in i["actions"])
            if incidents and all(i["status"] == "monitoring" for i in incidents) and watch_only and rnd >= 2:
                outcome = "monitoring"
                break
        return {"scenario": scenario_id, "activation": activation, "outcome": outcome, "rounds": len(timeline),
                "timeline": timeline, "duration_s": round(time.time() - started, 1)}
