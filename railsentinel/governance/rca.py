"""Autonomous root-cause analysis.

The platform is a dependency graph: upstream feeds and configurations feed
models, models feed agents. Given the findings of a monitoring cycle, the
analyser:

1. attributes each finding to graph nodes (a drifted feature to the feed that
   owns it, blocked records to the feed whose contract failed, a regression to
   a recent deployment, tool errors to the failing tool),
2. classifies the failure pattern (data drift vs concept drift vs bad
   deployment vs corrupted feed vs failing dependency vs behaviour change),
3. walks the graph downstream to measure blast radius,
4. recommends remediations in policy-ranked order with the evidence chain.
"""
from __future__ import annotations

import time

STATIC_DEPENDENCIES = {
    "train_recommender": ["user_profile_store", "timetable_feed", "recommender_config"],
    "support_assistant": ["kb_index", "prompt_config", "booking_feed", "confirmation_predictor",
                          "travel_planner_agent"],
    "travel_planner_agent": ["timetable_search", "agent_config", "confirmation_predictor", "delay_predictor",
                             "fare_forecaster", "train_recommender"],
}
DEPLOYMENT_WINDOW_S = 30 * 60


class RootCauseAnalyzer:
    def __init__(self, platform):
        self.p = platform

    # ------------------------------------------------------------ graph
    def graph(self) -> dict[str, list[str]]:
        """node -> upstream dependencies."""
        g: dict[str, list[str]] = {}
        for name, spec in self.p.specs.items():
            feeds = sorted(set(spec.contract.feeds().values()) - {"client", "unknown"})
            g[name] = feeds + [f"{name}@deployment"]
        for name, deps in STATIC_DEPENDENCIES.items():
            g[name] = sorted(set(g.get(name, []) + deps + [f"{name}@deployment"]))
        return g

    def downstream(self, node: str) -> list[str]:
        g = self.graph()
        out, frontier = set(), {node}
        while frontier:
            nxt = {a for a, deps in g.items() for d in frontier if d in deps and a not in out}
            out |= nxt
            frontier = nxt
        return sorted(out)

    def dot(self) -> str:
        lines = ["digraph G {", "rankdir=LR;", 'node [shape=box, style="rounded,filled", fillcolor="#eef2f7", '
                 'fontname="Helvetica", fontsize=10];']
        for asset, deps in self.graph().items():
            lines.append(f'"{asset}" [fillcolor="#d9e8ff"];')
            for d in deps:
                if d.endswith("@deployment"):
                    continue
                lines.append(f'"{d}" -> "{asset}";')
        lines.append("}")
        return "\n".join(lines)

    # ------------------------------------------------------------ evidence helpers
    def recent_deployment(self, asset: str) -> dict | None:
        """The deployment of the version serving now, if it happened within the deployment window and was
        not made by the supervisor itself. A release that has since been replaced is not a suspect."""
        prod = self.p.registry.production(asset)
        if not prod:
            return None
        since = time.time() - DEPLOYMENT_WINDOW_S
        for ev in self.p.audit.recent(limit=200, target=asset, since=since):
            d = ev["details"]
            if ev["action"] == "stage_changed" and d.get("stage") == "production":
                if d.get("version") != prod["version"]:
                    continue
                if ev["actor"] == "supervisor-agent" or any(w in str(d.get("reason", ""))
                                                            for w in ("rollback", "initial", "demo reset")):
                    return None
                return {"version": d.get("version"), "actor": ev["actor"], "ts": ev["ts"], "reason": d.get("reason")}
        return None

    # ------------------------------------------------------------ analysis
    def analyze(self, asset: str, findings: list[dict], report: dict) -> dict:
        kinds = {f["kind"] for f in findings}
        chain: list[str] = [f"{f['kind']}: {f['message']}" for f in findings]
        deployment = self.recent_deployment(asset)
        checks = report.get("checks", {})
        drift = checks.get("drift", {})
        drifted = drift.get("drifted_features", []) if drift.get("status") == "breach" else []
        actions: list[dict] = []
        root, root_kind, cause, confidence = asset, "model", "unknown", 0.4

        if "data_quality" in kinds:
            feeds: dict[str, int] = {}
            for f in findings:
                if f["kind"] == "data_quality":
                    for i, feed in enumerate(f.get("feeds", [])):
                        feeds[feed] = feeds.get(feed, 0) + (len(f.get("feeds", [])) - i)
            root = max(feeds, key=feeds.get) if feeds else "unknown_feed"
            root_kind, cause, confidence = "feed", "corrupted_upstream_feed", 0.9
            chain.append(f"contract violations concentrate on {root}; bad records are already blocked at inference")
            actions = [{"action": "quarantine_feed", "params": {"feed": root},
                        "reason": f"switch {root} to its backup source until the primary passes health probes"}]
        elif kinds & {"hallucination", "quality", "csat"}:
            if deployment:
                root, root_kind, cause, confidence = "prompt_config", "config", "prompt_regression", 0.85
                chain.append(f"hallucinations started after {deployment['actor']} deployed prompt "
                             f"v{deployment['version']}")
                actions = [{"action": "rollback", "params": {}, "reason": "restore the last grounded prompt version"}]
            else:
                root, root_kind, cause, confidence = "kb_index", "feed", "ungrounded_generation", 0.6
                chain.append("no recent prompt change; answers are drifting from the retrieved evidence")
                actions = [{"action": "enable_strict_grounding", "params": {},
                            "reason": "force extractive, source-only answers while the cause is investigated"}]
        elif kinds & {"tool_errors", "agent_abnormal", "agent_violation"}:
            failed = {}
            for f in findings:
                failed.update(f.get("evidence", {}).get("failed_tools", {}) or {})
            if "tool_errors" in kinds or failed:
                tool = max(failed, key=failed.get) if failed else "timetable_search"
                root, root_kind, cause, confidence = tool, "tool", "failing_dependency", 0.85
                chain.append(f"agent sessions are retrying failed {tool} calls; decisions themselves are valid")
                actions = [{"action": "circuit_breaker", "params": {"tool": tool},
                            "reason": f"stop calling {tool} and serve the cached timetable until probes pass"}]
            elif deployment:
                root, root_kind, cause, confidence = f"{asset}@deployment", "config", "deployment_regression", 0.8
                actions = [{"action": "rollback", "params": {}, "reason": "restore the previous agent configuration"}]
            else:
                root, root_kind, cause, confidence = asset, "agent", "agent_decision_errors", 0.6
                actions = [{"action": "escalate", "params": {}, "reason": "agent decisions break constraints"}]
        elif kinds & {"performance", "fairness", "exposure_bias"} and deployment and self._previous_healthy(asset):
            root, root_kind, cause, confidence = f"{asset}@deployment", "config", "deployment_regression", 0.9
            chain.append(f"degradation began after {deployment['actor']} promoted v{deployment['version']}; the "
                         "previous version was healthy on live traffic")
            actions = [{"action": "rollback", "params": {}, "reason": "return the last healthy version to service"}]
        elif "false_positives" in kinds:
            root, root_kind, cause, confidence = "session_stream", "feed", "user_behaviour_change", 0.7
            chain.append("labelled outcomes show the flagged sessions are genuine passengers behaving differently "
                         "(e.g. Tatkal-window autofill), not a model fault")
            actions = [{"action": "adjust_threshold", "params": {},
                        "reason": "raise the block threshold within recall limits; affects passengers so needs approval"}]
            if "performance" in kinds:
                actions.append({"action": "retrain", "params": {}, "reason": "learn the new genuine behaviour"})
        elif "performance" in kinds:
            perf = checks.get("performance", {})
            if drifted:
                spec = self.p.specs[asset]
                feeds = sorted({self.p.monitors._feed(spec, f) for f in drifted})
                root, root_kind, cause, confidence = feeds[0], "feed", "data_drift_degradation", 0.8
                chain.append(f"inputs shifted on {', '.join(drifted[:3])} (owned by {', '.join(feeds)}) and accuracy "
                             "fell with them")
            elif asset == "fraud_detector":
                root, root_kind, cause, confidence = "fraud_detector:concept", "concept", "adversary_adaptation", 0.75
                chain.append("missed fraud rose while session features still look normal: attackers changed "
                             "behaviour to evade the current model (adversarial concept drift)")
            else:
                root, root_kind, cause, confidence = f"{asset}:concept", "concept", "concept_drift", 0.75
                chain.append("inputs look like the training data but outcomes changed: the input-outcome "
                             "relationship moved (change point "
                             f"{'localised' if perf.get('change_ts') else 'not localised'})")
            actions = [{"action": "retrain", "params": {"since_ts": perf.get("change_ts")},
                        "reason": "retrain on post-change labelled data, evaluate, then deploy as shadow"}]
        elif "fairness" in kinds or "exposure_bias" in kinds:
            root, root_kind, cause, confidence = asset, "model", "fairness_degradation", 0.6
            actions = [{"action": "escalate", "params": {}, "reason": "fairness change without a deployment needs "
                        "human review of the affected segments"}]
        elif "drift" in kinds:
            spec = self.p.specs.get(asset)
            feeds = sorted({self.p.monitors._feed(spec, f) for f in drifted}) if spec else []
            root = feeds[0] if feeds else asset
            root_kind, cause, confidence = "feed", "benign_distribution_shift", 0.65
            perf = checks.get("performance", {})
            chain.append(f"accuracy is holding ({perf.get('metric')}={perf.get('current')}), so the shift is "
                         "being watched rather than acted on")
            actions = [{"action": "watch", "params": {}, "reason": "drift without degradation; retrain only if "
                        "labelled performance drops"}]
        elif "health" in kinds:
            root, root_kind, cause, confidence = asset, "service", "capacity_or_latency", 0.5
            actions = [{"action": "escalate", "params": {}, "reason": "latency above SLA"}]
        elif "user_behaviour_change" in kinds:
            root, root_kind, cause, confidence = "passenger_behaviour", "feed", "user_behaviour_change", 0.6
            actions = [{"action": "watch", "params": {}, "reason": "new question mix; review knowledge base coverage"}]

        impacted = self.downstream(asset)
        return {"asset": asset, "cause_type": cause, "root_cause": root, "root_kind": root_kind,
                "confidence": confidence, "evidence_chain": chain, "impacted": impacted,
                "drifted_features": drifted, "recent_deployment": deployment, "recommended_actions": actions,
                "analysed_at": time.time()}

    def _previous_healthy(self, asset: str) -> bool:
        """Was the version before the current production one meeting its targets on live traffic?"""
        versions = self.p.registry.versions(asset)
        prod = next((v for v in versions if v["stage"] == "production"), None)
        prev = next((v for v in versions if v["stage"] in ("archived", "rolled_back") and v["promoted_ts"]
                     and prod and v["version"] != prod["version"]), None)
        if not prev:
            return False
        if asset not in self.p.specs:
            return True
        spec = self.p.specs[asset]
        from railsentinel.ml.runtime import recent_predictions
        rows = recent_predictions(self.p.db, asset, 300, version=prev["version"], labelled_only=True)
        if len(rows) < 50:
            return True  # no evidence against the previous version
        m = spec.metrics([r["output"] for r in rows], [r["label"] for r in rows], [r["segment"] for r in rows])
        pol = self.p.policies.for_asset(asset)
        metric = pol["performance"]["metric"]
        base, cur = (prev["metrics"] or {}).get(metric), m.get(metric)
        if base is None or cur is None:
            return True
        degr = (base - cur) / base if spec.higher_is_better else (cur - base) / base
        return degr <= pol["performance"]["max_relative_degradation"]
