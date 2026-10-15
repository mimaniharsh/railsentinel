"""Supervisor agent: the closed governance loop.

Every cycle it observes all assets (monitors), opens or updates incidents,
diagnoses them (root-cause analysis), prioritises them, executes remediations
the policy allows, queues the rest for human approval, verifies fixes on fresh
data and closes incidents only when the checks that raised them pass again.
Every step is written to the hash-chained audit log.

Decisions come from policy and evidence. When an API key is configured,
Gemini writes the human-readable incident narrative; it does not choose actions.
"""
from __future__ import annotations

import logging
import threading
import time
import uuid
from datetime import datetime

import numpy as np

from railsentinel.config import IST
from railsentinel.db import dumps, loads
from railsentinel.governance.monitors import set_window_floor, window_floor
from railsentinel.ml.runtime import recent_predictions

log = logging.getLogger("railsentinel.supervisor")

ACTIVE = ("open", "mitigating", "awaiting_approval", "monitoring")
ACTOR = "supervisor-agent"
KIND_TO_CHECK = {"data_quality": "data_quality", "drift": "drift", "performance": "performance",
                 "fairness": "fairness", "exposure_bias": "exposure", "false_positives": "false_positives",
                 "hallucination": "hallucination", "quality": "quality", "csat": "quality",
                 "tool_errors": "agent", "agent_abnormal": "agent", "agent_violation": "agent", "health": "health"}
SEVERITY_ORDER = {"info": 0, "warning": 1, "critical": 2}
MAX_ACTIONS = 6
FEED_ASSETS = {"fare_feed": ["fare_forecaster"]}

NARRATIVE_SYSTEM = ("You write incident summaries for an ML on-call engineer at a railway platform. Use only the "
                    "facts in the JSON. Three sentences: what is wrong and its impact, the most likely root cause "
                    "with the key evidence, and what the supervisor is doing about it. No speculation.")


class Supervisor:
    def __init__(self, platform):
        self.p = platform
        self._lock = threading.Lock()
        self._thread: threading.Thread | None = None
        self._stop = threading.Event()
        # Held by the background loop for each traffic+cycle iteration, and by an autopilot run for its whole
        # duration, so the two never interleave.
        self._background = threading.Lock()

    # ------------------------------------------------------------ background loop
    def start(self, interval_s: int, background_traffic: bool = False) -> None:
        if interval_s <= 0 or (self._thread and self._thread.is_alive()):
            return
        self._stop.clear()

        def loop():
            while not self._stop.wait(interval_s):
                if not self._background.acquire(blocking=False):
                    continue  # an autopilot run is driving traffic and cycles itself
                try:
                    if background_traffic:
                        self.p.traffic.run(scale=0.5)
                    self.run_cycle()
                except Exception:  # keep the loop alive; the error is logged
                    log.exception("supervisor cycle failed")
                finally:
                    self._background.release()

        self._thread = threading.Thread(target=loop, name="supervisor", daemon=True)
        self._thread.start()

    def stop(self) -> None:
        self._stop.set()

    def pause(self) -> None:
        """Hold the background loop; waits for an iteration already in progress to finish."""
        self._background.acquire()

    def resume(self) -> None:
        self._background.release()

    # ------------------------------------------------------------ one cycle
    def run_cycle(self, actor: str = ACTOR) -> dict:
        with self._lock:
            t0 = time.perf_counter()
            reports = self.p.monitors.collect_all()
            events: list[dict] = []
            acted = self._handle_shadows(reports, events)
            self._probe_mitigations(events)
            for asset, rep in reports.items():
                if asset in acted:
                    continue  # production changed this cycle; judge it on fresh data next cycle
                findings = [f for f in rep.get("findings", []) if f["severity"] != "info"]
                inc = self.open_incident(asset)
                if findings and inc is None:
                    self._open(asset, findings, rep, events)
                elif findings and inc is not None:
                    self._update(inc, findings, rep, events)
                elif inc is not None:
                    self._maybe_resolve(inc, rep, events)
            self._execute_planned(events)
            health = self._fleet_health(reports)
            resources = self.p.monitors.resources()
            duration = (time.perf_counter() - t0) * 1000
            summary = {"events": events, "fleet": health, "resources": resources}
            self.p.db.execute("INSERT INTO supervisor_cycles(ts, duration_ms, summary) VALUES(?,?,?)",
                              (time.time(), round(duration, 1), dumps(summary)))
            return {"ts": time.time(), "duration_ms": round(duration, 1), "events": events, "fleet": health,
                    "reports": reports, "resources": resources}

    # ------------------------------------------------------------ incidents
    def open_incident(self, asset: str) -> dict | None:
        row = self.p.db.query_one(f"SELECT * FROM incidents WHERE model=? AND status IN ({','.join('?' * len(ACTIVE))}) "
                                  "ORDER BY opened_ts DESC LIMIT 1", (asset, *ACTIVE))
        return self._decode(row) if row else None

    def incidents(self, active_only: bool = False, limit: int = 100) -> list[dict]:
        if active_only:
            rows = self.p.db.query(f"SELECT * FROM incidents WHERE status IN ({','.join('?' * len(ACTIVE))}) "
                                   "ORDER BY priority DESC, opened_ts DESC LIMIT ?", (*ACTIVE, limit))
        else:
            rows = self.p.db.query("SELECT * FROM incidents ORDER BY opened_ts DESC LIMIT ?", (limit,))
        return [self._decode(r) for r in rows]

    @staticmethod
    def _decode(row: dict) -> dict:
        for k in ("evidence", "rca", "actions"):
            row[k] = loads(row[k], [] if k != "rca" else {})
        return row

    def _save(self, inc: dict) -> None:
        self.p.db.execute(
            "UPDATE incidents SET updated_ts=?, closed_ts=?, severity=?, priority=?, status=?, title=?, evidence=?, "
            "rca=?, actions=?, narrative=? WHERE id=?",
            (time.time(), inc.get("closed_ts"), inc["severity"], inc["priority"], inc["status"], inc["title"],
             dumps(inc["evidence"]), dumps(inc["rca"]), dumps(inc["actions"]), inc.get("narrative"), inc["id"]))

    def _open(self, asset: str, findings: list[dict], report: dict, events: list[dict]) -> dict:
        rca = self.p.rca.analyze(asset, findings, report)
        rca["handled_kinds"] = sorted({f["kind"] for f in findings})
        severity = max((f["severity"] for f in findings), key=lambda s: SEVERITY_ORDER[s])
        tier =(self.p.registry.model(asset) or {}).get("risk_tier", "medium")
        priority = self.p.policies.priority(severity, tier, len(rca["impacted"]))
        inc = {"id": "INC-" + uuid.uuid4().hex[:8].upper(), "opened_ts": time.time(), "updated_ts": time.time(),
               "closed_ts": None, "model": asset, "kind": rca["cause_type"], "severity": severity,
               "priority": priority, "status": "open",
               "title": f"{asset.replace('_', ' ')}: {rca['cause_type'].replace('_', ' ')}",
               "evidence": findings, "rca": rca,
               "actions": [{**a, "status": "planned", "planned_ts": time.time()} for a in rca["recommended_actions"]],
               "narrative": None, "scenario": self.p.world.scenario}
        inc["narrative"] = self._narrative(inc)
        self.p.db.execute(
            "INSERT INTO incidents(id, opened_ts, updated_ts, closed_ts, model, kind, severity, priority, status, title, "
            "evidence, rca, actions, narrative, scenario) VALUES(?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)",
            (inc["id"], inc["opened_ts"], inc["updated_ts"], None, asset, inc["kind"], severity, priority, "open",
             inc["title"], dumps(findings), dumps(rca), dumps(inc["actions"]), inc["narrative"], inc["scenario"]))
        self.p.audit.append(ACTOR, "incident_opened", asset, {"incident": inc["id"], "cause": rca["cause_type"],
                                                              "root_cause": rca["root_cause"], "severity": severity,
                                                              "priority": priority,
                                                              "findings": [f["message"] for f in findings]})
        events.append({"type": "incident_opened", "incident": inc["id"], "asset": asset, "cause": rca["cause_type"],
                       "root_cause": rca["root_cause"]})
        return inc

    def _update(self, inc: dict, findings: list[dict], report: dict, events: list[dict]) -> None:
        inc["evidence"] = findings
        inc["severity"] = max((f["severity"] for f in findings), key=lambda s: SEVERITY_ORDER[s])
        busy = (inc["status"] == "awaiting_approval" or self.p.registry.shadow(inc["model"]) is not None
                or any(a["status"] in ("planned", "awaiting_approval", "waiting") for a in inc["actions"]))
        if not busy:
            new_kinds = {f["kind"] for f in findings}
            handled = set(inc["rca"].get("handled_kinds", []))
            done = [a for a in inc["actions"] if a["status"] in ("executed", "failed", "rejected")]
            last = done[-1]["action"] if done else None
            # Watching and escalations wait for new evidence; a remediation that did not work gets a new plan.
            if not new_kinds <= handled or last not in ("watch", "escalate"):
                self._replan(inc, findings, report, events)
        self._save(inc)

    def _replan(self, inc: dict, findings: list[dict], report: dict, events: list[dict]) -> None:
        rca = self.p.rca.analyze(inc["model"], findings, report)
        max_attempts = int(self.p.policies.for_asset(inc["model"]).get("retraining", {}).get("max_attempts", 2))
        attempts: dict[str, int] = {}
        for a in inc["actions"]:
            if a["status"] in ("executed", "failed"):
                attempts[a["action"]] = attempts.get(a["action"], 0) + 1
        new_actions = [a for a in rca["recommended_actions"]
                       if a["action"] not in ("escalate",) and attempts.get(a["action"], 0) < max_attempts]
        if (not new_actions or len(inc["actions"]) >= MAX_ACTIONS) and not any(
                a["action"] == "escalate" for a in inc["actions"]):
            new_actions = [{"action": "escalate", "params": {},
                            "reason": "automated remediation did not restore the asset; human review needed"}]
        elif len(inc["actions"]) >= MAX_ACTIONS:
            new_actions = []
        rca["handled_kinds"] = sorted({f["kind"] for f in findings})
        inc["rca"] = rca
        inc["kind"] = rca["cause_type"]
        if not new_actions:
            return
        inc["actions"].extend({**a, "status": "planned", "planned_ts": time.time()} for a in new_actions)
        inc["status"] = "open"
        self.p.audit.append(ACTOR, "incident_replanned", inc["model"], {"incident": inc["id"],
                                                                         "cause": rca["cause_type"],
                                                                         "actions": [a["action"] for a in new_actions]})
        events.append({"type": "incident_replanned", "incident": inc["id"], "asset": inc["model"],
                       "actions": [a["action"] for a in new_actions]})

    def _maybe_resolve(self, inc: dict, report: dict, events: list[dict]) -> None:
        if inc["status"] == "awaiting_approval" or any(a["status"] in ("planned", "awaiting_approval", "waiting")
                                                        for a in inc["actions"]):
            return
        if self.p.registry.shadow(inc["model"]) is not None:
            return
        checks = report.get("checks", {})
        needed = {KIND_TO_CHECK.get(f["kind"]) for f in inc["evidence"]} - {None}
        if any(checks.get(c, {}).get("status") != "ok" for c in needed):
            return  # verification needs fresh data from after the fix
        inc["status"] = "resolved"
        inc["closed_ts"] = time.time()
        self._save(inc)
        mins = (inc["closed_ts"] - inc["opened_ts"]) / 60
        self.p.audit.append(ACTOR, "incident_resolved", inc["model"],
                            {"incident": inc["id"], "time_to_resolve_min": round(mins, 2),
                             "verified_checks": sorted(needed)})
        events.append({"type": "incident_resolved", "incident": inc["id"], "asset": inc["model"],
                       "minutes": round(mins, 2)})

    # ------------------------------------------------------------ actions
    def _execute_planned(self, events: list[dict]) -> None:
        for inc in self.incidents(active_only=True):
            changed = False
            for a in inc["actions"]:
                if a["status"] not in ("planned", "waiting"):
                    continue
                changed = True
                if a["action"] == "watch":
                    a.update(status="executed", executed_ts=time.time(), result={"status": "monitoring"})
                    inc["status"] = "monitoring"
                    continue
                if self.p.policies.requires_approval(inc["model"], a["action"]):
                    approval_id = "APR-" + uuid.uuid4().hex[:8].upper()
                    self.p.db.execute(
                        "INSERT INTO approvals(id, ts, incident_id, model, action, params, reason, status) "
                        "VALUES(?,?,?,?,?,?,?,?)", (approval_id, time.time(), inc["id"], inc["model"], a["action"],
                                                    dumps(a.get("params", {})), a.get("reason", ""), "pending"))
                    a.update(status="awaiting_approval", approval_id=approval_id)
                    inc["status"] = "awaiting_approval"
                    self.p.audit.append(ACTOR, "approval_requested", inc["model"],
                                        {"incident": inc["id"], "approval": approval_id, "action": a["action"],
                                         "risk": self.p.policies.action_risk(a["action"])})
                    events.append({"type": "approval_requested", "incident": inc["id"], "asset": inc["model"],
                                   "action": a["action"], "approval": approval_id})
                    continue
                self._run_action(inc, a, ACTOR, events)
            if changed:
                if inc["status"] == "open":
                    inc["status"] = "mitigating"
                self._save(inc)

    def _run_action(self, inc: dict, a: dict, actor: str, events: list[dict]) -> None:
        try:
            result = self.execute(inc["model"], a["action"], a.get("params", {}), actor,
                                  reason=f"{inc['id']}: {a.get('reason', '')}")
            if result.get("status") == "waiting_for_labels":
                if a["status"] != "waiting":
                    events.append({"type": "retrain_queued", "incident": inc["id"], "asset": inc["model"],
                                   "result": result})
                a.update(status="waiting", result=result)
                return
            a.update(status="executed", executed_ts=time.time(), result=result, executed_by=actor)
        except Exception as exc:
            result = {"ok": False, "error": f"{type(exc).__name__}: {exc}"}
            a.update(status="failed", executed_ts=time.time(), result=result, executed_by=actor)
        self.p.audit.append(actor, "remediation_executed", inc["model"],
                            {"incident": inc["id"], "action": a["action"], "result": result})
        events.append({"type": "remediation", "incident": inc["id"], "asset": inc["model"], "action": a["action"],
                       "result": result})

    def execute(self, asset: str, action: str, params: dict, actor: str, reason: str = "") -> dict:
        ops = self.p.world.ops()
        if action in ("watch", "escalate"):
            return {"ok": True, "status": "monitoring" if action == "watch" else "escalated to on-call ML engineer"}
        if action == "retrain":
            return self.p.lifecycle.retrain(asset, actor=actor, since_ts=params.get("since_ts"), reason=reason)
        if action == "promote":
            return self.p.lifecycle.promote(asset, int(params["version"]), actor, reason)
        if action == "rollback":
            return self.p.lifecycle.rollback(asset, actor, reason, params.get("to_version"))
        if action == "quarantine_feed":
            feed = params["feed"]
            feeds = sorted(set(ops.quarantined_feeds) | {feed})
            self.p.world.update_ops(quarantined_feeds=feeds)
            for a in FEED_ASSETS.get(feed, [asset]):
                set_window_floor(self.p.db, a)
            return {"ok": True, "quarantined": feeds, "note": f"{feed} traffic now served from its backup source"}
        if action == "circuit_breaker":
            tool = params.get("tool", "timetable_search")
            breakers = sorted(set(ops.circuit_breakers) | {tool})
            self.p.world.update_ops(circuit_breakers=breakers)
            set_window_floor(self.p.db, "travel_planner_agent")
            return {"ok": True, "circuit_breakers": breakers}
        if action == "close_circuit_breaker":
            tool = params.get("tool", "timetable_search")
            self.p.world.update_ops(circuit_breakers=sorted(set(ops.circuit_breakers) - {tool}))
            return {"ok": True, "closed": tool}
        if action == "enable_strict_grounding":
            self.p.world.update_ops(support_strict_mode=True)
            set_window_floor(self.p.db, "support_assistant")
            return {"ok": True, "support_strict_mode": True}
        if action == "adjust_threshold":
            return self._adjust_fraud_threshold(params)
        raise ValueError(f"unknown action {action}")

    def _adjust_fraud_threshold(self, params: dict) -> dict:
        pol = self.p.policies.for_asset("fraud_detector")
        max_fpr = pol["false_positives"]["max_fpr"]
        prod = self.p.registry.production("fraud_detector")
        base_recall = (prod["metrics"] or {}).get("recall") or 0.95
        # Calibrate on the same post-change window the monitors judged, not on older traffic.
        rows = recent_predictions(self.p.db, "fraud_detector", 1500, version=prod["version"], labelled_only=True,
                                  since=window_floor(self.p.db, "fraud_detector"), sources=("traffic", "api"))
        if params.get("threshold") is not None:
            chosen = float(params["threshold"])
        elif len(rows) < 100:
            return {"ok": False, "reason": "not enough labelled sessions to recalibrate"}
        else:
            from scipy.stats import beta

            risk = np.array([r["output"]["risk_score"] for r in rows])
            fraud = np.array([r["label"] != "legit" for r in rows])
            genuine = int((~fraud).sum())
            chosen, best = None, None
            for t in np.round(np.arange(0.5, 0.96, 0.025), 3):
                flag = risk >= t
                fp = int((flag & ~fraud).sum())
                fpr = fp / max(1, genuine)
                # One-sided 95% Clopper-Pearson upper bound: the true rate is under the limit with 95% confidence,
                # which matters when the calibration window holds few genuine sessions.
                fpr_upper = float(beta.ppf(0.95, fp + 1, genuine - fp)) if fp < genuine else 1.0
                rec = float((flag & fraud).sum() / max(1, fraud.sum()))
                if chosen is None and fpr <= 0.7 * max_fpr and fpr_upper <= max_fpr and rec >= 0.85 * base_recall:
                    chosen = float(t)
                score = rec - 5 * fpr
                if best is None or score > best[0]:
                    best = (score, float(t))
            chosen = chosen if chosen is not None else best[1]
        old = self.p.world.ops().fraud_threshold
        self.p.world.update_ops(fraud_threshold=chosen)
        # Adapting the threshold accepts the new behaviour as normal, so drift is measured against it from now on.
        accepted = [r["features"] for r in rows][:1500]
        if accepted:
            self.p.db.set_kv(f"drift_reference:fraud_detector:{prod['version']}", accepted)
        set_window_floor(self.p.db, "fraud_detector")
        return {"ok": True, "old_threshold": old, "new_threshold": chosen, "rebaselined_on": len(accepted)}

    # ------------------------------------------------------------ approvals
    def approvals(self, status: str | None = None) -> list[dict]:
        if status:
            rows = self.p.db.query("SELECT * FROM approvals WHERE status=? ORDER BY ts DESC", (status,))
        else:
            rows = self.p.db.query("SELECT * FROM approvals ORDER BY ts DESC LIMIT 100")
        for r in rows:
            r["params"] = loads(r["params"], {})
        return rows

    def decide(self, approval_id: str, approve: bool, actor: str) -> dict:
        with self._lock:
            row = self.p.db.query_one("SELECT * FROM approvals WHERE id=?", (approval_id,))
            if not row:
                raise KeyError("approval not found")
            if row["status"] != "pending":
                raise ValueError(f"approval already {row['status']}")
            inc = self._decode(self.p.db.query_one("SELECT * FROM incidents WHERE id=?", (row["incident_id"],)))
            action = next((a for a in inc["actions"] if a.get("approval_id") == approval_id), None)
            self.p.db.execute("UPDATE approvals SET status=?, decided_ts=?, decided_by=? WHERE id=?",
                              ("approved" if approve else "rejected", time.time(), actor, approval_id))
            self.p.audit.append(actor, "approval_" + ("granted" if approve else "rejected"), row["model"],
                                {"approval": approval_id, "incident": inc["id"], "action": row["action"]})
            events: list[dict] = []
            if approve and action is not None:
                self._run_action(inc, action, actor, events)
                inc["status"] = "mitigating"
            elif action is not None:
                action.update(status="rejected", executed_ts=time.time(), executed_by=actor)
                inc["status"] = "open" if any(a["status"] == "planned" for a in inc["actions"]) else "mitigating"
            self._save(inc)
            return {"approval": approval_id, "decision": "approved" if approve else "rejected", "events": events}

    # ------------------------------------------------------------ shadows and probes
    def _handle_shadows(self, reports: dict, events: list[dict]) -> set[str]:
        acted: set[str] = set()
        for asset, rep in reports.items():
            sh = rep.get("checks", {}).get("shadow")
            if not sh or sh.get("status") != "ready":
                continue
            acted.add(asset)
            inc = self.open_incident(asset)
            if sh["promote"]:
                self.p.lifecycle.promote(asset, sh["shadow_version"], ACTOR,
                                         f"shadow beat production on {sh['labels']} live requests "
                                         f"({sh['metric']} {sh['relative_gain']:+.1%}, paired test "
                                         f"p={sh['p_value']:.3g})", live_metrics=sh.get("shadow_metrics"))
                entry = {"action": "promote", "params": {"version": sh["shadow_version"]}, "status": "executed",
                         "executed_ts": time.time(), "result": sh, "reason": "shadow evaluation passed",
                         "executed_by": ACTOR}
                events.append({"type": "promoted", "asset": asset, "version": sh["shadow_version"], "comparison": sh})
            else:
                self.p.lifecycle.reject_shadow(asset, sh["shadow_version"], ACTOR,
                                               f"shadow did not beat production on live traffic "
                                               f"({sh['metric']} {sh['relative_gain']:+.1%}, paired test "
                                               f"p={sh['p_value']:.3g})")
                entry = {"action": "reject_shadow", "params": {"version": sh["shadow_version"]}, "status": "executed",
                         "executed_ts": time.time(), "result": sh, "reason": "shadow evaluation failed",
                         "executed_by": ACTOR}
                events.append({"type": "shadow_rejected", "asset": asset, "version": sh["shadow_version"]})
            if inc:
                inc["actions"].append(entry)
                inc["status"] = "mitigating"
                self._save(inc)
        return acted

    def _probe_mitigations(self, events: list[dict]) -> None:
        ops = self.p.world.ops()
        if "timetable_search" in ops.circuit_breakers:
            if all(self.p.planner.tools.probe_search() for _ in range(5)):
                self.execute("travel_planner_agent", "close_circuit_breaker", {"tool": "timetable_search"}, ACTOR)
                self.p.audit.append(ACTOR, "circuit_breaker_closed", "timetable_search",
                                    {"probes": 5, "note": "primary timetable search healthy again"})
                events.append({"type": "circuit_breaker_closed", "tool": "timetable_search"})
        if "fare_feed" in ops.quarantined_feeds:
            spec = self.p.specs["fare_forecaster"]
            sample = spec.generate(self.p.network, 200, self.p.rng(), self.p.world.params())
            bad = sum(1 for r in spec.frame_to_records(sample) if spec.contract.validate(r))
            if bad / 200 <= 0.01:
                self.p.world.update_ops(quarantined_feeds=sorted(set(ops.quarantined_feeds) - {"fare_feed"}))
                self.p.audit.append(ACTOR, "feed_released", "fare_feed",
                                    {"probe_records": 200, "violations": bad, "note": "primary feed healthy again"})
                events.append({"type": "feed_released", "feed": "fare_feed"})

    # ------------------------------------------------------------ reporting
    def _fleet_health(self, reports: dict) -> dict:
        open_by = {}
        for inc in self.incidents(active_only=True):
            open_by[inc["model"]] = open_by.get(inc["model"], 0) + 1
        now = datetime.now(IST).isoformat(timespec="seconds")
        fleet = {a: {"status": r.get("status", "unknown"), "open_incidents": open_by.get(a, 0), "checked_at": now}
                 for a, r in reports.items()}
        self.p.db.set_kv("fleet_health", fleet)
        return fleet

    def _narrative(self, inc: dict) -> str:
        rca = inc["rca"]
        actions = ", ".join(f"{a['action']} ({a['reason']})" for a in inc["actions"]) or "none"
        template = (f"{inc['title'].capitalize()} [{inc['severity']}]. "
                    f"Root cause: {rca['root_cause']} ({rca['cause_type'].replace('_', ' ')}, confidence "
                    f"{rca['confidence']:.0%}). Evidence: {'; '.join(rca['evidence_chain'][:3])}. "
                    f"Downstream impact: {', '.join(rca['impacted']) or 'none'}. Planned: {actions}.")
        if self.p.llm.enabled:
            facts = {"incident": inc["title"], "severity": inc["severity"], "findings": [f["message"] for f in inc["evidence"]],
                     "root_cause": rca["root_cause"], "cause_type": rca["cause_type"], "evidence": rca["evidence_chain"],
                     "impacted": rca["impacted"], "actions": [a["action"] + ": " + a["reason"] for a in inc["actions"]]}
            text = self.p.llm.text(NARRATIVE_SYSTEM, dumps(facts))
            if text:
                return text
        return template
