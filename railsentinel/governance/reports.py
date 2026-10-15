"""Governance reports: model cards, incident history, ethics checks and
control mapping, rendered as Markdown for download from the Control Tower."""
from __future__ import annotations

from datetime import datetime

from railsentinel.config import IST
from railsentinel.db import loads
from railsentinel.security import mask_pii

NIST_MAPPING = [
    ("GOVERN", "Declarative policies (policies.yaml), risk tiers, approval rules for high-risk actions, "
               "hash-chained audit log, RBAC on governance endpoints."),
    ("MAP", "Model cards with purpose, data provenance and limitations; dependency graph of feeds, models and agents."),
    ("MEASURE", "Continuous monitors: data contracts, PSI/KS drift, labelled performance, Page-Hinkley change "
                "detection, fairness parity, exposure bias, hallucination grounding, agent behaviour."),
    ("MANAGE", "Supervisor agent: incident triage and prioritisation, root-cause analysis, retrain-shadow-promote, "
               "rollback, feed quarantine, circuit breakers, human approval queue, verification before closure."),
]


def ethics_checks(platform) -> list[dict]:
    """Checks computed from the running system, not self-declared."""
    checks = []
    specs = platform.specs
    checks.append({"check": "Every ML prediction carries an explanation",
                   "passed": all(s.supports_shapley or s.name == "train_recommender" for s in specs.values()),
                   "detail": "Shapley attributions (tabular models) and per-option rationale (recommender)."})
    pol = platform.policies
    fair_assets = [a for a in specs if pol.for_asset(a).get("fairness", {}).get("enabled")]
    checks.append({"check": "Fairness monitored for passenger-impacting decisions",
                   "passed": {"confirmation_predictor", "fraud_detector", "train_recommender"} <= set(fair_assets),
                   "detail": f"Fairness monitors enabled on: {', '.join(fair_assets)}."})
    checks.append({"check": "Passenger-affecting threshold changes need human approval",
                   "passed": pol.requires_approval("fraud_detector", "adjust_threshold"),
                   "detail": "adjust_threshold is listed in approvals_required for the fraud detector."})
    agent_pnrs = {r["pnr"] for r in platform.db.query("SELECT pnr FROM bookings WHERE source='agent'")}
    approved_pnrs = {p for r in platform.db.query("SELECT pnr FROM booking_intents WHERE status='booked'")
                     for p in (r["pnr"] or "").split(",") if p}
    unapproved = sorted(agent_pnrs - approved_pnrs)
    checks.append({"check": "Agent never books without passenger approval",
                   "passed": not unapproved,
                   "detail": f"{len(agent_pnrs)} agent-created PNRs; {len(unapproved)} without an approved booking "
                             "intent." + (f" Unapproved: {', '.join(unapproved[:5])}." if unapproved else "")})
    sample = platform.db.query("SELECT query, response FROM llm_interactions ORDER BY ts DESC LIMIT 200")
    leaks = sum(1 for r in sample for text in (r["query"], r["response"]) if text and mask_pii(text) != text)
    checks.append({"check": "PII masked in conversation logs", "passed": leaks == 0,
                   "detail": f"{leaks} unmasked e-mail, phone, ID or card patterns in the last {len(sample)} "
                             "logged conversations."})
    audit = platform.audit.verify()
    checks.append({"check": "Audit trail intact", "passed": audit["valid"],
                   "detail": f"{audit['records']} records verified by hash chain."})
    graded = platform.db.query("SELECT grounding FROM llm_interactions ORDER BY ts DESC LIMIT 200")
    ungraded = sum(1 for r in graded if "grounding_score" not in (loads(r["grounding"], {}) or {}))
    checks.append({"check": "Generative answers checked for hallucination", "passed": ungraded == 0,
                   "detail": f"{len(graded) - ungraded} of the last {len(graded)} support answers carry a grounding "
                             "grade (numeric grounding and sentence support)."})
    return checks


def governance_report(platform) -> str:
    now = datetime.now(IST).strftime("%d %b %Y %H:%M IST")
    fleet = platform.db.get_kv("fleet_health", {})
    lines = ["# RailSentinel governance report", f"Generated {now}.", "",
             "All data in this report comes from RailSentinel's digital twin (synthetic passengers on the real "
             "station network).", "", "## Fleet status", "",
             "| Asset | Kind | Production version | Status | Open incidents |", "|---|---|---|---|---|"]
    for m in platform.registry.models():
        prod = platform.registry.production(m["name"])
        f = fleet.get(m["name"], {})
        lines.append(f"| {m['name']} | {m['kind']} | v{prod['version'] if prod else '-'} | {f.get('status', 'unknown')} "
                     f"| {f.get('open_incidents', 0)} |")
    lines += ["", "## Model cards", ""]
    for m in platform.registry.models():
        prod = platform.registry.production(m["name"])
        spec = platform.specs.get(m["name"])
        lines += [f"### {m['name']}", f"- Purpose: {m['description']}", f"- Kind: {m['kind']}; risk tier: "
                  f"{m['risk_tier']}; owner: {m['owner']}",
                  f"- Production version: v{prod['version']} (promoted "
                  f"{datetime.fromtimestamp(prod['promoted_ts'] or prod['created_ts'], IST).strftime('%d %b %H:%M')})"]
        if spec:
            base = prod["metrics"] or {}
            shown = {k: v for k, v in base.items() if isinstance(v, (int, float)) and v is not None}
            lines.append("- Validation metrics: " + ", ".join(f"{k}={v}" for k, v in list(shown.items())[:6]))
            lines.append(f"- Inputs: {', '.join(spec.features) if spec.features else 'request summary'}")
            lines.append("- Training data: synthetic records from the digital twin's documented generators; "
                         "retrained versions add labelled production outcomes.")
            lines.append("- Limitations: trained on simulated behaviour; real deployment needs historical "
                         "Indian Railways data and recalibration.")
        versions = platform.registry.versions(m["name"])
        lines.append("- Version history: " + "; ".join(f"v{v['version']} {v['stage']}" for v in versions[:6]))
        lines.append("")
    lines += ["## Incidents", "", "| ID | Asset | Cause | Severity | Status | Opened | Actions |", "|---|---|---|---|---|---|---|"]
    for inc in platform.supervisor.incidents(limit=50):
        acts = ", ".join(f"{a['action']}({a['status']})" for a in inc["actions"])
        opened = datetime.fromtimestamp(inc["opened_ts"], IST).strftime("%d %b %H:%M")
        lines.append(f"| {inc['id']} | {inc['model']} | {inc['kind']} | {inc['severity']} | {inc['status']} | "
                     f"{opened} | {acts} |")
    lines += ["", "## Ethical AI checks", "", "| Check | Result | Detail |", "|---|---|---|"]
    for c in ethics_checks(platform):
        lines.append(f"| {c['check']} | {'pass' if c['passed'] else 'FAIL'} | {c['detail']} |")
    audit = platform.audit.verify()
    lines += ["", "## Audit trail", "", f"Hash chain valid: **{audit['valid']}** across {audit['records']} records."
              + (f" Head hash `{audit.get('head', '')[:16]}...`." if audit["valid"] else
                 f" Broken at record {audit.get('broken_at_seq')}: {audit.get('reason')}."), "",
              "## Control mapping (NIST AI RMF functions)", "",
              "Used as an organising frame for the controls in this platform; this is not a certification.", "",
              "| Function | Controls |", "|---|---|"]
    lines += [f"| {f} | {c} |" for f, c in NIST_MAPPING]
    approvals = platform.supervisor.approvals()
    if approvals:
        lines += ["", "## Human approvals", "", "| ID | Asset | Action | Status | Decided by |", "|---|---|---|---|---|"]
        for a in approvals[:30]:
            lines.append(f"| {a['id']} | {a['model']} | {a['action']} | {a['status']} | {a['decided_by'] or '-'} |")
    return "\n".join(lines) + "\n"
