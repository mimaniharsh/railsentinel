"""Autonomous governance end to end: each chaos scenario must be detected, diagnosed and handled."""
import pytest

EXPECT = {
    # scenario: (expected root-cause type, expected autopilot outcome)
    "fare_feed_glitch": ("corrupted_upstream_feed", "resolved"),
    "recommender_bad_release": ("deployment_regression", "resolved"),
    "prompt_regression": ("prompt_regression", "resolved"),
    "search_api_outage": ("failing_dependency", "resolved"),
    "cancellation_shift": ("concept_drift", "resolved"),
    "stealth_bots": ("adversary_adaptation", "resolved"),
    "fog_surge": ("data_drift_degradation", "resolved"),
    "festival_rush": ("benign_distribution_shift", "monitoring"),
}


def _incidents_since(platform, ts):
    return [platform.supervisor._decode(r) for r in
            platform.db.query("SELECT * FROM incidents WHERE opened_ts>=?", (ts,))]


def test_baseline_fleet_is_healthy(clean):
    clean.traffic.run()
    cycle = clean.supervisor.run_cycle()
    assert all(v["status"] == "healthy" for v in cycle["fleet"].values()), cycle["fleet"]


@pytest.mark.slow
@pytest.mark.parametrize("scenario", list(EXPECT))
def test_scenario_is_detected_diagnosed_and_handled(clean, scenario):
    import time
    started = time.time()
    run = clean.chaos.autopilot(scenario, actor="test", max_rounds=8)
    cause, outcome = EXPECT[scenario]
    assert run["outcome"] == outcome, run["timeline"]
    from railsentinel.governance.chaos import SCENARIOS
    targets = SCENARIOS[scenario]["targets"]
    incidents = [i for i in _incidents_since(clean, started) if i["model"] in targets]
    assert incidents and any(i["kind"] == cause for i in incidents), [(i["model"], i["kind"]) for i in incidents]
    assert clean.audit.verify()["valid"]


@pytest.mark.slow
def test_passenger_affecting_fix_waits_for_human_approval(clean):
    run = clean.chaos.autopilot("tatkal_rush_false_positives", actor="test", max_rounds=4)
    assert run["outcome"] == "awaiting_human_approval"
    pending = clean.supervisor.approvals("pending")
    assert pending and pending[0]["action"] == "adjust_threshold"
    before = clean.world.ops().fraud_threshold
    clean.supervisor.decide(pending[0]["id"], approve=True, actor="admin:test")
    assert clean.world.ops().fraud_threshold > before
    for _ in range(3):
        clean.traffic.run()
        clean.supervisor.run_cycle()
    inc = [i for i in clean.supervisor.incidents() if i["model"] == "fraud_detector"][0]
    assert inc["status"] == "resolved"
    actions = {a["action"]: a for a in inc["actions"]}
    assert actions["adjust_threshold"]["executed_by"] == "admin:test"


def test_manual_rollback_and_registry_history(clean):
    p = clean
    spec = p.specs["delay_predictor"]
    art = p.registry.load("delay_predictor", 1)
    v2 = p.registry.add_version("delay_predictor", art, art["baseline_metrics"], stage="production",
                                notes="manual test release", actor="admin:test")
    assert p.registry.production("delay_predictor")["version"] == v2
    res = p.lifecycle.rollback("delay_predictor", actor="admin:test", reason="test")
    assert res["ok"] and res["to"] == 1
    assert p.registry.production("delay_predictor")["version"] == 1
    actions = [e["action"] for e in p.audit.recent(limit=20, target="delay_predictor")]
    assert "rollback" in actions and "version_created" in actions
    assert spec.name == "delay_predictor"


def test_governance_report_and_ethics(platform):
    from railsentinel.governance.reports import ethics_checks, governance_report
    report = governance_report(platform)
    assert "Model cards" in report and "NIST AI RMF" in report
    assert all(c["passed"] for c in ethics_checks(platform))
