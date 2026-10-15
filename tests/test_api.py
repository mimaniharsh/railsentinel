"""REST API: role-based access, passenger endpoints and governance endpoints."""
from datetime import timedelta

import pytest
from fastapi.testclient import TestClient

from railsentinel.twin.generators import today_ist

P = {"X-API-Key": "rs-passenger-demo"}
O = {"X-API-Key": "rs-operator-demo"}
A = {"X-API-Key": "rs-admin-demo"}


@pytest.fixture(scope="module")
def api(platform):
    from railsentinel.api.main import app
    with TestClient(app) as client:
        yield client


def test_health_and_rbac(api):
    assert api.get("/health").json()["bootstrapped"]
    assert api.get("/meta").status_code == 401
    assert api.get("/gov/fleet", headers=P).status_code == 403
    assert api.post("/chaos/activate", headers=O, json={"scenario": "fog_surge"}).status_code == 403
    assert api.get("/gov/fleet", headers=O).status_code == 200


def test_passenger_endpoints(api):
    jd = (today_ist() + timedelta(days=15)).isoformat()
    r = api.post("/predict/confirmation", headers=P, json={"train_number": "RS101", "origin": "PUNE",
                                                          "destination": "NZM", "travel_class": "SL",
                                                          "journey_date": jd, "booking_status": "WL",
                                                          "queue_position": 30})
    body = r.json()
    assert r.status_code == 200 and 0 <= body["output"]["p_confirm"] <= 1
    assert api.post("/passports/verify", json={"passport": body["passport"]}).json()["valid"]
    tampered = {**body["passport"], "confidence": 1.0}
    assert not api.post("/passports/verify", json={"passport": tampered}).json()["valid"]
    r = api.post("/recommend", headers=P, json={"origin": "Mumbai", "destination": "Delhi", "journey_date": jd})
    assert r.status_code == 200 and r.json()["options"]
    r = api.post("/predict/delay", headers=P, json={"train_number": "RS101", "station": "NZM"})
    assert r.status_code == 200 and r.json()["status"] == "ok"
    r = api.post("/predict/fare", headers=P, json={"train_number": "RS109", "origin": "PUNE", "destination": "NDLS",
                                                  "travel_class": "3A", "journey_date": jd})
    assert r.status_code == 200 and r.json()["curve"]
    demand = r.json()["demand_outlook"]
    assert demand["usual_demand"] > 0 and isinstance(demand["surge"], bool)
    r = api.get("/insights/confirmation-trends", headers=P, params={"train_number": "RS101", "travel_class": "SL"})
    trends = r.json()
    assert r.status_code == 200 and trends["by_waitlist_position"] and trends["booking_curve"]
    first, last = trends["by_waitlist_position"][0], trends["by_waitlist_position"][-1]
    assert first["confirmed_rate"] >= last["confirmed_rate"]  # deeper waitlists clear less often
    assert api.get("/insights/confirmation-trends", headers=P,
                   params={"train_number": "RS101", "travel_class": "2S"}).status_code == 400
    assert api.get("/stations/NDLS/congestion", headers=P).json()["trains"] > 0
    r = api.post("/agent/plan", headers=P, json={"query": "Cheapest way from Bengaluru to Chennai this friday"})
    assert r.status_code == 200 and r.json()["status"] == "proposed"
    r = api.post("/support/chat", headers=P, json={"message": "When is the reservation chart prepared?"})
    assert r.status_code == 200 and not r.json()["grounding"]["hallucination"]


def test_governance_endpoints(api):
    assert api.get("/gov/audit/verify", headers=O).json()["valid"]
    assert "RailSentinel governance report" in api.get("/gov/report", headers=O).text
    assert "digraph" in api.get("/gov/graph", headers=O).json()["dot"]
    assert len(api.get("/chaos/scenarios", headers=O).json()) == 9
    r = api.post("/gov/cycle", headers=A)
    assert r.status_code == 200 and "fleet" in r.json()
