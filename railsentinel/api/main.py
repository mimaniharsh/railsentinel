"""RailSentinel REST API (FastAPI).

Roles (X-API-Key header): passenger < operator < admin.
* passenger: travel services (predictions, planner, support, PNR)
* operator: read the governance layer (fleet, incidents, audit, reports)
* admin: change the system (approvals, rollbacks, retraining, chaos scenarios)
"""
from __future__ import annotations

import json
import logging
import time
from contextlib import asynccontextmanager
from datetime import date, datetime

from fastapi import Depends, FastAPI, Header, HTTPException, Request
from fastapi.responses import JSONResponse, PlainTextResponse

from railsentinel import __version__
from railsentinel.agents.tools import confirmation_record
from railsentinel.api import schemas as s
from railsentinel.config import IST, ROLE_LEVELS
from railsentinel.core import Platform
from railsentinel.db import dumps, loads
from railsentinel.governance.chaos import SCENARIOS
from railsentinel.governance.reports import ethics_checks, governance_report
from railsentinel.ml.runtime import recent_predictions
from railsentinel.security import RateLimiter, verify
from railsentinel.twin import inventory as inv
from railsentinel.twin.generators import today_ist
from railsentinel.twin.network import CATEGORIES, TRAVEL_CLASSES

log = logging.getLogger("railsentinel.api")


class SafeJSONResponse(JSONResponse):
    """JSON response that also serialises numpy values and other non-standard types."""

    def render(self, content) -> bytes:
        return dumps(content).encode("utf-8")


def platform() -> Platform:
    return Platform.get()


@asynccontextmanager
async def lifespan(app: FastAPI):
    p = Platform.get()
    if not p.is_bootstrapped():
        log.warning("No trained models found: bootstrapping (about a minute on a laptop)...")
        p.bootstrap(progress=log.warning)
    p.refresh_demo_bookings()
    p.supervisor.start(p.settings.supervisor_interval_s, background_traffic=p.settings.background_traffic)
    app.state.limiter = RateLimiter(p.settings.rate_limit_per_minute)
    yield
    p.supervisor.stop()


app = FastAPI(title="RailSentinel API", version=__version__, lifespan=lifespan,
              default_response_class=SafeJSONResponse,
              description="AI rail-travel services governed by an autonomous AI control tower.")


# ---------------------------------------------------------------- security
def require(role: str):
    def dependency(request: Request, x_api_key: str | None = Header(default=None)) -> str:
        p = platform()
        if not p.settings.auth_enabled:
            return "anonymous:admin"
        granted = p.settings.api_keys.get(x_api_key or "")
        if granted is None:
            raise HTTPException(status_code=401, detail="missing or invalid X-API-Key")
        if ROLE_LEVELS[granted] < ROLE_LEVELS[role]:
            raise HTTPException(status_code=403, detail=f"requires role '{role}'")
        if granted == "passenger":
            limiter: RateLimiter = request.app.state.limiter
            client = f"{x_api_key}:{request.client.host if request.client else 'unknown'}"
            if not limiter.allow(client):
                p.audit.append("security", "rate_limited", "api", {"client_role": granted,
                                                                    "path": request.url.path})
                raise HTTPException(status_code=429, detail="rate limit exceeded")
        return f"{granted}:{(x_api_key or '')[-4:]}"
    return dependency


passenger = require("passenger")
operator = require("operator")
admin = require("admin")


def _bad_request(exc: Exception) -> HTTPException:
    return HTTPException(status_code=400, detail=str(exc))


# ---------------------------------------------------------------- meta
@app.get("/health", tags=["meta"])
def health():
    p = platform()
    return {"status": "ok", "version": __version__, "bootstrapped": p.is_bootstrapped(),
            "llm": {"enabled": p.llm.enabled, "model": p.settings.llm_model if p.llm.enabled else None,
                    "last_error": p.llm.last_error},
            "scenario": p.world.scenario, "time": datetime.now(IST).isoformat(timespec="seconds")}


@app.get("/meta", tags=["meta"])
def meta(_: str = Depends(passenger)):
    p = platform()
    cities = sorted({st.city for st in p.network.stations.values()})
    return {"stations": [p.network.station_dict(c) for c in sorted(p.network.stations)], "cities": cities,
            "classes": {k: v["name"] for k, v in TRAVEL_CLASSES.items()},
            "categories": {k: v["label"] for k, v in CATEGORIES.items()}, "trains": len(p.network.trains),
            "today": today_ist().isoformat()}


@app.get("/trains", tags=["meta"])
def trains(origin: str, destination: str, _: str = Depends(passenger)):
    p = platform()
    segs = p.network.direct_segments(p.network.resolve_place(origin), p.network.resolve_place(destination))
    return [{"train_number": sg.train.number, "train_name": sg.train.name, "category": sg.train.category,
             "origin": sg.origin, "destination": sg.destination, "departure": f"{(sg.dep_offset % 1440) // 60:02d}:"
             f"{sg.dep_offset % 60:02d}", "duration_min": sg.duration_min, "distance_km": round(sg.distance_km),
             "classes": list(sg.train.classes)} for sg in segs]


@app.get("/trains/{number}", tags=["meta"])
def train(number: str, _: str = Depends(passenger)):
    p = platform()
    if number not in p.network.trains:
        raise HTTPException(404, "train not found")
    return p.network.trains[number].to_dict()


# ---------------------------------------------------------------- passenger services
@app.post("/predict/confirmation", tags=["passenger"])
def predict_confirmation(req: s.ConfirmationRequest, _: str = Depends(passenger)):
    p = platform()
    if req.pnr:
        b = p.bookings.get(req.pnr)
        if not b:
            raise HTTPException(404, "PNR not found")
        res = p.pnr_confirmation(b, explain=True)
        if res is None:
            if b["current_status"].split()[0] in ("WL", "RAC"):
                msg = f"{b['travel_class']} is not covered by the confirmation model (it covers SL, CC, 3A, 2A, 1A)."
            else:
                msg = f"Ticket status is {b['current_status']}; no prediction needed."
            return {"pnr": req.pnr, "booking": b, "status": "not_applicable", "message": msg}
        return {"pnr": req.pnr, "booking": b, **res}
    if not all([req.train_number, req.origin, req.destination, req.travel_class, req.journey_date]):
        raise HTTPException(400, "give a pnr, or train_number, origin, destination, travel_class and journey_date")
    try:
        seg = p.segment_for(req.train_number, req.origin, req.destination)
    except (ValueError, KeyError) as exc:
        raise _bad_request(exc)
    snap = inv.snapshot(p.network, seg, req.travel_class, req.journey_date, today_ist(), p.world.effective_params())
    snap = {**snap, "status": req.booking_status, "position": req.queue_position}
    record = confirmation_record(snap, p.network.stations[seg.origin].region, quota=req.quota)
    res = p.runtime("confirmation_predictor").predict([record], source="api", explain=True)[0]
    return {**res, "inputs": record}


@app.get("/insights/confirmation-trends", tags=["passenger"])
def confirmation_trends(train_number: str, travel_class: str, _: str = Depends(passenger)):
    try:
        return platform().confirmation_trends(train_number, travel_class)
    except KeyError:
        raise HTTPException(404, "train not found")
    except ValueError as exc:
        raise _bad_request(exc)


@app.post("/recommend", tags=["passenger"])
def recommend(req: s.RecommendRequest, _: str = Depends(passenger)):
    p = platform()
    o, d = p.network.resolve_place(req.origin), p.network.resolve_place(req.destination)
    if not o or not d:
        raise HTTPException(400, "unknown origin or destination")
    its = p.planner.tools.search(o, d, req.journey_date)
    options = p.planner.tools.assess(its, today_ist(), classes=req.classes)
    if not options:
        return {"options": [], "message": "no bookable options for that date"}
    budget = req.budget or float(sorted(x["fare"] for x in options)[len(options) // 2]) * 1.3
    res = p.runtime("train_recommender").predict(
        [{"user": req.profile.model_dump(), "budget": budget, "options": options, "origin": o[0], "destination": d[0],
          "journey_date": req.journey_date.isoformat()}], source="api")[0]
    if res["status"] != "ok":
        return {"status": "blocked", "reasons": res["output"]["reasons"], "passport": res["passport"]}
    ranked = []
    for item in res["output"]["top"]:
        opt = options[item["index"]]
        ranked.append({**{k: v for k, v in opt.items() if k != "legs"}, "legs": opt["legs"], "rank": item["rank"],
                       "match": item["match"], "rationale": item["rationale"]})
    return {"budget": budget, "options": ranked, "candidates": len(options), "prediction_id": res["prediction_id"],
            "passport": res["passport"]}


@app.post("/predict/delay", tags=["passenger"])
def predict_delay(req: s.DelayRequest, _: str = Depends(passenger)):
    p = platform()
    try:
        return p.predict_train_delay(req.train_number, req.station, req.run_date or today_ist())
    except (ValueError, KeyError) as exc:
        raise _bad_request(exc)


@app.get("/stations/{code}/congestion", tags=["passenger"])
def congestion(code: str, day: date | None = None, _: str = Depends(passenger)):
    try:
        return platform().station_congestion(code.upper(), day or today_ist())
    except KeyError:
        raise HTTPException(404, "station not found")


@app.post("/predict/fare", tags=["passenger"])
def predict_fare(req: s.FareRequest, _: str = Depends(passenger)):
    try:
        return platform().fare_forecast(req.train_number, req.origin, req.destination, req.travel_class,
                                        req.journey_date, quoted_fare=req.quoted_fare)
    except (ValueError, KeyError) as exc:
        raise _bad_request(exc)


@app.post("/agent/plan", tags=["passenger"])
def plan(req: s.PlanRequest, _: str = Depends(passenger)):
    return platform().planner.plan(req.query, session_id=req.session_id,
                                   profile=req.profile.model_dump() if req.profile else None)


@app.post("/agent/intents/{intent_id}/approve", tags=["passenger"])
def approve_intent(intent_id: str, req: s.ApproveBooking, who: str = Depends(passenger)):
    try:
        return platform().planner.approve(intent_id, req.passenger_name, actor=who)
    except KeyError as exc:
        raise HTTPException(404, str(exc))
    except ValueError as exc:
        raise HTTPException(409, str(exc))


@app.post("/agent/intents/{intent_id}/reject", tags=["passenger"])
def reject_intent(intent_id: str, who: str = Depends(passenger)):
    platform().planner.reject(intent_id, actor=who)
    return {"intent_id": intent_id, "status": "rejected"}


@app.post("/support/chat", tags=["passenger"])
def support_chat(req: s.ChatRequest, _: str = Depends(passenger)):
    return platform().support.answer(req.message, session_id=req.session_id)


@app.post("/support/feedback", tags=["passenger"])
def support_feedback(req: s.FeedbackRequest, _: str = Depends(passenger)):
    platform().support.feedback(req.interaction_id, req.helpful)
    return {"ok": True}


@app.get("/pnr", tags=["passenger"])
def list_pnrs(limit: int = 50, _: str = Depends(passenger)):
    return platform().bookings.list(limit)


@app.get("/pnr/{pnr}", tags=["passenger"])
def get_pnr(pnr: str, _: str = Depends(passenger)):
    p = platform()
    b = p.bookings.get(pnr)
    if not b:
        raise HTTPException(404, "PNR not found")
    return {"booking": b, "refund": p.bookings.refund_status(pnr)}


@app.get("/pnr/{pnr}/alerts", tags=["passenger"])
def pnr_alerts(pnr: str, _: str = Depends(passenger)):
    try:
        return platform().journey_alerts(pnr)
    except KeyError:
        raise HTTPException(404, "PNR not found")


@app.post("/pnr/{pnr}/cancel", tags=["passenger"])
def cancel_pnr(pnr: str, who: str = Depends(passenger)):
    p = platform()
    try:
        result = p.bookings.cancel(pnr)
    except KeyError:
        raise HTTPException(404, "PNR not found")
    p.audit.append(who, "booking_cancelled", "bookings", {"pnr": pnr, "refund": result.get("refund")})
    return result


@app.post("/fraud/score", tags=["passenger"])
def fraud_score(req: s.FraudSession, _: str = Depends(passenger)):
    rec = req.model_dump()
    rec["account_tenure"] = rec.get("account_tenure") or ("new" if rec["account_age_days"] < 90 else "established")
    return platform().runtime("fraud_detector").predict([rec], source="api", explain=True)[0]


@app.post("/passports/verify", tags=["meta"])
def verify_passport(req: s.PassportVerify):
    return {"valid": verify(req.passport, platform().settings.passport_secret)}


# ---------------------------------------------------------------- governance (read)
@app.get("/gov/fleet", tags=["governance"])
def fleet(_: str = Depends(operator)):
    p = platform()
    health = p.db.get_kv("fleet_health", {})
    out = []
    for m in p.registry.models():
        prod = p.registry.production(m["name"])
        shadow = p.registry.shadow(m["name"])
        latest = {}
        for row in p.db.query("SELECT metric, value FROM metrics WHERE model=? AND ts>=? ORDER BY ts DESC LIMIT 40",
                              (m["name"], time.time() - 3600)):
            latest.setdefault(row["metric"], row["value"])
        out.append({**m, "production_version": prod["version"] if prod else None,
                    "shadow_version": shadow["version"] if shadow else None,
                    "baseline_metrics": {k: v for k, v in (prod["metrics"] or {}).items()
                                         if isinstance(v, (int, float))} if prod else {},
                    "health": health.get(m["name"], {"status": "unknown"}), "latest_metrics": latest})
    last = p.db.query_one("SELECT ts, duration_ms, summary FROM supervisor_cycles ORDER BY id DESC LIMIT 1")
    return {"assets": out, "world": p.world.snapshot(),
            "last_cycle": {"ts": last["ts"], "duration_ms": last["duration_ms"],
                           "resources": loads(last["summary"], {}).get("resources")} if last else None,
            "llm_enabled": p.llm.enabled}


@app.get("/gov/assets/{name}", tags=["governance"])
def asset(name: str, _: str = Depends(operator)):
    p = platform()
    if not p.registry.model(name):
        raise HTTPException(404, "asset not found")
    return {"model": p.registry.model(name), "versions": p.registry.versions(name),
            "policy": p.policies.for_asset(name), "report": p.monitors.collect(name, record=False),
            "dependencies": p.rca.graph().get(name, []), "downstream": p.rca.downstream(name)}


@app.get("/gov/metrics", tags=["governance"])
def metrics(asset: str, metric: str, hours: float = 6, _: str = Depends(operator)):
    rows = platform().db.query("SELECT ts, value FROM metrics WHERE model=? AND metric=? AND ts>=? ORDER BY ts",
                               (asset, metric, time.time() - hours * 3600))
    return rows


@app.get("/gov/incidents", tags=["governance"])
def incidents(active: bool = False, limit: int = 50, _: str = Depends(operator)):
    return platform().supervisor.incidents(active_only=active, limit=limit)


@app.get("/gov/approvals", tags=["governance"])
def approvals(status: str | None = None, _: str = Depends(operator)):
    return platform().supervisor.approvals(status)


@app.get("/gov/audit", tags=["governance"])
def audit(limit: int = 200, target: str | None = None, _: str = Depends(operator)):
    return platform().audit.recent(limit=limit, target=target)


@app.get("/gov/audit/verify", tags=["governance"])
def audit_verify(_: str = Depends(operator)):
    return platform().audit.verify()


@app.get("/gov/report", tags=["governance"], response_class=PlainTextResponse)
def report(_: str = Depends(operator)):
    return governance_report(platform())


@app.get("/gov/ethics", tags=["governance"])
def ethics(_: str = Depends(operator)):
    return ethics_checks(platform())


@app.get("/gov/graph", tags=["governance"])
def graph(_: str = Depends(operator)):
    p = platform()
    return {"dot": p.rca.dot(), "graph": p.rca.graph()}


@app.get("/gov/policies", tags=["governance"])
def policies(_: str = Depends(operator)):
    return platform().policies.describe()


@app.get("/gov/cycles", tags=["governance"])
def cycles(limit: int = 20, _: str = Depends(operator)):
    rows = platform().db.query("SELECT * FROM supervisor_cycles ORDER BY id DESC LIMIT ?", (limit,))
    for r in rows:
        r["summary"] = loads(r["summary"], {})
    return rows


@app.get("/gov/fraud/alerts", tags=["governance"])
def fraud_alerts(limit: int = 50, _: str = Depends(operator)):
    rows = recent_predictions(platform().db, "fraud_detector", 2000)
    flagged = [r for r in rows if r["status"] == "ok" and r["output"].get("risk_score", 0) >= 0.3][:limit]
    return [{"prediction_id": r["id"], "ts": r["ts"], "version": r["version"], "source": r["source"],
             "risk_score": r["output"]["risk_score"], "decision": r["output"]["decision"],
             "predicted_type": r["output"]["predicted_type"], "reason_codes": r["output"]["reason_codes"],
             "anomaly_score": r["output"]["anomaly_score"], "outcome": r["label"], "segment": r["segment"]}
            for r in flagged]


@app.get("/gov/agent/traces", tags=["governance"])
def agent_traces(limit: int = 30, _: str = Depends(operator)):
    rows = platform().db.query("SELECT * FROM agent_traces ORDER BY ts DESC LIMIT ?", (limit,))
    for r in rows:
        for k in ("constraints", "steps", "plan", "verdict"):
            r[k] = loads(r[k], {})
    return rows


@app.get("/gov/support/interactions", tags=["governance"])
def support_interactions(limit: int = 50, _: str = Depends(operator)):
    rows = platform().db.query("SELECT * FROM llm_interactions ORDER BY ts DESC LIMIT ?", (limit,))
    for r in rows:
        for k in ("sources", "context", "grounding"):
            r[k] = loads(r[k], {})
    return rows


@app.get("/world", tags=["governance"])
def world(_: str = Depends(operator)):
    return platform().world.snapshot()


@app.get("/chaos/scenarios", tags=["chaos"])
def scenarios(_: str = Depends(operator)):
    return platform().chaos.scenarios()


# ---------------------------------------------------------------- governance (act)
@app.post("/gov/cycle", tags=["governance"])
def run_cycle(_: str = Depends(admin)):
    cycle = platform().supervisor.run_cycle()
    return {k: v for k, v in cycle.items() if k != "reports"}


@app.post("/gov/approvals/{approval_id}", tags=["governance"])
def decide(approval_id: str, req: s.ApprovalDecision, who: str = Depends(admin)):
    try:
        return platform().supervisor.decide(approval_id, req.approve, actor=who)
    except KeyError as exc:
        raise HTTPException(404, str(exc))
    except ValueError as exc:
        raise HTTPException(409, str(exc))


@app.post("/gov/assets/{name}/rollback", tags=["governance"])
def rollback(name: str, req: s.RollbackRequest, who: str = Depends(admin)):
    res = platform().lifecycle.rollback(name, actor=who, reason="manual rollback from Control Tower",
                                        to_version=req.to_version)
    if not res.get("ok"):
        raise HTTPException(409, res.get("reason", "rollback failed"))
    return res


@app.post("/gov/assets/{name}/promote", tags=["governance"])
def promote(name: str, req: s.PromoteRequest, who: str = Depends(admin)):
    p = platform()
    if not any(v["version"] == req.version for v in p.registry.versions(name)):
        raise HTTPException(404, "version not found")
    return p.lifecycle.promote(name, req.version, actor=who, reason="manual promotion from Control Tower")


@app.post("/gov/assets/{name}/retrain", tags=["governance"])
def retrain(name: str, who: str = Depends(admin)):
    p = platform()
    if name not in p.specs:
        raise HTTPException(400, "only ML models can be retrained")
    return p.lifecycle.retrain(name, actor=who, reason="manual retrain from Control Tower")


@app.post("/chaos/activate", tags=["chaos"])
def chaos_activate(req: s.ScenarioRequest, who: str = Depends(admin)):
    if req.scenario not in SCENARIOS:
        raise HTTPException(404, "unknown scenario")
    return platform().chaos.activate(req.scenario, actor=who)


@app.post("/chaos/clear", tags=["chaos"])
def chaos_clear(who: str = Depends(admin)):
    return platform().chaos.clear(actor=who)


@app.post("/chaos/reset", tags=["chaos"])
def chaos_reset(who: str = Depends(admin)):
    return platform().chaos.reset(actor=who)


@app.post("/chaos/traffic", tags=["chaos"])
def chaos_traffic(req: s.TrafficRequest, _: str = Depends(admin)):
    return platform().traffic.run(scale=req.scale)


@app.post("/chaos/autopilot", tags=["chaos"])
def chaos_autopilot(req: s.ScenarioRequest, who: str = Depends(admin)):
    if req.scenario not in SCENARIOS:
        raise HTTPException(404, "unknown scenario")
    p = platform()
    if req.reset_first:
        p.chaos.reset(actor=who)
    result = p.chaos.autopilot(req.scenario, actor=who, max_rounds=req.max_rounds)
    return json.loads(dumps(result))
