"""Passenger services: models, passports, planner agent (offline and Gemini), support assistant."""
from datetime import timedelta

import pytest
from google.genai import types

from railsentinel.agents.llm import LLM
from railsentinel.agents.tools import confirmation_record
from railsentinel.security import verify
from railsentinel.twin import inventory as inv
from railsentinel.twin.generators import today_ist


def _response(*parts: types.Part) -> types.GenerateContentResponse:
    """A real SDK response object with one model candidate."""
    return types.GenerateContentResponse(candidates=[types.Candidate(
        content=types.Content(role="model", parts=list(parts)), finish_reason=types.FinishReason.STOP)])


def _call(call_id: str, name: str, args: dict) -> types.Part:
    return types.Part(function_call=types.FunctionCall(id=call_id, name=name, args=args))


def _text(text: str) -> types.Part:
    return types.Part.from_text(text=text)


class FakeClient:
    """Stands in for google.genai.Client: `script(kwargs, call_no)` returns a GenerateContentResponse."""

    def __init__(self, script):
        self.calls = []
        outer = self

        class _Models:
            def generate_content(self, **kwargs):
                outer.calls.append(kwargs)
                return script(kwargs, len(outer.calls))

        self.models = _Models()


def _last_function_output(kwargs) -> dict:
    """The output the loop sent back for the most recent function call."""
    return kwargs["contents"][-1].parts[0].function_response.response["output"]


@pytest.fixture()
def with_fake_gemini(platform):
    original = platform.llm

    def install(script):
        platform.llm = LLM(platform.settings, client=FakeClient(script))
        return platform.llm._client

    yield install
    platform.llm = original


def test_every_model_returns_a_signed_passport(platform):
    for name, spec in platform.specs.items():
        data = spec.generate(platform.network, 3, platform.rng(), platform.world.params())
        records = spec.frame_to_records(data)
        res = platform.runtime(name).predict(records[:1], source="api", explain=spec.supports_shapley)[0]
        assert res["status"] == "ok", name
        assert verify(res["passport"], platform.settings.passport_secret)
        if spec.supports_shapley:
            assert res["passport"]["top_factors"], name


def test_data_contract_blocks_bad_input(platform):
    spec = platform.specs["fare_forecaster"]
    rec = spec.frame_to_records(spec.generate(platform.network, 1, platform.rng(), platform.world.params()))[0]
    rec["quoted_fare"] = -50
    res = platform.runtime("fare_forecaster").predict([rec], source="api")[0]
    assert res["status"] == "blocked"
    assert any(v["feed"] == "fare_feed" for v in res["output"]["reasons"])
    assert res["passport"]["data_quality"]["contract"] == "fail"


def test_confirmation_probability_falls_with_waitlist_position(platform):
    net = platform.network
    seg = net.direct_segments(["PUNE"], ["NZM"])[0]
    snap = inv.snapshot(net, seg, "SL", today_ist() + timedelta(days=20), today_ist(), platform.world.params())
    near = confirmation_record({**snap, "status": "WL", "position": 3}, "West")
    far = confirmation_record({**snap, "status": "WL", "position": 400}, "West")
    res = platform.runtime("confirmation_predictor").predict([near, far], source="api")
    assert res[0]["output"]["p_confirm"] > res[1]["output"]["p_confirm"]


def test_planner_offline_proposes_audited_itinerary_and_books_after_approval(platform):
    plan = platform.planner.plan("I want to travel from Pune to Delhi tomorrow under ₹1500 and reach before 10:00 AM.")
    assert plan["status"] == "proposed"
    assert plan["verdict"]["valid"], plan["verdict"]
    assert plan["recommended"]["legs"][0]["origin"] == "PUNE"
    tools = {s["tool"] for s in plan["steps"]}
    assert {"timetable_search", "predict_confirmation", "predict_delay", "rank_options"} <= tools
    assert plan["booking_intent_id"]
    booked = platform.planner.approve(plan["booking_intent_id"], "Test Passenger")
    assert booked["bookings"] and len(booked["bookings"][0]["pnr"]) == 10
    with pytest.raises(ValueError):
        platform.planner.approve(plan["booking_intent_id"], "Test Passenger")  # cannot book twice


def test_planner_asks_when_origin_missing(platform):
    plan = platform.planner.plan("I want to go to Goa")
    assert plan["status"] == "needs_clarification" and plan["booking_intent_id"] is None


def test_planner_with_gemini_function_calling(platform, with_fake_gemini):
    tomorrow = (today_ist() + timedelta(days=1)).isoformat()

    def script(kw, n):
        if n == 1:
            return _response(_call("c1", "search_options",
                                   {"origin": "Pune", "destination": "Delhi", "date_str": tomorrow}))
        if n == 2:
            result = _last_function_output(kw)
            best = min(result["options"], key=lambda o: (-o["confirm_prob"], o["recommender_rank"]))
            return _response(_call("c2", "propose_itinerary",
                                   {"option_id": best["option_id"], "explanation": "Best ranked option."}))
        return _response(_text("I recommend the best-ranked option."))

    client = with_fake_gemini(script)
    plan = platform.planner.plan("Pune to Delhi tomorrow")
    assert plan["mode"] == "gemini" and plan["status"] == "proposed"
    assert plan["verdict"]["valid"], plan["verdict"]
    assert len(client.calls) == 3 and client.calls[0]["config"].tools
    # Every model turn is sent back unchanged, followed by the function result with the call's id.
    second = client.calls[1]["contents"]
    assert second[1].role == "model" and second[1].parts[0].function_call.name == "search_options"
    assert second[2].parts[0].function_response.id == "c1"
    assert client.calls[0]["config"].automatic_function_calling.disable


def test_auditor_catches_fabricated_option_from_llm(platform, with_fake_gemini):
    def script(kw, n):
        if n == 1:
            return _response(_call("c1", "propose_itinerary", {"option_id": "OPT999", "explanation": "Trust me."}))
        return _response(_text("Booked the best train."))

    with_fake_gemini(script)
    plan = platform.planner.plan("Pune to Delhi tomorrow")
    assert not plan["verdict"]["valid"]
    assert any("never returned" in v for v in plan["verdict"]["violations"])
    assert plan["booking_intent_id"] is None


def test_planner_falls_back_offline_when_gemini_fails(platform, with_fake_gemini):
    def script(kw, n):
        raise ConnectionError("quota exceeded")

    with_fake_gemini(script)
    plan = platform.planner.plan("I want to travel from Pune to Delhi tomorrow under ₹1500 and reach before 10:00 AM.")
    assert plan["mode"] == "offline" and plan["status"] == "proposed"
    assert any(s["tool"] == "gemini_call" and not s["ok"] for s in plan["steps"])


def test_planner_never_counts_a_two_day_journey_as_on_time(platform):
    from datetime import datetime

    plan = platform.planner.plan("I want to travel from Pune to Delhi tomorrow under ₹1500 and reach before 10:00 AM.",
                                 create_intent=False)
    shown = [plan["recommended"]] + plan["alternatives"]
    for o in [o for o in shown if o and o.get("on_time_prob") is not None]:
        days = (datetime.fromisoformat(o["arrival"]).date() - datetime.fromisoformat(o["departure"]).date()).days
        if days >= 2:
            assert o["on_time_prob"] < 0.01, o["summary"]
    assert plan["verdict"]["valid"], plan["verdict"]


def test_one_change_routes_never_double_back(platform):
    from railsentinel.agents.tools import MAX_DETOUR
    from railsentinel.twin.network import haversine_km

    net = platform.network
    its = platform.planner.tools.search(["SBC"], ["MAS"], today_ist() + timedelta(days=5))
    assert its
    for it in its:
        a, b = net.stations[it["legs"][0]["origin"]], net.stations[it["legs"][-1]["destination"]]
        assert sum(leg["distance_km"] for leg in it["legs"]) <= MAX_DETOUR * haversine_km(a.lat, a.lon, b.lat, b.lon)


def test_fully_waitlisted_request_gets_alternatives_not_a_dead_end(platform, monkeypatch):
    # Every option in the requested class counts as unbookable, as on a peak festival date.
    monkeypatch.setattr(platform.planner, "_select", lambda options, constraints, min_confirm: ([], []))
    plan = platform.planner.plan("Delhi to Lucknow in 3A on " + (today_ist() + timedelta(days=4)).isoformat())
    assert plan["status"] == "no_options" and plan["booking_intent_id"] is None
    assert "No realistic seat in 3A" in plan["message"] and plan["alternatives"]
    assert any(a.get("note") == "Best of the waitlisted options" for a in plan["alternatives"])
    assert plan["verdict"]["valid"], plan["verdict"]


def test_support_offline_answers_are_grounded(platform):
    pnr = next(b["pnr"] for b in platform.bookings.list(100) if b["current_status"] != "CANCELLED")
    res = platform.support.answer(f"What is the status of PNR {pnr}?")
    assert pnr in res["answer"] and not res["grounding"]["hallucination"]
    res = platform.support.answer("What are the cancellation charges for a confirmed 3A ticket with fare Rs 2100 "
                                  "cancelled 20 hours before departure?")
    assert "525" in res["answer"] and not res["grounding"]["hallucination"]
    assert any("indianrail.gov.in" in (s.get("url") or "") for s in res["sources"])


def test_support_with_gemini_flags_invented_numbers(platform, with_fake_gemini):
    pnr = next(b["pnr"] for b in platform.bookings.list(100) if b["current_status"] != "CANCELLED")

    def script(kw, n):
        if not kw["config"].tools:  # optional LLM-judge call
            return _response(_text('{"faithful": false, "score": 1, "unsupported_claims": []}'))
        if n == 1:
            return _response(_call("c1", "lookup_pnr", {"pnr": pnr}))
        return _response(_text(f"PNR {pnr} is confirmed and you will get a refund of Rs 98765."))

    with_fake_gemini(script)
    res = platform.support.answer(f"Status of {pnr}?")
    assert res["mode"] == "gemini"
    assert res["grounding"]["hallucination"] and "98765" in res["grounding"]["unsupported_numbers"]


@pytest.mark.parametrize("name, world_kwargs", [
    ("confirmation_predictor", {"cancel_mult": 0.45}),
    ("delay_predictor", {"weather_shift": 0.4, "weather_effect_mult": 2.2}),
])
def test_recalibration_adapts_production_model_to_a_changed_world(platform, name, world_kwargs):
    import numpy as np

    from railsentinel.twin.world import WorldParams

    spec = platform.specs[name]
    prod = platform.registry.load(name, 1)
    world = WorldParams(**world_kwargs)
    recent = spec.frame_to_records(spec.generate(platform.network, 450, np.random.default_rng(1), world))
    test = spec.frame_to_records(spec.generate(platform.network, 2000, np.random.default_rng(2), world))
    labels = [spec.label_of(r) for r in test]
    adapted = spec.recalibrate(prod, recent)
    before = spec.sample_losses(spec.predict(prod, spec.prepare(test), {}), labels).mean()
    after = spec.sample_losses(spec.predict(adapted, spec.prepare(test), {}), labels).mean()
    assert after < before
    assert "recalibration" not in prod and len(adapted["recalibration"]) == 1  # production artifact untouched
