"""AI customer-support assistant (English).

Answers PNR, refund, cancellation, Tatkal, chart and journey questions. Every
answer is built from evidence (tool results and retrieved policy passages),
checked for hallucinations, logged for audit with PII masked, and carries the
prompt version that produced it.

Two engines share the same tools and the same checks:
* Gemini (when GEMINI_API_KEY is set): a function-calling loop over the support tools.
* Offline: intent routing plus answers composed from tool results and extracted policy text.
"""
from __future__ import annotations

import json
import random
import re
import time
import uuid
from datetime import datetime

from railsentinel.agents.llm import run_tool_loop
from railsentinel.agents.refunds import cancellation_refund
from railsentinel.config import IST
from railsentinel.db import dumps
from railsentinel.governance.hallucination import NUMBER_RE, check_grounding, llm_judge
from railsentinel.security import mask_pii, stable_hash
from railsentinel.twin.network import TRAVEL_CLASSES

NAME = "support_assistant"

PROMPT_V1 = {
    "mode": "grounded",
    "system_prompt": (
        "You are RailSentinel's railway customer-support assistant. Answer in clear, friendly English.\n"
        "Rules:\n"
        "1. Use the tools for every fact: PNR details, refunds, cancellation charges, policies and journey options.\n"
        "2. Only state facts, amounts, times and percentages that appear in tool results. Never estimate or invent them.\n"
        "3. If the tools do not cover the question, say you do not have that information and suggest checking IRCTC.\n"
        "4. RailSentinel PNRs are demo records in RailSentinel's own system, not IRCTC records.\n"
        "5. Mention the policy source when you state a rule. Keep answers under 120 words."),
    "fault_injection": False,
}
PROMPT_V2_FAULTY = {
    "mode": "ungrounded",
    "system_prompt": (
        "You are a confident railway expert. Answer quickly from your own knowledge. Avoid tool calls unless "
        "unavoidable and never say you are unsure."),
    # Used only by the prompt_regression chaos scenario: the offline engine rewrites numbers in its answers
    # to reproduce what an ungrounded prompt does, so the hallucination monitor has something real to catch.
    "fault_injection": True,
}

INTENT_PATTERNS = [
    ("chart", r"\bchart"),
    ("tatkal", r"\btatkal\b"),
    ("waitlist_codes", r"\b(?:gnwl|rlwl|pqwl|tqwl|waiting list code|wl code|what does wl|meaning of (?:wl|rac))\b"),
    ("refund_status", r"\brefund\b.*\b(?:status|track|credited|received|when|where)\b|\b(?:status|track)\b.*\brefund"),
    ("cancellation", r"\b(?:cancel\w*|refund rules?|refund|cancellation charges?|how much refund)\b"),
    ("journey_planning", r"\b(?:from\s+\w+.*\bto\b|train to|trains to|plan (?:a|my) (?:trip|journey)|travel to)\b"),
    ("travel_recommendation", r"\b(?:recommend|best train|suggest|which train)\b"),
    ("booking_help", r"\b(?:book|booking|reserve|approve)\b"),
    ("greeting", r"^(?:hi|hello|hey|namaste|good (?:morning|evening|afternoon))\b"),
]


def classify(message: str) -> str:
    lower = message.lower()
    has_pnr = re.search(r"\b\d{10}\b", message) is not None
    if has_pnr:
        if re.search(r"\bcancel", lower):
            return "pnr_cancel_estimate"
        if re.search(r"\brefund", lower):
            return "refund_status"
        return "pnr_status"
    if re.search(r"\bpnr\b", lower):
        return "pnr_status"
    for intent, pattern in INTENT_PATTERNS:
        if re.search(pattern, lower):
            return intent
    return "other"


def inject_faults(answer: str, seed: int) -> str:
    """Rewrite numbers the way an ungrounded prompt drifts (chaos scenario only)."""
    rng = random.Random(seed)

    def bump(m: re.Match) -> str:
        raw = m.group(0)
        digits = m.group(1).replace(",", "")
        if len(digits) == 10:  # keep PNRs intact
            return raw
        value = int(digits)
        new = value + max(1, int(round(value * rng.choice([0.25, 0.5, 1.0]))))
        return raw.replace(m.group(1), f"{new:,}" if value >= 1000 else str(new))

    return NUMBER_RE.sub(bump, answer) + " Refunds are always credited within 24 hours."


TOPIC_DOCS = {"tatkal": "tatkal_refunds", "chart": "chart_preparation", "waitlist_codes": "waitlist_codes"}

CAPABILITIES = ("I can check a PNR, track a refund, explain cancellation charges, Tatkal or chart rules, and plan "
                "a journey.")


def describe_hours(hours: float) -> str:
    """Time before departure in words: days once it is two days or more, hours below that (the refund
    rules change at 48, 12 and 4 hours, so hours are what matter close to departure)."""
    if hours < 1:
        return "less than an hour"
    if hours < 48:
        return f"about {hours:.0f} hours"
    return f"about {round(hours / 24)} days"


class SupportAssistant:
    def __init__(self, platform):
        self.p = platform

    # ------------------------------------------------------------ tools (shared by both engines)
    def tool_lookup_pnr(self, pnr: str) -> dict:
        b = self.p.bookings.get(str(pnr).strip())
        if not b:
            return {"found": False, "pnr": pnr}
        out = {"found": True, "pnr": b["pnr"], "train": f"{b['train_number']} {b['train_name']}",
               "from": b["origin"], "to": b["destination"], "journey_date": b["journey_date"],
               "departure": b["departure"].replace("T", " "), "arrival": b["arrival"].replace("T", " "),
               "class": b["travel_class"], "passengers": b["passengers"], "booking_status": b["booking_status"],
               "current_status": b["current_status"], "status_detail": b["status_detail"], "fare": b["fare"]}
        status = b["current_status"].split()[0]
        if status in ("WL", "RAC"):
            pred = self.p.pnr_confirmation(b, source="agent")
            if pred:
                out["confirmation_probability_pct"] = round(pred["output"]["p_confirm"] * 100)
                out["predicted_final_status"] = pred["output"]["predicted_final_status"]
                out["prediction_model"] = f"{pred['model']} v{pred['version']}"
                out["prediction_id"] = pred["prediction_id"]
        return out

    def tool_refund_status(self, pnr: str) -> dict:
        r = self.p.bookings.refund_status(str(pnr).strip())
        return r or {"pnr": pnr, "refund_found": False}

    def tool_estimate_cancellation(self, pnr: str | None = None, fare: float | None = None,
                                   travel_class: str | None = None, status: str = "CNF",
                                   hours_before_departure: float | None = None, quota: str = "GN") -> dict:
        if pnr:
            b = self.p.bookings.get(str(pnr).strip())
            if not b:
                return {"found": False, "pnr": pnr}
            if b["current_status"] == "CANCELLED":
                return {"pnr": pnr, "already_cancelled": True}
            now = datetime.now(IST).replace(tzinfo=None)
            hours = (datetime.fromisoformat(b["departure"]) - now).total_seconds() / 3600
            per_pax = b["fare"] / max(1, b["passengers"])
            calc = cancellation_refund(per_pax, b["travel_class"], b["current_status"].split()[0], hours,
                                       quota=b["quota"])
            return {"pnr": pnr, "hours_before_departure": round(hours, 1),
                    "time_before_departure": describe_hours(hours), "passengers": b["passengers"],
                    "fare_per_passenger": round(per_pax), "refund_per_passenger": calc["refund"],
                    "charge_per_passenger": calc["charges"], "total_refund": round(calc["refund"] * b["passengers"]),
                    "rule": calc["rule"], "source": calc["source"]}
        if fare is None or travel_class is None or hours_before_departure is None:
            return {"error": "need either a PNR or fare, travel_class and hours_before_departure"}
        calc = cancellation_refund(float(fare), travel_class, status, float(hours_before_departure), quota=quota)
        return {"fare_per_passenger": fare, "travel_class": travel_class, "status": status,
                "hours_before_departure": hours_before_departure, "refund_per_passenger": calc["refund"],
                "charge_per_passenger": calc["charges"], "rule": calc["rule"], "source": calc["source"]}

    def tool_search_policy(self, query: str) -> list[dict]:
        return [{"title": h["title"], "section": h["heading"], "text": h["text"], "url": h["url"]}
                for h in self.p.kb.search(query, k=3)]

    def tool_plan_journey(self, request: str) -> dict:
        plan = self.p.planner.plan(request, source="support", use_llm=False, create_intent=False)
        best = plan.get("recommended")
        return {"message": plan["message"], "recommended": None if not best else
                {k: best[k] for k in ("summary", "travel_class", "fare", "departure", "arrival", "availability",
                                      "confirm_prob")},
                "trace_id": plan["trace_id"]}

    # ------------------------------------------------------------ public entry point
    def answer(self, message: str, session_id: str | None = None, source: str = "api",
               use_llm: bool | None = None) -> dict:
        t0 = time.perf_counter()
        version, config = self.p.registry.load_production(NAME)
        intent = classify(message)
        llm_ok = self.p.llm.enabled and (use_llm if use_llm is not None else
                                         (source == "api" or self.p.settings.llm_for_synthetic_traffic))
        mode = "offline"
        evidence: list[str] = []
        sources: list[dict] = []
        if llm_ok:
            try:
                text, evidence, sources = self._answer_with_llm(message, config)
                mode = "gemini"
            except Exception as exc:  # API failure falls back to the offline engine
                self.p.llm.last_error = f"{type(exc).__name__}: {exc}"
                text, evidence, sources = self._answer_offline(message, intent)
        else:
            text, evidence, sources = self._answer_offline(message, intent)
        if config.get("fault_injection") and mode == "offline":
            text = inject_faults(text, seed=stable_hash(message))
        grounding = check_grounding(text, evidence, message)
        strict = self.p.world.ops().support_strict_mode
        if strict and grounding["hallucination"]:
            text = self._extractive_fallback(message, evidence)
            grounding = check_grounding(text, evidence, message)
            grounding["strict_mode_rewrite"] = True
        judge = None
        if mode == "gemini" and random.random() < self.p.settings.llm_judge_sample_rate:
            judge = llm_judge(self.p.llm, message, text, evidence)
            if judge and not judge["faithful"]:
                grounding["hallucination"] = True
            grounding["llm_judge"] = judge
        quality = round(0.6 * grounding["grounding_score"] + 0.4 * min(1.0, grounding["relevance"] * 2), 3)
        latency = (time.perf_counter() - t0) * 1000
        iid = uuid.uuid4().hex
        self.p.db.execute(
            "INSERT INTO llm_interactions(id, service, version, ts, session_id, query, intent, response, sources, "
            "context, grounding, hallucination, quality, feedback, latency_ms, mode, source, scenario) "
            "VALUES(?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)",
            (iid, NAME, version, time.time(), session_id, mask_pii(message), intent, mask_pii(text), dumps(sources),
             dumps([mask_pii(e) for e in evidence]), dumps(grounding), int(grounding["hallucination"]), quality, None,
             round(latency, 1), mode, source, self.p.world.scenario))
        return {"interaction_id": iid, "answer": text, "intent": intent, "sources": sources, "grounding": grounding,
                "quality": quality, "mode": mode, "prompt_version": version, "latency_ms": round(latency, 1)}

    def feedback(self, interaction_id: str, helpful: bool) -> None:
        self.p.db.execute("UPDATE llm_interactions SET feedback=? WHERE id=?", (1 if helpful else -1, interaction_id))

    # ------------------------------------------------------------ Gemini engine
    def _answer_with_llm(self, message: str, config: dict) -> tuple[str, list[str], list[dict]]:
        evidence: list[str] = []
        sources: list[dict] = []

        def wrap(fn):
            def handler(args):
                result = fn(**args)
                evidence.append(json.dumps(result, default=str))
                if isinstance(result, list):
                    for r in result:
                        evidence.append(r.get("text", ""))
                        sources.append({"title": r.get("title"), "url": r.get("url")})
                elif isinstance(result, dict) and result.get("source"):
                    sources.append({"title": "Indian Railways refund rules", "url": result["source"]})
                return result
            return handler

        tools = [
            {"name": "lookup_pnr", "description": "Look up a RailSentinel demo PNR (10 digits): train, journey, "
             "class, current status and, for RAC/WL, the model's confirmation probability.",
             "parameters": {"type": "object", "properties": {"pnr": {"type": "string"}}, "required": ["pnr"]}},
            {"name": "refund_status", "description": "Refund status of a cancelled RailSentinel PNR.",
             "parameters": {"type": "object", "properties": {"pnr": {"type": "string"}}, "required": ["pnr"]}},
            {"name": "estimate_cancellation_refund", "description": "Compute cancellation charges and refund with the "
             "published Indian Railways rules, either for a PNR or for a fare, class, status and hours before departure.",
             "parameters": {"type": "object", "properties": {
                 "pnr": {"type": "string"}, "fare": {"type": "number"},
                 "travel_class": {"type": "string", "enum": list(TRAVEL_CLASSES)},
                 "status": {"type": "string", "enum": ["CNF", "RAC", "WL"]},
                 "hours_before_departure": {"type": "number"}, "quota": {"type": "string", "enum": ["GN", "TQ"]}}}},
            {"name": "search_policy_kb", "description": "Search the policy knowledge base (refund rules, Tatkal, "
             "chart timing, waiting list codes, how RailSentinel works).",
             "parameters": {"type": "object", "properties": {"query": {"type": "string"}}, "required": ["query"]}},
            {"name": "plan_journey", "description": "Plan a journey from a natural-language request and return the "
             "recommended option.",
             "parameters": {"type": "object", "properties": {"request": {"type": "string"}},
                              "required": ["request"]}},
        ]
        handlers = {"lookup_pnr": wrap(lambda pnr: self.tool_lookup_pnr(pnr)),
                    "refund_status": wrap(lambda pnr: self.tool_refund_status(pnr)),
                    "estimate_cancellation_refund": wrap(lambda **kw: self.tool_estimate_cancellation(**kw)),
                    "search_policy_kb": wrap(lambda query: self.tool_search_policy(query)),
                    "plan_journey": wrap(lambda request: self.tool_plan_journey(request))}
        today = datetime.now(IST).date().isoformat()
        system = f"{config['system_prompt']}\nToday's date: {today}."
        result = run_tool_loop(self.p.llm, system, message, tools, handlers, max_steps=6)
        return result["text"] or "I could not produce an answer. Please try again.", evidence, _dedupe(sources)

    # ------------------------------------------------------------ offline engine
    def _answer_offline(self, message: str, intent: str) -> tuple[str, list[str], list[dict]]:
        pnr_match = re.search(r"\b\d{10}\b", message)
        pnr = pnr_match.group(0) if pnr_match else None
        if intent == "pnr_status":
            if not pnr:
                return ("Please share your 10-digit PNR and I will look up its status.", [], [])
            info = self.tool_lookup_pnr(pnr)
            if not info["found"]:
                return (f"I could not find PNR {pnr} in RailSentinel's demo reservation system.", [dumps(info)], [])
            fact = (f"PNR {pnr}: train {info['train']} from {info['from']} to {info['to']}, departing "
                    f"{info['departure']}, class {info['class']}, {info['passengers']} passenger(s). "
                    f"Current status: {info['current_status']}.")
            parts = [fact]
            if "confirmation_probability_pct" in info:
                parts.append(f"The confirmation model ({info['prediction_model']}) estimates a "
                             f"{info['confirmation_probability_pct']}% chance that this ticket is confirmed by chart "
                             f"preparation; its most likely final status is {info['predicted_final_status']}.")
            kb = self.p.kb.search("PNR status RailSentinel demo records", k=1)
            return " ".join(parts), [fact, " ".join(parts), dumps(info)] + [h["text"] for h in kb], _src(kb)
        if intent == "refund_status":
            if not pnr:
                return ("Please share the 10-digit PNR of the cancelled ticket so I can check its refund.", [], [])
            r = self.tool_refund_status(pnr)
            if not r.get("status"):
                return (f"I could not find a refund request for PNR {pnr}. Refunds appear here after a RailSentinel "
                        "booking is cancelled.", [dumps(r)], [])
            text = (f"Refund for PNR {pnr}: Rs {r['amount']:,.0f} after charges of Rs {r['charges']:,.0f}. "
                    f"Status: {r['status']}. Rule applied: {r['rule']}.")
            return text, [text, dumps(r)], []
        if intent == "pnr_cancel_estimate":
            est = self.tool_estimate_cancellation(pnr=pnr)
            if est.get("found") is False:
                return (f"I could not find PNR {pnr} in RailSentinel's demo reservation system.", [dumps(est)], [])
            if est.get("already_cancelled"):
                return (f"PNR {pnr} is already cancelled. Ask me for its refund status.", [dumps(est)], [])
            text = (f"If you cancel PNR {pnr} now, {est['time_before_departure']} before departure, "
                    f"the charge is Rs {est['charge_per_passenger']:,.0f} per passenger and the total refund is "
                    f"Rs {est['total_refund']:,.0f} for {est['passengers']} passenger(s). Rule: {est['rule']}.")
            return text, [text, dumps(est)], [{"title": "Indian Railways refund rules", "url": est["source"]}]
        if intent == "cancellation":
            calc_text = self._maybe_calculate(message)
            hits = self.p.kb.search(message, k=2)
            policy = self._extract(message, hits, 2)
            text = " ".join(t for t in (calc_text, policy) if t) or "I could not find a matching cancellation rule."
            evidence = [h["text"] for h in hits] + ([calc_text] if calc_text else [])
            return text + " Please confirm current rules on IRCTC before cancelling.", evidence, _src(hits)
        if intent in ("chart", "tatkal", "waitlist_codes", "other", "booking_help"):
            hits = self.p.kb.search(message, k=3)
            topical = [h for h in hits if h["doc"] == TOPIC_DOCS.get(intent)]
            if topical:  # a topic question is answered from its own policy document first
                hits = topical + [h for h in self.p.kb.search(message, k=6) if h["doc"] == TOPIC_DOCS[intent]
                                  and h not in topical]
                hits = hits[:3]
            if intent == "other" and (not hits or hits[0]["score"] < 0.12):
                return (f"{CAPABILITIES} What would you like to know?", [CAPABILITIES], [])
            return self._extract(message, hits, 3) or "I do not have that information.", [h["text"] for h in hits], \
                _src(hits)
        if intent in ("journey_planning", "travel_recommendation"):
            plan = self.tool_plan_journey(message)
            return plan["message"], [plan["message"], dumps(plan)], []
        if intent == "greeting":
            return (f"Hello! {CAPABILITIES} How can I help?", [CAPABILITIES], [])
        hits = self.p.kb.search(message, k=3)
        return self._extract(message, hits, 2) or "I do not have that information.", [h["text"] for h in hits], \
            _src(hits)

    def _maybe_calculate(self, message: str) -> str | None:
        lower = message.lower()
        cls = None
        for code in TRAVEL_CLASSES:
            if re.search(rf"\b{code.lower()}\b", lower):
                cls = code
        if cls is None:
            for word, code in (("sleeper", "SL"), ("chair car", "CC"), ("first ac", "1A"), ("second ac", "2A"),
                               ("third ac", "3A")):
                if word in lower:
                    cls = code
        hours = re.search(r"(\d+(?:\.\d+)?)\s*(?:hours|hrs|hour|h)\b", lower)
        days = re.search(r"(\d+)\s*days?\b", lower)
        fare = re.search(r"(?:₹|rs\.?|inr|fare(?: of| is)?)\s*([\d,]+)", lower)
        status = "WL" if re.search(r"\b(?:wl|waitlist|waiting)", lower) else "RAC" if "rac" in lower else "CNF"
        quota = "TQ" if "tatkal" in lower else "GN"
        if not (cls and fare and (hours or days)):
            return None
        h = float(hours.group(1)) if hours else float(days.group(1)) * 24
        est = self.tool_estimate_cancellation(fare=float(fare.group(1).replace(",", "")), travel_class=cls,
                                              status=status, hours_before_departure=h, quota=quota)
        return (f"For a {status} {cls} ticket with fare Rs {est['fare_per_passenger']:,.0f} cancelled {h:g} hours "
                f"before departure, the charge is Rs {est['charge_per_passenger']:,.0f} and the refund is "
                f"Rs {est['refund_per_passenger']:,.0f} per passenger ({est['rule']}).")

    def _extract(self, question: str, hits: list[dict], max_sentences: int) -> str:
        """Pick the passage sentences most similar to the question, keeping their original wording."""
        from sklearn.feature_extraction.text import TfidfVectorizer
        from sklearn.metrics.pairwise import cosine_similarity

        sentences = []
        for h in hits:
            for s in re.split(r"(?<=[.!?])\s+", h["text"]):
                if len(s.split()) >= 5:
                    sentences.append(s.strip())
        if not sentences:
            return ""
        vec = TfidfVectorizer(stop_words="english", ngram_range=(1, 2)).fit(sentences + [question])
        sims = cosine_similarity(vec.transform([question]), vec.transform(sentences))[0]
        top = sorted(range(len(sentences)), key=lambda i: -sims[i])[:max_sentences]
        return " ".join(sentences[i] for i in top)

    def _extractive_fallback(self, question: str, evidence: list[str]) -> str:
        hits = [{"text": e} for e in evidence if e and not e.startswith("{") and not e.startswith("[")]
        text = self._extract(question, hits, 2)
        return text or "I do not have verified information for that. Please check IRCTC."


def _src(hits: list[dict]) -> list[dict]:
    return _dedupe([{"title": h["title"], "url": h["url"]} for h in hits])


def _dedupe(items: list[dict]) -> list[dict]:
    seen, out = set(), []
    for i in items:
        key = (i.get("title"), i.get("url"))
        if key not in seen:
            seen.add(key)
            out.append(i)
    return out
