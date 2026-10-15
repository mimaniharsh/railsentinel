"""AI travel planner agent.

Understands a natural-language request, searches direct and one-change
itineraries, coordinates the specialist agents (availability/confirmation,
punctuality/delay, fare, ranking), checks every option against the passenger's
constraints and proposes the best one with alternatives. It never books: it
creates a pending booking intent that only the passenger can approve.

Runs with Gemini (function calling) when an API key is configured, otherwise with a
deterministic planner that uses the same tools. Both are audited the same way.
"""
from __future__ import annotations

import math
import time
import uuid
from datetime import date, datetime, timedelta

from railsentinel.agents.llm import run_tool_loop
from railsentinel.agents.nlu import parse_request
from railsentinel.agents.tools import JourneyTools, ToolError, Trace
from railsentinel.db import dumps, loads
from railsentinel.twin.generators import today_ist
from railsentinel.twin.network import CATEGORIES, TRAVEL_CLASSES, Segment
from railsentinel.twin.seasons import season_of

NAME = "travel_planner_agent"

AGENT_V1 = {
    "max_steps": 12,
    "max_tool_retries": 3,
    "min_confirm_prob": 0.4,   # options below this are proposed only if nothing better exists, with a warning
    "system_prompt": (
        "You are RailSentinel's travel planner agent for Indian rail journeys. Use the tools to search and compare "
        "real options from the timetable; never invent trains, fares, times or probabilities. Respect every "
        "constraint the passenger gives (date, budget per passenger, arrival or departure time, class). If no option "
        "meets all constraints, propose the closest one and say exactly which constraint it misses. You cannot book: "
        "finish by calling propose_itinerary (the passenger approves booking separately) or ask_clarification if "
        "origin or destination is unclear. Keep the final message under 150 words."),
}

PREFERENCE_SORT = {
    "cheapest": lambda o: (o["fare"], o["duration_min"]),
    "fastest": lambda o: (o["duration_min"], o["fare"]),
    "reliable": lambda o: (-(o.get("on_time_prob") or o["punctuality"]) * o["confirm_prob"], o["fare"]),
    "comfort": lambda o: (-o["comfort"], o["fare"]),
}


def _fmt_dt(iso: str) -> str:
    return datetime.fromisoformat(iso).strftime("%a %d %b %H:%M")


class TravelPlanner:
    def __init__(self, platform):
        self.p = platform
        self.tools = JourneyTools(platform)

    # ------------------------------------------------------------ public API
    def plan(self, query: str, session_id: str | None = None, source: str = "api", use_llm: bool | None = None,
             create_intent: bool = True, profile: dict | None = None) -> dict:
        t0 = time.perf_counter()
        version, config = self.p.registry.load_production(NAME)
        today = today_ist()
        trace = Trace()
        tp = time.perf_counter()
        constraints = parse_request(query, self.p.network, today)
        trace.add("planner", "parse_request", {"query": query[:200]}, True, (time.perf_counter() - tp) * 1000,
                  self._constraint_summary(constraints))
        llm_ok = self.p.llm.enabled and (use_llm if use_llm is not None else
                                         (source == "api" or self.p.settings.llm_for_synthetic_traffic))
        mode = "offline"
        result = None
        if llm_ok:
            try:
                result = self._plan_with_llm(query, constraints, config, trace, today, source, profile)
                mode = "gemini"
            except Exception as exc:  # API failure falls back to the deterministic planner
                self.p.llm.last_error = f"{type(exc).__name__}: {exc}"
                trace.add("planner", "gemini_call", {}, False, 0, f"Gemini unavailable, using offline planner: {exc}")
        if result is None:
            result = self._plan_offline(constraints, config, trace, today, source, profile)
        verdict = self.p.agent_auditor.validate(constraints, result, trace.steps, config,
                                                result.get("returned_option_ids"))
        trace_id = uuid.uuid4().hex
        latency = (time.perf_counter() - t0) * 1000
        plan_record = {k: v for k, v in result.items() if k != "returned_option_ids"}
        self.p.db.execute(
            "INSERT INTO agent_traces(id, ts, session_id, version, query, constraints, steps, plan, verdict, mode, "
            "latency_ms, source, scenario) VALUES(?,?,?,?,?,?,?,?,?,?,?,?,?)",
            (trace_id, time.time(), session_id, version, query, dumps(constraints), dumps(trace.steps),
             dumps(plan_record), dumps(verdict), mode, round(latency, 1), source, self.p.world.scenario))
        intent_id = None
        if create_intent and result.get("recommended") and verdict["valid"]:
            intent_id = uuid.uuid4().hex[:12]
            self.p.db.execute("INSERT INTO booking_intents(id, ts, trace_id, itinerary, status) VALUES(?,?,?,?,?)",
                              (intent_id, time.time(), trace_id,
                               dumps({"option": result["recommended"], "passengers": constraints["passengers"]}),
                               "pending"))
        return {"trace_id": trace_id, "mode": mode, "agent_version": version, "status": result["status"],
                "message": result["message"], "constraints": constraints, "recommended": result.get("recommended"),
                "alternatives": result.get("alternatives", []), "relaxed": result.get("relaxed", []),
                "fare_outlook": result.get("fare_outlook"), "date_flex": result.get("date_flex"),
                "steps": trace.steps, "verdict": verdict,
                "booking_intent_id": intent_id, "latency_ms": round(latency, 1)}

    def approve(self, intent_id: str, passenger_name: str, actor: str = "passenger") -> dict:
        intent = self.p.db.query_one("SELECT * FROM booking_intents WHERE id=?", (intent_id,))
        if not intent:
            raise KeyError("booking intent not found")
        if intent["status"] != "pending":
            raise ValueError(f"booking intent is {intent['status']}")
        data = loads(intent["itinerary"])
        option, passengers = data["option"], int(data.get("passengers", 1))
        pnrs = []
        rng = self.p.rng()
        for leg in option["legs"]:
            tr = self.p.network.trains[leg["train_number"]]
            seg = Segment(tr, tr.index_of(leg["origin"]), tr.index_of(leg["destination"]))
            boarding = datetime.fromisoformat(leg["departure"]).date()
            row = self.p.bookings.create(seg, leg["travel_class"], boarding, today_ist(),
                                         self.p.world.effective_params(), passenger_name, passengers, rng)
            pnrs.append({"pnr": row["pnr"], "train": f"{row['train_number']} {row['train_name']}",
                         "status": row["current_status"], "detail": row["status_detail"], "fare": row["fare"]})
        self.p.db.execute("UPDATE booking_intents SET status='booked', pnr=?, decided_ts=? WHERE id=?",
                          (",".join(p["pnr"] for p in pnrs), time.time(), intent_id))
        self.p.audit.append(actor, "booking_approved", NAME, {"intent_id": intent_id, "pnrs": [p["pnr"] for p in pnrs]})
        return {"intent_id": intent_id, "bookings": pnrs}

    def reject(self, intent_id: str, actor: str = "passenger") -> None:
        self.p.db.execute("UPDATE booking_intents SET status='rejected', decided_ts=? WHERE id=? AND status='pending'",
                          (time.time(), intent_id))

    # ------------------------------------------------------------ shared steps
    @staticmethod
    def _constraint_summary(c: dict) -> str:
        bits = [f"{c.get('origin_text') or '?'} → {c.get('destination_text') or '?'}", f"on {c['date']}"]
        if c.get("budget"):
            bits.append(f"budget Rs {c['budget']:.0f}")
        if c.get("arrive_by"):
            bits.append(f"arrive by {c['arrive_by']}")
        if c.get("depart_after"):
            bits.append(f"depart after {c['depart_after']}")
        if c.get("classes"):
            bits.append("class " + "/".join(c["classes"]))
        if c.get("passengers", 1) > 1:
            bits.append(f"{c['passengers']} passengers")
        return ", ".join(bits)

    def _search_with_retries(self, constraints, config, trace, jdate) -> list[dict] | None:
        for _ in range(int(config.get("max_tool_retries", 3)) + 1):
            try:
                return self.tools.search(constraints["origin_codes"], constraints["destination_codes"], jdate, trace)
            except ToolError:
                continue
        return None

    def _assess_and_rank(self, constraints, trace, today, source, profile, itineraries) -> list[dict]:
        # Specialist-model calls made on the agent's behalf are logged as "agent" traffic, so they are
        # monitored for health but kept out of the passenger distributions used for drift and accuracy.
        options = self.tools.assess(itineraries, today, classes=constraints["classes"] or None, trace=trace)
        if not options:
            return []
        self._apply_party_size(options, constraints)
        t0 = time.perf_counter()
        budget = constraints.get("budget") or float(sorted(o["fare"] for o in options)[len(options) // 2]) * 1.3
        user = profile or {"hist_class_mode": (constraints["classes"] or ["SL"])[0], "hist_night_share": 0.6,
                           "hist_fare_per_km": TRAVEL_CLASSES[(constraints["classes"] or ["SL"])[0]]["per_km"],
                           "segment": "unknown"}
        ranked = self.p.runtime("train_recommender").predict(
            [{"user": user, "budget": budget, "options": options, "origin": constraints["origin_codes"][0],
              "destination": constraints["destination_codes"][0], "journey_date": constraints["date"]}],
            source="agent")[0]
        if ranked["status"] == "ok":
            out = ranked["output"]
            scores = {item["index"]: item for item in out["top"]}
            for rank, idx in enumerate(out["ranking"]):
                options[idx]["rec_rank"] = rank + 1
                if idx in scores:
                    options[idx]["rationale"] = scores[idx]["rationale"]
                    options[idx]["rec_score"] = scores[idx]["score"]
            rec_pid = ranked["prediction_id"]
        else:
            rec_pid = None
        for o in options:
            o.setdefault("rec_rank", len(options))
            o["on_time_prob"] = self.tools.on_time_probability(o, constraints.get("arrive_by"))
            o["recommender_prediction_id"] = rec_pid
        trace.add("ranking-agent", "rank_options", {"options": len(options), "budget": round(budget)}, True,
                  (time.perf_counter() - t0) * 1000, "recommender ranked options by learned passenger utility")
        return options

    @staticmethod
    def _apply_party_size(options: list[dict], constraints: dict) -> None:
        """With fewer free berths than travellers, the rest of the party would be waitlisted, so the
        chance that everyone travels confirmed is capped by the share of the party that fits."""
        party = int(constraints.get("passengers") or 1)
        if party <= 1:
            return
        for o in options:
            for leg in o["legs"]:
                if leg["availability"].startswith("AVAILABLE-"):
                    free = int(leg["availability"].split("-")[1])
                    if free < party:
                        leg["confirm_prob"] = min(leg["confirm_prob"], free / party)
                        leg["confirm_note"] = f"only {free} berth(s) free for {party} passengers"
            o["confirm_prob"] = round(float(math.prod(leg["confirm_prob"] for leg in o["legs"])), 4)

    @staticmethod
    def _meets(o: dict, c: dict, skip: set[str] = frozenset(), min_confirm: float = 0.4) -> dict:
        checks = {}
        if c.get("budget") is not None and "budget" not in skip:
            checks["budget"] = o["fare"] <= c["budget"]
        if c.get("arrive_by") and "arrive_by" not in skip:
            checks["arrive_by"] = o["arrival"][11:16] <= c["arrive_by"]
        if c.get("depart_after"):
            checks["depart_after"] = o["departure"][11:16] >= c["depart_after"]
        if c.get("classes") and "classes" not in skip:
            checks["classes"] = o["travel_class"] in c["classes"]
        checks["bookable"] = o["confirm_prob"] >= (0.05 if "confirmation" in skip else min_confirm)
        return checks

    def _select(self, options, constraints, min_confirm: float) -> tuple[list[dict], list[str]]:
        """Hard constraints first; relax them one at a time (and say so) only when nothing fits."""
        relax_order = [[], ["arrive_by"], ["budget"], ["arrive_by", "budget"], ["confirmation"],
                       ["arrive_by", "confirmation"], ["budget", "confirmation"],
                       ["arrive_by", "budget", "confirmation"]]
        requested = set(constraints.get("classes") or [])

        def comfort_penalty(o):
            # A seating class for an overnight journey is a poor default unless the passenger asked for it.
            return 6 if o["overnight"] and o["travel_class"] in ("2S", "CC") and o["travel_class"] not in requested else 0

        for skip in relax_order:
            feasible = [o for o in options if all(self._meets(o, constraints, set(skip), min_confirm).values())]
            if feasible:
                pref = constraints.get("preference")
                if pref in PREFERENCE_SORT:
                    feasible.sort(key=PREFERENCE_SORT[pref])
                else:
                    feasible.sort(key=lambda o: (o["arrival"][:10], o["rec_rank"] - 8 * (o.get("on_time_prob") or 0)
                                                 - 6 * o["confirm_prob"] + 10 * o["connection_risk"]
                                                 + comfort_penalty(o)))
                return feasible, list(skip)
        return [], []

    @staticmethod
    def _alternatives(best: dict, ranked: list[dict], n: int = 2, already: list[dict] | None = None) -> list[dict]:
        alts = list(already or [])
        seen = {(best["summary"], best["travel_class"])} | {(a["summary"], a["travel_class"]) for a in alts}
        for o in ranked:
            if len(alts) >= n:
                break
            key = (o["summary"], o["travel_class"])
            if key in seen or o["summary"] == best["summary"]:
                continue
            seen.add(key)
            alts.append(o)
        return alts[:n]

    @staticmethod
    def _shortfall(o: dict, c: dict) -> dict:
        """How far an option misses the budget (Rs) and the arrival time (minutes); 0 when it meets them.
        Arrival is compared with the clock time on the day the train arrives, like `_meets`."""
        out = {}
        if c.get("budget") is not None:
            out["budget"] = max(0.0, float(o["fare"]) - float(c["budget"]))
        if c.get("arrive_by"):
            arr = datetime.fromisoformat(o["arrival"])
            hh, mm = (int(x) for x in c["arrive_by"].split(":"))
            out["arrive_by"] = max(0.0, (arr - arr.replace(hour=hh, minute=mm)).total_seconds() / 60)
        return out

    def _trade_offs(self, o: dict, c: dict, min_confirm: float) -> list[str]:
        sf = self._shortfall(o, c)
        parts = []
        if sf.get("budget"):
            parts.append(f"costs Rs {sf['budget']:,.0f} more than your Rs {c['budget']:,.0f} budget")
        if sf.get("arrive_by"):
            h, m = divmod(int(round(sf["arrive_by"])), 60)
            parts.append(f"arrives {f'{h} h {m:02d} min' if h else f'{m} min'} after {c['arrive_by']}")
        if o["confirm_prob"] < min_confirm:
            parts.append(f"has only a {o['confirm_prob'] * 100:.0f}% chance of a confirmed berth")
        return parts

    def _situation(self, options: list[dict], c: dict, min_confirm: float) -> str:
        """Why no option meets every constraint: how full the date is and how the literal matches look."""
        jdate = date.fromisoformat(c["date"])
        waitlisted = sum(1 for o in options if o["confirm_prob"] < min_confirm)
        season = season_of(jdate)
        note = {"festival": " (festival-season demand)", "summer_vacation": " (summer-holiday demand)",
                "winter_holiday": " (winter-holiday demand)"}.get(season, "")
        text = f"{waitlisted} of {len(options)} train and class options on {jdate:%a %d %b} are waitlisted{note}."
        literal = [o for o in options if all(v for k, v in self._meets(o, c, frozenset({"confirmation"})).items()
                                             if k != "bookable")]
        if literal:
            top = self._select(literal, c, 0.0)[0][0]  # same ordering as the "closest match" alternative
            text += (f" {len(literal)} option(s) fit your budget and timing; the best of them is {top['summary']} in "
                     f"{top['travel_class']} at {top['availability']} ({top['confirm_prob'] * 100:.0f}% chance of "
                     "confirmation).")
        else:
            text += " No train fits your budget and timing on this date."
        return text

    def _date_flex(self, constraints: dict, trace: Trace, jdate: date, today: date, min_confirm: float,
                   days: int = 3) -> dict | None:
        """Look up to `days` later dates for an option that meets every constraint."""
        t0 = time.perf_counter()
        checked: list[str] = []
        found = None
        for k in range(1, days + 1):
            day = jdate + timedelta(days=k)
            local = {**constraints, "date": day.isoformat()}
            try:
                its = self.tools.search(constraints["origin_codes"], constraints["destination_codes"], day)
            except ToolError:
                continue
            checked.append(day.isoformat())
            opts = self.tools.assess(its, today, classes=constraints["classes"] or None, source="agent")
            self._apply_party_size(opts, constraints)
            for o in opts:
                o["on_time_prob"] = self.tools.on_time_probability(o, constraints.get("arrive_by"))
            fits = [o for o in opts if all(self._meets(o, local, frozenset(), min_confirm).values())]
            if fits:
                found = max(fits, key=lambda o: (o["confirm_prob"], -o["fare"]))
                found["option_id"] = f"FLEX-{day.isoformat()}-{found['option_id']}"
                found["note"] = f"Meets every constraint if you travel on {day:%a %d %b}"
                break
        summary = (f"{found['summary']} on {found['departure'][:10]} meets every constraint" if found
                   else f"no option meets every constraint on {', '.join(checked) or 'the next days'}")
        trace.add("date-flex-agent", "search_nearby_dates", {"dates": checked}, True,
                  (time.perf_counter() - t0) * 1000, summary)
        return {"option": found, "checked": checked}

    def _describe(self, o: dict, c: dict) -> str:
        net = self.p.network
        legs = o["legs"]
        first, last = legs[0], legs[-1]
        if o["transfers"]:
            h, m = divmod(int(o["layover_min"]), 60)
            route = (f"{first['train_number']} {first['train_name']} to {net.stations[o['hub']].name}, then "
                     f"{last['train_number']} {last['train_name']} (change trains, {h} h {m:02d} min layover, "
                     f"missed-connection risk {o['connection_risk'] * 100:.0f}%)")
        else:
            route = f"{first['train_number']} {first['train_name']} ({CATEGORIES[first['category']]['label']})"
        text = (f"{route} in {o['travel_class']}: departs {net.stations[first['origin']].name} {_fmt_dt(o['departure'])}, "
                f"arrives {net.stations[last['destination']].name} {_fmt_dt(o['arrival'])}. Rs {o['fare']:,} per "
                f"passenger · {o['availability']} · confirmation chance {o['confirm_prob'] * 100:.0f}% · expected "
                f"arrival delay about {o['expected_delay_min']:.0f} min.")
        if o.get("on_time_prob") is not None:
            text += f" Chance of arriving before {c['arrive_by']}: {o['on_time_prob'] * 100:.0f}%."
        return text

    def _compose(self, best, alts, relaxed, c, outlook, min_confirm: float, situation: str | None = None,
                 flex: dict | None = None) -> str:
        net = self.p.network
        o_name = c.get("origin_text") or net.stations[best["legs"][0]["origin"]].city
        d_name = c.get("destination_text") or net.stations[best["legs"][-1]["destination"]].city
        jdate = date.fromisoformat(c["date"])
        lines = [f"**Recommended for {o_name.title()} → {d_name.title()} on {jdate:%a %d %b}:** {self._describe(best, c)}"]
        if relaxed:
            gaps = self._trade_offs(best, c, min_confirm)
            lines.append("**Trade-off:** no option meets every constraint" +
                         (f"; this one {' and '.join(gaps)}." if gaps else "."))
            if situation:
                lines.append(f"**Why:** {situation}")
            if flex is not None and flex["option"] is None and flex["checked"]:
                days = [date.fromisoformat(d) for d in flex["checked"]]
                lines.append(f"**Other dates:** I also checked {days[0]:%d %b} to {days[-1]:%d %b}; nothing meets every "
                             "constraint on those days either.")
        if best["confirm_prob"] < 0.5:
            lines.append("Seats are heavily waitlisted on this date. Consider another date, the Tatkal quota, or one "
                         "of the alternatives below.")
        if best.get("rationale"):
            lines.append("**Why this option:** " + "; ".join(r["factor"] for r in best["rationale"]) + ".")
        if outlook and outlook.get("fare_change_pct") is not None:
            if outlook["fare_change_pct"] > 2:
                lines.append(f"**Fare outlook:** likely to rise about {outlook['fare_change_pct']:.0f}% within "
                             f"{outlook['days_from_now']} days, so booking now saves money.")
            elif outlook.get("sellout_probability", 0) > 0.5 and best["confirm_prob"] >= 1.0:
                lines.append(f"**Sell-out risk:** {outlook['sellout_probability'] * 100:.0f}% chance this class fills "
                             f"up within {outlook['days_from_now']} days.")
        for i, a in enumerate(alts, 1):
            label = a.get("note") or "Also worth considering"
            gaps = self._trade_offs(a, {**c, "date": a["departure"][:10]}, min_confirm) if relaxed else []
            lines.append(f"**Alternative {i} ({label}):** {self._describe(a, c)}"
                         + (f" Trade-off: it {' and '.join(gaps)}." if gaps else ""))
        lines.extend(c.get("assumptions", []))
        if c.get("arrive_by"):
            lines.append(f"Arrival times are checked against {c['arrive_by']} on the day the train arrives.")
        lines.append("Approve to book the recommended itinerary (creates a demo PNR in RailSentinel).")
        return "\n".join(lines)

    # ------------------------------------------------------------ offline engine
    def _plan_offline(self, constraints, config, trace, today, source, profile) -> dict:
        if constraints["missing"]:
            q = ("Where are you travelling from?" if "origin" in constraints["missing"]
                 else "Where would you like to go?")
            return {"status": "needs_clarification", "message": q, "returned_option_ids": set()}
        jdate = date.fromisoformat(constraints["date"])
        if jdate < today:
            return {"status": "invalid_request", "message": "That date has passed. Which date would you like to travel?",
                    "returned_option_ids": set()}
        itineraries = self._search_with_retries(constraints, config, trace, jdate)
        if itineraries is None:
            return {"status": "tool_failure", "message": "The timetable service is not responding right now. "
                    "Please try again in a few minutes.", "returned_option_ids": set()}
        if not itineraries:
            return {"status": "no_options", "message": "I could not find trains between those places on that date.",
                    "returned_option_ids": set()}
        options = self._assess_and_rank(constraints, trace, today, source, profile, itineraries)
        ids = {o["option_id"] for o in options}
        if not options:
            return {"status": "no_options", "message": "Bookings are not open for that date yet, or no class matched.",
                    "returned_option_ids": ids}
        min_confirm = float(config.get("min_confirm_prob", 0.4))
        feasible, relaxed = self._select(options, constraints, min_confirm)
        if not feasible:
            return {"status": "no_options", "message": "None of the options can be booked with a realistic chance of "
                    "confirmation. Try another date or class.", "returned_option_ids": ids}
        best = feasible[0]
        pool = feasible[1:] + sorted(options, key=lambda o: o["rec_rank"])
        picked: list[dict] = []
        situation = flex = None
        if relaxed:
            situation = self._situation(options, constraints, min_confirm)
            # Interactive requests also get a look at the next few days (skipped for simulated traffic).
            if source in ("api", "support"):
                flex = self._date_flex(constraints, trace, jdate, today, min_confirm)
                if flex["option"]:
                    picked.append(flex["option"])
            if "arrive_by" in relaxed:
                confirmable = [o for o in options if o["confirm_prob"] >= min_confirm and o["summary"] != best["summary"]]
                if confirmable:
                    soonest = min(confirmable, key=lambda o: (self._shortfall(o, constraints).get("arrive_by", 0.0),
                                                              o["arrival"], o["fare"]))
                    soonest["note"] = "Closest to your arrival time with a confirmable seat"
                    picked.append(soonest)
            if "confirmation" not in relaxed:
                literal, _ = self._select(options, constraints, 0.0)
                if literal and literal[0]["summary"] != best["summary"]:
                    literal[0]["note"] = "Closest match to your exact request, but seats are waitlisted"
                    picked.append(literal[0])
        alts = self._alternatives(best, pool, n=3 if relaxed else 2, already=picked)
        t0 = time.perf_counter()
        outlook = self.tools.fare_outlook(best, today)
        trace.add("fare-agent", "fare_outlook", {"option": best["option_id"]}, True, (time.perf_counter() - t0) * 1000,
                  "fare forecaster checked whether waiting would cost more" if outlook else "booking window too short")
        trace.add("planner", "propose_itinerary", {"option": best["option_id"]}, True, 0.0, best["summary"])
        return {"status": "proposed",
                "message": self._compose(best, alts, relaxed, constraints, outlook, min_confirm, situation, flex),
                "recommended": best, "alternatives": alts, "relaxed": relaxed, "fare_outlook": outlook,
                "date_flex": None if flex is None else {"checked": flex["checked"],
                                                        "found": flex["option"]["departure"][:10]
                                                        if flex["option"] else None},
                "returned_option_ids": ids}

    # ------------------------------------------------------------ Gemini engine
    def _plan_with_llm(self, query, constraints, config, trace, today, source, profile) -> dict | None:
        state: dict = {"options": {}, "proposal": None, "clarification": None}

        def search_options(origin: str, destination: str, date_str: str | None = None,
                           classes: list[str] | None = None) -> dict:
            o_codes = self.p.network.resolve_place(origin)
            d_codes = self.p.network.resolve_place(destination)
            if not o_codes or not d_codes:
                return {"error": f"unknown place: {origin if not o_codes else destination}"}
            jdate = date.fromisoformat(date_str or constraints["date"])
            local = {**constraints, "origin_codes": o_codes, "destination_codes": d_codes, "date": jdate.isoformat(),
                     "classes": classes or constraints["classes"]}
            its = self._search_with_retries(local, config, trace, jdate)
            if its is None:
                return {"error": "timetable search failed after retries"}
            opts = self._assess_and_rank(local, trace, today, source, profile, its)
            for o in opts:
                state["options"][o["option_id"]] = o
            opts.sort(key=lambda o: o["rec_rank"])
            return {"options": [{"option_id": o["option_id"], "route": o["summary"], "class": o["travel_class"],
                                 "fare_per_passenger": o["fare"], "departure": o["departure"], "arrival": o["arrival"],
                                 "duration_min": o["duration_min"], "transfers": o["transfers"],
                                 "availability": o["availability"], "confirm_prob": o["confirm_prob"],
                                 "expected_delay_min": o["expected_delay_min"], "on_time_prob": o["on_time_prob"],
                                 "connection_risk": o["connection_risk"], "recommender_rank": o["rec_rank"]}
                                for o in opts[:25]], "total_options": len(opts)}

        def fare_outlook(option_id: str) -> dict:
            o = state["options"].get(option_id)
            if not o:
                return {"error": "unknown option_id"}
            return self.tools.fare_outlook(o, today) or {"note": "departure too soon to forecast"}

        def propose_itinerary(option_id: str, explanation: str, alternative_ids: list[str] | None = None) -> dict:
            state["proposal"] = {"option_id": option_id, "explanation": explanation,
                                 "alternative_ids": alternative_ids or []}
            return {"recorded": True, "note": "The passenger must approve before anything is booked."}

        def ask_clarification(question: str) -> dict:
            state["clarification"] = question
            return {"recorded": True}

        tools = [
            {"name": "search_options", "description": "Search direct and one-change itineraries and return priced "
             "options with availability, confirmation probability, expected delay, on-time probability and "
             "recommender rank.",
             "parameters": {"type": "object", "properties": {
                 "origin": {"type": "string", "description": "city or station code"},
                 "destination": {"type": "string"}, "date_str": {"type": "string", "description": "YYYY-MM-DD"},
                 "classes": {"type": "array", "items": {"type": "string", "enum": list(TRAVEL_CLASSES)}}},
                 "required": ["origin", "destination"]}},
            {"name": "fare_outlook", "description": "Forecast fare and sell-out risk for an option if the passenger waits.",
             "parameters": {"type": "object", "properties": {"option_id": {"type": "string"}},
                              "required": ["option_id"]}},
            {"name": "propose_itinerary", "description": "Propose the chosen option (and up to two alternatives) to the "
             "passenger. Does not book.",
             "parameters": {"type": "object", "properties": {
                 "option_id": {"type": "string"}, "explanation": {"type": "string"},
                 "alternative_ids": {"type": "array", "items": {"type": "string"}}},
                 "required": ["option_id", "explanation"]}},
            {"name": "ask_clarification", "description": "Ask the passenger a question when the request is ambiguous.",
             "parameters": {"type": "object", "properties": {"question": {"type": "string"}},
                              "required": ["question"]}},
        ]

        def logged(name, fn, agent):
            def handler(args):
                t0 = time.perf_counter()
                out = fn(**args)
                trace.add(agent, name, args, "error" not in out, (time.perf_counter() - t0) * 1000,
                          (out.get("error") or f"{name} ok") if isinstance(out, dict) else f"{name} ok")
                return out
            return handler

        handlers = {"search_options": logged("search_options", search_options, "planner(gemini)"),
                    "fare_outlook": logged("fare_outlook", fare_outlook, "planner(gemini)"),
                    "propose_itinerary": logged("propose_itinerary", propose_itinerary, "planner(gemini)"),
                    "ask_clarification": logged("ask_clarification", ask_clarification, "planner(gemini)")}
        system = (f"{config['system_prompt']}\nToday's date: {today.isoformat()} (India Standard Time).\n"
                  f"A rule-based parser read the request as: {self._constraint_summary(constraints)}. "
                  "Use it as a hint but trust the passenger's words.")
        out = run_tool_loop(self.p.llm, system, query, tools, handlers, max_steps=int(config.get("max_steps", 12)))
        ids = set(state["options"])
        if state["clarification"] and not state["proposal"]:
            return {"status": "needs_clarification", "message": state["clarification"], "returned_option_ids": ids}
        prop = state["proposal"]
        if not prop:
            return None  # the deterministic planner takes over; the trace keeps the Gemini steps
        best = state["options"].get(prop["option_id"])
        if best is None:
            # A fabricated option id is kept as-is so the auditor flags it.
            best = {"option_id": prop["option_id"], "legs": [], "fare": 0, "arrival": f"{constraints['date']}T00:00",
                    "travel_class": "?", "summary": "unknown option"}
        alts = [state["options"][a] for a in prop["alternative_ids"] if a in state["options"]][:2]
        relaxed = [k for k, v in self._meets(best, constraints).items() if not v and k != "bookable"] if best.get("legs") else []
        if best.get("legs") and best.get("confirm_prob", 1) < float(config.get("min_confirm_prob", 0.4)):
            relaxed.append("confirmation")
        message = (out["text"] or prop["explanation"]) + "\nApprove to book this itinerary (creates a demo PNR)."
        return {"status": "proposed", "message": message, "recommended": best, "alternatives": alts,
                "relaxed": relaxed if relaxed and any(r in message.lower() for r in ("budget", "arriv", "class", "miss", "confirm", "waitlist"))
                else [], "returned_option_ids": ids}
