"""Synthetic passenger traffic.

Draws requests from the digital twin under the current world (including any
active chaos scenario), sends them through the real serving paths of every
service, and attaches the outcomes the twin later observes (final ticket
status, actual delay, final occupancy, investigation result, the train the
passenger actually chose). This is the stream the monitors watch.
"""
from __future__ import annotations

import time
from datetime import timedelta

import numpy as np

from railsentinel.twin.generators import today_ist

BATCH = {"confirmation_predictor": 300, "delay_predictor": 300, "fare_forecaster": 300, "fraud_detector": 600,
         "train_recommender": 300, "support_assistant": 60, "travel_planner_agent": 15}

CITY_PAIRS = [("Pune", "Delhi"), ("Mumbai", "Delhi"), ("Delhi", "Kolkata"), ("Chennai", "Bengaluru"),
              ("Mumbai", "Ahmedabad"), ("Delhi", "Lucknow"), ("Bengaluru", "Delhi"), ("Hyderabad", "Chennai"),
              ("Mumbai", "Goa"), ("Delhi", "Amritsar"), ("Delhi", "Patna"), ("Kolkata", "Guwahati"),
              ("Mumbai", "Chennai"), ("Delhi", "Jaipur"), ("Kochi", "Bengaluru"), ("Pune", "Bengaluru")]


def planner_queries(rng: np.random.Generator, n: int) -> list[str]:
    templates = [
        "I want to travel from {o} to {d} {when} under ₹{budget} and reach before {hour}:00 AM.",
        "Need a {cls} ticket from {o} to {d} {when}",
        "Cheapest way from {o} to {d} {when}",
        "{o} to {d} {when}, must arrive by {hour}:00 am",
        "Find the most reliable train from {o} to {d} {when}",
        "{n} people travelling from {o} to {d} {when} in sleeper, budget Rs {budget}",
        "Fastest train {o} to {d} {when}",
    ]
    out = []
    for _ in range(n):
        o, d = CITY_PAIRS[int(rng.integers(0, len(CITY_PAIRS)))]
        if rng.random() < 0.5:
            o, d = d, o
        when = str(rng.choice(["tomorrow", "day after tomorrow", "this friday", "next sunday", "on " +
                               (today_ist() + timedelta(days=int(rng.integers(3, 40)))).strftime("%d %b")]))
        out.append(str(rng.choice(templates)).format(
            o=o, d=d, when=when, budget=int(rng.choice([800, 1200, 1500, 2500, 4000])), hour=int(rng.integers(6, 12)),
            cls=str(rng.choice(["sleeper", "3AC", "2AC", "chair car"])), n=int(rng.integers(2, 5))))
    return out


def support_questions(rng: np.random.Generator, n: int, pnrs: list[str], cancelled: list[str]) -> list[str]:
    o, d = CITY_PAIRS[int(rng.integers(0, len(CITY_PAIRS)))]
    templates = [
        lambda: f"What is the status of PNR {rng.choice(pnrs)}?",
        lambda: f"Will my ticket {rng.choice(pnrs)} get confirmed?",
        lambda: f"How much refund will I get if I cancel PNR {rng.choice(pnrs)}?",
        lambda: (f"What is the refund status for PNR {rng.choice(cancelled)}?" if cancelled
                 else "When is the reservation chart prepared?"),
        lambda: (f"What are the cancellation charges for a confirmed {rng.choice(['SL', '3A', '2A', '1A'])} ticket "
                 f"with fare Rs {int(rng.choice([650, 1200, 2100, 3400]))} cancelled "
                 f"{int(rng.choice([2, 6, 20, 30, 72]))} hours before departure?"),
        lambda: "Can I get a refund on a confirmed Tatkal ticket?",
        lambda: "When is the reservation chart prepared?",
        lambda: str(rng.choice(["What does GNWL mean?", "What is the difference between RLWL and PQWL?",
                                "What is TQWL?"])),
        lambda: "What are the refund rules for a waitlisted e-ticket?",
        lambda: f"Find me a train from {o} to {d} tomorrow",
        lambda: "Hello",
    ]
    return [str(templates[int(rng.integers(0, len(templates)))]()) for _ in range(n)]


class TrafficGenerator:
    def __init__(self, platform):
        self.p = platform

    def run(self, scale: float = 1.0, services: list[str] | None = None) -> dict:
        t0 = time.perf_counter()
        world = self.p.world.effective_params()
        rng = self.p.rng()
        counts: dict[str, int] = {}
        wanted = services or list(BATCH)
        for name, spec in self.p.specs.items():
            if name not in wanted:
                continue
            n = max(20, int(BATCH[name] * scale))
            data = spec.generate(self.p.network, n, rng, world)
            records = spec.frame_to_records(data)
            labels = [spec.label_of(r) for r in records]
            self.p.runtime(name).predict(records, source="traffic", labels=labels)
            counts[name] = n
        if "support_assistant" in wanted:
            n = max(10, int(BATCH["support_assistant"] * scale))
            pnrs = [b["pnr"] for b in self.p.bookings.list(200) if b["current_status"] != "CANCELLED"]
            cancelled = [b["pnr"] for b in self.p.bookings.list(200) if b["current_status"] == "CANCELLED"]
            use_llm = self.p.settings.llm_for_synthetic_traffic
            for q in support_questions(rng, n, pnrs or ["2000000000"], cancelled):
                res = self.p.support.answer(q, source="traffic", use_llm=use_llm)
                if rng.random() < 0.4:  # some passengers leave feedback
                    helpful = rng.random() < (0.25 if res["grounding"]["hallucination"] else 0.88)
                    self.p.support.feedback(res["interaction_id"], helpful)
            counts["support_assistant"] = n
        if "travel_planner_agent" in wanted:
            n = max(5, int(BATCH["travel_planner_agent"] * scale))
            for q in planner_queries(rng, n):
                self.p.planner.plan(q, source="traffic", use_llm=self.p.settings.llm_for_synthetic_traffic,
                                    create_intent=False)
            counts["travel_planner_agent"] = n
        summary = {"requests": counts, "scenario": self.p.world.scenario,
                   "duration_s": round(time.perf_counter() - t0, 2)}
        self.p.audit.append("traffic-simulator", "traffic_batch", "platform", summary)
        return summary
