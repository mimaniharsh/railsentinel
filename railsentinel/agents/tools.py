"""Railway tools used by the planner agent and the support assistant.

`JourneyTools` is the coordination layer: it searches the timetable (direct
and one-change itineraries), then asks the specialist models for seat
availability, confirmation probability, delay risk and fare outlook, and
returns options in the same shape the recommender ranks.
"""
from __future__ import annotations

import math
import random
import time
from dataclasses import dataclass, field
from datetime import date, datetime, timedelta

from railsentinel.twin import inventory as inv
from railsentinel.twin import live
from railsentinel.twin.generators import delay_features, is_overnight
from railsentinel.twin.network import TRAVEL_CLASSES, Segment

MIN_TRANSFER_MIN = 45
MAX_TRANSFER_MIN = 8 * 60
TRANSFER_BUFFER_MIN = 20
CONFIRMATION_CLASSES = {"SL", "CC", "3A", "2A", "1A"}


class ToolError(RuntimeError):
    pass


def normal_cdf(x: float) -> float:
    return 0.5 * (1 + math.erf(x / math.sqrt(2)))


def prob_delay_within(slack_min: float, p10: float, p50: float, p90: float) -> float:
    """P(delay <= slack) from the predicted p10/p50/p90, using a normal approximation."""
    sd = max((p90 - p10) / 2.563, 5.0)
    return float(min(max(normal_cdf((slack_min - p50) / sd), 0.0), 1.0))


@dataclass
class Trace:
    steps: list[dict] = field(default_factory=list)

    def add(self, agent: str, tool: str, args: dict, ok: bool, latency_ms: float, summary: str) -> None:
        self.steps.append({"step": len(self.steps) + 1, "agent": agent, "tool": tool, "args": args, "ok": ok,
                           "latency_ms": round(latency_ms, 1), "summary": summary})


class JourneyTools:
    def __init__(self, platform):
        self.p = platform
        self._rng = random.Random(platform.settings.seed)

    # ------------------------------------------------------------ timetable search
    def search(self, origins: list[str], destinations: list[str], jdate: date, trace: Trace | None = None,
               agent: str = "search-agent") -> list[dict]:
        world = self.p.world.params()
        ops = self.p.world.ops()
        started = time.perf_counter()
        use_cache = "timetable_search" in ops.circuit_breakers
        simulated_latency = 0 if use_cache else world.search_latency_ms
        if not use_cache and world.search_failure_rate > 0 and self._rng.random() < world.search_failure_rate:
            elapsed = (time.perf_counter() - started) * 1000 + simulated_latency
            if trace:
                trace.add(agent, "timetable_search", {"origins": origins, "destinations": destinations,
                                                      "date": jdate.isoformat()}, False, elapsed,
                          "timetable search API timed out")
            raise ToolError("timetable search API timed out")
        itineraries = self._direct(origins, destinations, jdate) + self._one_change(origins, destinations, jdate)
        itineraries.sort(key=lambda it: it["arrival"])
        elapsed = (time.perf_counter() - started) * 1000 + simulated_latency
        if trace:
            source = "cached timetable (circuit breaker open)" if use_cache else "live timetable"
            trace.add(agent, "timetable_search", {"origins": origins, "destinations": destinations,
                                                  "date": jdate.isoformat()}, True, elapsed,
                      f"{len(itineraries)} itineraries from {source}")
        return itineraries

    def probe_search(self) -> bool:
        """Health probe used by the supervisor before closing a circuit breaker."""
        world = self.p.world.params()
        return not (world.search_failure_rate > 0 and self._rng.random() < world.search_failure_rate)

    def _leg(self, seg: Segment, boarding: date) -> dict:
        return {"train_number": seg.train.number, "train_name": seg.train.name, "category": seg.train.category,
                "origin": seg.origin, "destination": seg.destination,
                "departure": seg.departure_dt(boarding).isoformat(timespec="minutes"),
                "arrival": seg.arrival_dt(boarding).isoformat(timespec="minutes"),
                "distance_km": round(seg.distance_km), "duration_min": seg.duration_min,
                "classes": list(seg.train.classes), "_seg": seg, "_boarding": boarding}

    def _direct(self, origins, destinations, jdate) -> list[dict]:
        out = []
        for seg in self.p.network.direct_segments(origins, destinations):
            leg = self._leg(seg, jdate)
            out.append({"legs": [leg], "transfers": 0, "departure": leg["departure"], "arrival": leg["arrival"]})
        return out

    def _one_change(self, origins, destinations, jdate, limit: int = 25) -> list[dict]:
        net = self.p.network
        dest_set = set(destinations)
        firsts: dict[str, list[Segment]] = {}
        for o in origins:
            for tr in net.trains_at(o):
                i = tr.index_of(o)
                for j in range(i + 1, len(tr.stops)):
                    hub = tr.stops[j].code
                    if hub in dest_set or hub in origins:
                        continue
                    firsts.setdefault(hub, []).append(Segment(tr, i, j))
        out = []
        for hub, legs1 in firsts.items():
            seconds = net.direct_segments([hub], list(dest_set))
            if not seconds:
                continue
            for s1 in legs1:
                arr1 = s1.arrival_dt(jdate)
                for s2 in seconds:
                    if s2.train.number == s1.train.number:
                        continue
                    for boarding2 in (arr1.date(), arr1.date() + timedelta(days=1)):
                        dep2 = s2.departure_dt(boarding2)
                        layover = (dep2 - arr1).total_seconds() / 60
                        if MIN_TRANSFER_MIN <= layover <= MAX_TRANSFER_MIN:
                            l1, l2 = self._leg(s1, jdate), self._leg(s2, boarding2)
                            out.append({"legs": [l1, l2], "transfers": 1, "hub": hub, "layover_min": int(layover),
                                        "departure": l1["departure"], "arrival": l2["arrival"]})
        out.sort(key=lambda it: it["arrival"])
        return out[:limit]

    # ------------------------------------------------------------ specialist assessment
    def assess(self, itineraries: list[dict], today: date, classes: list[str] | None = None,
               trace: Trace | None = None, source: str = "agent") -> list[dict]:
        """Expand itineraries into priced, risk-scored options (one per travel class)."""
        world = self.p.world.effective_params()
        options: list[dict] = []
        t0 = time.perf_counter()
        for k, it in enumerate(itineraries):
            common = set(it["legs"][0]["classes"])
            for leg in it["legs"][1:]:
                common &= set(leg["classes"])
            wanted = [c for c in it["legs"][0]["classes"] if c in common and (not classes or c in classes)]
            for cls in wanted:
                legs = []
                for leg in it["legs"]:
                    seg, boarding = leg["_seg"], leg["_boarding"]
                    snap = inv.snapshot(self.p.network, seg, cls, boarding, today, world)
                    base = self.p.network.fare(seg.train, cls, seg.distance_km)
                    fare = base * (inv.flexi_multiplier(snap["occupancy"]) if seg.train.dynamic_fare else 1.0)
                    legs.append({**{k2: v for k2, v in leg.items() if not k2.startswith("_")}, "travel_class": cls,
                                 "fare": round(fare), "base_fare": base, "availability": inv.availability_label(snap),
                                 "_snap": snap, "_seg": seg, "_boarding": boarding})
                if any(l["_snap"]["status"] in ("DEPARTED", "NOT_OPEN") for l in legs):
                    continue
                dep = datetime.fromisoformat(it["departure"])
                arr = datetime.fromisoformat(it["arrival"])
                options.append({
                    "option_id": f"OPT{len(options) + 1}", "itinerary_index": k,
                    "summary": " + ".join(f"{l['train_number']} {l['origin']}→{l['destination']}" for l in legs),
                    "legs": legs, "transfers": it["transfers"], "hub": it.get("hub"),
                    "layover_min": it.get("layover_min"), "travel_class": cls,
                    "fare": int(sum(l["fare"] for l in legs)), "departure": it["departure"], "arrival": it["arrival"],
                    "duration_min": int((arr - dep).total_seconds() / 60), "dep_hour": dep.hour,
                    "overnight": is_overnight(dep, arr),
                    "punctuality": min(inv.punctuality_index(l["_seg"].train) for l in legs),
                    "comfort": TRAVEL_CLASSES[cls]["comfort"],
                    "is_premium": int(any(l["category"] == "premium" for l in legs)),
                    "distance_km": int(sum(l["distance_km"] for l in legs)),
                    "train_number": legs[0]["train_number"], "train_name": legs[0]["train_name"],
                    "category": legs[0]["category"],
                })
        if trace:
            trace.add("availability-agent", "seat_availability", {"itineraries": len(itineraries),
                                                                  "classes": classes or "all"}, True,
                      (time.perf_counter() - t0) * 1000, f"{len(options)} priced options")
        self._attach_confirmation(options, today, trace, source)
        self._attach_delay(options, trace, source)
        for o in options:
            for leg in o["legs"]:
                leg.pop("_seg", None)
                leg.pop("_boarding", None)
                leg.pop("_snap", None)
        return options

    def _attach_confirmation(self, options, today, trace, source):
        t0 = time.perf_counter()
        records, where = [], []
        for oi, o in enumerate(options):
            for li, leg in enumerate(o["legs"]):
                snap = leg["_snap"]
                leg["confirm_prob"] = 1.0
                if snap["status"] in ("RAC", "WL"):
                    if leg["travel_class"] in CONFIRMATION_CLASSES:
                        records.append(confirmation_record(snap, self.p.network.stations[leg["origin"]].region))
                        where.append((oi, li))
                    else:
                        leg["confirm_prob"] = max(0.05, 0.75 - 0.01 * snap["position"])
                        leg["confirm_note"] = "heuristic (class not covered by the confirmation model)"
        if records:
            results = self.p.runtime("confirmation_predictor").predict(records, source=source)
            for (oi, li), res in zip(where, results):
                leg = options[oi]["legs"][li]
                if res["status"] == "ok":
                    leg["confirm_prob"] = res["output"]["p_confirm"]
                    leg["confirm_prediction_id"] = res["prediction_id"]
                else:
                    leg["confirm_prob"] = 0.3
                    leg["confirm_note"] = "input blocked by data contract; conservative default used"
        for o in options:
            o["confirm_prob"] = round(float(math.prod(l["confirm_prob"] for l in o["legs"])), 4)
            o["availability"] = " / ".join(l["availability"] for l in o["legs"])
        if trace:
            trace.add("availability-agent", "predict_confirmation", {"waitlisted_legs": len(records)}, True,
                      (time.perf_counter() - t0) * 1000,
                      f"confirmation model scored {len(records)} waitlisted/RAC legs")

    def _attach_delay(self, options, trace, source):
        t0 = time.perf_counter()
        world = self.p.world.effective_params()
        records, where = [], []
        for oi, o in enumerate(options):
            for li, leg in enumerate(o["legs"]):
                seg, boarding = leg["_seg"], leg["_boarding"]
                arr_dt = datetime.fromisoformat(leg["arrival"])
                region = self.p.network.stations[seg.destination].region
                weather = live.weather_feed(region, arr_dt.date(), world)
                run_date = seg.origin_run_date(boarding)
                feats = delay_features(self.p.network, seg.train, seg.to_idx, None, 0.0, run_date.month,
                                       run_date.weekday(), weather)
                records.append(feats)
                where.append((oi, li))
        results = self.p.runtime("delay_predictor").predict(records, source=source) if records else []
        for (oi, li), res in zip(where, results):
            leg = options[oi]["legs"][li]
            if res["status"] == "ok":
                out = res["output"]
                leg["delay_p50"] = out["arrival_delay_min"]
                leg["delay_interval"] = out["interval_p10_p90"]
            else:
                leg["delay_p50"], leg["delay_interval"] = 30.0, [5.0, 90.0]
        for o in options:
            last = o["legs"][-1]
            o["expected_delay_min"] = last["delay_p50"]
            o["delay_interval"] = last["delay_interval"]
            if o["transfers"]:
                first = o["legs"][0]
                lo, hi = first["delay_interval"]
                p_ok = prob_delay_within(o["layover_min"] - TRANSFER_BUFFER_MIN, lo, first["delay_p50"], hi)
                o["connection_risk"] = round(1 - p_ok, 4)
            else:
                o["connection_risk"] = 0.0
        if trace:
            trace.add("punctuality-agent", "predict_delay", {"legs": len(records)}, True,
                      (time.perf_counter() - t0) * 1000, f"delay model scored {len(records)} legs")

    def on_time_probability(self, option: dict, arrive_by: str | None) -> float | None:
        if not arrive_by:
            return None
        arr = datetime.fromisoformat(option["arrival"])
        hh, mm = (int(x) for x in arrive_by.split(":"))
        deadline = arr.replace(hour=hh, minute=mm)
        slack = (deadline - arr).total_seconds() / 60
        lo, hi = option["delay_interval"]
        p = prob_delay_within(slack, lo, option["expected_delay_min"], hi)
        return round(p * (1 - option["connection_risk"]), 4)

    def fare_outlook(self, option: dict, today: date, source: str = "agent") -> dict | None:
        """Fare/sell-out forecast for the first leg if the passenger waits a week (or until 2 days out)."""
        leg = option["legs"][0]
        net = self.p.network
        tr = net.trains[leg["train_number"]]
        seg = Segment(tr, tr.index_of(leg["origin"]), tr.index_of(leg["destination"]))
        jdate = datetime.fromisoformat(leg["departure"]).date()
        snap = inv.snapshot(net, seg, leg["travel_class"], jdate, today, self.p.world.effective_params())
        lead = snap["days_to_departure"]
        if lead < 2:
            return None
        target = max(0, min(lead - 1, lead - 7))
        rec = fare_record(snap, net.fare(tr, leg["travel_class"], seg.distance_km), tr.dynamic_fare, target,
                          net.stations[seg.origin].region)
        res = self.p.runtime("fare_forecaster").predict([rec], source=source)[0]
        if res["status"] != "ok":
            return {"status": "blocked", "reasons": res["output"]["reasons"]}
        return {"days_from_now": lead - target, **res["output"], "prediction_id": res["prediction_id"]}


def confirmation_record(snap: dict, region: str, quota: str = "GN") -> dict:
    return {"train_category": snap["train_category"], "travel_class": snap["travel_class"], "quota": quota,
            "booking_status": snap["status"], "season": snap["season"], "queue_position": int(snap["position"]),
            "days_to_departure": int(max(0, snap["days_to_departure"])), "distance_km": snap["distance_km"],
            "route_demand": snap["route_demand"], "hist_cancel_rate": snap["hist_cancel_rate"],
            "weekday": snap["weekday"], "month": snap["month"], "capacity": snap["capacity"], "region": region}


def fare_record(snap: dict, base_fare: float, dynamic: bool, target_lead: int, region: str,
                quoted_fare: float | None = None) -> dict:
    occ = snap["occupancy"]
    quote = quoted_fare if quoted_fare is not None else base_fare * (inv.flexi_multiplier(occ) if dynamic else 1.0)
    return {"train_category": snap["train_category"], "travel_class": snap["travel_class"],
            "season": snap["season"], "weekday": snap["weekday"], "month": snap["month"],
            "route_demand": snap["route_demand"], "distance_km": snap["distance_km"], "dynamic_fare": int(dynamic),
            "current_lead": int(snap["days_to_departure"]), "current_occupancy": round(min(occ, 1.8), 4),
            "target_lead": int(target_lead), "base_fare": base_fare, "quoted_fare": round(quote, 2),
            "region": region}
