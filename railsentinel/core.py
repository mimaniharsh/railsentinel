"""Platform: wires the twin, the governed services and the governance layer together."""
from __future__ import annotations

import logging
import threading
import time
from datetime import date, datetime, timedelta

import numpy as np
import pandas as pd

from railsentinel.agents.kb import KnowledgeBase
from railsentinel.agents.llm import LLM
from railsentinel.agents.planner import AGENT_V1, TravelPlanner
from railsentinel.agents.support import PROMPT_V1, SupportAssistant
from railsentinel.agents.tools import confirmation_record, fare_record
from railsentinel.config import IST, Settings, load_settings
from railsentinel.db import Database
from railsentinel.governance.agent_audit import AgentAuditor
from railsentinel.governance.audit import AuditLog
from railsentinel.governance.chaos import ChaosEngine
from railsentinel.governance.lifecycle import Lifecycle
from railsentinel.governance.monitors import MonitorSuite, set_window_floor
from railsentinel.governance.policy import PolicyEngine
from railsentinel.governance.rca import RootCauseAnalyzer
from railsentinel.governance.registry import Registry
from railsentinel.governance.supervisor import Supervisor
from railsentinel.ml.confirmation import ConfirmationPredictor
from railsentinel.ml.delay import DelayPredictor
from railsentinel.ml.fare import FareForecaster
from railsentinel.ml.fraud import FraudDetector
from railsentinel.ml.recommender import TrainRecommender
from railsentinel.ml.runtime import ModelRuntime
from railsentinel.traffic import TrafficGenerator
from railsentinel.twin import inventory as inv
from railsentinel.twin import live
from railsentinel.security import stable_hash
from railsentinel.twin.generators import (CONFIRMATION_CLASSES, REFERENCE_YEAR, confirmation_samples, delay_features,
                                          today_ist)
from railsentinel.twin.network import Segment, load_network
from railsentinel.twin.seasons import season_of, season_of_month
from railsentinel.twin.world import World, WorldParams

log = logging.getLogger("railsentinel")

GENERATIVE_ASSETS = {
    "support_assistant": ("prompt", "Customer-support assistant prompt and grounding configuration", "cx-ai", "high",
                          PROMPT_V1),
    "travel_planner_agent": ("agent", "Agentic travel planner configuration (limits, retries, instructions)",
                             "agents-team", "high", AGENT_V1),
}


class Platform:
    _instance: "Platform | None" = None
    _instance_lock = threading.Lock()

    def __init__(self, settings: Settings | None = None, llm_client=None):
        self.settings = settings or load_settings()
        self.db = Database(self.settings.db_path)
        self.audit = AuditLog(self.db)
        self.registry = Registry(self.db, self.settings.models_dir, self.audit)
        self.world = World(self.db)
        self.network = load_network()
        from railsentinel.twin.bookings import BookingStore
        self.bookings = BookingStore(self.db, self.network)
        self.llm = LLM(self.settings, client=llm_client)
        self.kb = KnowledgeBase()
        self.specs = {s.name: s for s in (ConfirmationPredictor(), TrainRecommender(), DelayPredictor(),
                                          FareForecaster(), FraudDetector())}
        self._runtimes = {name: ModelRuntime(self, spec) for name, spec in self.specs.items()}
        self.policies = PolicyEngine()
        self.agent_auditor = AgentAuditor()
        self.support = SupportAssistant(self)
        self.planner = TravelPlanner(self)
        self.monitors = MonitorSuite(self)
        self.rca = RootCauseAnalyzer(self)
        self.lifecycle = Lifecycle(self)
        self.supervisor = Supervisor(self)
        self.chaos = ChaosEngine(self)
        self.traffic = TrafficGenerator(self)
        self._rng_lock = threading.Lock()
        self._rng_counter = 0

    # ------------------------------------------------------------ singleton used by the API
    @classmethod
    def get(cls) -> "Platform":
        with cls._instance_lock:
            if cls._instance is None:
                cls._instance = Platform()
            return cls._instance

    @classmethod
    def set_instance(cls, platform: "Platform | None") -> None:
        with cls._instance_lock:
            cls._instance = platform

    # ------------------------------------------------------------ helpers
    def runtime(self, name: str) -> ModelRuntime:
        return self._runtimes[name]

    def asset_names(self) -> list[str]:
        return list(self.specs) + list(GENERATIVE_ASSETS)

    def rng(self) -> np.random.Generator:
        with self._rng_lock:
            self._rng_counter += 1
            return np.random.default_rng([self.settings.seed, self._rng_counter, int(time.time() * 1000) % 1_000_000])

    def fleet_health(self, asset: str) -> dict:
        return (self.db.get_kv("fleet_health", {}) or {}).get(asset, {"status": "unknown", "open_incidents": 0})

    def is_bootstrapped(self) -> bool:
        return all(self.registry.production(a) for a in self.asset_names())

    # ------------------------------------------------------------ bootstrap
    def bootstrap(self, scale: float | None = None, warmup_batches: int = 2, progress=print) -> dict:
        scale = self.settings.bootstrap_scale if scale is None else scale
        started = time.time()
        self.world.reset()
        rng = np.random.default_rng(self.settings.seed)
        for name, spec in self.specs.items():
            if self.registry.production(name):
                progress(f"  {name}: already trained (v{self.registry.production(name)['version']})")
                continue
            self.registry.register_model(name, "ml", spec.description, spec.owner, spec.risk_tier)
            n = max(600 if name == "train_recommender" else 4000, int(spec.default_train_size * scale))
            t0 = time.time()
            data = spec.generate(self.network, n, rng, WorldParams())
            artifact = spec.fit(data, seed=self.settings.seed)
            version = self.registry.add_version(name, artifact, artifact["baseline_metrics"],
                                                params={"training_rows": n}, stage="production",
                                                notes="initial training on baseline twin data", actor="bootstrap")
            progress(f"  {name}: trained v{version} on {n} records in {time.time() - t0:.1f}s "
                     f"({spec.primary_metric}={artifact['baseline_metrics'].get(spec.primary_metric)})")
        for name, (kind, desc, owner, tier, config) in GENERATIVE_ASSETS.items():
            if self.registry.production(name):
                continue
            self.registry.register_model(name, kind, desc, owner, tier)
            self.registry.add_version(name, dict(config), {}, stage="production", notes="initial release",
                                      actor="bootstrap")
            progress(f"  {name}: registered v1")
        if not self.bookings.list(1):
            today = today_ist()
            pnrs = self.bookings.seed_samples(rng, today, WorldParams(), n=30)
            for pnr in pnrs[-5:]:
                self.bookings.cancel(pnr)
            progress(f"  seeded {len(pnrs)} demo PNRs (5 cancelled with refunds in progress)")
        for asset in self.asset_names():
            set_window_floor(self.db, asset, 0.0)
        for i in range(warmup_batches):
            summary = self.traffic.run()
            progress(f"  warm-up traffic batch {i + 1}: {sum(summary['requests'].values())} requests "
                     f"in {summary['duration_s']}s")
        intents = self.db.query("SELECT intent, COUNT(*) AS n FROM llm_interactions GROUP BY intent")
        self.db.set_kv("support_intent_reference", {r["intent"]: r["n"] for r in intents})
        cycle = self.supervisor.run_cycle()
        self.audit.append("bootstrap", "platform_bootstrapped", "platform",
                          {"duration_s": round(time.time() - started, 1), "scale": scale,
                           "fleet": {a: f["status"] for a, f in cycle["fleet"].items()}})
        statuses = ", ".join(f"{a}={f['status']}" for a, f in cycle["fleet"].items())
        progress(f"  first supervisor cycle: {statuses}")
        return {"duration_s": round(time.time() - started, 1), "fleet": cycle["fleet"]}

    def refresh_demo_bookings(self) -> int:
        """Seed fresh demo PNRs when every seeded journey date has passed (long-running deployments)."""
        today = today_ist()
        upcoming = self.db.scalar("SELECT COUNT(*) FROM bookings WHERE journey_date>=? AND current_status<>'CANCELLED'",
                                  (today.isoformat(),)) or 0
        if upcoming >= 10:
            return 0
        pnrs = self.bookings.seed_samples(self.rng(), today, WorldParams(), n=30)
        for pnr in pnrs[-5:]:
            self.bookings.cancel(pnr)
        self.audit.append("platform", "demo_bookings_refreshed", "bookings", {"seeded": len(pnrs)})
        return len(pnrs)

    # ------------------------------------------------------------ passenger-facing composites
    def segment_for(self, train_number: str, origin: str, destination: str) -> Segment:
        tr = self.network.trains[train_number]
        i, j = tr.index_of(origin), tr.index_of(destination)
        if i is None or j is None or i >= j:
            raise ValueError(f"{train_number} does not run {origin} → {destination}")
        return Segment(tr, i, j)

    def pnr_confirmation(self, booking: dict, explain: bool = False, source: str = "api") -> dict | None:
        status = booking["current_status"].split()
        if status[0] not in ("WL", "RAC") or booking["travel_class"] not in {"SL", "CC", "3A", "2A", "1A"}:
            return None
        seg = self.segment_for(booking["train_number"], booking["origin"], booking["destination"])
        jdate = date.fromisoformat(booking["journey_date"])
        snap = inv.snapshot(self.network, seg, booking["travel_class"], jdate, today_ist(),
                            self.world.effective_params())
        snap = {**snap, "status": status[0], "position": int(status[1]) if len(status) > 1 else 1}
        rec = confirmation_record(snap, self.network.stations[seg.origin].region, quota=booking["quota"])
        return self.runtime("confirmation_predictor").predict([rec], source=source, explain=explain)[0]

    def confirmation_trends(self, train_number: str, travel_class: str, n: int = 3000) -> dict:
        """Historical booking and cancellation trends for one train and class.

        Analyses the twin's booking history for this train (a reproducible sample of past waitlisted/RAC
        bookings with their outcome at chart time, and a year of daily seat-inventory snapshots) under the
        baseline world: how far waitlists usually clear, how outcomes depend on booking lead time and season,
        how fast seats sell, and the cancellation statistic the operator reports per month."""
        if train_number not in self.network.trains:
            raise KeyError(train_number)
        tr = self.network.trains[train_number]
        if travel_class not in tr.classes or travel_class not in CONFIRMATION_CLASSES:
            raise ValueError(f"{train_number} has no {travel_class} covered by the confirmation model "
                             f"(its waitlisted classes: {', '.join(c for c in tr.classes if c in CONFIRMATION_CLASSES)})")
        rng = np.random.default_rng(stable_hash("trends", train_number, travel_class))
        hist = confirmation_samples(self.network, n, rng, WorldParams(), only_train=train_number,
                                    only_class=travel_class)
        hist["confirmed"] = hist["final_status"] == "CNF"
        hist["at_least_rac"] = hist["final_status"].isin(["CNF", "RAC"])

        def rates(frame, column, bins, labels):
            cut = pd.cut(frame[column], bins=bins, labels=labels, right=True)
            out = []
            for label in labels:
                part = frame[cut == label]
                if len(part) >= 15:
                    out.append({"band": label, "bookings": int(len(part)),
                                "confirmed_rate": round(float(part["confirmed"].mean()), 3),
                                "at_least_rac_rate": round(float(part["at_least_rac"].mean()), 3)})
            return out

        wl = hist[hist["booking_status"] == "WL"]
        by_position = rates(wl, "queue_position", [0, 10, 25, 50, 100, 200, 10_000],
                            ["WL 1-10", "WL 11-25", "WL 26-50", "WL 51-100", "WL 101-200", "WL 200+"])
        confirmed_bands = [b["band"] for b in by_position if b["confirmed_rate"] >= 0.5]
        rac_bands = [b["band"] for b in by_position if b["at_least_rac_rate"] >= 0.5]
        by_lead = rates(hist, "days_to_departure", [-1, 2, 7, 15, 30, 60],
                        ["0-2 days", "3-7 days", "8-15 days", "16-30 days", "31-60 days"])
        by_month = []
        for m in range(1, 13):
            part = hist[hist["month"] == m]
            by_month.append({"month": m, "bookings": int(len(part)),
                             "confirmed_rate": round(float(part["confirmed"].mean()), 3) if len(part) >= 15 else None,
                             "reported_cancellation_rate": inv.hist_cancel_rate(tr, travel_class,
                                                                                season_of_month(m))})
        # Booking curve: share of seats sold by days before departure, from a year of inventory snapshots.
        seg = Segment(tr, 0, len(tr.stops) - 1)
        leads = [60, 45, 30, 21, 14, 10, 7, 5, 3, 2, 1, 0]
        curves: dict[str, list[list[float]]] = {}
        start = date(REFERENCE_YEAR, 1, 1)
        for week in range(52):
            jdate = start + timedelta(days=7 * week + int(rng.integers(0, 7)))
            season = season_of(jdate)
            occ = [inv.snapshot(self.network, seg, travel_class, jdate, jdate - timedelta(days=lead),
                                WorldParams())["occupancy"] for lead in leads]
            curves.setdefault(season, []).append(occ)
        # demand_vs_capacity above 1.0 means more requests than berths (the excess is waitlisted)
        booking_curve = [{"days_before_departure": lead, "season": season,
                          "demand_vs_capacity": round(float(np.mean([c[i] for c in rows])), 3)}
                         for season, rows in sorted(curves.items()) for i, lead in enumerate(leads)]
        return {"train_number": train_number, "train_name": tr.name, "travel_class": travel_class,
                "history_bookings": int(len(hist)), "waitlisted_bookings": int(len(wl)),
                "overall": {"confirmed_rate": round(float(hist["confirmed"].mean()), 3),
                            "at_least_rac_rate": round(float(hist["at_least_rac"].mean()), 3),
                            "rac_start_confirmed_rate": round(float(hist[hist["booking_status"] == "RAC"]["confirmed"]
                                                                    .mean()), 3)
                            if (hist["booking_status"] == "RAC").sum() >= 15 else None},
                # deepest starting-position band that still ends confirmed (or at least RAC) at least half the time
                "usually_confirmed_up_to": confirmed_bands[-1] if confirmed_bands else None,
                "usually_rac_or_better_up_to": rac_bands[-1] if rac_bands else None,
                "by_waitlist_position": by_position, "by_booking_lead": by_lead, "by_month": by_month,
                "booking_curve": booking_curve,
                "data_note": "Twin history for this train and class under baseline conditions (simulated, "
                             "reproducible); not Indian Railways records."}

    def predict_train_delay(self, train_number: str, station: str, run_date: date, explain: bool = True) -> dict:
        tr = self.network.trains[train_number]
        j = tr.index_of(station)
        if j is None:
            raise ValueError(f"{train_number} does not stop at {station}")
        world = self.world.effective_params()
        now = datetime.now(IST).replace(tzinfo=None)
        status = live.running_status(self.network, tr, run_date, now, world)
        known_idx = status.get("last_index") if status["started"] and status["last_index"] < j else None
        arrival_day = (datetime.combine(run_date, datetime.min.time())
                       + timedelta(minutes=tr.stops[j].arr_min or tr.stops[j].dep_min)).date()
        weather = live.weather_feed(self.network.stations[station].region, arrival_day, world)
        feats = delay_features(self.network, tr, j, known_idx, status["delay_min"] if known_idx is not None else 0.0,
                               run_date.month, run_date.weekday(), weather)
        res = self.runtime("delay_predictor").predict([feats], source="api", explain=explain)[0]
        stop = tr.stops[j]
        sched = datetime.combine(run_date, datetime.min.time()) + timedelta(minutes=stop.arr_min or stop.dep_min)
        res["context"] = {"train": tr.to_dict()["name"], "train_number": tr.number, "station": station,
                          "scheduled_arrival": sched.isoformat(timespec="minutes"), "running_status": status,
                          "weather_severity": weather, "features": feats}
        if res["status"] == "ok":
            res["context"]["expected_arrival"] = (sched + timedelta(minutes=res["output"]["arrival_delay_min"])
                                                  ).isoformat(timespec="minutes")
        return res

    def station_congestion(self, code: str, day: date) -> dict:
        """Expected trains on platforms per hour, using predicted delays to shift arrivals."""
        if code not in self.network.stations:
            raise KeyError(code)
        st = self.network.stations[code]
        world = self.world.effective_params()
        entries, records = [], []
        for tr in self.network.trains_at(code):
            j = tr.index_of(code)
            stop = tr.stops[j]
            t = stop.arr_min if stop.arr_min is not None else stop.dep_min
            run_date = day - timedelta(days=int(t // 1440))
            weather = live.weather_feed(st.region, day, world)
            records.append(delay_features(self.network, tr, j, None, 0.0, run_date.month, run_date.weekday(),
                                          weather) if j > 0 else None)
            entries.append((tr, t % 1440, j))
        valid = [r for r in records if r is not None]
        preds = iter(self.runtime("delay_predictor").predict(valid, source="agent")) if valid else iter([])
        hours = np.zeros(24)
        disruptions = []
        for (tr, minute, j), rec in zip(entries, records):
            delay = 0.0
            if rec is not None:
                res = next(preds)
                delay = res["output"]["arrival_delay_min"] if res["status"] == "ok" else 0.0
                if delay >= 60:
                    disruptions.append({"train_number": tr.number, "train_name": tr.name,
                                        "scheduled": f"{int(minute // 60):02d}:{int(minute % 60):02d}",
                                        "predicted_delay_min": delay})
            occupancy_min = 20 if j in (0, len(tr.stops) - 1) else 10
            start = (minute + delay) % 1440
            for m in range(int(start), int(start + occupancy_min)):
                hours[(m // 60) % 24] += 1 / 60
        load = hours / st.platforms
        levels = ["low" if x < 0.35 else "moderate" if x < 0.7 else "high" for x in load]
        return {"station": self.network.station_dict(code), "date": day.isoformat(),
                "hourly_platform_utilisation": [round(float(x), 3) for x in load], "hourly_level": levels,
                "peak_hour": int(np.argmax(load)), "peak_level": levels[int(np.argmax(load))],
                "trains": len(entries), "predicted_disruptions": sorted(disruptions,
                                                                        key=lambda d: -d["predicted_delay_min"])}

    def journey_alerts(self, pnr: str) -> dict:
        """Proactive alerts for a booking: late arrival, low confirmation chance, and alternatives."""
        b = self.bookings.get(pnr)
        if not b:
            raise KeyError(pnr)
        alerts = []
        seg = self.segment_for(b["train_number"], b["origin"], b["destination"])
        dep = datetime.fromisoformat(b["departure"])
        run_date = seg.origin_run_date(dep.date())
        delay = self.predict_train_delay(b["train_number"], b["destination"], run_date, explain=False)
        if delay["status"] == "ok" and delay["output"]["arrival_delay_min"] >= 60:
            alerts.append({"type": "late_arrival", "severity": "high",
                           "message": f"Expected to arrive about {delay['output']['arrival_delay_min']:.0f} minutes "
                                      f"late at {self.network.stations[b['destination']].name}."})
        conf = self.pnr_confirmation(b, source="agent") if b["current_status"] != "CANCELLED" else None
        if conf and conf["status"] == "ok" and conf["output"]["p_confirm"] < 0.35:
            alerts.append({"type": "low_confirmation", "severity": "high",
                           "message": f"Only {conf['output']['p_confirm'] * 100:.0f}% chance this ticket is confirmed "
                                      "by chart preparation."})
        alternatives = []
        if alerts:
            its = self.planner.tools.search([b["origin"]], [b["destination"]], dep.date())
            its = [i for i in its if i["legs"][0]["train_number"] != b["train_number"]][:15]
            opts = self.planner.tools.assess(its, today_ist(), classes=[b["travel_class"]], source="agent")
            opts.sort(key=lambda o: (-o["confirm_prob"], o["expected_delay_min"], o["fare"]))
            alternatives = [{k: o[k] for k in ("summary", "travel_class", "fare", "departure", "arrival",
                                               "availability", "confirm_prob", "expected_delay_min")}
                            for o in opts[:3]]
        return {"pnr": pnr, "booking": b, "alerts": alerts, "delay_prediction": delay,
                "confirmation_prediction": conf, "alternatives": alternatives}

    def fare_forecast(self, train_number: str, origin: str, destination: str, travel_class: str, journey_date: date,
                      explain: bool = True, quoted_fare: float | None = None) -> dict:
        seg = self.segment_for(train_number, origin, destination)
        if travel_class not in seg.train.classes:
            raise ValueError(f"{train_number} has no {travel_class}")
        today = today_ist()
        snap = inv.snapshot(self.network, seg, travel_class, journey_date, today, self.world.effective_params())
        lead = snap["days_to_departure"]
        if lead < 1 or snap["status"] in ("NOT_OPEN", "DEPARTED"):
            raise ValueError("forecasts need a journey date 1 to 60 days ahead")
        base = self.network.fare(seg.train, travel_class, seg.distance_km)
        region = self.network.stations[seg.origin].region
        curve_records = [fare_record(snap, base, seg.train.dynamic_fare, t, region, quoted_fare)
                         for t in range(lead - 1, -1, -1)]
        res = self.runtime("fare_forecaster").predict(curve_records, source="api")
        headline_idx = max(0, len(curve_records) - 1 - max(0, lead - 7)) if lead > 7 else len(curve_records) - 1
        headline = self.runtime("fare_forecaster").predict([curve_records[min(headline_idx, len(curve_records) - 1)]],
                                                           source="api", explain=explain)[0]
        curve = [{"days_before_departure": r["target_lead"], **(x["output"] if x["status"] == "ok" else {})}
                 for r, x in zip(curve_records, res)]
        ok = [c for c in curve if "fare_forecast" in c]
        now_fare = ok[0]["fare_now"] if ok else None
        cheapest = min(ok, key=lambda c: c["fare_forecast"]) if ok else None
        sellout_soon = next((c for c in ok if c["sellout_probability"] >= 0.5), None)
        if not ok:
            advice = "Forecast blocked by data-quality checks; showing no recommendation."
        elif snap["status"] in ("WL", "RAC"):
            advice = (f"Already {inv.availability_label(snap)}: this class is sold out, so waiting will not help. "
                      "Check the confirmation chance or the cheaper options below.")
        elif sellout_soon and sellout_soon["days_before_departure"] >= lead - 7:
            advice = (f"Book now: {sellout_soon['sellout_probability'] * 100:.0f}% chance this class sells out by "
                      f"{sellout_soon['days_before_departure']} days before departure.")
        elif seg.train.dynamic_fare and ok[-1]["fare_forecast"] > now_fare * 1.03:
            advice = (f"Book now: the fare is forecast to rise from Rs {now_fare:,} to about "
                      f"Rs {ok[-1]['fare_forecast']:,} by departure.")
        elif not seg.train.dynamic_fare:
            advice = "Fixed-fare train: price will not change, so book when the sell-out risk starts rising."
        else:
            advice = f"Low urgency: fare likely stays near Rs {cheapest['fare_forecast']:,} for now."
        savings = []
        for alt in self.network.direct_segments([origin], [destination]):
            if alt.train.number == train_number or travel_class not in alt.train.classes:
                continue
            s2 = inv.snapshot(self.network, alt, travel_class, journey_date, today, self.world.effective_params())
            f2 = self.network.fare(alt.train, travel_class, alt.distance_km) * (
                inv.flexi_multiplier(s2["occupancy"]) if alt.train.dynamic_fare else 1)
            if now_fare and f2 < now_fare * 0.95 and s2["status"] in ("AVAILABLE", "RAC"):
                savings.append({"train_number": alt.train.number, "train_name": alt.train.name, "fare": round(f2),
                                "saving": round(now_fare - f2), "availability": inv.availability_label(s2)})
        savings.sort(key=lambda s: -s["saving"])
        demand = None
        if ok:
            # Demand surge: forecast demand at departure against this train's usual level for the same
            # weekday in a normal season (the twin's long-run expectation for this route and class).
            typical = inv.expected_final_ratio(inv.route_demand(seg.train, seg.from_idx, seg.to_idx), "normal",
                                               journey_date.weekday(), travel_class, WorldParams())
            final = ok[-1]["forecast_occupancy"]
            index = final / typical if typical else None
            surge = bool(index is not None and index >= 1.2)
            week = next((c for c in ok if c["days_before_departure"] <= max(0, lead - 7)), ok[-1])
            seats_next_week = max(0, round((week["forecast_occupancy"] - snap["occupancy"]) * snap["capacity"]))
            weekday = journey_date.strftime("%A")
            demand = {"forecast_demand_at_departure": round(final, 3), "usual_demand": round(typical, 3),
                      "demand_index": round(index, 2) if index else None, "surge": surge,
                      "seats_booked_next_7_days": seats_next_week,
                      "message": (f"Demand surge: forecast demand at departure is {final:.0%} of seats, "
                                  f"{index:.1f}x this train's usual level for a {weekday} departure."
                                  if surge else
                                  f"Demand is in the usual range: forecast {final:.0%} of seats at departure "
                                  f"vs a usual {typical:.0%} for a {weekday}.")}
        return {"train": seg.train.to_dict()["name"], "train_number": train_number, "travel_class": travel_class,
                "journey_date": journey_date.isoformat(), "days_to_departure": lead, "dynamic_fare": seg.train.dynamic_fare,
                "base_fare": base, "availability": inv.availability_label(snap), "curve": curve,
                "headline": headline, "advice": advice, "demand_outlook": demand,
                "cost_saving_options": savings[:3]}
