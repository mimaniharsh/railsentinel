"""Synthetic data generators of the digital twin.

Each generator draws records from a documented stochastic process that depends
on the current `WorldParams`. With default parameters they produce the
"baseline" world used for initial training. Chaos scenarios change the
parameters, which changes the data the live services receive (data drift) and,
for some scenarios, the relationship between inputs and outcomes (concept drift).

Nothing here is real passenger data. The processes are designed to have the
structure of the real problem (waitlists clear through cancellations, delays
accumulate along a route, bots act faster than people) so the models and the
governance layer have something meaningful to learn and to monitor.
"""
from __future__ import annotations

import math
from datetime import date, datetime, timedelta

import numpy as np
import pandas as pd

from railsentinel.config import IST
from railsentinel.twin import inventory as inv
from railsentinel.twin.network import TRAVEL_CLASSES, Network, Segment, Train
from railsentinel.twin.seasons import BOOKING_SHAPE, season_of, weather_mean
from railsentinel.twin.world import WorldParams

REFERENCE_YEAR = 2026


def today_ist() -> date:
    return datetime.now(IST).date()


# ---------------------------------------------------------------- shared helpers
def sample_segment(network: Network, rng: np.random.Generator, min_stops: int = 1,
                   trains: list[Train] | None = None) -> Segment:
    pool = trains or list(network.trains.values())
    train = pool[int(rng.integers(0, len(pool)))]
    n = len(train.stops)
    i = int(rng.integers(0, max(1, n - min_stops)))
    j = int(rng.integers(i + min_stops, n)) if i + min_stops < n else n - 1
    if rng.random() < 0.45:  # many journeys are end-to-end
        i, j = 0, n - 1
    return Segment(train, i, j)


def sample_journey_date(rng: np.random.Generator, world: WorldParams) -> date:
    if world.festival_share > 0 and rng.random() < world.festival_share:
        start = date(REFERENCE_YEAR, 10, 12)
        return start + timedelta(days=int(rng.integers(0, 32)))
    return date(REFERENCE_YEAR, 1, 1) + timedelta(days=int(rng.integers(0, 365)))


def sample_weather(rng: np.random.Generator, region: str, month: int, world: WorldParams) -> float:
    mean = weather_mean(region, month)
    if region in world.weather_regions:
        mean += world.weather_shift
    mean = float(np.clip(mean, 0.03, 0.95))
    k = 9.0
    return float(rng.beta(mean * k, (1 - mean) * k))


# ---------------------------------------------------------------- confirmation
CONFIRMATION_CLASSES = ("SL", "CC", "3A", "2A", "1A")


def confirmation_samples(network: Network, n: int, rng: np.random.Generator, world: WorldParams,
                         only_train: str | None = None, only_class: str | None = None) -> pd.DataFrame:
    """Waitlisted / RAC bookings and how they ended at chart preparation.

    `only_train` / `only_class` restrict the history to one train and class (used for trend analysis)."""
    trains = [t for t in network.trains.values() if any(c in CONFIRMATION_CLASSES for c in t.classes)
              and (only_train is None or t.number == only_train)]
    if not trains:
        raise ValueError(f"no train {only_train} with a waitlisted class")
    rows = []
    for _ in range(n):
        seg = sample_segment(network, rng, trains=trains)
        train = seg.train
        allowed = [c for c in train.classes if c in CONFIRMATION_CLASSES and (only_class is None or c == only_class)]
        if not allowed:
            raise ValueError(f"train {train.number} has no class {only_class} covered by the confirmation model")
        travel_class = str(rng.choice(allowed))
        jdate = sample_journey_date(rng, world)
        season = season_of(jdate)
        quota = "TQ" if rng.random() < 0.15 else "GN"
        lead = 1 if quota == "TQ" else 1 + int(rng.beta(1.2, 1.8) * (inv.ARP_DAYS - 1))
        cls = TRAVEL_CLASSES[travel_class]
        capacity = cls["capacity"]
        rac_cap = int(round(capacity * cls["rac_share"]))
        rd = inv.route_demand(train, seg.from_idx, seg.to_idx)
        final = inv.expected_final_ratio(rd, season, jdate.weekday(), travel_class, world) * rng.lognormal(0, 0.2)
        occ_now = inv.occupancy_at(final, lead, BOOKING_SHAPE[season]) * capacity
        if occ_now < capacity:
            # Condition on the booking being RAC/WL: excess demand grows with route popularity.
            occ_now = capacity + rng.gamma(1.3, 0.05 * capacity * max(final, 0.3) * world.demand_mult)
        k = int(occ_now - capacity) + 1
        if rac_cap and k <= rac_cap:
            status, position = "RAC", k
        else:
            status, position = "WL", max(1, k - rac_cap)

        lead_factor = 1.0 - math.exp(-lead / 10.0) + 0.15
        q = (inv.base_cancel_rate(travel_class, train.category, season) * lead_factor
             * (0.35 if quota == "TQ" else 1.0) * inv.weekday_factor(jdate.weekday())
             * rng.lognormal(0, 0.25) * world.cancel_mult)
        q = float(np.clip(q, 0.004, 0.6))
        q_wl = min(0.9, q * 1.8)
        freed = rng.binomial(capacity, q)
        if status == "RAC":
            ahead = 1 + rng.binomial(position - 1, 1 - q) if position > 1 else 1
            final_status = "CNF" if freed >= ahead else "RAC"
        else:
            rac_left = rng.binomial(rac_cap, 1 - q) if rac_cap else 0
            w_eff = 1 + rng.binomial(position - 1, 1 - q_wl) if position > 1 else 1
            remaining = rac_left + w_eff - freed
            if remaining <= 0:
                final_status = "CNF"
            elif rac_cap and remaining <= rac_cap:
                final_status = "RAC"
            else:
                final_status = "WL"
        rows.append({
            "train_category": train.category, "travel_class": travel_class, "quota": quota,
            "booking_status": status, "season": season, "queue_position": int(position),
            "days_to_departure": int(lead), "distance_km": round(seg.distance_km, 1), "route_demand": rd,
            "hist_cancel_rate": inv.hist_cancel_rate(train, travel_class, season),
            "weekday": jdate.weekday(), "month": jdate.month, "capacity": capacity,
            "region": network.stations[seg.origin].region, "train_number": train.number,
            "final_status": final_status,
        })
    return pd.DataFrame(rows)


# ---------------------------------------------------------------- delays
def simulate_run(network: Network, train: Train, rng: np.random.Generator, month: int,
                 world: WorldParams) -> tuple[np.ndarray, float]:
    """Arrival delay (minutes) at every stop of one run, plus the weather severity."""
    region = network.stations[train.stops[len(train.stops) // 2].code].region
    w = sample_weather(rng, region, month, world)
    mult = inv.category_delay_mult(train.category)
    eff = world.weather_effect_mult if region in world.weather_regions else 1.0
    delays = np.zeros(len(train.stops))
    d = rng.exponential(4.0 * mult) + 15.0 * w * eff * rng.random()
    delays[0] = d
    for k in range(1, len(train.stops)):
        seg_km = train.stops[k].km - train.stops[k - 1].km
        stop = train.stops[k]
        t = stop.arr_min if stop.arr_min is not None else stop.dep_min
        load = network.scheduled_load(stop.code, (t // 60) % 24)
        mean_add = 0.022 * seg_km * mult * (1 + 2.2 * w * eff) * (1 + 0.5 * load)
        add = rng.gamma(1.4, mean_add / 1.4)
        recovery = 0.010 * seg_km * (1.6 if train.category == "premium" else 1.0)
        d = max(0.0, d + add - recovery)
        delays[k] = d
    return delays, w


def hist_mean_delay(train: Train) -> float:
    """Long-run mean arrival delay at destination reported for a train (twin history)."""
    return round(0.012 * train.total_km * inv.category_delay_mult(train.category) + 4, 1)


def delay_features(network: Network, train: Train, target_idx: int, known_idx: int | None, known_delay: float,
                   month: int, weekday: int, weather: float) -> dict:
    stop = train.stops[target_idx]
    t = stop.arr_min if stop.arr_min is not None else stop.dep_min
    started = known_idx is not None
    from_km = train.stops[known_idx].km if started else 0.0
    return {
        "train_category": train.category,
        "region": network.stations[stop.code].region,
        "month": int(month), "weekday": int(weekday),
        "sched_hour": int((t // 60) % 24),
        "stop_index": int(target_idx),
        "frac_route": round(stop.km / train.total_km, 4),
        "distance_ahead_km": round(stop.km - from_km, 1),
        "stops_ahead": int(target_idx - (known_idx if started else 0)),
        "known_delay_min": round(float(known_delay) if started else 0.0, 1),
        "journey_started": int(started),
        "weather_severity": round(float(weather), 4),
        "station_load": round(network.scheduled_load(stop.code, (t // 60) % 24), 4),
        "hist_mean_delay": hist_mean_delay(train),
    }


def delay_samples(network: Network, n: int, rng: np.random.Generator, world: WorldParams) -> pd.DataFrame:
    trains = list(network.trains.values())
    rows = []
    while len(rows) < n:
        train = trains[int(rng.integers(0, len(trains)))]
        jdate = sample_journey_date(rng, world)
        delays, w = simulate_run(network, train, rng, jdate.month, world)
        # several targets per run keeps generation fast and realistic (correlated within a run)
        for _ in range(3):
            j = int(rng.integers(1, len(train.stops)))
            started = rng.random() < 0.6
            i = int(rng.integers(0, j)) if started else None
            observed_w = float(np.clip(w + rng.normal(0, 0.03), 0, 1))
            feats = delay_features(network, train, j, i, delays[i] if started else 0.0,
                                   jdate.month, jdate.weekday(), observed_w)
            feats["train_number"] = train.number
            feats["delay_min"] = round(float(delays[j]), 1)
            rows.append(feats)
    return pd.DataFrame(rows[:n])


# ---------------------------------------------------------------- fares
def fare_samples(network: Network, n: int, rng: np.random.Generator, world: WorldParams) -> pd.DataFrame:
    rows = []
    trains = list(network.trains.values())
    for _ in range(n):
        seg = sample_segment(network, rng, trains=trains)
        train = seg.train
        travel_class = str(rng.choice(train.classes))
        jdate = sample_journey_date(rng, world)
        season = season_of(jdate)
        rd = inv.route_demand(train, seg.from_idx, seg.to_idx)
        final = inv.expected_final_ratio(rd, season, jdate.weekday(), travel_class, world) * rng.lognormal(0, 0.18)
        shape = BOOKING_SHAPE[season] * rng.lognormal(0, 0.15)
        current_lead = int(rng.integers(2, inv.ARP_DAYS + 1))
        target_lead = int(rng.integers(0, current_lead))
        cur = max(0.0, inv.occupancy_at(final, current_lead, shape) + rng.normal(0, 0.02))
        tgt = max(cur, inv.occupancy_at(final, target_lead, shape) + rng.normal(0, 0.02))
        base_fare = network.fare(train, travel_class, seg.distance_km)
        quote = base_fare * (inv.flexi_multiplier(cur) if train.dynamic_fare else 1.0)
        if world.fare_glitch_rate > 0 and rng.random() < world.fare_glitch_rate:
            quote = float(rng.choice([0.0, -base_fare, base_fare * 10, base_fare * 25]))
        rows.append({
            "train_category": train.category, "travel_class": travel_class, "season": season,
            "weekday": jdate.weekday(), "month": jdate.month, "route_demand": rd,
            "distance_km": round(seg.distance_km, 1), "dynamic_fare": int(train.dynamic_fare),
            "current_lead": current_lead, "current_occupancy": round(min(cur, 1.8), 4),
            "target_lead": target_lead, "base_fare": base_fare, "quoted_fare": round(quote, 2),
            "region": network.stations[seg.origin].region, "train_number": train.number,
            "target_occupancy": round(min(tgt, 1.8), 4),
        })
    return pd.DataFrame(rows)


# ---------------------------------------------------------------- fraud / bots
FRAUD_CLASSES = ("legit", "bot", "payment_fraud", "account_misuse")


def fraud_sessions(n: int, rng: np.random.Generator, world: WorldParams) -> pd.DataFrame:
    p_bot = world.bot_rate
    probs = np.array([max(0.0, 1 - p_bot - 0.02 - 0.012), p_bot, 0.02, 0.012])
    probs = probs / probs.sum()
    labels = rng.choice(FRAUD_CLASSES, size=n, p=probs)
    rows = []
    for label in labels:
        r = {}
        # During a Tatkal opening rush most genuine sessions are Tatkal bookings too.
        tatkal = rng.random() < (0.85 if label == "bot" else min(0.9, 0.15 + world.legit_fast_share))
        human_like = label != "bot" or world.stealth_bots
        fast_legit = label == "legit" and tatkal and rng.random() < world.legit_fast_share * 4
        if human_like and not fast_legit:
            r["requests_per_min"] = rng.lognormal(math.log(4 if label != "bot" else 6), 0.5)
            r["interarrival_cv"] = rng.uniform(0.6, 1.4)
            r["form_fill_seconds"] = rng.lognormal(math.log(35 if label != "bot" else 28), 0.4)
            r["headless_signals"] = int(rng.random() < 0.01)
        elif fast_legit:  # genuine Tatkal rush: autofill extensions and shared counter / cyber-cafe PCs
            r["requests_per_min"] = rng.lognormal(math.log(24), 0.4)
            r["interarrival_cv"] = rng.uniform(0.15, 0.5)
            r["form_fill_seconds"] = rng.lognormal(math.log(5), 0.35)
            r["headless_signals"] = 0
        else:
            r["requests_per_min"] = rng.lognormal(math.log(60), 0.4)
            r["interarrival_cv"] = rng.uniform(0.02, 0.25)
            r["form_fill_seconds"] = rng.lognormal(math.log(2), 0.5)
            r["headless_signals"] = int(rng.random() < 0.6)
        r["is_tatkal_window"] = int(tatkal)
        if tatkal:
            if label == "bot":
                ms = rng.lognormal(math.log(400 if not world.stealth_bots else 2500), 0.5)
            elif fast_legit:
                ms = rng.lognormal(math.log(3000), 0.5)
            else:
                ms = rng.lognormal(math.log(25000), 0.6)
            r["ms_after_tatkal_open"] = round(float(ms), 1)
        else:
            r["ms_after_tatkal_open"] = -1.0
        bot = label == "bot"
        loud_bot = bot and not world.stealth_bots
        stealth_bot = bot and world.stealth_bots
        r["booking_attempts"] = int(1 + rng.poisson(6 if loud_bot else (3 if bot else 1)))
        r["passengers_per_booking"] = int(rng.integers(4, 7) if loud_bot else rng.integers(1, 5))
        if label == "account_misuse":
            r["accounts_on_device"] = int(rng.integers(3, 9))
        elif loud_bot:
            r["accounts_on_device"] = int(rng.integers(5, 31))
        elif fast_legit:
            r["accounts_on_device"] = int(rng.integers(2, 7))  # shared counter / cyber-cafe machines
        else:
            r["accounts_on_device"] = int(1 + rng.binomial(2, 0.1))
        if label == "payment_fraud":
            r["accounts_on_payment_instrument"] = int(rng.integers(5, 16))
        elif loud_bot:
            r["accounts_on_payment_instrument"] = int(rng.integers(4, 21))
        else:  # stealth bots rotate fresh payment instruments
            r["accounts_on_payment_instrument"] = int(1 + rng.binomial(2, 0.15))
        age_mu = 10 if loud_bot else 300 if stealth_bot else 25 if label == "payment_fraud" else 700
        r["account_age_days"] = round(float(rng.lognormal(math.log(age_mu), 0.9)), 1)
        r["failed_payments"] = int(rng.poisson(4 if label == "payment_fraud" else 0.1))
        if label == "account_misuse":
            r["distinct_passenger_names_30d"] = int(rng.integers(25, 81))
        elif loud_bot:
            r["distinct_passenger_names_30d"] = int(rng.integers(15, 61))
        else:
            r["distinct_passenger_names_30d"] = int(rng.integers(1, 9) if not stealth_bot else rng.integers(3, 12))
        ip_mu = 20 if loud_bot else 10 if fast_legit else 1.5 if stealth_bot else 0.5
        r["ip_bookings_last_hour"] = int(1 + rng.poisson(ip_mu))
        # Realistic overlap between classes: families share one card, power users autofill
        # forms, and some bots run "low and slow" to blend in.
        if label == "legit":
            if rng.random() < 0.05:
                r["accounts_on_payment_instrument"] = int(rng.integers(3, 7))
            if rng.random() < 0.04:
                r["form_fill_seconds"] = rng.lognormal(math.log(6), 0.4)
                r["requests_per_min"] = rng.lognormal(math.log(14), 0.4)
            if rng.random() < 0.03:
                r["account_age_days"] = round(float(rng.uniform(1, 20)), 1)
        elif label == "bot" and not world.stealth_bots and rng.random() < 0.12:
            r["requests_per_min"] = rng.lognormal(math.log(8), 0.4)
            r["interarrival_cv"] = rng.uniform(0.4, 1.0)
            r["form_fill_seconds"] = rng.lognormal(math.log(15), 0.4)
            r["accounts_on_device"] = int(rng.integers(1, 4))
            r["headless_signals"] = 0
        r["requests_per_min"] = round(float(r["requests_per_min"]), 2)
        r["interarrival_cv"] = round(float(r["interarrival_cv"]), 3)
        r["form_fill_seconds"] = round(float(r["form_fill_seconds"]), 2)
        r["account_tenure"] = "new" if r["account_age_days"] < 90 else "established"
        r["label"] = str(label)
        rows.append(r)
    return pd.DataFrame(rows)


# ---------------------------------------------------------------- recommendation
USER_SEGMENTS = {
    "budget": {"share": 0.35, "w": {"price": 3.0, "duration": 0.6, "punctuality": 0.6, "confirm": 1.5,
                                    "comfort": 0.4, "time": 0.5}, "budget_classes": ("SL", "2S", "CC")},
    "business": {"share": 0.25, "w": {"price": 0.6, "duration": 2.0, "punctuality": 1.6, "confirm": 1.8,
                                      "comfort": 1.4, "time": 0.9}, "budget_classes": ("2A", "1A", "3A")},
    "family": {"share": 0.25, "w": {"price": 1.4, "duration": 0.8, "punctuality": 0.8, "confirm": 2.2,
                                    "comfort": 1.0, "time": 1.0}, "budget_classes": ("3A", "SL")},
    "senior": {"share": 0.15, "w": {"price": 1.0, "duration": 0.6, "punctuality": 1.0, "confirm": 2.0,
                                    "comfort": 1.6, "time": 1.2}, "budget_classes": ("3A", "2A")},
}


def sample_user(rng: np.random.Generator, user_id: str | None = None) -> dict:
    segs = list(USER_SEGMENTS)
    seg = str(rng.choice(segs, p=[USER_SEGMENTS[s]["share"] for s in segs]))
    spec = USER_SEGMENTS[seg]
    w = {k: float(v * rng.lognormal(0, 0.2)) for k, v in spec["w"].items()}
    night_pref = float(np.clip(rng.beta(4, 2) if seg in ("budget", "family", "business") else rng.beta(2, 4), 0, 1))
    hist_class = str(rng.choice(spec["budget_classes"]))
    fare_per_km = TRAVEL_CLASSES[hist_class]["per_km"] * float(rng.lognormal(0, 0.1))
    return {"user_id": user_id or f"U{int(rng.integers(10000, 99999))}", "segment": seg, "latent_w": w,
            "hist_class_mode": hist_class, "hist_night_share": round(night_pref, 3),
            "hist_fare_per_km": round(fare_per_km, 3), "budget_factor": float(rng.uniform(1.0, 1.5))}


def is_overnight(dep: datetime, arr: datetime) -> int:
    t = dep
    while t < arr:
        if 1 <= t.hour < 5:
            return 1
        t += timedelta(minutes=30)
    return 0


def candidate_options(network: Network, segments: list[Segment], jdate: date, today: date,
                      world: WorldParams, confirm_fn=None) -> list[dict]:
    """All bookable train/class options for an origin-destination pair on a date.

    `confirm_fn(snapshot_dict) -> probability` lets the live service plug in the
    confirmation model; the generator uses a simple heuristic instead.
    """
    out = []
    for seg in segments:
        dep = seg.departure_dt(jdate)
        arr = seg.arrival_dt(jdate)
        for travel_class in seg.train.classes:
            snap = inv.snapshot(network, seg, travel_class, jdate, today, world)
            if snap["status"] in ("DEPARTED", "NOT_OPEN"):
                continue
            if snap["status"] == "AVAILABLE":
                p = 1.0
            elif confirm_fn is not None:
                p = float(confirm_fn(snap))
            elif snap["status"] == "RAC":
                p = 0.8
            else:
                p = max(0.03, 0.78 - 0.012 * snap["position"])
            base = network.fare(seg.train, travel_class, seg.distance_km)
            fare = base * (inv.flexi_multiplier(snap["occupancy"]) if seg.train.dynamic_fare else 1.0)
            out.append({
                "option_id": f"{seg.train.number}-{travel_class}-{seg.origin}-{seg.destination}",
                "train_number": seg.train.number, "train_name": seg.train.name, "category": seg.train.category,
                "travel_class": travel_class, "origin": seg.origin, "destination": seg.destination,
                "departure": dep.isoformat(timespec="minutes"), "arrival": arr.isoformat(timespec="minutes"),
                "fare": round(fare), "base_fare": base, "duration_min": seg.duration_min,
                "dep_hour": dep.hour, "overnight": is_overnight(dep, arr),
                "punctuality": inv.punctuality_index(seg.train), "availability": inv.availability_label(snap),
                "confirm_prob": round(p, 4), "comfort": TRAVEL_CLASSES[travel_class]["comfort"],
                "is_premium": int(seg.train.category == "premium"), "distance_km": round(seg.distance_km),
                "snapshot": snap,
            })
    return out


def od_pairs(network: Network, min_direct: int = 3) -> list[tuple[str, str]]:
    counts: dict[tuple[str, str], int] = {}
    for tr in network.trains.values():
        codes = [s.code for s in tr.stops]
        for a in range(len(codes)):
            for b in range(a + 1, len(codes)):
                counts[(codes[a], codes[b])] = counts.get((codes[a], codes[b]), 0) + 1
    return sorted(k for k, v in counts.items() if v >= min_direct)


def true_utility(user: dict, option: dict, budget: float, min_duration: float) -> float:
    w = user["latent_w"]
    night_match = option["overnight"] * user["hist_night_share"] + (1 - option["overnight"]) * (1 - user["hist_night_share"])
    return (-w["price"] * option["fare"] / budget - 4.0 * (option["fare"] > budget)
            - w["duration"] * (option["duration_min"] / min_duration - 1)
            + w["punctuality"] * option["punctuality"] + w["confirm"] * option["confirm_prob"]
            + w["comfort"] * option["comfort"] / 6 + w["time"] * night_match
            + 0.8 * (option["travel_class"] == user["hist_class_mode"]))


def user_budget(user: dict, options: list[dict]) -> float:
    pref = [o["fare"] for o in options if o["travel_class"] == user["hist_class_mode"]]
    ref = float(np.median(pref)) if pref else float(np.median([o["fare"] for o in options]))
    return round(ref * user["budget_factor"], -1)


def recommendation_episodes(network: Network, n: int, rng: np.random.Generator, world: WorldParams,
                            today: date | None = None) -> list[dict]:
    pairs = od_pairs(network)
    today = today or date(REFERENCE_YEAR, 1, 1)
    episodes = []
    while len(episodes) < n:
        o, d = pairs[int(rng.integers(0, len(pairs)))]
        jdate = today + timedelta(days=int(rng.integers(1, inv.ARP_DAYS)))
        segments = network.direct_segments([o], [d])
        options = candidate_options(network, segments, jdate, today, world)
        if len(options) < 3:
            continue
        user = sample_user(rng)
        budget = user_budget(user, options)
        min_dur = min(op["duration_min"] for op in options)
        utils = [true_utility(user, op, budget, min_dur) + rng.gumbel(0, 0.6) for op in options]
        chosen = int(np.argmax(utils))
        for op in options:
            op.pop("snapshot", None)
        episodes.append({"user": {k: v for k, v in user.items() if k != "latent_w"}, "latent_w": user["latent_w"],
                         "budget": budget, "origin": o, "destination": d, "journey_date": jdate.isoformat(),
                         "options": options, "chosen_index": chosen})
    return episodes


def choose_for_user(user: dict, latent_w: dict, options: list[dict], budget: float,
                    rng: np.random.Generator) -> int:
    min_dur = min(op["duration_min"] for op in options)
    u = {**user, "latent_w": latent_w}
    utils = [true_utility(u, op, budget, min_dur) + rng.gumbel(0, 0.6) for op in options]
    return int(np.argmax(utils))
