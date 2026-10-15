"""Seat inventory model of the twin: demand, booking curves, availability and
cancellation propensity. Shared by the dataset generators (training data) and
the live services (current availability), so both see the same world.
"""
from __future__ import annotations

import math
from datetime import date

import numpy as np

from railsentinel.security import stable_hash
from railsentinel.twin.network import CATEGORIES, MAJOR_HUBS, TRAVEL_CLASSES, Network, Segment, Train
from railsentinel.twin.seasons import BOOKING_SHAPE, CANCEL_MULT, DEMAND_MULT, season_of
from railsentinel.twin.world import WorldParams

ARP_DAYS = 60  # twin's advance reservation period

CLASS_DEMAND = {"2S": 1.05, "SL": 1.15, "CC": 0.95, "3A": 1.05, "2A": 0.85, "1A": 0.7}
CLASS_CANCEL = {"2S": 0.10, "SL": 0.12, "CC": 0.08, "3A": 0.10, "2A": 0.11, "1A": 0.09}
CATEGORY_CANCEL = {"premium": 0.75, "superfast": 1.0, "express": 1.1, "intercity": 0.9}


def route_demand(train: Train, i: int, j: int) -> float:
    """Popularity index of an origin-destination pair on a train (0.35 - 1.15)."""
    h = stable_hash("demand", train.corridor, train.stops[i].code, train.stops[j].code)
    value = 0.35 + (h % 1000) / 1000 * 0.6
    if train.stops[j].code in MAJOR_HUBS:
        value += 0.12
    if train.stops[i].code in MAJOR_HUBS:
        value += 0.05
    if train.category == "premium":
        value += 0.05
    return round(min(value, 1.15), 4)


def weekday_factor(weekday: int) -> float:
    # Friday and Sunday departures are busiest, Tuesday/Wednesday quietest.
    return {4: 1.12, 6: 1.12, 1: 0.92, 2: 0.92}.get(weekday, 1.0)


def expected_final_ratio(rd: float, season: str, weekday: int, travel_class: str,
                         world: WorldParams) -> float:
    """Expected final demand as a share of capacity (can exceed 1: waitlist)."""
    return rd * DEMAND_MULT[season] * weekday_factor(weekday) * CLASS_DEMAND[travel_class] * world.demand_mult


def occupancy_at(final_ratio: float, lead_days: float, shape: float) -> float:
    lead = min(max(lead_days, 0.0), ARP_DAYS)
    return final_ratio * (1.0 - (lead / ARP_DAYS) ** shape)


def flexi_multiplier(occupancy: float) -> float:
    """Flexi-style pricing used by the twin for premium trains: +10% for every
    10% of berths sold beyond the first 10%, capped at 1.5x."""
    sold = min(max(occupancy, 0.0), 1.0)
    return 1.0 + 0.1 * min(5, math.floor(sold * 10))


def base_cancel_rate(travel_class: str, category: str, season: str) -> float:
    return CLASS_CANCEL[travel_class] * CATEGORY_CANCEL[category] * CANCEL_MULT[season]


def hist_cancel_rate(train: Train, travel_class: str, season: str) -> float:
    """What the operator's 90-day rolling statistic reports. It is computed from
    past behaviour, so it does not react to a sudden change in the world."""
    noise = 0.9 + (stable_hash("hist", train.number, travel_class) % 200) / 1000
    return round(base_cancel_rate(travel_class, train.category, season) * noise, 4)


def snapshot(network: Network, segment: Segment, travel_class: str, journey_date: date, today: date,
             world: WorldParams) -> dict:
    """Current availability for a segment/class on a date, as the booking system would show it."""
    train = segment.train
    lead = (journey_date - today).days
    cls = TRAVEL_CLASSES[travel_class]
    capacity = cls["capacity"]
    rac_cap = int(round(capacity * cls["rac_share"]))
    season = season_of(journey_date)
    rd = route_demand(train, segment.from_idx, segment.to_idx)
    base = {"train_number": train.number, "travel_class": travel_class, "journey_date": journey_date.isoformat(),
            "days_to_departure": lead, "season": season, "route_demand": rd, "capacity": capacity,
            "rac_capacity": rac_cap, "hist_cancel_rate": hist_cancel_rate(train, travel_class, season),
            "distance_km": round(segment.distance_km, 1), "train_category": train.category,
            "weekday": journey_date.weekday(), "month": journey_date.month}
    if lead < 0:
        return {**base, "status": "DEPARTED", "position": 0, "available": 0, "occupancy": 1.0}
    if lead > ARP_DAYS:
        return {**base, "status": "NOT_OPEN", "position": 0, "available": capacity, "occupancy": 0.0}
    rng = np.random.default_rng(stable_hash("inv", train.number, travel_class, journey_date.isoformat(),
                                            segment.from_idx, segment.to_idx))
    final = expected_final_ratio(rd, season, journey_date.weekday(), travel_class, world)
    final *= float(rng.lognormal(0.0, 0.18))
    occ = occupancy_at(final, lead, BOOKING_SHAPE[season])
    booked = occ * capacity
    if booked < capacity:
        # a partly free berth still counts as one available (never show AVAILABLE-0)
        status, position, available = "AVAILABLE", 0, int(math.ceil(capacity - booked))
    elif booked < capacity + rac_cap:
        status, position, available = "RAC", int(booked - capacity) + 1, 0
    else:
        status, position, available = "WL", int(booked - capacity - rac_cap) + 1, 0
    return {**base, "status": status, "position": position, "available": available, "occupancy": round(occ, 4)}


def availability_label(snap: dict) -> str:
    if snap["status"] == "AVAILABLE":
        return f"AVAILABLE-{snap['available']}"
    if snap["status"] in ("RAC", "WL"):
        return f"{snap['status']} {snap['position']}"
    return snap["status"]


def punctuality_index(train: Train) -> float:
    """Share of runs arriving within 15 minutes, as published by the twin's history (0-1)."""
    base = {"premium": 0.86, "superfast": 0.72, "express": 0.58, "intercity": 0.8}[train.category]
    jitter = ((stable_hash("punct", train.number) % 100) - 50) / 1000
    return round(min(max(base + jitter - train.total_km / 40000, 0.3), 0.97), 3)


def category_delay_mult(category: str) -> float:
    return CATEGORIES[category]["delay_mult"]
