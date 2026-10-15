"""Mock reservation system of the twin: PNRs, statuses and refunds.

PNRs created here are demo records in the local database. Nothing is booked
on any real railway system.
"""
from __future__ import annotations

import time
from datetime import date, datetime, timedelta

import numpy as np

from railsentinel.agents.refunds import cancellation_refund
from railsentinel.config import IST
from railsentinel.db import Database
from railsentinel.twin import inventory as inv
from railsentinel.twin.network import Network, Segment
from railsentinel.twin.world import WorldParams

COACH_PREFIX = {"SL": "S", "3A": "B", "2A": "A", "1A": "H", "CC": "C", "2S": "D"}
COACH_COUNT = {"SL": 8, "3A": 6, "2A": 4, "1A": 1, "CC": 6, "2S": 10}
BERTHS = {"SL": 72, "3A": 64, "2A": 46, "1A": 24, "CC": 73, "2S": 100}

# Refund pipeline timing in the twin (seconds after the request) so a demo can watch it move.
REFUND_STAGES = [(0, "initiated"), (90, "processing"), (300, "credited to original payment method")]


def new_pnr(rng: np.random.Generator, db: Database) -> str:
    while True:
        pnr = str(int(rng.integers(2, 9))) + "".join(str(int(d)) for d in rng.integers(0, 10, size=9))
        if not db.query_one("SELECT pnr FROM bookings WHERE pnr=?", (pnr,)):
            return pnr


def berth_for(rng: np.random.Generator, travel_class: str) -> str:
    coach = f"{COACH_PREFIX[travel_class]}{int(rng.integers(1, COACH_COUNT[travel_class] + 1))}"
    return f"{coach}/{int(rng.integers(1, BERTHS[travel_class] + 1))}"


class BookingStore:
    def __init__(self, db: Database, network: Network):
        self.db = db
        self.network = network

    def get(self, pnr: str) -> dict | None:
        return self.db.query_one("SELECT * FROM bookings WHERE pnr=?", (pnr,))

    def list(self, limit: int = 50) -> list[dict]:
        return self.db.query("SELECT * FROM bookings ORDER BY created_ts DESC LIMIT ?", (limit,))

    def create(self, segment: Segment, travel_class: str, journey_date: date, today: date, world: WorldParams,
               passenger_name: str, passengers: int, rng: np.random.Generator, source: str = "agent",
               quota: str = "GN") -> dict:
        snap = inv.snapshot(self.network, segment, travel_class, journey_date, today, world)
        if snap["status"] == "AVAILABLE":
            status = "CNF"
            detail = berth_for(rng, travel_class)
        elif snap["status"] in ("RAC", "WL"):
            status = f"{snap['status']} {snap['position']}"
            detail = f"{'GNWL' if snap['status'] == 'WL' else 'RAC'} {snap['position']}"
        else:
            raise ValueError(f"booking not possible: {snap['status']}")
        base = self.network.fare(segment.train, travel_class, segment.distance_km)
        fare = base * (inv.flexi_multiplier(snap["occupancy"]) if segment.train.dynamic_fare else 1.0)
        pnr = new_pnr(rng, self.db)
        row = {
            "pnr": pnr, "created_ts": time.time(), "passenger_name": passenger_name, "passengers": passengers,
            "train_number": segment.train.number, "train_name": segment.train.name,
            "travel_class": travel_class, "quota": quota, "origin": segment.origin,
            "destination": segment.destination, "journey_date": journey_date.isoformat(),
            "departure": segment.departure_dt(journey_date).isoformat(timespec="minutes"),
            "arrival": segment.arrival_dt(journey_date).isoformat(timespec="minutes"),
            "fare": round(fare) * passengers, "booking_status": status, "current_status": status,
            "status_detail": detail, "source": source,
        }
        self.db.execute(f"INSERT INTO bookings({','.join(row)}) VALUES({','.join('?' * len(row))})",
                        tuple(row.values()))
        return row

    def seed_samples(self, rng: np.random.Generator, today: date, world: WorldParams, n: int = 30) -> list[str]:
        """Create demo PNRs with a mix of confirmed, RAC and waitlisted tickets."""
        pnrs = []
        trains = [t for t in self.network.trains.values() if any(c in ("SL", "3A", "2A") for c in t.classes)]
        names = ["A. Sharma", "R. Iyer", "S. Khan", "P. Das", "M. Patel", "K. Reddy", "N. Singh", "J. Fernandes"]
        attempts = 0
        while len(pnrs) < n and attempts < n * 20:
            attempts += 1
            tr = trains[int(rng.integers(0, len(trains)))]
            seg = Segment(tr, 0, len(tr.stops) - 1)
            cls = str(rng.choice([c for c in tr.classes if c in ("SL", "3A", "2A")]))
            jdate = today + timedelta(days=int(rng.integers(1, 40)))
            snap = inv.snapshot(self.network, seg, cls, jdate, today, world)
            want_wl = len(pnrs) % 3 != 0
            if want_wl and snap["status"] == "AVAILABLE":
                continue
            row = self.create(seg, cls, jdate, today, world, names[len(pnrs) % len(names)],
                              int(rng.integers(1, 4)), rng, source="seed")
            pnrs.append(row["pnr"])
        return pnrs

    def cancel(self, pnr: str, now: datetime | None = None) -> dict:
        b = self.get(pnr)
        if not b:
            raise KeyError(pnr)
        if b["current_status"] == "CANCELLED":
            return {"pnr": pnr, "already_cancelled": True, **(self.refund_status(pnr) or {})}
        now = now or datetime.now(IST).replace(tzinfo=None)
        dep = datetime.fromisoformat(b["departure"])
        hours = (dep - now).total_seconds() / 3600
        status = b["current_status"].split()[0]
        per_pax = b["fare"] / max(1, b["passengers"])
        calc = cancellation_refund(per_pax, b["travel_class"], status, hours, quota=b["quota"])
        refund = calc["refund"] * b["passengers"]
        charges = calc["charges"] * b["passengers"]
        self.db.execute("UPDATE bookings SET current_status='CANCELLED' WHERE pnr=?", (pnr,))
        self.db.execute("INSERT OR REPLACE INTO refunds(pnr, requested_ts, amount, charges, rule, status, updated_ts) "
                        "VALUES(?,?,?,?,?,?,?)", (pnr, time.time(), refund, charges, calc["rule"], "initiated",
                                                  time.time()))
        return {"pnr": pnr, "refund": refund, "charges": charges, "rule": calc["rule"], "source": calc["source"]}

    def refund_status(self, pnr: str) -> dict | None:
        r = self.db.query_one("SELECT * FROM refunds WHERE pnr=?", (pnr,))
        if not r:
            return None
        elapsed = time.time() - r["requested_ts"]
        stage = REFUND_STAGES[0][1]
        for secs, name in REFUND_STAGES:
            if elapsed >= secs:
                stage = name
        if stage != r["status"]:
            self.db.execute("UPDATE refunds SET status=?, updated_ts=? WHERE pnr=?", (stage, time.time(), pnr))
        return {"pnr": pnr, "amount": r["amount"], "charges": r["charges"], "rule": r["rule"], "status": stage,
                "requested_at": datetime.fromtimestamp(r["requested_ts"], IST).isoformat(timespec="seconds")}
