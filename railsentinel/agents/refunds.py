"""Cancellation refund calculator.

Implements the Indian Railways refund rules for reserved tickets as published
on https://www.indianrail.gov.in/refund_Rules.html ("Revised Refund Rules
w.e.f. 12.11.2015"):

* Confirmed ticket, cancelled more than 48 h before scheduled departure:
  flat charge per passenger - Rs 240 (AC First / Executive), Rs 200 (AC 2-tier /
  First Class), Rs 180 (AC 3-tier / AC Chair Car / AC 3 Economy),
  Rs 120 (Sleeper), Rs 60 (Second Class).
* 48 h to 12 h before departure: 25% of the fare, subject to that minimum.
* 12 h to 4 h before departure: 50% of the fare, subject to that minimum.
* Within 4 h of departure: no refund.
* RAC / waitlisted ticket: refundable up to 30 minutes before scheduled
  departure, less clerkage of Rs 60 per passenger (Rs 30 for unreserved).

Confirmed Tatkal tickets are treated as non-refundable on passenger
cancellation, following the summary at
https://www.ixigo.com/travel-stories/can-a-tatkal-ticket-be-cancelled-refunded-online-story-1180562
(exceptions such as a train late by more than 3 hours are handled through TDR).

The calculator returns the rule it applied so answers can cite it. Rules can
change; the knowledge base tells passengers to confirm on IRCTC.
"""
from __future__ import annotations

OFFICIAL_SOURCE = "https://www.indianrail.gov.in/refund_Rules.html"
TATKAL_SOURCE = "https://www.ixigo.com/travel-stories/can-a-tatkal-ticket-be-cancelled-refunded-online-story-1180562"

FLAT_CHARGE = {"1A": 240, "EC": 240, "2A": 200, "FC": 200, "3A": 180, "CC": 180, "3E": 180, "SL": 120, "2S": 60}
CLERKAGE = 60
CLERKAGE_UNRESERVED = 30


def cancellation_refund(fare: float, travel_class: str, status: str, hours_before_departure: float,
                        quota: str = "GN") -> dict:
    """Refund for one passenger. `status` is CNF, RAC or WL."""
    fare = float(fare)
    status = status.upper()
    hours = float(hours_before_departure)
    if status in ("WL", "RAC"):
        if hours < 0.5:
            return {"refund": 0.0, "charges": fare, "rule": f"{status} ticket cancelled less than 30 minutes before "
                    "departure: no refund", "source": OFFICIAL_SOURCE}
        clerk = min(CLERKAGE, fare)
        return {"refund": round(fare - clerk, 2), "charges": float(clerk),
                "rule": f"{status} ticket cancelled at least 30 minutes before departure: fare minus clerkage of "
                        f"Rs {CLERKAGE} per passenger", "source": OFFICIAL_SOURCE}
    if status != "CNF":
        raise ValueError("status must be CNF, RAC or WL")
    if quota.upper() == "TQ":
        return {"refund": 0.0, "charges": fare, "rule": "Confirmed Tatkal ticket: no refund on cancellation by the "
                "passenger", "source": TATKAL_SOURCE}
    flat = FLAT_CHARGE.get(travel_class, 240)
    if hours > 48:
        charge, rule = flat, f"Confirmed ticket cancelled more than 48 hours before departure: flat Rs {flat}"
    elif hours > 12:
        charge = max(0.25 * fare, flat)
        rule = f"Confirmed ticket cancelled 48 to 12 hours before departure: 25% of fare (minimum Rs {flat})"
    elif hours > 4:
        charge = max(0.5 * fare, flat)
        rule = f"Confirmed ticket cancelled 12 to 4 hours before departure: 50% of fare (minimum Rs {flat})"
    else:
        return {"refund": 0.0, "charges": fare, "rule": "Confirmed ticket cancelled within 4 hours of departure: "
                "no refund", "source": OFFICIAL_SOURCE}
    charge = min(charge, fare)
    return {"refund": round(fare - charge, 2), "charges": round(charge, 2), "rule": rule, "source": OFFICIAL_SOURCE}
