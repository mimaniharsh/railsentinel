"""Refund rules (Indian Railways, w.e.f. 12.11.2015) and natural-language parsing."""
from datetime import date, datetime

import pytest

from railsentinel.agents.nlu import arrival_deadline, arrival_shortfall_min, parse_request
from railsentinel.agents.refunds import cancellation_refund
from railsentinel.agents.support import describe_hours
from railsentinel.twin.network import load_network


@pytest.mark.parametrize("cls,flat", [("1A", 240), ("2A", 200), ("3A", 180), ("CC", 180), ("SL", 120), ("2S", 60)])
def test_flat_charge_more_than_48_hours(cls, flat):
    r = cancellation_refund(1000, cls, "CNF", 72)
    assert r["charges"] == flat and r["refund"] == 1000 - flat


def test_25_and_50_percent_windows_respect_minimum():
    assert cancellation_refund(2100, "3A", "CNF", 20)["charges"] == 525      # 25% of 2100
    assert cancellation_refund(400, "3A", "CNF", 20)["charges"] == 180       # minimum flat charge applies
    assert cancellation_refund(2100, "3A", "CNF", 8)["charges"] == 1050      # 50%
    assert cancellation_refund(2100, "3A", "CNF", 3)["refund"] == 0          # within 4 hours


def test_waitlisted_and_tatkal_rules():
    wl = cancellation_refund(800, "SL", "WL", 10)
    assert wl["refund"] == 740 and wl["charges"] == 60                      # clerkage Rs 60
    assert cancellation_refund(800, "SL", "RAC", 0.25)["refund"] == 0        # less than 30 minutes before
    assert cancellation_refund(1500, "3A", "CNF", 72, quota="TQ")["refund"] == 0


def test_brief_example_query_is_understood():
    net = load_network()
    c = parse_request("I want to travel from Pune to Delhi tomorrow under ₹1500 and reach before 10:00 AM.", net,
                      date(2026, 10, 14))
    assert c["origin_codes"] == ["PUNE"]
    assert set(c["destination_codes"]) == {"NDLS", "NZM", "DEE"}
    assert c["date"] == "2026-10-15"
    assert c["budget"] == 1500 and c["arrive_by"] == "10:00"


def test_parser_handles_dates_classes_and_passengers():
    net = load_network()
    c = parse_request("Need 2 sleeper tickets Mumbai to Kolkata on 20 Oct, arrive by 6 pm", net, date(2026, 10, 14))
    assert c["date"] == "2026-10-20" and c["classes"] == ["SL"] and c["passengers"] == 2 and c["arrive_by"] == "18:00"
    c = parse_request("I want to go to Goa", net, date(2026, 10, 14))
    assert c["missing"] == ["origin"] and c["destination_codes"] == ["MAO"]


@pytest.mark.parametrize("text,expected", [
    ("Pune to Delhi tomorrow by 6 am", "06:00"),
    ("Pune to Delhi tomorrow by 6am", "06:00"),
    ("Pune to Delhi tomorrow before 6 a.m.", "06:00"),
    ("Pune to Delhi tomorrow latest by 06:00", "06:00"),
    ("Pune to Delhi tomorrow no later than 9:30", "09:30"),
    ("Pune to Delhi tomorrow, need to be there by 6 pm", "18:00"),
    ("Pune to Delhi tomorrow, leave before 6 am", None),     # about departure, not arrival
    ("Pune to Delhi tomorrow by 6", None),                   # no am/pm or minutes: too ambiguous
])
def test_arrival_time_phrasings(text, expected):
    c = parse_request(text, load_network(), date(2026, 10, 14))
    assert c["arrive_by"] == expected


def test_arrival_deadline_counts_the_date():
    saturday_evening = datetime(2026, 10, 17, 22, 35)
    # Overnight: arriving Sunday 07:00 meets "by 10:00" (Sunday 10:00 is the deadline).
    assert arrival_deadline(saturday_evening, datetime(2026, 10, 18, 7, 0), "10:00") == datetime(2026, 10, 18, 10, 0)
    # Arriving Monday 07:14 is late: the deadline stays Sunday 10:00, never the arrival day.
    late = {"departure": "2026-10-17T22:35", "arrival": "2026-10-19T07:14"}
    assert arrival_shortfall_min(late, "10:00") == 21 * 60 + 14
    # Same-day trip: leaving 06:00 and arriving 14:00 misses "by 10:00" on the travel day by 4 hours.
    same_day = {"departure": "2026-10-17T06:00", "arrival": "2026-10-17T14:00"}
    assert arrival_shortfall_min(same_day, "10:00") == 240
    assert arrival_shortfall_min({"departure": "2026-10-17T22:35", "arrival": "2026-10-18T07:00"}, "10:00") == 0


def test_time_before_departure_reads_naturally():
    assert describe_hours(578.5) == "about 24 days"
    assert describe_hours(30.4) == "about 30 hours"
    assert describe_hours(0.5) == "less than an hour"
