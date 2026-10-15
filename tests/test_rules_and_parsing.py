"""Refund rules (Indian Railways, w.e.f. 12.11.2015) and natural-language parsing."""
from datetime import date

import pytest

from railsentinel.agents.nlu import parse_request
from railsentinel.agents.refunds import cancellation_refund
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
