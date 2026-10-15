"""Demand seasons and weather severity used by the twin.

These windows are approximations chosen to give the simulator realistic
seasonality (festival travel peaks in Oct-Nov, summer holidays, winter fog in
the north and east, monsoon on the west coast). They are not an official
calendar and do not track exact festival dates.
"""
from __future__ import annotations

from datetime import date

SEASONS = ("normal", "festival", "summer_vacation", "winter_holiday")

DEMAND_MULT = {"normal": 1.0, "festival": 1.45, "summer_vacation": 1.3, "winter_holiday": 1.2}
# Booking-curve shape: >1 means people book earlier in the advance-reservation window.
BOOKING_SHAPE = {"normal": 1.0, "festival": 2.0, "summer_vacation": 1.6, "winter_holiday": 1.3}
CANCEL_MULT = {"normal": 1.0, "festival": 0.6, "summer_vacation": 0.75, "winter_holiday": 0.8}


def season_of(d: date) -> str:
    md = (d.month, d.day)
    if (10, 10) <= md <= (11, 15):
        return "festival"
    if (4, 25) <= md <= (6, 30):
        return "summer_vacation"
    if md >= (12, 20) or md <= (1, 5):
        return "winter_holiday"
    return "normal"


def season_of_month(month: int, day: int = 15) -> str:
    return season_of(date(2026, month, min(day, 28)))


# Mean weather severity (0 = clear, 1 = severe) by region and month.
_FOG_REGIONS = {"North", "East", "Central"}
_MONSOON_REGIONS = {"West", "South"}


def weather_mean(region: str, month: int) -> float:
    base = 0.12
    if region in _FOG_REGIONS and month in (12, 1):
        base += 0.33
    elif region in _FOG_REGIONS and month in (11, 2):
        base += 0.12
    if region in _MONSOON_REGIONS and month in (7, 8):
        base += 0.3
    elif region in _MONSOON_REGIONS and month in (6, 9):
        base += 0.15
    if region == "Northeast" and month in (6, 7, 8):
        base += 0.25
    return min(base, 0.85)
