"""Deterministic parser for travel requests.

Used in two ways: as the offline planner's understanding step, and as an
independent second reading of the request that the agent auditor compares the
LLM planner's decisions against (so the LLM cannot quietly drop a constraint).
"""
from __future__ import annotations

import re
from datetime import date, datetime, time, timedelta

from railsentinel.twin.network import Network

MONTHS = {m: i for i, m in enumerate(["jan", "feb", "mar", "apr", "may", "jun", "jul", "aug", "sep", "oct", "nov",
                                      "dec"], start=1)}
WEEKDAYS = {d: i for i, d in enumerate(["monday", "tuesday", "wednesday", "thursday", "friday", "saturday",
                                        "sunday"])}
CLASS_PATTERNS = [
    (r"\b(?:1a|1ac|first ac|ac first|first class)\b", ["1A"]),
    (r"\b(?:2a|2ac|second ac|ac 2 ?tier|2 ?tier)\b", ["2A"]),
    (r"\b(?:3a|3ac|third ac|ac 3 ?tier|3 ?tier)\b", ["3A"]),
    (r"\b(?:sleeper|sl class)\b", ["SL"]),
    (r"\b(?:chair car|cc)\b", ["CC"]),
    (r"\b(?:2s|second sitting|general class)\b", ["2S"]),
]
PREFERENCES = [
    (r"\b(?:cheapest|lowest fare|cheap|budget option)\b", "cheapest"),
    (r"\b(?:fastest|quickest|shortest)\b", "fastest"),
    (r"\b(?:most reliable|on time|punctual|reliable)\b", "reliable"),
    (r"\b(?:comfortable|comfort|luxury)\b", "comfort"),
]


def _time_from(h: str, m: str | None, ampm: str | None) -> time | None:
    hour = int(h)
    minute = int(m) if m else 0
    if ampm:
        ampm = ampm.lower().replace(".", "")
        if ampm == "pm" and hour < 12:
            hour += 12
        if ampm == "am" and hour == 12:
            hour = 0
    if 0 <= hour <= 23 and 0 <= minute <= 59:
        return time(hour, minute)
    return None


def arrival_deadline(departure: datetime, arrival: datetime, arrive_by: str) -> datetime:
    """What "reach by HH:MM" means for one journey: by HH:MM on the travel day or, for an overnight
    journey, by HH:MM the next day, never later. A train leaving on Saturday evening that arrives at
    07:00 on Sunday meets "before 10:00"; one that arrives at 07:00 on Monday misses it by 21 hours."""
    hh, mm = (int(x) for x in arrive_by.split(":"))
    day = min(arrival.date(), departure.date() + timedelta(days=1))
    return datetime.combine(day, time(hh, mm))


def arrival_shortfall_min(option: dict, arrive_by: str) -> float:
    """Minutes an option's scheduled arrival is past its `arrival_deadline` (0 when it arrives in time)."""
    departure = datetime.fromisoformat(option["departure"])
    arrival = datetime.fromisoformat(option["arrival"])
    return max(0.0, (arrival - arrival_deadline(departure, arrival, arrive_by)).total_seconds() / 60)


def _find_places(text: str, network: Network) -> list[tuple[int, str, list[str]]]:
    """All place mentions with their position, longest names first, no overlaps."""
    lower = text.lower()
    names = sorted(set(network.city_index) | {s.name.lower() for s in network.stations.values()}, key=len,
                   reverse=True)
    taken = [False] * len(lower)
    found = []
    for name in names:
        for m in re.finditer(r"\b" + re.escape(name) + r"\b", lower):
            if any(taken[m.start():m.end()]):
                continue
            codes = network.resolve_place(name)
            if codes:
                for k in range(m.start(), m.end()):
                    taken[k] = True
                found.append((m.start(), name, codes))
    for m in re.finditer(r"\b([A-Z]{2,5})\b", text):
        code = m.group(1)
        if code in network.stations and not any(taken[m.start():m.end()]):
            found.append((m.start(), code, [code]))
    return sorted(found)


def parse_date(text: str, today: date) -> tuple[date | None, str | None]:
    lower = text.lower()
    if "day after tomorrow" in lower:
        return today + timedelta(days=2), "day after tomorrow"
    if re.search(r"\btomorrow\b", lower):
        return today + timedelta(days=1), "tomorrow"
    if re.search(r"\b(?:today|tonight)\b", lower):
        return today, "today"
    m = re.search(r"\b(\d{4})-(\d{2})-(\d{2})\b", lower)
    if m:
        try:
            return date(int(m.group(1)), int(m.group(2)), int(m.group(3))), m.group(0)
        except ValueError:
            pass
    day_month = None
    for m in re.finditer(r"\b(\d{1,2})(?:st|nd|rd|th)?\s+(?:of\s+)?([a-z]{3})[a-z]*\b", lower):
        if m.group(2) in MONTHS:
            day_month = (int(m.group(1)), MONTHS[m.group(2)], m.group(0))
            break
    if day_month is None:
        for m in re.finditer(r"\b([a-z]{3})[a-z]*\s+(\d{1,2})(?:st|nd|rd|th)?\b", lower):
            if m.group(1) in MONTHS:
                day_month = (int(m.group(2)), MONTHS[m.group(1)], m.group(0))
                break
    if day_month is None:
        m3 = re.search(r"\b(\d{1,2})[/-](\d{1,2})(?:[/-](\d{2,4}))?\b", lower)
        if m3:
            day_month = (int(m3.group(1)), int(m3.group(2)), m3.group(0))
    if day_month:
        d, mo, raw = day_month
        try:
            candidate = date(today.year, mo, d)
            if candidate < today:
                candidate = date(today.year + 1, mo, d)
            return candidate, raw
        except ValueError:
            pass
    m = re.search(r"\b(?:on|this|next|coming)?\s*(monday|tuesday|wednesday|thursday|friday|saturday|sunday)\b", lower)
    if m:
        target = WEEKDAYS[m.group(1)]
        delta = (target - today.weekday()) % 7 or 7
        return today + timedelta(days=delta), m.group(0).strip()
    return None, None


def parse_request(text: str, network: Network, today: date) -> dict:
    lower = text.lower()
    assumptions: list[str] = []
    places = _find_places(text, network)
    origin = destination = None
    m_from = re.search(r"\bfrom\s+", lower)
    m_to = re.search(r"\bto\s+", lower)
    if m_from and m_to and places:
        after_from = [p for p in places if p[0] >= m_from.end() - 1]
        after_to = [p for p in places if p[0] >= m_to.end() - 1]
        if after_from:
            origin = after_from[0]
        if after_to:
            destination = after_to[0]
        if origin and destination and origin == destination and len(places) >= 2:
            origin, destination = places[0], places[1]
    if (origin is None or destination is None) and len(places) >= 2:
        origin, destination = places[0], places[1]
    if len(places) == 1 and origin is None and destination is None:
        before = lower[:places[0][0]].rstrip()
        if before.endswith(" to") or before.endswith("for") or before.endswith("reach"):
            destination = places[0]
        else:
            origin = places[0]
    jdate, date_text = parse_date(text, today)
    if jdate is None:
        jdate = today + timedelta(days=1)
        assumptions.append("No travel date given, so I assumed tomorrow.")
    budget = None
    m = re.search(r"(?:under|below|less than|within|max(?:imum)?|budget(?: of)?|up ?to|upto|not more than)\s*"
                  r"(?:₹|rs\.?|inr)?\s*([\d,]{2,7})", lower)
    if not m:
        m = re.search(r"(?:₹|rs\.?\s*|inr\s*)([\d,]{2,7})", lower)
    if m:
        budget = float(m.group(1).replace(",", ""))
    arrive_by = depart_after = None
    m = re.search(r"(?:reach|arrive|arriving|get there|be there|land)\w*\s*(?:\w+\s){0,2}?(?:by|before)\s*"
                  r"(\d{1,2})(?::(\d{2}))?\s*(a\.?m\.?|p\.?m\.?)?", lower)
    if m:
        arrive_by = _time_from(m.group(1), m.group(2), m.group(3))
    if arrive_by is None:
        # Without a verb such as "reach", "by 6 am" or "before 06:30" still sets the arrival time, as long as
        # the time is unambiguous (am/pm or hh:mm) and is not about leaving ("leave before 6 am").
        for m in re.finditer(r"\b(?:no later than|latest by|by|before)\s*(\d{1,2})(?::(\d{2}))?\s*"
                             r"(a\.?m\.?|p\.?m\.?)?(?![\w:])", lower):
            if not (m.group(2) or m.group(3)):
                continue
            if re.search(r"(?:leave|leaving|depart|departing|start|starting)\w*\s*$", lower[:m.start()]):
                continue
            arrive_by = _time_from(m.group(1), m.group(2), m.group(3))
            if arrive_by:
                break
    m = re.search(r"(?:leave|leaving|depart|departing|start|starting)\w*\s*(?:after)\s*(\d{1,2})(?::(\d{2}))?\s*"
                  r"(a\.?m\.?|p\.?m\.?)?", lower)
    if m:
        depart_after = _time_from(m.group(1), m.group(2), m.group(3))
    classes: list[str] = []
    for pattern, cls in CLASS_PATTERNS:
        if re.search(pattern, lower):
            classes.extend(cls)
    if not classes and re.search(r"\bac\b", lower):
        classes = ["CC", "3A", "2A", "1A"]
    passengers = 1
    m = re.search(r"\b(\d{1,2})\s*(?:[a-z0-9]+\s+)?(?:passengers|people|persons|adults|tickets|seats|berths|travellers|travelers|of us)\b", lower)
    if m:
        passengers = max(1, min(6, int(m.group(1))))
    preference = None
    for pattern, pref in PREFERENCES:
        if re.search(pattern, lower):
            preference = pref
            break
    if budget is not None and passengers > 1:
        assumptions.append(f"Budget read as Rs {budget:,.0f} per passenger.")
    return {
        "origin_text": origin[1] if origin else None,
        "destination_text": destination[1] if destination else None,
        "origin_codes": origin[2] if origin else [],
        "destination_codes": destination[2] if destination else [],
        "date": jdate.isoformat(), "date_text": date_text,
        "budget": budget,
        "arrive_by": arrive_by.strftime("%H:%M") if arrive_by else None,
        "depart_after": depart_after.strftime("%H:%M") if depart_after else None,
        "classes": classes, "passengers": passengers, "preference": preference,
        "assumptions": assumptions,
        "missing": [k for k, v in (("origin", origin), ("destination", destination)) if v is None],
    }
