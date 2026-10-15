"""Railway network of the digital twin.

Stations are real (coordinates from the datameet CC0 dataset). Trains are
generated deterministically from `corridors.yaml`: distances come from the
station coordinates (great-circle distance times a detour factor), running
times from category speeds, and departure times from a seeded generator.
Fares use the twin's own tariff table below, which produces realistic-looking
magnitudes but is not an official fare chart.
"""
from __future__ import annotations

import csv
import math
from dataclasses import dataclass, field
from datetime import date, datetime, timedelta
from functools import lru_cache

import numpy as np
import yaml

from railsentinel.config import STATIC_DATA_DIR
from railsentinel.security import stable_hash

DETOUR_FACTOR = 1.12  # rail routes are longer than the straight line between stations

CATEGORIES: dict[str, dict] = {
    "premium": {"label": "Premium Express", "speed": 74.0, "halt": 5, "classes": ("3A", "2A", "1A"),
                "delay_mult": 0.6, "dynamic_fare": True, "loading": 1.15},
    "superfast": {"label": "Superfast Express", "speed": 60.0, "halt": 8, "classes": ("SL", "3A", "2A", "1A"),
                  "delay_mult": 1.0, "dynamic_fare": False, "loading": 1.0},
    "express": {"label": "Mail/Express", "speed": 50.0, "halt": 8, "classes": ("2S", "SL", "3A", "2A"),
                "delay_mult": 1.35, "dynamic_fare": False, "loading": 1.0},
    "intercity": {"label": "Intercity Day Express", "speed": 64.0, "halt": 5, "classes": ("2S", "CC"),
                  "delay_mult": 0.8, "dynamic_fare": False, "loading": 1.0},
}

TRAVEL_CLASSES: dict[str, dict] = {
    "2S": {"name": "Second Sitting", "fixed": 15, "per_km": 0.30, "comfort": 1, "capacity": 600, "rac_share": 0.0},
    "SL": {"name": "Sleeper", "fixed": 40, "per_km": 0.48, "comfort": 2, "capacity": 576, "rac_share": 0.10},
    "CC": {"name": "AC Chair Car", "fixed": 60, "per_km": 1.05, "comfort": 3, "capacity": 438, "rac_share": 0.0},
    "3A": {"name": "AC 3 Tier", "fixed": 60, "per_km": 1.32, "comfort": 4, "capacity": 384, "rac_share": 0.10},
    "2A": {"name": "AC 2 Tier", "fixed": 75, "per_km": 1.90, "comfort": 5, "capacity": 184, "rac_share": 0.08},
    "1A": {"name": "AC First Class", "fixed": 100, "per_km": 3.15, "comfort": 6, "capacity": 48, "rac_share": 0.0},
}
SUPERFAST_SURCHARGE = {"2S": 15, "SL": 30, "CC": 45, "3A": 45, "2A": 45, "1A": 75}

# Departure windows (minutes after midnight) per category, as (start, end) ranges.
DEPARTURE_WINDOWS = {
    "premium": [(16 * 60, 20 * 60)],
    "superfast": [(5 * 60 + 30, 9 * 60), (15 * 60, 22 * 60 + 30)],
    "express": [(0, 24 * 60 - 1)],
    "intercity": [(5 * 60 + 30, 8 * 60 + 30), (13 * 60 + 30, 16 * 60 + 30)],
}

MAJOR_HUBS = {"NDLS", "NZM", "CSMT", "LTT", "BDTS", "HWH", "SDAH", "MAS", "SBC", "SC", "PUNE", "ADI", "LKO",
              "CNB", "PNBE", "BPL", "NGP", "JP", "GHY", "ERS", "TVC", "BSB", "PRYJ", "KYN", "ET", "DDU"}


@dataclass(frozen=True)
class Station:
    code: str
    name: str
    city: str
    state: str
    region: str
    lat: float
    lon: float
    platforms: int


@dataclass(frozen=True)
class Stop:
    code: str
    arr_min: int | None  # minutes after 00:00 of the train's origin departure date
    dep_min: int | None
    km: float  # cumulative distance from origin


@dataclass
class Train:
    number: str
    name: str
    category: str
    corridor: str
    stops: list[Stop]
    classes: tuple[str, ...]
    _index: dict[str, int] = field(default_factory=dict, repr=False)

    def __post_init__(self) -> None:
        self._index = {s.code: i for i, s in enumerate(self.stops)}

    def index_of(self, code: str) -> int | None:
        return self._index.get(code)

    @property
    def origin(self) -> str:
        return self.stops[0].code

    @property
    def destination(self) -> str:
        return self.stops[-1].code

    @property
    def total_km(self) -> float:
        return self.stops[-1].km

    @property
    def dynamic_fare(self) -> bool:
        return CATEGORIES[self.category]["dynamic_fare"]

    def to_dict(self) -> dict:
        return {"number": self.number, "name": self.name, "category": self.category,
                "category_label": CATEGORIES[self.category]["label"], "corridor": self.corridor,
                "classes": list(self.classes), "origin": self.origin, "destination": self.destination,
                "total_km": round(self.total_km), "dynamic_fare": self.dynamic_fare,
                "stops": [{"code": s.code, "arr": _fmt(s.arr_min), "dep": _fmt(s.dep_min),
                           "day": 1 + (s.arr_min if s.arr_min is not None else s.dep_min) // 1440,
                           "km": round(s.km)} for s in self.stops]}


@dataclass(frozen=True)
class Segment:
    """A passenger's portion of a train run between two of its stops."""

    train: Train
    from_idx: int
    to_idx: int

    @property
    def origin(self) -> str:
        return self.train.stops[self.from_idx].code

    @property
    def destination(self) -> str:
        return self.train.stops[self.to_idx].code

    @property
    def distance_km(self) -> float:
        return self.train.stops[self.to_idx].km - self.train.stops[self.from_idx].km

    @property
    def dep_offset(self) -> int:
        return int(self.train.stops[self.from_idx].dep_min)

    @property
    def arr_offset(self) -> int:
        return int(self.train.stops[self.to_idx].arr_min)

    @property
    def duration_min(self) -> int:
        return self.arr_offset - self.dep_offset

    def departure_dt(self, boarding_date: date) -> datetime:
        """Departure from the boarding station on `boarding_date` (the train runs daily)."""
        return datetime.combine(boarding_date, datetime.min.time()) + timedelta(minutes=self.dep_offset % 1440)

    def arrival_dt(self, boarding_date: date) -> datetime:
        return self.departure_dt(boarding_date) + timedelta(minutes=self.duration_min)

    def origin_run_date(self, boarding_date: date) -> date:
        return boarding_date - timedelta(days=self.dep_offset // 1440)


def _fmt(minutes: int | None) -> str | None:
    if minutes is None:
        return None
    m = int(minutes) % 1440
    return f"{m // 60:02d}:{m % 60:02d}"


def haversine_km(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
    r = 6371.0
    p1, p2 = math.radians(lat1), math.radians(lat2)
    dp, dl = p2 - p1, math.radians(lon2 - lon1)
    a = math.sin(dp / 2) ** 2 + math.cos(p1) * math.cos(p2) * math.sin(dl / 2) ** 2
    return 2 * r * math.asin(math.sqrt(a))


class Network:
    def __init__(self, stations: dict[str, Station], trains: dict[str, Train]):
        self.stations = stations
        self.trains = trains
        self.city_index: dict[str, list[str]] = {}
        for st in stations.values():
            self.city_index.setdefault(st.city.lower(), []).append(st.code)
        for alias, city in CITY_ALIASES.items():
            if city.lower() in self.city_index:
                self.city_index.setdefault(alias, self.city_index[city.lower()])
        self._by_station: dict[str, list[Train]] = {}
        for tr in trains.values():
            for s in tr.stops:
                self._by_station.setdefault(s.code, []).append(tr)

    # ---------- lookup ----------
    def resolve_place(self, text: str) -> list[str]:
        """Map a city name, alias or station code to station codes."""
        if not text:
            return []
        key = text.strip().lower()
        if key.upper() in self.stations:
            return [key.upper()]
        if key in self.city_index:
            return list(self.city_index[key])
        for st in self.stations.values():
            if st.name.lower() == key:
                return [st.code]
        return []

    def trains_at(self, code: str) -> list[Train]:
        return self._by_station.get(code, [])

    def direct_segments(self, origins: list[str], destinations: list[str]) -> list[Segment]:
        out = []
        dest_set = set(destinations)
        seen = set()
        for o in origins:
            for tr in self.trains_at(o):
                i = tr.index_of(o)
                for j in range(i + 1, len(tr.stops)):
                    if tr.stops[j].code in dest_set and (tr.number, i, j) not in seen:
                        seen.add((tr.number, i, j))
                        out.append(Segment(tr, i, j))
        return out

    def fare(self, train: Train, travel_class: str, distance_km: float) -> float:
        cls = TRAVEL_CLASSES[travel_class]
        cat = CATEGORIES[train.category]
        base = cls["fixed"] + cls["per_km"] * distance_km * cat["loading"]
        if train.category in ("premium", "superfast"):
            base += SUPERFAST_SURCHARGE[travel_class]
        return float(round(base / 5.0) * 5)

    def station_dict(self, code: str) -> dict:
        st = self.stations[code]
        return {"code": st.code, "name": st.name, "city": st.city, "state": st.state,
                "region": st.region, "lat": st.lat, "lon": st.lon, "platforms": st.platforms}

    def scheduled_load(self, code: str, hour: int) -> float:
        """Trains scheduled at a station within +/-1h of `hour`, per platform."""
        n = 0
        for tr in self.trains_at(code):
            s = tr.stops[tr.index_of(code)]
            t = s.arr_min if s.arr_min is not None else s.dep_min
            h = (t // 60) % 24
            if min(abs(h - hour), 24 - abs(h - hour)) <= 1:
                n += 1
        return n / self.stations[code].platforms


CITY_ALIASES = {
    "new delhi": "Delhi", "bombay": "Mumbai", "calcutta": "Kolkata", "madras": "Chennai",
    "bangalore": "Bengaluru", "secunderabad": "Hyderabad", "allahabad": "Prayagraj", "benaras": "Varanasi",
    "banaras": "Varanasi", "kashi": "Varanasi", "baroda": "Vadodara", "nasik": "Nashik", "goa": "Madgaon",
    "margao": "Madgaon", "hubli": "Hubballi", "cochin": "Kochi", "ernakulam": "Kochi",
    "trivandrum": "Thiruvananthapuram", "vizag": "Visakhapatnam", "jamshedpur": "Jamshedpur",
    "tatanagar": "Jamshedpur", "new jalpaiguri": "Siliguri", "katpadi": "Vellore", "mughalsarai": "Mughalsarai",
    "palghat": "Palakkad", "amravati": "Badnera", "tirupati": "Renigunta", "poona": "Pune",
}


def _load_stations() -> dict[str, Station]:
    stations = {}
    with (STATIC_DATA_DIR / "stations.csv").open(newline="", encoding="utf-8") as fh:
        for row in csv.DictReader(fh):
            code = row["code"]
            # Platform counts are a twin attribute (not real data), used for congestion estimates.
            platforms = 16 if code in MAJOR_HUBS else 6 + stable_hash("platforms", code) % 4
            stations[code] = Station(code, row["name"], row["city"], row["state"], row["region"],
                                     float(row["lat"]), float(row["lon"]), platforms)
    return stations


def _sample_departure(rng: np.random.Generator, category: str) -> int:
    windows = DEPARTURE_WINDOWS[category]
    lo, hi = windows[int(rng.integers(0, len(windows)))]
    return int(rng.integers(lo, hi + 1) // 5 * 5)


def _build_train(number: str, corridor: str, category: str, codes: list[str],
                 stations: dict[str, Station], dep_origin: int) -> Train:
    cat = CATEGORIES[category]
    stops: list[Stop] = []
    t = dep_origin
    km = 0.0
    for i, code in enumerate(codes):
        if i == 0:
            stops.append(Stop(code, None, t, 0.0))
            continue
        a, b = stations[codes[i - 1]], stations[code]
        seg = haversine_km(a.lat, a.lon, b.lat, b.lon) * DETOUR_FACTOR
        km += seg
        t += int(round(seg / cat["speed"] * 60))
        if i == len(codes) - 1:
            stops.append(Stop(code, t, None, km))
        else:
            arr = t
            t += cat["halt"]
            stops.append(Stop(code, arr, t, km))
    first, last = stations[codes[0]], stations[codes[-1]]
    name = f"{first.city}–{last.city} {cat['label']}"
    return Train(number, name, category, corridor, stops, cat["classes"])


@lru_cache(maxsize=4)
def load_network(seed: int = 11) -> Network:
    stations = _load_stations()
    spec = yaml.safe_load((STATIC_DATA_DIR / "corridors.yaml").read_text(encoding="utf-8"))
    trains: dict[str, Train] = {}
    serial = 101
    for corridor, cfg in spec["corridors"].items():
        codes = list(cfg["stops"])
        for category, count in cfg["trains"].items():
            for k in range(int(count)):
                for direction, seq in (("down", codes), ("up", list(reversed(codes)))):
                    rng = np.random.default_rng(stable_hash(seed, corridor, category, k, direction))
                    number = f"RS{serial if direction == 'down' else serial + 1}"
                    trains[number] = _build_train(number, corridor, category, seq, stations,
                                                  _sample_departure(rng, category))
                serial += 2
    return Network(stations, trains)
