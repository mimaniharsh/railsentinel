"""SQLite persistence layer.

One database file holds the model registry, prediction logs, agent traces,
incidents, approvals, the hash-chained audit log and the twin's bookings.
Connections are per-thread; writes are serialised with a lock and the
database runs in WAL mode so the background supervisor and API threads can
read concurrently.
"""
from __future__ import annotations

import json
import sqlite3
import threading
from pathlib import Path
from typing import Any, Iterable

SCHEMA = """
CREATE TABLE IF NOT EXISTS registry_models (
    name TEXT PRIMARY KEY,
    kind TEXT NOT NULL,
    description TEXT,
    owner TEXT,
    risk_tier TEXT,
    created_ts REAL
);
CREATE TABLE IF NOT EXISTS registry_versions (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    name TEXT NOT NULL,
    version INTEGER NOT NULL,
    stage TEXT NOT NULL,
    created_ts REAL,
    promoted_ts REAL,
    metrics TEXT,
    params TEXT,
    artifact_path TEXT,
    train_rows INTEGER,
    notes TEXT,
    parent_version INTEGER,
    UNIQUE(name, version)
);
CREATE TABLE IF NOT EXISTS predictions (
    id TEXT PRIMARY KEY,
    request_id TEXT,
    model TEXT NOT NULL,
    version INTEGER,
    ts REAL NOT NULL,
    latency_ms REAL,
    status TEXT,
    features TEXT,
    output TEXT,
    segment TEXT,
    label TEXT,
    labelled_ts REAL,
    shadow INTEGER DEFAULT 0,
    source TEXT,
    scenario TEXT
);
CREATE INDEX IF NOT EXISTS ix_pred_model_ts ON predictions(model, ts);
CREATE INDEX IF NOT EXISTS ix_pred_request ON predictions(request_id);
CREATE TABLE IF NOT EXISTS llm_interactions (
    id TEXT PRIMARY KEY,
    service TEXT,
    version INTEGER,
    ts REAL,
    session_id TEXT,
    query TEXT,
    intent TEXT,
    response TEXT,
    sources TEXT,
    context TEXT,
    grounding TEXT,
    hallucination INTEGER,
    quality REAL,
    feedback INTEGER,
    latency_ms REAL,
    mode TEXT,
    source TEXT,
    scenario TEXT
);
CREATE INDEX IF NOT EXISTS ix_llm_ts ON llm_interactions(service, ts);
CREATE TABLE IF NOT EXISTS agent_traces (
    id TEXT PRIMARY KEY,
    ts REAL,
    session_id TEXT,
    version INTEGER,
    query TEXT,
    constraints TEXT,
    steps TEXT,
    plan TEXT,
    verdict TEXT,
    mode TEXT,
    latency_ms REAL,
    source TEXT,
    scenario TEXT
);
CREATE INDEX IF NOT EXISTS ix_trace_ts ON agent_traces(ts);
CREATE TABLE IF NOT EXISTS booking_intents (
    id TEXT PRIMARY KEY,
    ts REAL,
    trace_id TEXT,
    itinerary TEXT,
    status TEXT,
    pnr TEXT,
    decided_ts REAL
);
CREATE TABLE IF NOT EXISTS bookings (
    pnr TEXT PRIMARY KEY,
    created_ts REAL,
    passenger_name TEXT,
    passengers INTEGER,
    train_number TEXT,
    train_name TEXT,
    travel_class TEXT,
    quota TEXT,
    origin TEXT,
    destination TEXT,
    journey_date TEXT,
    departure TEXT,
    arrival TEXT,
    fare REAL,
    booking_status TEXT,
    current_status TEXT,
    status_detail TEXT,
    source TEXT
);
CREATE TABLE IF NOT EXISTS refunds (
    pnr TEXT PRIMARY KEY,
    requested_ts REAL,
    amount REAL,
    charges REAL,
    rule TEXT,
    status TEXT,
    updated_ts REAL
);
CREATE TABLE IF NOT EXISTS incidents (
    id TEXT PRIMARY KEY,
    opened_ts REAL,
    updated_ts REAL,
    closed_ts REAL,
    model TEXT,
    kind TEXT,
    severity TEXT,
    priority REAL,
    status TEXT,
    title TEXT,
    evidence TEXT,
    rca TEXT,
    actions TEXT,
    narrative TEXT,
    scenario TEXT
);
CREATE TABLE IF NOT EXISTS approvals (
    id TEXT PRIMARY KEY,
    ts REAL,
    incident_id TEXT,
    model TEXT,
    action TEXT,
    params TEXT,
    reason TEXT,
    status TEXT,
    decided_ts REAL,
    decided_by TEXT
);
CREATE TABLE IF NOT EXISTS audit_log (
    seq INTEGER PRIMARY KEY AUTOINCREMENT,
    ts REAL,
    actor TEXT,
    action TEXT,
    target TEXT,
    details TEXT,
    prev_hash TEXT,
    hash TEXT
);
CREATE TABLE IF NOT EXISTS metrics (
    ts REAL,
    model TEXT,
    metric TEXT,
    value REAL
);
CREATE INDEX IF NOT EXISTS ix_metrics_model ON metrics(model, metric, ts);
CREATE TABLE IF NOT EXISTS kv_state (
    key TEXT PRIMARY KEY,
    value TEXT
);
CREATE TABLE IF NOT EXISTS supervisor_cycles (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    ts REAL,
    duration_ms REAL,
    summary TEXT
);
"""


def dumps(value: Any) -> str:
    return json.dumps(value, default=_json_default, separators=(",", ":"))


def loads(value: str | None, default: Any = None) -> Any:
    if value is None or value == "":
        return default
    return json.loads(value)


def _json_default(obj: Any) -> Any:
    # numpy scalars and arrays, sets, paths
    if hasattr(obj, "item") and callable(obj.item):
        try:
            return obj.item()
        except (ValueError, TypeError):
            pass
    if hasattr(obj, "tolist"):
        return obj.tolist()
    if isinstance(obj, (set, frozenset)):
        return sorted(obj)
    if isinstance(obj, Path):
        return str(obj)
    raise TypeError(f"Object of type {type(obj).__name__} is not JSON serialisable")


class Database:
    def __init__(self, path: Path):
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._write_lock = threading.RLock()
        self._local = threading.local()
        conn = self._conn()
        with self._write_lock:
            conn.executescript(SCHEMA)
            conn.commit()

    def _conn(self) -> sqlite3.Connection:
        conn = getattr(self._local, "conn", None)
        if conn is None:
            conn = sqlite3.connect(self.path, timeout=60, check_same_thread=False)
            conn.row_factory = sqlite3.Row
            conn.execute("PRAGMA journal_mode=WAL")
            conn.execute("PRAGMA synchronous=NORMAL")
            conn.execute("PRAGMA busy_timeout=60000")
            self._local.conn = conn
        return conn

    def execute(self, sql: str, params: Iterable[Any] = ()) -> int:
        with self._write_lock:
            conn = self._conn()
            cur = conn.execute(sql, tuple(params))
            conn.commit()
            return cur.lastrowid or 0

    def executemany(self, sql: str, rows: Iterable[Iterable[Any]]) -> None:
        with self._write_lock:
            conn = self._conn()
            conn.executemany(sql, [tuple(r) for r in rows])
            conn.commit()

    def query(self, sql: str, params: Iterable[Any] = ()) -> list[dict]:
        cur = self._conn().execute(sql, tuple(params))
        return [dict(row) for row in cur.fetchall()]

    def query_one(self, sql: str, params: Iterable[Any] = ()) -> dict | None:
        cur = self._conn().execute(sql, tuple(params))
        row = cur.fetchone()
        return dict(row) if row else None

    def scalar(self, sql: str, params: Iterable[Any] = ()) -> Any:
        cur = self._conn().execute(sql, tuple(params))
        row = cur.fetchone()
        return row[0] if row else None

    # Small key/value store used for world state and operational flags.
    def get_kv(self, key: str, default: Any = None) -> Any:
        row = self.query_one("SELECT value FROM kv_state WHERE key=?", (key,))
        return loads(row["value"], default) if row else default

    def set_kv(self, key: str, value: Any) -> None:
        self.execute("INSERT INTO kv_state(key, value) VALUES(?, ?) "
                     "ON CONFLICT(key) DO UPDATE SET value=excluded.value", (key, dumps(value)))

    def close(self) -> None:
        conn = getattr(self._local, "conn", None)
        if conn is not None:
            conn.close()
            self._local.conn = None
