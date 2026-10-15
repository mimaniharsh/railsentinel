"""Tamper-evident audit trail.

Every governance-relevant event (predictions are logged separately) is appended
with the SHA-256 hash of the previous record, forming a hash chain. Changing,
deleting or reordering any past record breaks every hash after it, which
`verify()` detects and localises.
"""
from __future__ import annotations

import hashlib
import time
from typing import Any

from railsentinel.db import Database, dumps, loads
from railsentinel.security import canonical_json

GENESIS = "0" * 64


def _record_hash(prev_hash: str, ts: float, actor: str, action: str, target: str, details: Any) -> str:
    body = canonical_json({"prev": prev_hash, "ts": round(ts, 6), "actor": actor, "action": action,
                           "target": target, "details": details})
    return hashlib.sha256(body.encode("utf-8")).hexdigest()


class AuditLog:
    def __init__(self, db: Database):
        self.db = db

    def append(self, actor: str, action: str, target: str, details: Any | None = None) -> dict:
        details = details if details is not None else {}
        details = loads(dumps(details))  # normalise to plain JSON types before hashing
        with self.db._write_lock:  # read-then-write must not interleave with other appends
            last = self.db.query_one("SELECT hash FROM audit_log ORDER BY seq DESC LIMIT 1")
            prev = last["hash"] if last else GENESIS
            ts = round(time.time(), 6)
            h = _record_hash(prev, ts, actor, action, target, details)
            seq = self.db.execute(
                "INSERT INTO audit_log(ts, actor, action, target, details, prev_hash, hash) VALUES(?,?,?,?,?,?,?)",
                (ts, actor, action, target, dumps(details), prev, h))
        return {"seq": seq, "ts": ts, "actor": actor, "action": action, "target": target, "hash": h}

    def recent(self, limit: int = 200, target: str | None = None, since: float | None = None) -> list[dict]:
        sql = "SELECT * FROM audit_log WHERE 1=1"
        params: list[Any] = []
        if target:
            sql += " AND target=?"
            params.append(target)
        if since is not None:
            sql += " AND ts>=?"
            params.append(since)
        sql += " ORDER BY seq DESC LIMIT ?"
        params.append(limit)
        rows = self.db.query(sql, params)
        for r in rows:
            r["details"] = loads(r["details"], {})
        return rows

    def verify(self) -> dict:
        prev = GENESIS
        count = 0
        for row in self.db.query("SELECT * FROM audit_log ORDER BY seq ASC"):
            count += 1
            if row["prev_hash"] != prev:
                return {"valid": False, "records": count, "broken_at_seq": row["seq"],
                        "reason": "previous-hash link does not match"}
            expected = _record_hash(prev, row["ts"], row["actor"], row["action"], row["target"],
                                    loads(row["details"], {}))
            if expected != row["hash"]:
                return {"valid": False, "records": count, "broken_at_seq": row["seq"],
                        "reason": "record content does not match its hash"}
            prev = row["hash"]
        return {"valid": True, "records": count, "head": prev}
