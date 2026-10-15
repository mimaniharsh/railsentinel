"""Security helpers: signed explanation passports, PII masking and rate limiting."""
from __future__ import annotations

import hashlib
import hmac
import json
import re
import threading
import time
from collections import defaultdict, deque
from typing import Any

SIGNATURE_FIELD = "signature"


def canonical_json(payload: dict) -> str:
    return json.dumps(payload, sort_keys=True, separators=(",", ":"), ensure_ascii=False, default=str)


def sign(payload: dict, secret: str) -> str:
    body = {k: v for k, v in payload.items() if k != SIGNATURE_FIELD}
    digest = hmac.new(secret.encode("utf-8"), canonical_json(body).encode("utf-8"), hashlib.sha256)
    return "hmac-sha256:" + digest.hexdigest()


def verify(payload: dict, secret: str) -> bool:
    claimed = payload.get(SIGNATURE_FIELD)
    if not isinstance(claimed, str):
        return False
    return hmac.compare_digest(claimed, sign(payload, secret))


# PNRs are 10 digits, so bare 10-digit numbers are left alone. Phone numbers are
# masked only when written with a +91 / 0091 prefix or labelled as a phone.
_PII_PATTERNS = [
    (re.compile(r"[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}"), "[email]"),
    (re.compile(r"(?:\+91|0091)[\s-]?\d{5}[\s-]?\d{5}"), "[phone]"),
    (re.compile(r"(?i)(?:phone|mobile|contact)(?:\s*(?:no\.?|number))?\s*[:\-]?\s*\d{10}"), "[phone]"),
    (re.compile(r"\b\d{4}[\s-]\d{4}[\s-]\d{4}\b"), "[id-number]"),
    (re.compile(r"\b\d{12}\b"), "[id-number]"),
    (re.compile(r"\b(?:\d[ -]?){15}\d\b"), "[card]"),
]


def mask_pii(text: str | None) -> str:
    if not text:
        return text or ""
    masked = text
    for pattern, replacement in _PII_PATTERNS:
        masked = pattern.sub(replacement, masked)
    return masked


class RateLimiter:
    """Sliding one-minute window per client key."""

    def __init__(self, limit_per_minute: int):
        self.limit = max(1, limit_per_minute)
        self._hits: dict[str, deque] = defaultdict(deque)
        self._lock = threading.Lock()

    def allow(self, client: str, now: float | None = None) -> bool:
        now = time.time() if now is None else now
        with self._lock:
            window = self._hits[client]
            while window and now - window[0] > 60:
                window.popleft()
            if len(window) >= self.limit:
                return False
            window.append(now)
            return True


def stable_hash(*parts: Any) -> int:
    """Deterministic 32-bit hash (Python's built-in hash() is salted per process)."""
    import zlib

    return zlib.crc32("|".join(str(p) for p in parts).encode("utf-8")) & 0xFFFFFFFF
