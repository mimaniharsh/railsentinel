"""Small HTTP client the Streamlit pages use to talk to the RailSentinel API."""
from __future__ import annotations

import os

import httpx

API_URL = os.getenv("RS_API_URL", "http://localhost:8000").rstrip("/")
API_KEY = os.getenv("RS_UI_API_KEY", "rs-admin-demo")


class ApiError(RuntimeError):
    def __init__(self, status: int, detail: str):
        super().__init__(f"{status}: {detail}")
        self.status = status
        self.detail = detail


def _headers() -> dict:
    return {"X-API-Key": API_KEY}


def _handle(resp: httpx.Response):
    if resp.status_code >= 400:
        try:
            detail = resp.json().get("detail", resp.text)
        except ValueError:
            detail = resp.text
        raise ApiError(resp.status_code, str(detail))
    if resp.headers.get("content-type", "").startswith("application/json"):
        return resp.json()
    return resp.text


def get(path: str, timeout: float = 60, **params):
    with httpx.Client(timeout=timeout) as c:
        return _handle(c.get(f"{API_URL}{path}", params={k: v for k, v in params.items() if v is not None},
                             headers=_headers()))


def post(path: str, json: dict | None = None, timeout: float = 120):
    with httpx.Client(timeout=timeout) as c:
        return _handle(c.post(f"{API_URL}{path}", json=json or {}, headers=_headers()))


def health() -> dict | None:
    try:
        with httpx.Client(timeout=5) as c:
            return c.get(f"{API_URL}/health").json()
    except (httpx.HTTPError, ValueError):
        return None
