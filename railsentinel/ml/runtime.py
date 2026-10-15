"""Serving runtime shared by every governed model.

For each request the runtime:
1. validates the input against the model's data contract (bad records are blocked, not scored),
2. scores valid records with the production version,
3. on synthetic traffic, also scores any shadow version (logged, never returned),
4. logs inputs, outputs, latency and status for the monitors,
5. returns a signed explanation passport with every prediction.
"""
from __future__ import annotations

import time
import uuid
from datetime import datetime
from typing import Any

import numpy as np
from threadpoolctl import ThreadpoolController

from railsentinel.config import IST
from railsentinel.db import dumps, loads
from railsentinel.ml.base import ModelSpec
from railsentinel.ml.explain import shapley_explain, top_factors
from railsentinel.security import sign

_THREADPOOLS: ThreadpoolController | None = None


def single_threaded():
    """Context manager limiting OpenMP to one thread. The controller is created once: building it scans
    every loaded shared library, which would otherwise cost milliseconds on each request."""
    global _THREADPOOLS
    if _THREADPOOLS is None:
        _THREADPOOLS = ThreadpoolController()
    return _THREADPOOLS.limit(limits=1, user_api="openmp")


INSERT_PREDICTION = (
    "INSERT INTO predictions(id, request_id, model, version, ts, latency_ms, status, features, output, segment, "
    "label, labelled_ts, shadow, source, scenario) VALUES(?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)")


class ModelRuntime:
    def __init__(self, platform, spec: ModelSpec):
        self.platform = platform
        self.spec = spec

    @property
    def name(self) -> str:
        return self.spec.name

    def _context(self) -> dict:
        ctx: dict[str, Any] = {}
        if self.spec.name == "fraud_detector":
            ctx["threshold"] = self.platform.world.ops().fraud_threshold
        return ctx

    def predict(self, records: list[dict], *, source: str = "api", explain: bool = False,
                labels: list | None = None, request_ids: list[str] | None = None) -> list[dict]:
        spec = self.spec
        registry = self.platform.registry
        t0 = time.perf_counter()
        version, artifact = registry.load_production(spec.name)
        ctx = self._context()
        violations = [spec.contract.validate(spec.validation_record(r)) for r in records]
        ok_idx = [i for i, v in enumerate(violations) if not v]
        outputs: list[dict | None] = [None] * len(records)
        if ok_idx:
            # Single-threaded inference: batches are small and the API serves requests and the
            # supervisor concurrently, where OpenMP thread pools would oversubscribe the CPU.
            with single_threaded():
                preds = spec.predict(artifact, spec.prepare([records[i] for i in ok_idx]), ctx)
            for i, out in zip(ok_idx, preds):
                outputs[i] = out
        elapsed_ms = (time.perf_counter() - t0) * 1000
        per_record_ms = elapsed_ms / max(1, len(records)) if len(records) > 1 else elapsed_ms
        now = time.time()
        scenario = self.platform.world.scenario
        request_ids = request_ids or [uuid.uuid4().hex for _ in records]
        rows = []
        results = []
        health = self.platform.fleet_health(spec.name)
        for i, rec in enumerate(records):
            pid = uuid.uuid4().hex
            status = "ok" if not violations[i] else "blocked"
            out = outputs[i] if status == "ok" else {"blocked": True, "reasons": violations[i]}
            label = labels[i] if labels is not None else None
            rows.append((pid, request_ids[i], spec.name, version, now, round(per_record_ms, 3), status,
                         dumps(spec.log_features(rec)), dumps(out), spec.segment_of(rec),
                         dumps(label) if label is not None else None, now if label is not None else None,
                         0, source, scenario))
            factors = []
            if explain and status == "ok" and spec.supports_shapley:
                factors = self.explain(artifact, rec)
            passport = self.passport(pid, version, out, status, violations[i], factors, health, rec)
            results.append({"prediction_id": pid, "model": spec.name, "version": version, "status": status,
                            "output": out, "passport": passport})
        self.platform.db.executemany(INSERT_PREDICTION, rows)
        if source == "traffic":
            self._shadow_score(records, ok_idx, request_ids, labels, now, scenario)
        return results

    def _shadow_score(self, records, ok_idx, request_ids, labels, now, scenario) -> None:
        shadow = self.platform.registry.shadow(self.spec.name)
        if not shadow or not ok_idx:
            return
        artifact = self.platform.registry.load(self.spec.name, shadow["version"])
        t0 = time.perf_counter()
        with single_threaded():
            preds = self.spec.predict(artifact, self.spec.prepare([records[i] for i in ok_idx]), self._context())
        per = (time.perf_counter() - t0) * 1000 / max(1, len(ok_idx))
        rows = []
        for i, out in zip(ok_idx, preds):
            label = labels[i] if labels is not None else None
            rows.append((uuid.uuid4().hex, request_ids[i], self.spec.name, shadow["version"], now, round(per, 3),
                         "ok", dumps(self.spec.log_features(records[i])), dumps(out),
                         self.spec.segment_of(records[i]), dumps(label) if label is not None else None,
                         now if label is not None else None, 1, "traffic", scenario))
        self.platform.db.executemany(INSERT_PREDICTION, rows)

    def explain(self, artifact: dict, record: dict) -> list[dict]:
        spec = self.spec
        frame = spec.prepare([record])
        expl = shapley_explain(lambda f: spec.explain_output(artifact, f), frame[spec.features],
                               spec.explanation_background(artifact), spec.features, n_samples=48,
                               seed=self.platform.settings.seed)
        factors = top_factors(expl, record, spec.feature_labels, k=5, unit=spec.explain_unit)
        for f in factors:
            if isinstance(f["value"], (np.floating, np.integer)):
                f["value"] = f["value"].item()
        return [{**f, "base_value": round(expl["base_value"], 4)} for f in factors]

    def passport(self, pid: str, version: int, output: dict, status: str, violations: list, factors: list,
                 health: dict, record: dict) -> dict:
        settings = self.platform.settings
        conf = output.get("confidence") if isinstance(output, dict) else None
        label = None if conf is None else ("high" if conf >= 0.8 else "medium" if conf >= 0.6 else "low")
        body = {
            "passport_id": pid,
            "issued_at": datetime.now(IST).isoformat(timespec="seconds"),
            "service": self.spec.title,
            "model": self.spec.name,
            "model_version": version,
            "status": status,
            "confidence": conf,
            "confidence_label": label,
            "top_factors": factors,
            "data_quality": {"contract": "pass" if not violations else "fail", "violations": violations},
            "governance": {"model_health": health.get("status", "unknown"),
                           "open_incidents": health.get("open_incidents", 0),
                           "last_checked": health.get("checked_at")},
            "data_provenance": "RailSentinel digital twin (synthetic data, real station network)",
        }
        body["signature"] = sign(body, settings.passport_secret)
        return body


def attach_labels(db, prediction_ids: list[str], labels: list) -> None:
    now = time.time()
    db.executemany("UPDATE predictions SET label=?, labelled_ts=? WHERE id=?",
                   [(dumps(lbl), now, pid) for pid, lbl in zip(prediction_ids, labels)])


def recent_predictions(db, model: str, limit: int, version: int | None = None, shadow: int = 0,
                       labelled_only: bool = False, since: float | None = None,
                       sources: tuple[str, ...] | None = None) -> list[dict]:
    sql = "SELECT * FROM predictions WHERE model=? AND shadow=?"
    params: list[Any] = [model, shadow]
    if sources:
        sql += f" AND source IN ({','.join('?' * len(sources))})"
        params.extend(sources)
    if version is not None:
        sql += " AND version=?"
        params.append(version)
    if labelled_only:
        sql += " AND label IS NOT NULL AND status='ok'"
    if since is not None:
        sql += " AND ts>=?"
        params.append(since)
    sql += " ORDER BY ts DESC, rowid DESC LIMIT ?"
    params.append(limit)
    rows = db.query(sql, params)
    for r in rows:
        r["features"] = loads(r["features"], {})
        r["output"] = loads(r["output"], {})
        r["label"] = loads(r["label"]) if r["label"] is not None else None
    return rows
