"""Model registry: versions, stages and artifacts for every governed AI asset.

Stages: `production` (serving), `shadow` (scored on live traffic but not
returned to users), `archived` (previous production), `rejected` (failed
evaluation) and `rolled_back` (withdrawn from production). Assets include
ML models, prompt configurations and agent configurations.
"""
from __future__ import annotations

import threading
import time
from pathlib import Path
from typing import Any

import joblib

from railsentinel.db import Database, dumps, loads
from railsentinel.governance.audit import AuditLog


class Registry:
    def __init__(self, db: Database, models_dir: Path, audit: AuditLog):
        self.db = db
        self.models_dir = Path(models_dir)
        self.models_dir.mkdir(parents=True, exist_ok=True)
        self.audit = audit
        self._cache: dict[tuple[str, int], Any] = {}
        self._lock = threading.RLock()

    # ------------------------------------------------------------ models
    def register_model(self, name: str, kind: str, description: str, owner: str, risk_tier: str) -> None:
        if self.db.query_one("SELECT name FROM registry_models WHERE name=?", (name,)):
            return
        self.db.execute("INSERT INTO registry_models(name, kind, description, owner, risk_tier, created_ts) "
                        "VALUES(?,?,?,?,?,?)", (name, kind, description, owner, risk_tier, time.time()))
        self.audit.append("registry", "model_registered", name, {"kind": kind, "risk_tier": risk_tier})

    def models(self) -> list[dict]:
        return self.db.query("SELECT * FROM registry_models ORDER BY name")

    def model(self, name: str) -> dict | None:
        return self.db.query_one("SELECT * FROM registry_models WHERE name=?", (name,))

    # ------------------------------------------------------------ versions
    def add_version(self, name: str, artifact: Any, metrics: dict, params: dict | None = None,
                    stage: str = "candidate", notes: str = "", parent: int | None = None,
                    actor: str = "registry") -> int:
        with self._lock:
            current = self.db.scalar("SELECT MAX(version) FROM registry_versions WHERE name=?", (name,)) or 0
            version = int(current) + 1
            path = self.models_dir / name / f"v{version}.joblib"
            path.parent.mkdir(parents=True, exist_ok=True)
            joblib.dump(artifact, path)
            train_rows = artifact.get("train_rows") if isinstance(artifact, dict) else None
            self.db.execute(
                "INSERT INTO registry_versions(name, version, stage, created_ts, metrics, params, artifact_path, "
                "train_rows, notes, parent_version) VALUES(?,?,?,?,?,?,?,?,?,?)",
                (name, version, "candidate", time.time(), dumps(metrics), dumps(params or {}),
                 str(path.relative_to(self.models_dir)), train_rows, notes, parent))
            self._cache[(name, version)] = artifact
        self.audit.append(actor, "version_created", name, {"version": version, "metrics": metrics, "notes": notes})
        if stage != "candidate":
            self.set_stage(name, version, stage, actor=actor, reason=notes or "initial registration")
        return version

    def set_stage(self, name: str, version: int, stage: str, actor: str, reason: str = "") -> None:
        with self._lock:
            now = time.time()
            if stage == "production":
                for row in self.db.query("SELECT version FROM registry_versions WHERE name=? AND stage='production'",
                                         (name,)):
                    if row["version"] != version:
                        self.db.execute("UPDATE registry_versions SET stage='archived' WHERE name=? AND version=?",
                                        (name, row["version"]))
                self.db.execute("UPDATE registry_versions SET stage='production', promoted_ts=? "
                                "WHERE name=? AND version=?", (now, name, version))
            elif stage == "shadow":
                self.db.execute("UPDATE registry_versions SET stage='rejected' WHERE name=? AND stage='shadow' "
                                "AND version<>?", (name, version))
                self.db.execute("UPDATE registry_versions SET stage='shadow' WHERE name=? AND version=?",
                                (name, version))
            else:
                self.db.execute("UPDATE registry_versions SET stage=? WHERE name=? AND version=?",
                                (stage, name, version))
        self.audit.append(actor, "stage_changed", name, {"version": version, "stage": stage, "reason": reason})

    def update_metrics(self, name: str, version: int, metrics: dict, actor: str, reason: str) -> None:
        """Replace a version's reference metrics (the baseline monitors compare live performance with).
        The previous figures are kept under `previous_baseline`."""
        row = self.db.query_one("SELECT metrics FROM registry_versions WHERE name=? AND version=?", (name, version))
        if not row:
            raise KeyError(f"{name} v{version} not found")
        old = loads(row["metrics"], {}) or {}
        new = {**metrics, "previous_baseline": {k: v for k, v in old.items() if k != "previous_baseline"}}
        self.db.execute("UPDATE registry_versions SET metrics=? WHERE name=? AND version=?",
                        (dumps(new), name, version))
        self.audit.append(actor, "baseline_updated", name, {"version": version, "reason": reason,
                                                             "metrics": {k: v for k, v in metrics.items()
                                                                         if isinstance(v, (int, float))}})

    def versions(self, name: str) -> list[dict]:
        rows = self.db.query("SELECT * FROM registry_versions WHERE name=? ORDER BY version DESC", (name,))
        for r in rows:
            r["metrics"] = loads(r["metrics"], {})
            r["params"] = loads(r["params"], {})
        return rows

    def version_row(self, name: str, stage: str) -> dict | None:
        row = self.db.query_one("SELECT * FROM registry_versions WHERE name=? AND stage=? "
                                "ORDER BY version DESC LIMIT 1", (name, stage))
        if row:
            row["metrics"] = loads(row["metrics"], {})
            row["params"] = loads(row["params"], {})
        return row

    def production(self, name: str) -> dict | None:
        return self.version_row(name, "production")

    def shadow(self, name: str) -> dict | None:
        return self.version_row(name, "shadow")

    def load(self, name: str, version: int) -> Any:
        key = (name, int(version))
        with self._lock:
            if key not in self._cache:
                row = self.db.query_one("SELECT artifact_path FROM registry_versions WHERE name=? AND version=?",
                                        (name, int(version)))
                if not row:
                    raise KeyError(f"{name} v{version} not found")
                self._cache[key] = joblib.load(self.models_dir / row["artifact_path"])
            return self._cache[key]

    def load_production(self, name: str) -> tuple[int, Any]:
        row = self.production(name)
        if not row:
            raise KeyError(f"no production version for {name}")
        return row["version"], self.load(name, row["version"])

    def rollback(self, name: str, actor: str, reason: str, to_version: int | None = None) -> dict:
        """Return the previous production version to service."""
        current = self.production(name)
        if to_version is None:
            row = self.db.query_one(
                "SELECT version FROM registry_versions WHERE name=? AND stage='archived' AND promoted_ts IS NOT NULL "
                "ORDER BY promoted_ts DESC LIMIT 1", (name,))
            if not row:
                return {"ok": False, "reason": "no previous production version to roll back to"}
            to_version = row["version"]
        self.set_stage(name, int(to_version), "production", actor=actor, reason=f"rollback: {reason}")
        if current and current["version"] != to_version:
            self.db.execute("UPDATE registry_versions SET stage='rolled_back' WHERE name=? AND version=?",
                            (name, current["version"]))
        self.audit.append(actor, "rollback", name, {"from": current["version"] if current else None,
                                                    "to": int(to_version), "reason": reason})
        return {"ok": True, "from": current["version"] if current else None, "to": int(to_version)}
