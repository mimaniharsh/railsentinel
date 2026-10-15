"""Backup and restore of the platform state (disaster recovery).

A backup is a directory holding:
* `railsentinel.db`: a consistent snapshot taken with SQLite's online backup API,
  so it is safe to run while the API is serving traffic;
* `models/`: every model artifact referenced by the registry;
* `manifest.json`: SHA-256 checksums of every file, the audit-chain head and the
  production version of every asset at backup time.

`verify_backup` re-computes the checksums and re-verifies the audit hash chain
inside the backup. `restore_backup` verifies first, moves the current data
directory aside (never deletes it) and copies the backup into place.
"""
from __future__ import annotations

import hashlib
import json
import shutil
import sqlite3
import time
from datetime import datetime
from pathlib import Path

from railsentinel.config import IST
from railsentinel.governance.audit import AuditLog

DB_NAME = "railsentinel.db"
MANIFEST = "manifest.json"


def _sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


class _ReadOnlyDB:
    """Read-only view of a backup database. `immutable=1` opens the file without locks or WAL
    side files, so inspecting a backup never changes a byte of it."""

    def __init__(self, path: Path):
        self._conn = sqlite3.connect(f"{Path(path).resolve().as_uri()}?immutable=1", uri=True)
        self._conn.row_factory = sqlite3.Row

    def query(self, sql: str, params=()) -> list[dict]:
        return [dict(r) for r in self._conn.execute(sql, tuple(params)).fetchall()]

    def query_one(self, sql: str, params=()) -> dict | None:
        row = self._conn.execute(sql, tuple(params)).fetchone()
        return dict(row) if row else None

    def close(self) -> None:
        self._conn.close()


def _checksums(root: Path) -> dict[str, str]:
    return {str(p.relative_to(root)).replace("\\", "/"): _sha256(p)
            for p in sorted(root.rglob("*")) if p.is_file() and p.name != MANIFEST}


def create_backup(data_dir: Path, out_dir: Path) -> Path:
    data_dir, out_dir = Path(data_dir), Path(out_dir)
    db_path = data_dir / DB_NAME
    if not db_path.exists():
        raise FileNotFoundError(f"no database at {db_path}; run scripts/bootstrap.py first")
    stamp = datetime.now(IST).strftime("%Y%m%d-%H%M%S")
    target = out_dir / f"railsentinel-{stamp}"
    suffix = 1
    while target.exists():
        target = out_dir / f"railsentinel-{stamp}-{suffix}"
        suffix += 1
    target.mkdir(parents=True)
    src = sqlite3.connect(db_path, timeout=60)
    dst = sqlite3.connect(target / DB_NAME)
    src.backup(dst)  # consistent snapshot even while other connections write
    dst.execute("PRAGMA journal_mode=DELETE")  # self-contained file: no WAL side files in the backup
    src.close()
    dst.close()
    models_src = data_dir / "models"
    if models_src.exists():
        shutil.copytree(models_src, target / "models")
    snapshot = _ReadOnlyDB(target / DB_NAME)
    audit = AuditLog(snapshot).verify()
    production = {r["name"]: r["version"] for r in
                  snapshot.query("SELECT name, version FROM registry_versions WHERE stage='production'")}
    snapshot.close()
    manifest = {"created_at": datetime.now(IST).isoformat(timespec="seconds"), "created_ts": time.time(),
                "source_data_dir": str(data_dir), "audit_chain": audit, "production_versions": production,
                "files": _checksums(target)}
    (target / MANIFEST).write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    return target


def verify_backup(backup_dir: Path) -> dict:
    backup_dir = Path(backup_dir)
    manifest_path = backup_dir / MANIFEST
    if not manifest_path.exists():
        return {"valid": False, "reason": "manifest.json missing"}
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    actual = _checksums(backup_dir)
    expected = manifest.get("files", {})
    missing = sorted(set(expected) - set(actual))
    changed = sorted(f for f in expected if f in actual and actual[f] != expected[f])
    extra = sorted(set(actual) - set(expected))
    if missing or changed or extra:
        return {"valid": False, "reason": "file checksums do not match the manifest", "missing": missing,
                "changed": changed, "unexpected": extra}
    db = _ReadOnlyDB(backup_dir / DB_NAME)
    audit = AuditLog(db).verify()
    db.close()
    if not audit["valid"]:
        return {"valid": False, "reason": "audit hash chain inside the backup is broken", "audit_chain": audit}
    return {"valid": True, "files": len(actual), "audit_chain": audit,
            "production_versions": manifest.get("production_versions", {}), "created_at": manifest.get("created_at")}


def restore_backup(backup_dir: Path, data_dir: Path) -> dict:
    """Restore `backup_dir` into `data_dir`. Stop the API before restoring."""
    backup_dir, data_dir = Path(backup_dir), Path(data_dir)
    check = verify_backup(backup_dir)
    if not check["valid"]:
        raise ValueError(f"refusing to restore an invalid backup: {check['reason']}")
    moved_to = None
    if data_dir.exists() and any(data_dir.iterdir()):
        moved_to = data_dir.with_name(f"{data_dir.name}.before-restore-{datetime.now(IST).strftime('%Y%m%d-%H%M%S')}")
        data_dir.rename(moved_to)
    data_dir.mkdir(parents=True, exist_ok=True)
    shutil.copy2(backup_dir / DB_NAME, data_dir / DB_NAME)
    if (backup_dir / "models").exists():
        shutil.copytree(backup_dir / "models", data_dir / "models")
    return {"restored_from": str(backup_dir), "data_dir": str(data_dir),
            "previous_data_moved_to": str(moved_to) if moved_to else None,
            "production_versions": check["production_versions"], "audit_records": check["audit_chain"]["records"]}
