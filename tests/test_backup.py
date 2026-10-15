"""Disaster recovery: backups are consistent, verifiable, tamper-evident and restorable."""
import sqlite3

import pytest

from railsentinel.backup import create_backup, restore_backup, verify_backup
from railsentinel.db import Database
from railsentinel.governance.audit import AuditLog


def test_backup_verify_restore_and_tamper_detection(platform, tmp_path):
    backup = create_backup(platform.settings.data_dir, tmp_path / "backups")
    check = verify_backup(backup)
    assert check["valid"] and check["audit_chain"]["valid"]
    assert check["production_versions"]["confirmation_predictor"] >= 1
    assert verify_backup(backup)["valid"]  # verifying never changes the backup

    target = tmp_path / "restored"
    result = restore_backup(backup, target)
    assert result["audit_records"] == check["audit_chain"]["records"]
    db = Database(target / "railsentinel.db")
    assert AuditLog(db).verify()["valid"]
    assert (target / "models" / "confirmation_predictor").exists()
    db.close()

    conn = sqlite3.connect(backup / "railsentinel.db")
    conn.execute("UPDATE audit_log SET actor='intruder' WHERE seq=2")
    conn.commit()
    conn.close()
    assert not verify_backup(backup)["valid"]
    with pytest.raises(ValueError):
        restore_backup(backup, tmp_path / "second_restore")
