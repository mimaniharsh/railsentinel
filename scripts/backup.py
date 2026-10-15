"""Back up, verify and restore RailSentinel's state (database and model artifacts).

Usage:
    python scripts/backup.py create                 # writes backups/railsentinel-<timestamp>/
    python scripts/backup.py create --out /mnt/safe # somewhere else
    python scripts/backup.py verify backups/railsentinel-20261014-171500
    python scripts/backup.py restore backups/railsentinel-20261014-171500   # stop the API first

`create` is safe while the API is running (SQLite online backup). `restore` checks
every checksum and the audit hash chain before touching anything, and moves the
current data directory aside instead of deleting it.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from railsentinel.backup import create_backup, restore_backup, verify_backup  # noqa: E402
from railsentinel.config import REPO_ROOT, load_settings  # noqa: E402


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = parser.add_subparsers(dest="command", required=True)
    c = sub.add_parser("create", help="snapshot the database and model artifacts")
    c.add_argument("--out", type=Path, default=REPO_ROOT / "backups", help="parent directory for the backup")
    v = sub.add_parser("verify", help="check a backup's checksums and audit chain")
    v.add_argument("backup", type=Path)
    r = sub.add_parser("restore", help="restore a backup into the data directory (stop the API first)")
    r.add_argument("backup", type=Path)
    args = parser.parse_args()
    settings = load_settings()
    if args.command == "create":
        target = create_backup(settings.data_dir, args.out)
        print(f"Backup written to {target}")
        print(json.dumps(verify_backup(target), indent=2))
        return 0
    if args.command == "verify":
        result = verify_backup(args.backup)
        print(json.dumps(result, indent=2))
        return 0 if result["valid"] else 1
    try:
        result = restore_backup(args.backup, settings.data_dir)
    except ValueError as exc:
        print(exc)
        return 1
    print(json.dumps(result, indent=2))
    print("Restore complete. Start the API again.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
