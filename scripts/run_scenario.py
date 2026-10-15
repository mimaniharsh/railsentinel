"""Run a chaos scenario end to end from the command line and print the timeline.

Works directly on the local database (no API needed). Handy for recording a demo
or checking the governance loop on a new machine.

Usage:
    python scripts/run_scenario.py --list
    python scripts/run_scenario.py fog_surge
    python scripts/run_scenario.py tatkal_rush_false_positives --approve
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from railsentinel.core import Platform  # noqa: E402
from railsentinel.governance.chaos import SCENARIOS  # noqa: E402

ICONS = {"incident_opened": "[INCIDENT]", "remediation": "[ACTION]  ", "retrain_queued": "[WAIT]    ",
         "promoted": "[PROMOTE] ", "shadow_rejected": "[REJECT]  ", "incident_resolved": "[RESOLVED]",
         "approval_requested": "[HUMAN]   ", "incident_replanned": "[REPLAN]  ", "circuit_breaker_closed": "[BREAKER] ",
         "feed_released": "[FEED]    "}


def line(e: dict) -> str:
    keep = {k: v for k, v in e.items() if k not in ("type", "comparison", "incident")}
    if isinstance(keep.get("result"), dict):
        keep["result"] = {k: v for k, v in keep["result"].items()
                          if k in ("status", "version", "relative_gain", "from", "to", "quarantined",
                                   "circuit_breakers", "new_threshold", "labels", "needed")}
    return f"  {ICONS.get(e['type'], '[EVENT]   ')} {keep}"


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("scenario", nargs="?", help="scenario id")
    parser.add_argument("--list", action="store_true", help="list scenarios")
    parser.add_argument("--rounds", type=int, default=8, help="maximum traffic+supervisor rounds")
    parser.add_argument("--approve", action="store_true", help="approve pending human approvals and continue")
    parser.add_argument("--keep", action="store_true", help="leave the scenario active afterwards")
    args = parser.parse_args()
    if args.list or not args.scenario:
        for k, v in SCENARIOS.items():
            print(f"{k:30s} {v['title']}")
        return 0
    if args.scenario not in SCENARIOS:
        print(f"Unknown scenario {args.scenario}. Use --list.")
        return 1
    p = Platform()
    if not p.is_bootstrapped():
        print("Run python scripts/bootstrap.py first.")
        return 1
    sc = SCENARIOS[args.scenario]
    print(f"Scenario: {sc['title']}\n  {sc['story']}\n  Expected: {sc['expected']}\n")
    p.chaos.reset(actor="cli")
    run = p.chaos.autopilot(args.scenario, actor="cli", max_rounds=args.rounds)
    for r in run["timeline"]:
        print(f"Round {r['round']}: {sum(r['traffic']['requests'].values())} passenger requests, status {r['status']}")
        for e in r["events"]:
            print(line(e))
    print(f"\nOutcome: {run['outcome']} after {run['rounds']} round(s) in {run['duration_s']} s")
    if run["outcome"] == "awaiting_human_approval" and args.approve:
        for ap in p.supervisor.approvals("pending"):
            print(f"\nApproving {ap['action']} on {ap['model']} ({ap['id']}) as admin:cli")
            p.supervisor.decide(ap["id"], approve=True, actor="admin:cli")
        for i in range(3):
            p.traffic.run()
            cycle = p.supervisor.run_cycle()
            print(f"Round {run['rounds'] + i + 1}:")
            for e in cycle["events"]:
                print(line(e))
    if not args.keep:
        p.chaos.clear(actor="cli")
    print(f"Audit chain valid: {p.audit.verify()['valid']}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
