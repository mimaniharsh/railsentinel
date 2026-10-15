"""Loads `policies.yaml` and answers policy questions for the supervisor."""
from __future__ import annotations

import copy
from pathlib import Path
from typing import Any

import yaml

POLICY_PATH = Path(__file__).resolve().parent / "policies.yaml"


def _merge(base: dict, override: dict) -> dict:
    out = copy.deepcopy(base)
    for k, v in (override or {}).items():
        if isinstance(v, dict) and isinstance(out.get(k), dict):
            out[k] = _merge(out[k], v)
        else:
            out[k] = copy.deepcopy(v)
    return out


class PolicyEngine:
    def __init__(self, path: Path = POLICY_PATH):
        self.path = Path(path)
        self.reload()

    def reload(self) -> None:
        self.raw = yaml.safe_load(self.path.read_text(encoding="utf-8"))
        self.defaults = self.raw.get("defaults", {})
        self.assets = self.raw.get("assets", {})
        self.actions = self.raw.get("actions", {})
        self.autonomy = self.raw.get("autonomy", {})

    def for_asset(self, name: str) -> dict:
        return _merge(self.defaults, self.assets.get(name, {}))

    def action_risk(self, action: str) -> str:
        return self.actions.get(action, {}).get("risk", "high")

    def requires_approval(self, asset: str, action: str) -> bool:
        if action in self.for_asset(asset).get("approvals_required", []):
            return True
        order = self.autonomy.get("risk_order", ["none", "low", "medium", "high"])
        limit = self.autonomy.get("auto_execute_max_risk", "medium")
        risk = self.action_risk(action)
        return order.index(risk) > order.index(limit) if risk in order and limit in order else True

    def priority(self, severity: str, risk_tier: str, blast_radius: int) -> float:
        sw = self.autonomy.get("severity_weight", {}).get(severity, 1)
        tw = self.autonomy.get("tier_weight", {}).get(risk_tier, 1)
        return round(sw * tw * (1 + 0.5 * blast_radius), 2)

    def describe(self) -> dict[str, Any]:
        return copy.deepcopy(self.raw)
