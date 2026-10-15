"""Data contracts: validation applied to every record before inference.

A record that breaks its contract is not scored. The service returns a
"blocked" response with the reasons, the violation is logged, and the data
quality monitor tracks the violation rate per upstream feed.
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Any


@dataclass(frozen=True)
class FieldRule:
    name: str
    kind: str = "num"  # "num" or "cat"
    minimum: float | None = None
    maximum: float | None = None
    allowed: tuple = ()
    required: bool = True
    feed: str = "unknown"


@dataclass
class DataContract:
    rules: list[FieldRule]
    cross_checks: list = field(default_factory=list)  # callables(record) -> str | None

    def validate(self, record: dict) -> list[dict]:
        violations = []
        for rule in self.rules:
            value = record.get(rule.name)
            if value is None or (isinstance(value, float) and math.isnan(value)):
                if rule.required:
                    violations.append({"field": rule.name, "feed": rule.feed, "issue": "missing"})
                continue
            if rule.kind == "num":
                try:
                    num = float(value)
                except (TypeError, ValueError):
                    violations.append({"field": rule.name, "feed": rule.feed, "issue": "not numeric",
                                       "value": str(value)[:40]})
                    continue
                if math.isnan(num) or math.isinf(num):
                    violations.append({"field": rule.name, "feed": rule.feed, "issue": "not finite"})
                elif rule.minimum is not None and num < rule.minimum:
                    violations.append({"field": rule.name, "feed": rule.feed, "issue": f"below {rule.minimum}",
                                       "value": num})
                elif rule.maximum is not None and num > rule.maximum:
                    violations.append({"field": rule.name, "feed": rule.feed, "issue": f"above {rule.maximum}",
                                       "value": num})
            elif rule.allowed and str(value) not in rule.allowed:
                violations.append({"field": rule.name, "feed": rule.feed, "issue": "unexpected category",
                                   "value": str(value)[:40]})
        for check in self.cross_checks:
            problem = check(record)
            if problem:
                violations.append(problem)
        return violations

    def feeds(self) -> dict[str, str]:
        return {r.name: r.feed for r in self.rules}


def coerce(record: dict, contract: DataContract) -> dict:
    out = dict(record)
    for rule in contract.rules:
        v = out.get(rule.name)
        if v is None:
            continue
        if rule.kind == "num":
            try:
                out[rule.name] = float(v)
            except (TypeError, ValueError):
                pass
        else:
            out[rule.name] = str(v)
    return out


def summarize(violations: list[dict]) -> str:
    if not violations:
        return "pass"
    return "; ".join(f"{v['field']}: {v['issue']}" for v in violations[:4])


def as_float(value: Any, default: float = 0.0) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return default
