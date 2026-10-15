"""Agent auditor: validates every planner decision and flags abnormal agent behaviour.

The auditor never trusts the agent's own account of the request. It compares
the proposed itinerary with an independent deterministic reading of the
user's message, with the options the tools actually returned (so a fabricated
train is caught), and with the rule that nothing is booked without approval.
"""
from __future__ import annotations

import json
from datetime import datetime


class AgentAuditor:
    def validate(self, constraints: dict, result: dict, steps: list[dict], config: dict,
                 returned_option_ids: set[str] | None) -> dict:
        violations: list[str] = []
        disclosed: list[str] = []
        best = result.get("recommended")
        relaxed = set(result.get("relaxed", []))
        if best:
            if returned_option_ids is not None and best.get("option_id") not in returned_option_ids:
                violations.append("proposed an option that the search tools never returned")
            legs = best.get("legs", [])
            if legs and constraints.get("origin_codes") and legs[0]["origin"] not in constraints["origin_codes"]:
                violations.append(f"origin {legs[0]['origin']} does not match the request")
            if legs and constraints.get("destination_codes") and legs[-1]["destination"] not in constraints["destination_codes"]:
                violations.append(f"destination {legs[-1]['destination']} does not match the request")
            if legs and constraints.get("date") and legs[0]["departure"][:10] != constraints["date"]:
                violations.append("departure date differs from the requested date")
            budget = constraints.get("budget")
            if budget is not None and best["fare"] > budget:
                (disclosed if "budget" in relaxed else violations).append(
                    f"fare Rs {best['fare']} exceeds budget Rs {budget:.0f}")
            arrive_by = constraints.get("arrive_by")
            if arrive_by and datetime.fromisoformat(best["arrival"]).strftime("%H:%M") > arrive_by:
                (disclosed if "arrive_by" in relaxed else violations).append(
                    f"arrives at {best['arrival'][11:16]}, after the requested {arrive_by}")
            min_conf = float(config.get("min_confirm_prob", 0.4))
            if best.get("confirm_prob") is not None and best["confirm_prob"] < min_conf:
                (disclosed if "confirmation" in relaxed else violations).append(
                    f"confirmation chance {best['confirm_prob']:.0%} below {min_conf:.0%}")
            if constraints.get("classes") and best["travel_class"] not in constraints["classes"]:
                (disclosed if "classes" in relaxed else violations).append(
                    f"class {best['travel_class']} was not requested")
        if result.get("booked"):
            violations.append("booking executed without passenger approval")
        n_steps = len(steps)
        errors = sum(1 for s in steps if not s.get("ok"))
        signatures = [s["tool"] + json.dumps(s.get("args", {}), sort_keys=True, default=str) for s in steps]
        repeats = max((signatures.count(sig) for sig in set(signatures)), default=0)
        latency = sum(s.get("latency_ms", 0) for s in steps)
        abnormal = []
        if n_steps > config.get("max_steps", 12):
            abnormal.append(f"{n_steps} steps exceeds the limit of {config.get('max_steps', 12)}")
        if repeats > 2:
            abnormal.append(f"same tool call repeated {repeats} times (possible loop)")
        if errors >= 2:
            abnormal.append(f"{errors} failed tool calls")
        if latency > 15000:
            abnormal.append(f"total tool latency {latency / 1000:.1f}s")
        return {"valid": not violations, "violations": violations, "disclosed_deviations": disclosed,
                "abnormal": bool(abnormal), "abnormal_reasons": abnormal, "steps": n_steps, "tool_errors": errors,
                "tool_latency_ms": round(latency, 1)}
