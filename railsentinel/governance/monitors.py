"""Monitoring suite: health, data quality, drift, performance, concept drift,
fairness, exposure bias, false positives, hallucination, response quality,
customer satisfaction and agent behaviour.

Each check returns a status of "ok", "breach" or "insufficient" plus evidence.
A breach becomes a Finding that the supervisor turns into an incident.

Windows: checks look at the most recent `window` records, but never further
back than the asset's "window floor" (set whenever a remediation is applied),
so verification after a fix only judges data produced after the fix.
Distribution and outcome checks use the passenger traffic stream; calls made
internally by the planner agent are monitored for health only.
"""
from __future__ import annotations

import os
import time
from dataclasses import asdict, dataclass, field
from typing import Any

import numpy as np
import pandas as pd
import psutil

from railsentinel.db import loads
from railsentinel.governance.stats import change_point, drift_report, page_hinkley, psi_categorical
from railsentinel.ml.base import disparity_ratio
from railsentinel.ml.runtime import recent_predictions

PASSENGER_SOURCES = ("traffic", "api")
MIN_FAIRNESS_GROUP = 50   # groups with fewer labelled outcomes in the window are not judged
FPR_ALPHA = 0.01          # significance needed before a false-positive rate above the limit counts as a breach


def fpr_excess_p_value(false_positives: int, genuine: int, limit: float) -> float:
    """P(at least this many false positives | the true rate equals the limit): small means the rate is really higher."""
    from scipy.stats import binom

    if genuine <= 0:
        return 1.0
    return round(float(binom.sf(false_positives - 1, genuine, limit)), 6)


@dataclass
class Finding:
    asset: str
    kind: str
    severity: str
    metric: str
    value: Any
    threshold: Any
    message: str
    evidence: dict = field(default_factory=dict)
    feeds: list = field(default_factory=list)

    def to_dict(self) -> dict:
        return asdict(self)


def window_floor(db, asset: str) -> float:
    return float(db.get_kv(f"window_floor:{asset}", 0.0) or 0.0)


def set_window_floor(db, asset: str, ts: float | None = None) -> None:
    db.set_kv(f"window_floor:{asset}", ts if ts is not None else time.time())


def _pct(values, q) -> float | None:
    return round(float(np.percentile(values, q)), 2) if len(values) else None


class MonitorSuite:
    def __init__(self, platform):
        self.p = platform

    # ------------------------------------------------------------ entry points
    def collect_all(self, record: bool = True) -> dict[str, dict]:
        """Run every monitor. `record=True` (supervisor cycles) also advances alert state such as
        drift persistence; read-only views (dashboards) pass False."""
        reports = {}
        for asset in self.p.asset_names():
            try:
                reports[asset] = self.collect(asset, record=record)
            except Exception as exc:  # a broken monitor must not stop the others
                reports[asset] = {"asset": asset, "status": "unknown", "checks": {}, "findings": [],
                                  "error": f"{type(exc).__name__}: {exc}"}
        return reports

    def collect(self, asset: str, record: bool = False) -> dict:
        if asset in self.p.specs:
            report = self._ml(asset, record)
        elif asset == "support_assistant":
            report = self._support(asset)
        else:
            report = self._agent(asset)
        sev = {f.severity for f in report["findings"]}
        report["status"] = "critical" if "critical" in sev else "warning" if "warning" in sev else "healthy"
        report["findings"] = [f.to_dict() for f in report["findings"]]
        report["checked_at"] = time.time()
        return report

    def resources(self) -> dict:
        proc = psutil.Process(os.getpid())
        db_path = self.p.settings.db_path
        models_dir = self.p.settings.models_dir
        model_bytes = sum(f.stat().st_size for f in models_dir.rglob("*.joblib")) if models_dir.exists() else 0
        return {"process_rss_mb": round(proc.memory_info().rss / 1e6, 1),
                "process_cpu_percent": proc.cpu_percent(interval=None),
                "system_memory_percent": psutil.virtual_memory().percent,
                "db_size_mb": round(db_path.stat().st_size / 1e6, 2) if db_path.exists() else 0.0,
                "model_artifacts_mb": round(model_bytes / 1e6, 2), "threads": proc.num_threads()}

    def _metric(self, asset: str, name: str, value) -> None:
        if value is None:
            return
        try:
            v = float(value)
        except (TypeError, ValueError):
            return
        if np.isfinite(v):
            self.p.db.execute("INSERT INTO metrics(ts, model, metric, value) VALUES(?,?,?,?)",
                              (time.time(), asset, name, v))

    # ------------------------------------------------------------ ML models
    def _ml(self, asset: str, record: bool = False) -> dict:
        spec = self.p.specs[asset]
        pol = self.p.policies.for_asset(asset)
        db = self.p.db
        floor = window_floor(db, asset)
        prod = self.p.registry.production(asset)
        version = prod["version"]
        artifact = self.p.registry.load(asset, version)
        findings: list[Finding] = []
        checks: dict[str, dict] = {}
        window = int(pol["window"])

        # health (all sources)
        rows = recent_predictions(db, asset, window, since=floor)
        lat = [r["latency_ms"] for r in rows if r["latency_ms"] is not None]
        p95 = _pct(lat, 95)
        checks["health"] = {"status": "insufficient" if len(rows) < 10 else "ok", "requests": len(rows),
                            "latency_p50_ms": _pct(lat, 50), "latency_p95_ms": p95,
                            "sla_p95_ms": pol["sla"]["p95_latency_ms"]}
        if p95 is not None and len(rows) >= 10 and p95 > pol["sla"]["p95_latency_ms"]:
            checks["health"]["status"] = "breach"
            findings.append(Finding(asset, "health", "warning", "latency_p95_ms", p95, pol["sla"]["p95_latency_ms"],
                                    f"p95 latency {p95} ms exceeds SLA {pol['sla']['p95_latency_ms']} ms"))
        self._metric(asset, "latency_p95_ms", p95)

        # data quality (passenger stream)
        stream = recent_predictions(db, asset, window, since=floor, sources=PASSENGER_SOURCES)
        if len(stream) >= pol["min_samples"]:
            blocked = [r for r in stream if r["status"] == "blocked"]
            rate = len(blocked) / len(stream)
            feeds: dict[str, int] = {}
            fields: dict[str, int] = {}
            for r in blocked:
                for v in r["output"].get("reasons", []):
                    feeds[v["feed"]] = feeds.get(v["feed"], 0) + 1
                    fields[v["field"]] = fields.get(v["field"], 0) + 1
            dq = {"status": "ok", "records": len(stream), "blocked_rate": round(rate, 4),
                  "max_blocked_rate": pol["data_quality"]["max_blocked_rate"], "violations_by_feed": feeds,
                  "violations_by_field": fields}
            if rate > pol["data_quality"]["max_blocked_rate"]:
                dq["status"] = "breach"
                findings.append(Finding(asset, "data_quality", "critical", "blocked_rate", round(rate, 4),
                                        pol["data_quality"]["max_blocked_rate"],
                                        f"{rate:.1%} of requests failed their data contract and were blocked",
                                        {"violations_by_feed": feeds, "violations_by_field": fields},
                                        sorted(feeds, key=feeds.get, reverse=True)))
            if asset == "fare_forecaster":
                ok_out = [r["output"] for r in stream if r["status"] == "ok"]
                anom = float(np.mean([o.get("quote_anomaly", False) for o in ok_out])) if ok_out else 0.0
                limit = pol["data_quality"].get("max_quote_anomaly_rate", 0.02)
                dq["quote_anomaly_rate"] = round(anom, 4)
                if anom > limit:
                    dq["status"] = "breach"
                    findings.append(Finding(asset, "data_quality", "critical", "quote_anomaly_rate", round(anom, 4),
                                            limit, f"{anom:.1%} of fare quotes disagree with the pricing rule",
                                            {"check": "pricing anomaly"}, ["fare_feed"]))
                self._metric(asset, "quote_anomaly_rate", anom)
            checks["data_quality"] = dq
            self._metric(asset, "blocked_rate", rate)
        else:
            checks["data_quality"] = {"status": "insufficient", "records": len(stream)}

        # drift (passenger stream, valid records)
        valid = [r["features"] for r in stream if r["status"] == "ok"]
        if len(valid) >= pol["min_samples"]:
            cats, nums = spec.drift_columns()
            reference = artifact["reference"]
            accepted = db.get_kv(f"drift_reference:{asset}:{version}")
            if accepted:  # a confirmed behaviour change that a human approved adapting to
                reference = spec.drift_frame(accepted)
            report = drift_report(reference, spec.drift_frame(valid), cats, nums)
            max_psi = max((v["psi"] for v in report.values()), default=0.0)
            psi_alert = float(pol["drift"]["psi_alert"])
            p_alert = float(pol["drift"].get("p_value", 0.01))
            # An alert needs a large shift (PSI) that is also statistically distinguishable from sampling noise,
            # and it must either be strong (PSI at least twice the alert level) or persist into the next
            # supervisor cycle. Small or clustered windows can produce one large PSI by chance.
            candidates = [f for f, v in report.items()
                          if v["psi"] >= psi_alert and v.get("ks_p", v.get("chi2_p", 0.0)) < p_alert]
            state_key = f"drift_candidates:{asset}:{version}"
            previous = set(db.get_kv(state_key, []) or [])
            drifted = sorted([f for f in candidates if report[f]["psi"] >= 2 * psi_alert or f in previous],
                             key=lambda f: -report[f]["psi"])
            pending = sorted(f for f in candidates if f not in drifted)
            if record:
                db.set_kv(state_key, candidates)
            warned = [f for f, v in report.items() if pol["drift"]["psi_warn"] <= v["psi"] < psi_alert]
            checks["drift"] = {"status": "breach" if drifted else "pending" if pending else "ok",
                               "max_psi": round(max_psi, 4), "drifted_features": drifted,
                               "pending_features": pending, "warning_features": warned, "features": report}
            if drifted:
                feeds = sorted({self._feed(spec, f) for f in drifted})
                findings.append(Finding(asset, "drift", "warning", "max_psi", round(max_psi, 4),
                                        pol["drift"]["psi_alert"],
                                        f"input drift on {', '.join(drifted[:4])} (max PSI {max_psi:.2f})",
                                        {"drifted_features": {f: report[f] for f in drifted}}, feeds))
            self._metric(asset, "max_psi", max_psi)
        else:
            checks["drift"] = {"status": "insufficient", "records": len(valid)}

        # performance and concept drift (labelled passenger stream, current production version)
        labelled = recent_predictions(db, asset, window, version=version, labelled_only=True, since=floor,
                                      sources=PASSENGER_SOURCES)
        if len(labelled) >= pol["min_labels"]:
            chrono = list(reversed(labelled))
            outputs = [r["output"] for r in chrono]
            labels = [r["label"] for r in chrono]
            segments = [r["segment"] for r in chrono]
            m = spec.metrics(outputs, labels, segments)
            metric = pol["performance"]["metric"]
            base = (prod["metrics"] or {}).get(metric)
            cur = m.get(metric)
            perf = {"status": "ok", "metric": metric, "current": cur, "baseline": base, "labels": len(labelled),
                    "metrics": m, "version": version}
            if cur is not None and base:
                degr = (base - cur) / base if spec.higher_is_better else (cur - base) / base
                perf["relative_degradation"] = round(degr, 4)
                errors = self._error_stream(spec, outputs, labels, prod["metrics"])
                ph = page_hinkley(errors)
                perf["page_hinkley"] = ph
                if ph.get("drift"):
                    cp = change_point(errors)
                    if cp is not None:
                        perf["change_ts"] = chrono[cp]["ts"]
                        ph["change_point"] = cp
                if degr > pol["performance"]["max_relative_degradation"]:
                    perf["status"] = "breach"
                    findings.append(Finding(asset, "performance", "critical", metric, cur, base,
                                            f"{metric} {cur} vs baseline {base} ({degr:+.1%} worse)",
                                            {"relative_degradation": round(degr, 4), "page_hinkley": ph,
                                             "change_ts": perf.get("change_ts"), "labels": len(labelled)}))
            checks["performance"] = perf
            self._metric(asset, metric, cur)
            self._fairness_checks(asset, spec, pol, m, prod, findings, checks, record)
        else:
            checks["performance"] = {"status": "insufficient", "labels": len(labelled), "version": version}

        # shadow comparison
        shadow = self.p.registry.shadow(asset)
        if shadow:
            checks["shadow"] = self.p.lifecycle.compare_shadow(asset, prod, shadow)
        return {"asset": asset, "kind": "ml", "version": version, "checks": checks, "findings": findings}

    def _fairness_checks(self, asset, spec, pol, m, prod, findings, checks, record: bool = False) -> None:
        fair = pol.get("fairness", {})
        if fair.get("enabled"):
            rates, counts = self._segment_rates(m)
            eligible = {g: r for g, r in rates.items() if counts.get(g, 0) >= MIN_FAIRNESS_GROUP}
            # Parity over groups large enough to judge; for an error rate (FPR) compare the complement.
            ratio = disparity_ratio({g: 1 - r for g, r in eligible.items()} if fair.get("metric") == "fpr"
                                    else eligible)
            base_ratio = (prod["metrics"] or {}).get("fairness_ratio")
            status = "ok" if ratio is not None else "insufficient"
            reasons = []
            if ratio is not None and ratio < fair.get("min_ratio", 0.8):
                reasons.append(f"parity ratio {ratio:.2f} below {fair['min_ratio']}")
            if ratio is not None and base_ratio and "max_drop_vs_baseline" in fair:
                drop = (base_ratio - ratio) / base_ratio
                if drop > fair["max_drop_vs_baseline"]:
                    reasons.append(f"parity fell {drop:.0%} from validated baseline {base_ratio:.2f}")
            sig = self._disparity_significant({g: rates[g] for g in eligible}, counts)
            note = None
            if reasons and not sig["significant"]:
                reasons, note = [], f"gap not statistically significant (adjusted p={sig['p_value']})"
            # A significant gap must persist into the next supervisor cycle: per-group rates in one window
            # are noisy, and a release that suddenly biases outcomes is caught at once by the exposure check.
            state_key = f"fairness_candidate:{asset}:{prod['version']}"
            previous = bool(self.p.db.get_kv(state_key, False))
            if record:
                self.p.db.set_kv(state_key, bool(reasons))
            if reasons and previous:
                status = "breach"
                findings.append(Finding(asset, "fairness", "warning", "fairness_ratio", ratio,
                                        fair.get("min_ratio"), "; ".join(reasons),
                                        {"by_segment": rates, "n_by_segment": counts, "baseline_ratio": base_ratio,
                                         "significance": sig}))
            elif reasons:
                status, note = "pending", "significant gap seen once; confirming on the next cycle"
            checks["fairness"] = {"status": status, "ratio": ratio, "baseline_ratio": base_ratio,
                                  "metric": fair.get("metric"), "groups_judged": sorted(eligible),
                                  "significance": sig, "note": note}
            self._metric(asset, "fairness_ratio", ratio)
        if "exposure" in pol:
            exp = m.get("premium_exposure_ratio")
            limit = pol["exposure"]["max_premium_exposure_ratio"]
            st = "insufficient" if exp is None else "breach" if exp > limit else "ok"
            checks["exposure"] = {"status": st, "premium_exposure_ratio": exp, "limit": limit}
            if st == "breach":
                findings.append(Finding(asset, "exposure_bias", "critical", "premium_exposure_ratio", exp, limit,
                                        f"premium trains are recommended {exp:.2f}x their share of options",
                                        {"premium_exposure_ratio": exp}))
            self._metric(asset, "premium_exposure_ratio", exp)
        if "false_positives" in pol:
            fpr = m.get("fpr")
            limit = pol["false_positives"]["max_fpr"]
            # A window holds a few hundred genuine sessions, so an observed rate a little above the limit is often
            # noise. A breach needs the false positives to be significantly more than the limit allows
            # (one-sided binomial test).
            genuine = int(m.get("n", 0)) - int(m.get("fraud_n", 0))
            fp_count = int(m.get("false_positives") or 0)
            p_value = fpr_excess_p_value(fp_count, genuine, limit) if fpr is not None else None
            st = ("insufficient" if fpr is None else
                  "breach" if fpr > limit and p_value < FPR_ALPHA else "ok")
            note = (f"above the limit in this window but not significant (p={p_value:.3g})"
                    if st == "ok" and fpr is not None and fpr > limit else None)
            checks["false_positives"] = {"status": st, "fpr": fpr, "limit": limit, "p_value": p_value,
                                         "genuine_sessions": genuine, "note": note,
                                         "false_positives": m.get("false_positives"),
                                         "false_negatives": m.get("false_negatives")}
            if st == "breach":
                findings.append(Finding(asset, "false_positives", "critical", "fpr", fpr, limit,
                                        f"{fpr:.1%} of genuine passengers are being blocked",
                                        {"false_positives": m.get("false_positives")}))
            self._metric(asset, "fpr", fpr)

    @staticmethod
    def _segment_rates(m: dict) -> tuple[dict, dict]:
        rates = next((v for k, v in m.items() if k.endswith("_by_segment") and k != "n_by_segment"
                      and isinstance(v, dict)), None) or {}
        return rates, m.get("n_by_segment") or {}

    @staticmethod
    def _disparity_significant(rates: dict, counts: dict, alpha: float = 0.01) -> dict:
        """Is any group's rate different from all other groups pooled? Two-proportion z-test per group,
        Bonferroni-adjusted for the number of groups, so picking the most extreme group of many does not
        manufacture a significant gap."""
        from scipy.stats import norm

        groups = [g for g in rates if counts.get(g, 0) >= 2]
        if len(groups) < 2:
            return {"significant": False, "p_value": None}
        best = (1.0, None)
        for g in groups:
            n1, p1 = counts[g], rates[g]
            others = [h for h in groups if h != g]
            n2 = sum(counts[h] for h in others)
            p2 = sum(rates[h] * counts[h] for h in others) / n2
            pooled = (p1 * n1 + p2 * n2) / (n1 + n2)
            se = (pooled * (1 - pooled) * (1 / n1 + 1 / n2)) ** 0.5
            pval = float(2 * (1 - norm.cdf(abs(p1 - p2) / se))) if se > 0 else (0.0 if p1 != p2 else 1.0)
            if pval < best[0]:
                best = (pval, g)
        adjusted = min(1.0, best[0] * len(groups))
        return {"significant": adjusted < alpha, "p_value": round(adjusted, 5), "most_different_group": best[1],
                "groups": len(groups)}

    @staticmethod
    def _error_stream(spec, outputs, labels, base_metrics) -> list[float]:
        name = spec.name
        if name == "confirmation_predictor":
            return [float(o["predicted_final_status"] != y) for o, y in zip(outputs, labels)]
        if name in ("delay_predictor", "fare_forecaster"):
            key = "arrival_delay_min" if name == "delay_predictor" else "forecast_occupancy"
            mae = max(float(base_metrics.get("mae") or 1.0), 1e-6)
            return [float(abs(o[key] - float(y)) > 2 * mae) for o, y in zip(outputs, labels)]
        if name == "fraud_detector":
            return [float((o["decision"] == "block_and_review") != (y != "legit")) for o, y in zip(outputs, labels)]
        if name == "train_recommender":
            return [float(o["ranking"][0] != int(y)) if o["ranking"] else 1.0 for o, y in zip(outputs, labels)]
        return []

    @staticmethod
    def _feed(spec, feature: str) -> str:
        feed = spec.feed_of(feature)
        if feed == "unknown" and spec.name == "train_recommender":
            return "user_profile_store"
        return feed

    # ------------------------------------------------------------ generative assistant
    def _support(self, asset: str) -> dict:
        pol = self.p.policies.for_asset(asset)
        db = self.p.db
        floor = window_floor(db, asset)
        version = self.p.registry.production(asset)["version"]
        findings: list[Finding] = []
        checks: dict[str, dict] = {}
        rows = db.query("SELECT * FROM llm_interactions WHERE service=? AND version=? AND ts>=? "
                        "ORDER BY ts DESC LIMIT ?", (asset, version, floor, int(pol["window"])))
        lat = [r["latency_ms"] for r in rows]
        checks["health"] = {"status": "ok" if rows else "insufficient", "requests": len(rows),
                            "latency_p95_ms": _pct(lat, 95)}
        if len(rows) < pol["min_samples"]:
            checks["hallucination"] = {"status": "insufficient", "interactions": len(rows), "version": version}
            return {"asset": asset, "kind": "generative", "version": version, "checks": checks, "findings": findings}
        hall = float(np.mean([r["hallucination"] for r in rows]))
        grounding = [loads(r["grounding"], {}) for r in rows]
        g_mean = float(np.mean([g.get("grounding_score", 0) for g in grounding]))
        rated = [r["feedback"] for r in rows if r["feedback"] is not None]
        csat = float(np.mean([f > 0 for f in rated])) if rated else None
        quality = float(np.mean([r["quality"] for r in rows]))
        h_pol = pol["hallucination"]
        checks["hallucination"] = {"status": "breach" if hall > h_pol["max_rate"] else "ok", "rate": round(hall, 4),
                                   "limit": h_pol["max_rate"], "interactions": len(rows), "version": version,
                                   "examples": [g.get("unsupported_numbers") or g.get("unsupported_sentences")
                                                for g in grounding if g.get("hallucination")][:3]}
        if hall > h_pol["max_rate"]:
            findings.append(Finding(asset, "hallucination", "critical", "hallucination_rate", round(hall, 4),
                                    h_pol["max_rate"], f"{hall:.1%} of answers contain unsupported claims",
                                    {"examples": checks["hallucination"]["examples"]}, ["prompt_config"]))
        q_pol = pol["quality"]
        checks["quality"] = {"status": "ok", "grounding_mean": round(g_mean, 3), "quality_mean": round(quality, 3),
                             "csat": None if csat is None else round(csat, 3), "rated": len(rated)}
        if g_mean < q_pol["min_grounding"]:
            checks["quality"]["status"] = "breach"
            findings.append(Finding(asset, "quality", "warning", "grounding_mean", round(g_mean, 3),
                                    q_pol["min_grounding"], f"mean grounding score {g_mean:.2f} is below target"))
        if csat is not None and len(rated) >= 15 and csat < q_pol["min_csat"]:
            checks["quality"]["status"] = "breach"
            findings.append(Finding(asset, "csat", "warning", "csat", round(csat, 3), q_pol["min_csat"],
                                    f"customer satisfaction {csat:.0%} below target"))
        ref = db.get_kv("support_intent_reference")
        if ref:
            cur = pd.Series([r["intent"] for r in rows])
            ref_s = pd.Series([k for k, v in ref.items() for _ in range(int(v))])
            psi = psi_categorical(ref_s, cur)
            checks["user_behaviour"] = {"status": "ok" if psi < 0.25 else "breach", "intent_psi": round(psi, 4)}
            if psi >= 0.25:
                findings.append(Finding(asset, "user_behaviour_change", "info", "intent_psi", round(psi, 4), 0.25,
                                        "passengers are asking different kinds of questions than at launch"))
            self._metric(asset, "intent_psi", psi)
        for name, value in (("hallucination_rate", hall), ("grounding_mean", g_mean), ("csat", csat)):
            self._metric(asset, name, value)
        return {"asset": asset, "kind": "generative", "version": version, "checks": checks, "findings": findings}

    # ------------------------------------------------------------ planner agent
    def _agent(self, asset: str) -> dict:
        pol = self.p.policies.for_asset(asset)
        a_pol = pol["agent"]
        db = self.p.db
        floor = window_floor(db, asset)
        version = self.p.registry.production(asset)["version"]
        findings: list[Finding] = []
        rows = db.query("SELECT * FROM agent_traces WHERE ts>=? ORDER BY ts DESC LIMIT ?", (floor, int(pol["window"])))
        checks: dict[str, dict] = {"health": {"status": "ok" if rows else "insufficient", "sessions": len(rows),
                                              "latency_p95_ms": _pct([r["latency_ms"] for r in rows], 95)}}
        if len(rows) < pol["min_samples"]:
            checks["agent"] = {"status": "insufficient", "sessions": len(rows)}
            return {"asset": asset, "kind": "agent", "version": version, "checks": checks, "findings": findings}
        verdicts = [loads(r["verdict"], {}) for r in rows]
        plans = [loads(r["plan"], {}) for r in rows]
        steps = [loads(r["steps"], []) for r in rows]
        proposed = [v for v, p in zip(verdicts, plans) if p.get("status") == "proposed"]
        violation_rate = float(np.mean([not v["valid"] for v in proposed])) if proposed else 0.0
        abnormal_rate = float(np.mean([v["abnormal"] for v in verdicts]))
        total_steps = sum(len(s) for s in steps)
        tool_errors = sum(1 for s in steps for st in s if not st.get("ok"))
        error_rate = tool_errors / total_steps if total_steps else 0.0
        mean_steps = total_steps / len(rows)
        success = float(np.mean([p.get("status") == "proposed" for p in plans]))
        a = {"status": "ok", "sessions": len(rows), "violation_rate": round(violation_rate, 4),
             "abnormal_rate": round(abnormal_rate, 4), "tool_error_rate": round(error_rate, 4),
             "mean_steps": round(mean_steps, 2), "success_rate": round(success, 4),
             "examples": [v["violations"] or v["abnormal_reasons"] for v in verdicts
                          if v.get("violations") or v.get("abnormal_reasons")][:3]}
        if violation_rate > a_pol["max_violation_rate"]:
            findings.append(Finding(asset, "agent_violation", "critical", "violation_rate", round(violation_rate, 4),
                                    a_pol["max_violation_rate"],
                                    f"{violation_rate:.1%} of proposals broke a passenger constraint",
                                    {"examples": a["examples"]}))
        if error_rate > a_pol["max_tool_error_rate"]:
            findings.append(Finding(asset, "tool_errors", "critical", "tool_error_rate", round(error_rate, 4),
                                    a_pol["max_tool_error_rate"], f"{error_rate:.1%} of agent tool calls failed",
                                    {"failed_tools": self._failed_tools(steps)}, ["timetable_search"]))
        if abnormal_rate > a_pol["max_abnormal_rate"] or mean_steps > a_pol["max_mean_steps"]:
            findings.append(Finding(asset, "agent_abnormal", "warning", "abnormal_rate", round(abnormal_rate, 4),
                                    a_pol["max_abnormal_rate"],
                                    f"{abnormal_rate:.1%} of sessions behaved abnormally (mean {mean_steps:.1f} steps)",
                                    {"examples": a["examples"], "failed_tools": self._failed_tools(steps)}))
        if findings:
            a["status"] = "breach"
        checks["agent"] = a
        for name, value in (("violation_rate", violation_rate), ("abnormal_rate", abnormal_rate),
                            ("tool_error_rate", error_rate), ("success_rate", success)):
            self._metric(asset, name, value)
        return {"asset": asset, "kind": "agent", "version": version, "checks": checks, "findings": findings}

    @staticmethod
    def _failed_tools(steps: list[list[dict]]) -> dict[str, int]:
        out: dict[str, int] = {}
        for s in steps:
            for st in s:
                if not st.get("ok"):
                    out[st["tool"]] = out.get(st["tool"], 0) + 1
        return out
